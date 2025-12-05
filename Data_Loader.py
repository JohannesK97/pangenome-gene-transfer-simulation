import os
import random
import h5py
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from typing import List, Tuple, Dict, Any



def one_hot_encode(sequences, gene_present, max_number_of_snps, alphabet=['A','C','T','G','-']):
    """
    sequences: List of strings (DNA sequences)
    gene_present: np.array(bool) oder Torch Tensor, gleiche Länge wie sequences
    max_number_of_snps: int, fixe Länge für das Hot-Encoding
    alphabet: list, Zeichenalphabet
    """
    num_samples = len(sequences)
    num_chars = len(alphabet)
    char_to_idx = {c:i for i,c in enumerate(alphabet)}
    sequences_str = [s.decode('utf-8') for s in sequences]
    gene_present = np.array(gene_present, dtype=bool)
    
    # 1️⃣ Leere Batch-Matrix vorbereiten: (num_samples, max_number_of_snps, num_chars)
    batch = np.zeros((num_samples, max_number_of_snps, num_chars), dtype=np.float32)
    
    # 2️⃣ Hot-Encode alle Sequenzen
    for i, seq in enumerate(sequences_str):
        if gene_present[i]:
            L = min(len(seq), max_number_of_snps)  # abschneiden
            for j, c in enumerate(seq[:L]):
                if c in char_to_idx:
                    batch[i, j, char_to_idx[c]] = 1.0
        elif gene_present[i] == 0:
            batch[i, :, :] = -1.0
            
    # 3️⃣ Zufällige, aber konsistente Spaltenpermutation
    perm = np.random.permutation(max_number_of_snps)
    batch = batch[:, perm, :]
    
    # 4️⃣ Optional: Flatten zu Vektor (num_samples, max_number_of_snps*num_chars)
    batch_flat = batch.reshape(num_samples, -1)
    
    return torch.tensor(batch_flat)  # shape: (num_samples, max_number_of_snps*num_chars)


def load_file(file, max_number_of_snps = 300):
    
    with h5py.File(file, "r") as f:
            grp = f["results"]
            # Load graph_properties (pickle stored in dataset)
            graph_properties = pickle.loads(grp["graph_properties"][()])
    
            # Unpack graph properties
            nodes = torch.tensor(graph_properties[0])                # [num_nodes]
            edges = torch.tensor(graph_properties[1], dtype=torch.long)  # [2, num_edges]
            coords = torch.tensor(graph_properties[2].T)             # [2, num_nodes]
    
            gene_absence_presence_matrix = grp["gene_absence_presence_matrix"][()]
            nucleotide_sequences = grp["nucleotide_sequences"][()]
            #children_gene_nodes_loss_events = grp["children_gene_nodes_loss_events"][()]

            if gene_absence_presence_matrix.ndim == 2:
                raise ValueError(
                    f"Mehrere Gene gefunden (Matrix-Shape: {gene_absence_presence_matrix.shape}). "
                    "Dieses Skript ist nur für ein einzelnes Gen ausgelegt."
                )

        
            ##### Construct the graph
        
            G = nx.DiGraph()
            
            ### Füge Knoten hinzu (mit zugehörigen Zeiten)
            node_id_list = nodes.tolist()
            sorted_indices = sorted(range(len(node_id_list)), key=lambda i: node_id_list[i])
            sorted_node_ids = [node_id_list[i] for i in sorted_indices]
            
            for new_i, orig_i in enumerate(sorted_indices):
                node_id = node_id_list[orig_i]
                G.add_node(node_id, node_time=coords[:, orig_i].tolist()[5])

            
            ### Füge Kanten hinzu
            edge_list = edges.tolist()
            for src, dst in zip(edge_list[0], edge_list[1]):
                G.add_edge(src, dst)

            ### Tiefe der Nodes: Blätter habe Tiefe 0
            level = {n: 0 for n in G.nodes}
            for node in reversed(list(nx.topological_sort(G))):
                successors = list(G.successors(node))
                if successors:
                    level[node] = 1 + max(level[s] for s in successors)
            nx.set_node_attributes(G, level, "level")

            """
            ### Füge gene presence hinzu
            
            gene_presence = {}

            for i, node in enumerate(G.nodes()):
                if i < len(gene_absence_presence_matrix):
                    gene_presence[node] = gene_absence_presence_matrix[i]
                else:
                    gene_presence[node] = 1
            
            nx.set_node_attributes(G, gene_presence, "gene_presence")
            """
        
            ### Integriere HGT information

            # Initialisierung:
            default_hgt_events = {
                node: {
                    "recipient": {"is_parent_node": False, "events": []},
                    "donor":      {"is_child_node":  False, "events": []}
                }
                for node in G.nodes()
            }
            nx.set_node_attributes(G, default_hgt_events)
        
            hgt_events = {}
            hgt_grp_simpl = grp.get("nodes_hgt_events_simplified", None)
            if hgt_grp_simpl is not None:
                for site_id in hgt_grp_simpl.keys():
                    hgt_events[int(site_id)] = hgt_grp_simpl[site_id][()]
            else:
                hgt_events = {}

            if hgt_events:
                for event in hgt_events[0]:
                    recipient_parent_node = int(event['recipient_parent_node'])
                    recipient_child_node  = int(event['recipient_child_node'])
                    donor_parent_node     = int(event['donor_parent_node'])
                    donor_child_node      = int(event['donor_child_node'])
        
                    hgt_event = ((recipient_parent_node, recipient_child_node),
                                 (donor_parent_node, donor_child_node))
        
                    # recipient parent: Flag setzen und Event anhängen
                    G.nodes[recipient_parent_node]["recipient"]["is_parent_node"] = True
                    G.nodes[recipient_parent_node]["recipient"]["events"].append(hgt_event)
        
                    # donor child: Flag setzen und Event anhängen
                    G.nodes[donor_child_node]["donor"]["is_child_node"] = True
                    G.nodes[donor_child_node]["donor"]["events"].append(hgt_event)
                


            ### Add hot encoded nucleotide sequences
        
            hot_encoded_nucleotide_sequences = one_hot_encode(nucleotide_sequences, gene_absence_presence_matrix, max_number_of_snps)
            pad_rows = len(nodes) - len(nucleotide_sequences) # Fill the remaining nodes with zeros.
            pad = torch.zeros((pad_rows, hot_encoded_nucleotide_sequences.shape[1]), dtype=hot_encoded_nucleotide_sequences.dtype)
            sequences = torch.cat([hot_encoded_nucleotide_sequences, pad], dim=0) # Dimension: [num_nodes, 5 * max_number_of_snps]
            sequences_dict = {node: sequences[i] for i, node in enumerate(G.nodes())}
            nx.set_node_attributes(G, sequences_dict, "sequences")

            ### Add a flag for nodes where the sequences have to be processed and the information how many leaves are below this node 
            ### and how many of them have the gene present.

            for node in G.nodes():
                children = list(G.successors(node))
                is_leaf = len(children) == 0
            
                # Update-Flag: True, falls der Knoten Kinder hat
                G.nodes[node]['update_needed'] = not is_leaf
            
                if is_leaf:  # Leaf node
                    G.nodes[node]['num_leaves_below'] = 1
                    G.nodes[node]['num_leaves_below_gene_present'] = int(gene_absence_presence_matrix[node])
                else:
                    # Summe über Kinder
                    G.nodes[node]['num_leaves_below'] = sum(G.nodes[child]['num_leaves_below'] for child in children)
                    G.nodes[node]['num_leaves_below_gene_present'] = sum(G.nodes[child]['num_leaves_below_gene_present'] for child in children)

    return G


def aggregate_sequences(G: nx.DiGraph, recalc_all: bool = False, device: str = 'cpu') -> nx.DiGraph:
    """
    Aggregiere die Node-Sequenzen von den Blättern nach oben.
    Jeder interne Knoten erhält das elementweise Maximum seiner Kindersequenzen.

    Args:
        G: gerichteter Baum (networkx.DiGraph), erwartet Node-Attribute:
           - 'sequences': 1D-Tensor (torch.Tensor) pro Node, gleiche Länge für alle Nodes.
           - 'level': int
           - 'update_needed': bool (optional)
        recalc_all: bool, True -> berechne alle internen Knoten neu,
                          False -> nur Knoten mit update_needed == True
        device: 'cpu' oder 'cuda' (wenn verfügbar)

    Returns:
        num_updated: Anzahl der Knoten, deren 'sequences' neu berechnet wurden.
    """
    # Liste Nodes in stabiler Reihenfolge
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Hole die Länge der Sequenzen
    seq_len = G.nodes[nodes[0]].get('sequences', None).shape[0]

    # Baue einen großen Tensor mit allen Sequenzen (für vektorisierte Operationen)
    # Achtung: wir kopieren hier Daten; das ist absichtlich für Batch-Operationen.
    seqs_all = torch.stack([G.nodes[n]['sequences'].to(device) for n in nodes], dim=0)  # shape: [num_nodes, seq_len]

    # Gruppiere Knoten nach level
    levels = nx.get_node_attributes(G, 'level')
    max_level = max(levels.values())

    # Erzeuge Kind-Index-Listen: children[node_idx] -> List[int]
    children = [[] for _ in range(num_nodes)]
    for parent in nodes:
        for child in G.successors(parent):
            children[parent].append(child)

    # Prozessiere level-weise von niedrigstem Level (Blätter, level=0) nach oben: wir brauchen
    # nur interne Knoten, also level>=1. Für Aggregation gehen wir von den Kindern herauf, damit die
    # Kinder bereits final sind, wenn wir einen Parent berechnen.
    for lvl in range(1, max_level + 1):
        # Nodes auf diesem level in der selben Reihenfolge wie nodes-liste
        nodes_on_level = [n for n in nodes if levels[n] == lvl]
        if not nodes_on_level:
            raise ValueError(
                f"Level {lvl.shape} is without nodes."
            )

        # Entscheide, welche Knoten tatsächlich neu berechnet werden sollen
        if recalc_all:
            to_recalc = nodes_on_level
        else:
            to_recalc = []
            for node in nodes_on_level:
                if bool(G.nodes[node].get('update_needed', True)):
                    to_recalc.append(node)

        # Vectorized: baue zwei index-Arrays für Kind0 und Kind1
        child0_idxs = [children[i][0] for i in to_recalc]
        child1_idxs = [children[i][1] for i in to_recalc]

        child0 = seqs_all[torch.tensor(child0_idxs, dtype=torch.long, device=device)]  # [batch, seq_len]
        child1 = seqs_all[torch.tensor(child1_idxs, dtype=torch.long, device=device)]

        # elementweises Maximum über die beiden Kinder
        new_seqs = torch.maximum(child0, child1)  # [batch, seq_len]

        # Schreibe die Ergebnisse zurück in seqs_all (und in den Graph)
        for out_pos, node in enumerate(to_recalc):
            seqs_all[node] = new_seqs[out_pos]
            G.nodes[node]['sequences'] = new_seqs[out_pos].detach().cpu()
            # update flag zurücksetzen
            G.nodes[node]['update_needed'] = False

    return G

def graph_to_dataset(G: nx.DiGraph):

    nodes = list(G.nodes())
    edges = list(G.edges())
    
    if len(nodes) == 0:
        raise ValueError(
            f"Graph has no nodes."
        )
    elif len(edges) == 0:
        raise ValueError(
            f"Graph has no edges."
        )
    elif nodes != list(range(max(nodes) + 1)):
        raise ValueError(
            f"Nodes are in wrong order."
        )

    # Feature-Matrix: stack der 'sequences' Tensoren
    x = torch.stack([G.nodes[n]['sequences'].float() for n in nodes], dim=0)  # [num_nodes, feat_dim]
    sum_x = x.sum(dim=1, keepdim=True)

    # Edge Index (directed): 
    edge_index = torch.tensor([[child for parent, child in edges], [parent for parent, child in edges]], dtype=torch.long)

    num_leaves = torch.tensor([G.nodes[n]['num_leaves_below'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    num_present = torch.tensor([G.nodes[n]['num_leaves_below_gene_present'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    node_time = torch.tensor([G.nodes[n]['node_time'] for n in nodes], dtype=torch.float32).unsqueeze(1)

    y = torch.tensor(
        [1.0 if (G.nodes[n].get('recipient',{}).get('is_parent_node', False)) else 0.0 for n in nodes],
        dtype=torch.float32
    )

    x = torch.cat([sum_x, num_leaves, num_present, node_time, x], dim=1)
    #x = torch.cat([sum_x, num_leaves, num_present, node_time], dim=1)

    data = Data(x = x, edge_index = edge_index, y = y)

    return data


def expected_and_observed_nucleotide_variants(hot_encoded_nucleotide_sequence_parent, hot_encoded_nucleotide_sequence_child_1, hot_encoded_nucleotide_sequence_child_2, time_parent, time_child_1, time_child_2):

    nucleotide_mutation_rate = 0.1
    
    time = 2 * time_parent - time_child_1 - time_child_2
    print("time", time)
    gene_length = hot_encoded_nucleotide_sequence_child_1.shape[1]

    hot_encoded_nucleotide_sequence_parent =  torch.maximum(
        hot_encoded_nucleotide_sequence_parent, torch.tensor(0, dtype=torch.float32, device=hot_encoded_nucleotide_sequence_parent.device)
    )
    hot_encoded_nucleotide_sequence_child_1 = torch.maximum(
        hot_encoded_nucleotide_sequence_child_1, torch.tensor(0, dtype=torch.float32, device=hot_encoded_nucleotide_sequence_child_1.device)
    )
    hot_encoded_nucleotide_sequence_child_2 = torch.maximum(
        hot_encoded_nucleotide_sequence_child_2, torch.tensor(0, dtype=torch.float32, device=hot_encoded_nucleotide_sequence_child_2.device)
    )

    #nucleotide_variants_child_1 = hot_encoded_nucleotide_sequence_child_1.sum(dim=1)
    #nucleotide_variants_child_2 = hot_encoded_nucleotide_sequence_child_2.sum(dim=1)

    #hot_encoded_nucleotide_sequence_combined_children = min(hot_encoded_nucleotide_sequence_child_1 + hot_encoded_nucleotide_sequence_child_2, 1)

    #hot_encoded_nucleotide_sequence_combined_children = torch.minimum(
    #    hot_encoded_nucleotide_sequence_child_1 + hot_encoded_nucleotide_sequence_child_2,
    #    torch.tensor(1, dtype=torch.float32, device=hot_encoded_nucleotide_sequence_child_1.device)
    #)
    
    # Expected amount of new nucleotide variants (not regarding immediate back mutations):

    possible_new_nucleotide_variants = gene_length * 5 - hot_encoded_nucleotide_sequence_parent.sum(dim=1)

    expected_new_nucleotide_variants = possible_new_nucleotide_variants * (1 - torch.exp(-nucleotide_mutation_rate / 5 * time))
    
    # Observed nucleotide variants:

    observed_new_nucleotide_variants = 1 / 2 * ((hot_encoded_nucleotide_sequence_parent - hot_encoded_nucleotide_sequence_child_1).sum(dim=1) + (hot_encoded_nucleotide_sequence_parent - hot_encoded_nucleotide_sequence_child_2).sum(dim=1))
    
    #(hot_encoded_nucleotide_sequence_parent - hot_encoded_nucleotide_sequence_combined_children).sum(dim=1) #- nucleotide_variants_child_1 - nucleotide_variants_child_2 + gene_length

    print(expected_new_nucleotide_variants, observed_new_nucleotide_variants)

    return expected_new_nucleotide_variants, observed_new_nucleotide_variants


class ParentChildFusionLayer(MessagePassing):
    """
    A MessagePassing layer designed for tree-like graphs where each parent
    has either exactly two children or none. For each parent node i, the
    features of the parent and its two children (if present) are concatenated.

    This layer does not use any attention or permutation-invariant
    aggregation: child messages are collected explicitly and concatenated
    in a fixed order. The user must ensure that each parent node has either
    (0 or 2) incoming edges, and that the edge_index ordering encodes a
    consistent left/right child order.

    Input dimensions:
        - Node feature dimension: in_dim
        - Output feature dimension: out_dim

    Output:
        - New node embeddings of dimension out_dim
    """

    def __init__(self, in_dim):
        # We do not use built-in aggregation ("add", "mean", ...) because
        # we aggregate manually. Set aggr=None.
            
        # Each node will produce: [parent_features, child1, child2]
        # If a node has no children, child features are zero-padded.
        super().__init__(node_dim=0, aggr=None)

        self.in_dim = in_dim

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Collect exactly two child feature vectors per parent.

        inputs:  (num_edges, in_dim)
        index:   (num_edges,) target node for each edge

        Returns:
            A tensor of shape (num_nodes, 2 * in_dim) containing the two
            children features for each parent. Order is determined by
            edge ordering and should be consistent in the dataset.
        """

        # Determine number of nodes from dim_size (preferred), fall back to index
        if dim_size is not None:
            num_nodes = dim_size
        else:
            num_nodes = int(index.max().item()) + 1

        device = inputs.device

        # Preallocate storage
        children = torch.zeros(num_nodes, 2, self.in_dim, device=device)

        # Compute for each edge its "child slot" 0 or 1
        # Example: for index = [3,3,5,5], this produces [0,1,0,1]
        slot = torch.zeros_like(index)
        slot[1:] = (index[1:] == index[:-1]).long()

        # Vectorized scatter operation
        children[index, slot] = inputs

        return children.reshape(num_nodes, 2 * self.in_dim)

    def update(self, aggr_out, x):
        """
        aggr_out: (num_nodes, 2*in_dim) concatenated children features
        x:        (num_nodes, in_dim)   parent features

        Returns:
            Fused parent representation → out_dim
        """
        d_x = x.size(1)

        child_1 = aggr_out[:, :d_x]
        child_2 = aggr_out[:, d_x:]
        
        time_parent = x[:, 3]
        time_child_1 = child_1[:, 3]
        time_child_2 = child_2[:, 3]

        hot_encoded_nucleotide_sequence_parent = x[:, 4:]
        hot_encoded_nucleotide_sequence_child_1 = child_1[:, 4:]
        hot_encoded_nucleotide_sequence_child_2 = child_2[:, 4:]
        
        expected_and_observed_nucleotide_variants(hot_encoded_nucleotide_sequence_parent, hot_encoded_nucleotide_sequence_child_1, hot_encoded_nucleotide_sequence_child_2, time_parent, time_child_1, time_child_2)
        
        fused = torch.cat([x, aggr_out], dim=-1)
        return fused

    def forward(self, x, edge_index):
        """
        x: (num_nodes, in_dim)
        edge_index: (2, num_edges), where edges point child -> parent

        Returns:
            Updated node embeddings (num_nodes, out_dim)
        """
        return self.propagate(edge_index, x=x)
