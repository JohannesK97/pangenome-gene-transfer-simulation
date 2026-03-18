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
from collections import defaultdict, deque



def one_hot_encode(sequences, gene_present, max_number_of_snps = 300, alphabet=['A','C','T','G','-']):
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

    """
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

    num_samples = len(sequences)
    num_chars = len(alphabet)
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    sequences_str = [s.decode('utf-8') for s in sequences]
    gene_present = np.array(gene_present, dtype=bool)
    """
    
    # 1) Leere Batch-Matrix
    batch = np.zeros((num_samples, max_number_of_snps, num_chars), dtype=np.float32)
    
    # 2) Erzeuge globalen "Füll-Nukleotid"-Index
    filler_idx = np.random.randint(0, num_chars)
    
    # 3) Hot-Encoding + Füllung
    for i, seq in enumerate(sequences_str):
        if gene_present[i]:
            L = min(len(seq), max_number_of_snps)
    
            # Encode echte SNPs
            for j, c in enumerate(seq[:L]):
                if c in char_to_idx:
                    batch[i, j, char_to_idx[c]] = 1.0
    
            # Fülle restliche Positionen mit globalem Nukleotid
            if L < max_number_of_snps:
                batch[i, L:, filler_idx] = 1.0
    
        else:
            # Gen fehlt komplett → Missing-Token
            batch[i, :, :] = 0 #-1.0

    
    # 3️⃣ Zufällige, aber konsistente Spaltenpermutation
    perm = np.random.permutation(max_number_of_snps)
    batch = batch[:, perm, :]
    
    # 4️⃣ Optional: Flatten zu Vektor (num_samples, max_number_of_snps*num_chars)
    batch_flat = batch.reshape(num_samples, max_number_of_snps * num_chars)
    
    return torch.from_numpy(batch_flat)  # shape: (num_samples, max_number_of_snps*num_chars)


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
                G.add_node(node_id, time = coords[:, orig_i].tolist()[5], has_hgt_later_in_time = coords[:, orig_i].tolist()[6])

            
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

                # Set a tree length variable
                G.nodes[node]['tree_length'] = 0
            
                # Update-Flag: True, falls der Knoten Kinder hat
                G.nodes[node]['update_needed'] = not is_leaf
            
                if is_leaf:  # Leaf node
                    G.nodes[node]['num_leaves_below'] = 1
                    G.nodes[node]['num_leaves_below_gene_present'] = int(gene_absence_presence_matrix[node])
                    G.nodes[node]['gene_present_below_node'] = G.nodes[node]['num_leaves_below_gene_present'] > 0
                    G.nodes[node]['valid_node'] = G.nodes[node]['num_leaves_below_gene_present'] > 0
                    G.nodes[node]['time_only_valid_nodes'] = 0
                else: # internal nodes
                    G.nodes[node]['num_leaves_below'] = sum(G.nodes[child]['num_leaves_below'] for child in children)
                    G.nodes[node]['num_leaves_below_gene_present'] = sum(G.nodes[child]['num_leaves_below_gene_present'] for child in children)
                    G.nodes[node]['gene_present_below_node'] = G.nodes[node]['num_leaves_below_gene_present'] > 0
                    if G.nodes[children[0]]['gene_present_below_node'] == 1 and G.nodes[children[1]]['gene_present_below_node'] == 1: # valid node
                        G.nodes[node]['time_only_valid_nodes'] = G.nodes[node]['time']
                        G.nodes[node]['valid_node'] = 1
                    else:
                        G.nodes[node]['time_only_valid_nodes'] = max([G.nodes[child]['time_only_valid_nodes'] for child in children])
                        G.nodes[node]['valid_node'] = 0

            valid_nodes = [n for n in G.nodes() if G.nodes[n]['valid_node'] == 1]
            max_valid_node = max(valid_nodes)
        
            for node in list(reversed(list(G.nodes()))):
                if node >= max_valid_node:
                    G.nodes[node]['time_only_valid_nodes_from_parent'] = G.nodes[max_valid_node]['time']
                else:
                    parent = list(G.predecessors(node))[0] # only one parent except root
                    if G.nodes[node]['valid_node'] == 1: 
                        G.nodes[node]['time_only_valid_nodes_from_parent'] = G.nodes[node]['time'] 
                    else:
                        G.nodes[node]['time_only_valid_nodes_from_parent'] = G.nodes[parent]['time_only_valid_nodes_from_parent']

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

                    if G.nodes[donor_child_node].get("valid_node", 0) == 0:
                        # Find largest valid descendant of donor_child_node
                        descendants = nx.descendants(G, donor_child_node)
                        
                        # Include the node itself (optional, falls er selbst valid sein kann)
                        candidates = list(descendants) + [donor_child_node]
                        
                        valid_descendants = [
                            n for n in candidates
                            if G.nodes[n].get("valid_node", 0) == 1
                        ]
                        
                        if valid_descendants:
                            donor_child_node = max(valid_descendants)
        
                    hgt_event = ((recipient_parent_node, recipient_child_node),
                                 (donor_parent_node, donor_child_node))
        
                    # recipient parent: Flag setzen und Event anhängen
                    G.nodes[recipient_parent_node]["recipient"]["is_parent_node"] = True
                    G.nodes[recipient_parent_node]["recipient"]["events"].append(hgt_event)
        
                    # donor child: Flag setzen und Event anhängen
                    G.nodes[donor_child_node]["donor"]["is_child_node"] = True
                    G.nodes[donor_child_node]["donor"]["events"].append(hgt_event)
                

    return G


def aggregate_sequences(G: nx.DiGraph, recalc_all: bool = False, device: str = 'cpu', max_number_of_snps = 300) -> nx.DiGraph:
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
    tree_lengths_all = torch.tensor([G.nodes[n]['tree_length'] for n in nodes], dtype=torch.float, device=device)
    time_all = torch.tensor([G.nodes[n]['time'] for n in nodes], dtype=torch.float, device=device)
    gene_present_under_node = torch.tensor([G.nodes[n]['gene_present_below_node'] for n in nodes], dtype=torch.float, device=device)

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

        # Compute the tree lengths:
        child0_tree_length = tree_lengths_all[torch.tensor(child0_idxs, dtype=torch.long, device=device)]
        child1_tree_length = tree_lengths_all[torch.tensor(child1_idxs, dtype=torch.long, device=device)]

        child0_time = time_all[torch.tensor(child0_idxs, dtype=torch.long, device=device)]
        child1_time = time_all[torch.tensor(child1_idxs, dtype=torch.long, device=device)]
        parent_time = time_all[torch.tensor(to_recalc, dtype=torch.long, device=device)]
        child0_gene_present = gene_present_under_node[torch.tensor(child0_idxs, dtype=torch.long, device=device)]
        child1_gene_present = gene_present_under_node[torch.tensor(child1_idxs, dtype=torch.long, device=device)]
        
        tree_lengths_all[to_recalc] = child0_tree_length + child1_tree_length + (parent_time - child0_time) * child0_gene_present + (parent_time - child1_time) * child1_gene_present

        sum_seq = new_seqs.sum(dim=1, keepdim=True) - max_number_of_snps
        
        # Schreibe die Ergebnisse zurück in seqs_all (und in den Graph)
        for out_pos, node in enumerate(to_recalc):
            seqs_all[node] = new_seqs[out_pos]
            G.nodes[node]['sequences'] = new_seqs[out_pos].detach().cpu()
            G.nodes[node]['sum_seq'] = sum_seq[out_pos].detach().cpu()
            G.nodes[node]['tree_length'] = tree_lengths_all[node]
            # update flag zurücksetzen
            G.nodes[node]['update_needed'] = False

    return G

def aggregate_sequences_fitch(G: nx.DiGraph, device: str = "cpu", max_number_of_snps: int = 300, alphabet_size: int = 5):
    """
    Aggregiert Sequenzen mit dem Fitch-Parsimony-Algorithmus.
    Jeder interne Knoten erhält die mögliche Zustandsmenge pro SNP.

    Erwartet:
        G.nodes[n]['sequences'] -> flattened one-hot tensor
                                   shape = max_number_of_snps * alphabet_size
    """

    nodes = list(G.nodes())
    num_nodes = len(nodes)

    seq_len = max_number_of_snps * alphabet_size

    # Stack aller Sequenzen
    seqs_all = torch.stack(
        [G.nodes[n]['sequences'].to(device) for n in nodes], dim=0
    )

    # reshape → [num_nodes, snps, alphabet]
    seqs_all = seqs_all.view(num_nodes, max_number_of_snps, alphabet_size)

    tree_lengths_all = torch.tensor(
        [G.nodes[n]['tree_length'] for n in nodes],
        dtype=torch.float,
        device=device,
    )

    time_all = torch.tensor(
        [G.nodes[n]['time'] for n in nodes],
        dtype=torch.float,
        device=device,
    )

    gene_present_under_node = torch.tensor(
        [G.nodes[n]['gene_present_below_node'] for n in nodes],
        dtype=torch.float,
        device=device,
    )

    levels = nx.get_node_attributes(G, "level")
    max_level = max(levels.values())

    children = [[] for _ in range(num_nodes)]
    for parent in nodes:
        for child in G.successors(parent):
            children[parent].append(child)

    # Bottom-up traversal
    for lvl in range(1, max_level + 1):

        nodes_on_level = [n for n in nodes if levels[n] == lvl]
        to_recalc = [n for n in nodes_on_level if G.nodes[n].get("update_needed", True)]

        if not to_recalc:
            continue

        child0_idxs = torch.tensor(
            [children[i][0] for i in to_recalc], dtype=torch.long, device=device
        )

        child1_idxs = torch.tensor(
            [children[i][1] for i in to_recalc], dtype=torch.long, device=device
        )

        child0 = seqs_all[child0_idxs]  # [batch, snps, alphabet]
        child1 = seqs_all[child1_idxs]

        # Fitch Intersection
        intersection = torch.minimum(child0, child1)

        # Prüfen ob Intersection leer ist
        intersection_empty = intersection.sum(dim=2, keepdim=True) == 0

        # Fitch Union
        union = torch.maximum(child0, child1)

        # Wenn intersection leer → union, sonst intersection
        new_seqs = torch.where(intersection_empty, union, intersection)

        # --- Tree length update (wie vorher) ---
        child0_tree_length = tree_lengths_all[child0_idxs]
        child1_tree_length = tree_lengths_all[child1_idxs]

        child0_time = time_all[child0_idxs]
        child1_time = time_all[child1_idxs]
        parent_time = time_all[torch.tensor(to_recalc, device=device)]

        child0_gene_present = gene_present_under_node[child0_idxs]
        child1_gene_present = gene_present_under_node[child1_idxs]

        tree_lengths_all[to_recalc] = (
            child0_tree_length
            + child1_tree_length
            + (parent_time - child0_time) * child0_gene_present
            + (parent_time - child1_time) * child1_gene_present
        )

        # Flatten zurück
        new_seqs_flat = new_seqs.view(len(to_recalc), seq_len)

        for out_pos, node in enumerate(to_recalc):
            seqs_all[node] = new_seqs[out_pos]

            G.nodes[node]["sequences"] = new_seqs_flat[out_pos].detach().cpu()
            G.nodes[node]["tree_length"] = tree_lengths_all[node]
            G.nodes[node]["sum_seq"] = new_seqs[out_pos].sum().detach().cpu()
            G.nodes[node]["update_needed"] = False

    return G

def RecipientFinder_graph_to_dataset(G: nx.DiGraph, max_number_of_snps = 300):

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
    level = torch.tensor([G.nodes[n]['level'] for n in nodes], dtype=torch.float32)
    sum_x = x.sum(dim=1, keepdim=True) - max_number_of_snps
    sum_x = torch.maximum(sum_x, torch.tensor(0.0, device=sum_x.device))

    # Edge Index (directed): 
    edge_index = torch.tensor([[child for parent, child in edges], [parent for parent, child in edges]], dtype=torch.long)

    num_leaves_below = torch.tensor([G.nodes[n]['num_leaves_below'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    num_leaves_below_gene_present = torch.tensor([G.nodes[n]['num_leaves_below_gene_present'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    time = torch.tensor([G.nodes[n]['time'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    tree_length = torch.tensor([G.nodes[n]['tree_length'] for n in nodes], dtype=torch.float32).unsqueeze(1) * 100
    gene_present_below_node = torch.tensor([G.nodes[n]['gene_present_below_node'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    
    has_hgt_later_in_time = torch.tensor([G.nodes[n]['has_hgt_later_in_time'] for n in nodes], dtype=torch.float32)

    y = torch.tensor(
        [1.0 if (G.nodes[n].get('recipient',{}).get('is_parent_node', False)) else 0.0 for n in nodes],
        dtype=torch.float32
    )

    #x = torch.cat([sum_x, num_leaves_below, num_leaves_below_gene_present, time, tree_length, gene_present_below_node], dim=1)
    #x = torch.cat([sum_x, time, gene_present_below_node], dim=1)
    #x = torch.cat([sum_x, tree_length], dim=1)
    #x = torch.cat([sum_x, y.unsqueeze(1)], dim=1)
    #x = torch.cat([x], dim=1)
    #internal_node_data = torch.cat([sum_x, tree_length, num_leaves_below, num_leaves_below_gene_present, gene_present_below_node, time], dim=1)
    internal_node_data = torch.cat([num_leaves_below, num_leaves_below_gene_present, gene_present_below_node, time], dim=1)

    data = Data(x = x, edge_index = edge_index, internal_node_data = internal_node_data, level = level, y = y, has_hgt_later_in_time = has_hgt_later_in_time)

    return data

def DonorFinder_graph_to_dataset(G: nx.DiGraph, recipient_node = None, max_number_of_snps = 300, p_false = None, len_alphabet = 5):

    nodes = list(G.nodes())
    nodes_tensor = torch.tensor(nodes, dtype=torch.long)
    edges = list(G.edges())
    N = len(nodes)
    
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

    if p_false == None:
        p_false = 1.0 / len(nodes_tensor)

    data = []
    
    # Feature-Matrix: stack der 'sequences' Tensoren
    x = torch.stack([G.nodes[n]['sequences'].float() for n in nodes], dim=0)  # [num_nodes, feat_dim]
    level = torch.tensor([G.nodes[n]['level'] for n in nodes], dtype=torch.float32)
    sum_x = x.sum(dim=1, keepdim=True) - max_number_of_snps
    sum_x = torch.maximum(sum_x, torch.tensor(0.0, device=sum_x.device))

    # Edge Index (directed): 
    edge_index = torch.tensor([[child for parent, child in edges], [parent for parent, child in edges]], dtype=torch.long)
    
    # Sort edge indices, such that the one with the lower id comes first. This is important for the graph convolution/fusion later on.
    child = edge_index[0]
    parent = edge_index[1]
    sort_key = parent * x.size(0) + child # Sort by parent first, then by child ID
    perm = torch.argsort(sort_key)
    edge_index = edge_index[:, perm]

    num_leaves_below = torch.tensor([G.nodes[n]['num_leaves_below'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    num_leaves_below_gene_present = torch.tensor([G.nodes[n]['num_leaves_below_gene_present'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    time = torch.tensor([G.nodes[n]['time'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    time_only_valid_nodes = torch.tensor([G.nodes[n]['time_only_valid_nodes'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    time_only_valid_nodes_from_parent = torch.tensor([G.nodes[n]['time_only_valid_nodes_from_parent'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    tree_length = torch.tensor([G.nodes[n]['tree_length'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    gene_present_below_node = torch.tensor([G.nodes[n]['gene_present_below_node'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    valid_node = torch.tensor([G.nodes[n]['valid_node'] for n in nodes], dtype=torch.float32).unsqueeze(1)

    #valid_internal_node = torch.tensor([G.nodes[n]['valid_internal_node'] for n in nodes], dtype=torch.float32).unsqueeze(1)
    
    has_hgt_later_in_time = torch.tensor([G.nodes[n]['has_hgt_later_in_time'] for n in nodes], dtype=torch.float32)

    children_dict = {n: [] for n in nodes}
    for parent, child in edges:
        children_dict[parent].append(child)
    for parent in children_dict:
        children_dict[parent].sort()
        
    parent = torch.full((N,), -1, dtype=torch.long)
    for p, c in edges:
        parent[c] = p

    tin, tout = compute_dfs_intervals(nodes, edges)

    # ------------------------------------------------------------------
    # Determine which nodes are treated/assumed to be recipient nodes:
    # ------------------------------------------------------------------

    recipient_events = {}

    # ------------------------------------------------------------------
    # 1) Determine which nodes are true recipient_parent_id nodes
    # ------------------------------------------------------------------
    
    is_real_recipient = torch.zeros(N, dtype=torch.bool)
    
    for n in nodes:
        recipient_info = G.nodes[n].get("recipient", {})
        if recipient_info.get("is_parent_node", True):
            is_real_recipient[n] = True
            recipient_events[n] = recipient_info.get("events", [])

    if recipient_node is None:
        
        # ------------------------------------------------------------------
        # 2) Vectorized negative sampling
        # ------------------------------------------------------------------
        
        # Sample all nodes at once
        neg_mask = (torch.rand(N) < p_false) & (torch.arange(N) > N/2)
        
        # Always keep positive nodes
        include_mask = is_real_recipient | (~is_real_recipient & neg_mask)
        
        # Indices of nodes we will generate samples for
        selected_indices = np.array(torch.where(include_mask)[0], dtype=int)

    else:
        selected_indices = [recipient_node]

    for node in selected_indices:
        
        if not recipient_events.get(node):

            # Create "fake" recipient.
            recipient_events[recipient_node] = [((recipient_node, children_dict[recipient_node][(torch.rand(1) < 0.5).int()]), (N, N-1))]
    
    tin_t = torch.tensor([tin[i] for i in nodes])
    tout_t = torch.tensor([tout[i] for i in nodes])

    # ------------------------------------------------------------------
    # 3) Generate samples
    # ------------------------------------------------------------------


    for n in selected_indices:

        real_recipient = bool(is_real_recipient[n].item())
        
        # ==============================================================
        # CASE 1: Real recipient node → one sample PER event
        # ==============================================================
        #if real_recipient and n in recipient_events:
        if True:
        
            for event in recipient_events[n]:
        
                recipient_parent_id = event[0][0]
                recipient_child_id = event[0][1]
                donor_child_id = event[1][1]

                ### Get which nodes are ancestors and descendants of the recipient:
                recipient_tin = tin_t[recipient_parent_id]
                recipient_tout = tout_t[recipient_parent_id]
                
                is_ancestor_of_recipient = (tin_t <= recipient_tin) & (tout_t >= recipient_tout)
                is_descendant_of_recipient = (tin_t >= recipient_tin) & (tout_t <= recipient_tout)
                
                is_ancestor_of_recipient[recipient_parent_id] = False
                is_descendant_of_recipient[recipient_parent_id] = False
                is_side_branch = ~(is_ancestor_of_recipient | is_descendant_of_recipient)
                
                ancestor_feat = is_ancestor_of_recipient.float().unsqueeze(1)
                descendant_feat = is_descendant_of_recipient.float().unsqueeze(1)
                side_feat = is_side_branch.float().unsqueeze(1)
                
                recipient_topology = torch.cat(
                    [ancestor_feat, descendant_feat, side_feat],
                    dim=1
                )

                if not has_hgt_later_in_time[recipient_parent_id]:

                    # ------------------------------------------------------
                    # Calculate similarities between children of recipient parent node and all other nodes:
                    # ------------------------------------------------------
    
                    x_first_child = x[children_dict[recipient_parent_id][0]]
                    x_second_child = x[children_dict[recipient_parent_id][1]]
    
                    x_bin = (x > 0).float()
    
                    # Expand children for broadcasting
                    x_first = x_first_child.unsqueeze(0)      # [1, F]
                    x_second = x_second_child.unsqueeze(0)    # [1, F]
                    
                    # ----------------------------------------------------------
                    # Compare FIRST child against all nodes
                    # ----------------------------------------------------------
                    
                    # Both == 1
                    first_both_one = (x_bin == 1) & (x_first == 1)
                    first_both_one_count = first_both_one.sum(dim=1, keepdim=True)  # [N, 1]
                    
                    # Different
                    first_diff = (x_bin != x_first)
                    first_diff_count = first_diff.sum(dim=1, keepdim=True)          # [N, 1]
                    
                    # ----------------------------------------------------------
                    # Compare SECOND child against all nodes
                    # ----------------------------------------------------------
                    
                    second_both_one = (x_bin == 1) & (x_second == 1)
                    second_both_one_count = second_both_one.sum(dim=1, keepdim=True)
                    
                    second_diff = (x_bin != x_second)
                    second_diff_count = second_diff.sum(dim=1, keepdim=True)
    
                    # ----------------------------------------------------------
                    # Intersection-based comparison (set overlap per SNP)
                    # ----------------------------------------------------------
                    
                    # reshape to [N, max_number_of_snps, len_alphabet]
                    x_reshaped = x.view(N, max_number_of_snps, len_alphabet)
                    first_child_reshaped = x[children_dict[recipient_parent_id][0]].view(1, max_number_of_snps, len_alphabet)
                    second_child_reshaped = x[children_dict[recipient_parent_id][1]].view(1, max_number_of_snps, len_alphabet)
                    
                    x_bin_reshaped = (x_reshaped > 0).float()
                    first_bin = (first_child_reshaped > 0).float()
                    second_bin = (second_child_reshaped > 0).float()
                    
                    # Intersection per SNP (any overlap in the len_alphabet-dim alphabet)
                    first_intersection = (x_bin_reshaped * first_bin).sum(dim=2) > 0
                    second_intersection = (x_bin_reshaped * second_bin).sum(dim=2) > 0
                    
                    # Count SNP positions with NO intersection
                    first_no_intersection_count = (~first_intersection).sum(dim=1, keepdim=True)
                    second_no_intersection_count = (~second_intersection).sum(dim=1, keepdim=True)
    
                    child_similarity = torch.cat(
                        [
                            first_both_one_count,
                            first_diff_count,
                            first_no_intersection_count,
                            second_both_one_count,
                            second_diff_count,
                            second_no_intersection_count
                        ],
                        dim=1
                    ).float()  # shape: [N, 6]        
                
                    # ------------------------------------------------------
                    # Determine left / right class
                    # ------------------------------------------------------
            
                    if is_ancestor(children_dict[recipient_parent_id][0],
                                   recipient_child_id, tin, tout):
                        event_class = 1   # donor-left
            
                    elif is_ancestor(children_dict[recipient_parent_id][1],
                                     recipient_child_id, tin, tout):
                        event_class = 2   # donor-right
            
                    else:
                        raise ValueError("Error with left/right child.")
    
                    # ------------------------------------------------------
                    # Graph information (i.e. for the whole graph)
                    # ------------------------------------------------------
                    child1 = children_dict[recipient_parent_id][0]
                    child2 = children_dict[recipient_parent_id][1]
                    
                    tin1, tout1 = tin_t[child1], tout_t[child1]
                    tin2, tout2 = tin_t[child2], tout_t[child2]
                    
                    is_ancestor_child1 = (tin_t <= tin1) & (tout_t >= tout1)
                    is_descendant_child1 = (tin_t >= tin1) & (tout_t <= tout1)
                    
                    is_ancestor_child2 = (tin_t <= tin2) & (tout_t >= tout2)
                    is_descendant_child2 = (tin_t >= tin2) & (tout_t <= tout2)
                    
                    valid_mask_child1 = (~is_ancestor_child1) & (~is_descendant_child1) & valid_node.squeeze().bool()
                    valid_mask_child2 = (~is_ancestor_child2) & (~is_descendant_child2) & valid_node.squeeze().bool()
                    
                    valid_values_child1_min = child_similarity.clone()
                    valid_values_child2_min = child_similarity.clone()
                    valid_values_child1_max = child_similarity.clone()
                    valid_values_child2_max = child_similarity.clone()
                    
                    valid_values_child1_min[~valid_mask_child1] = float('inf')
                    valid_values_child2_min[~valid_mask_child2] = float('inf')
                    
                    valid_values_child1_max[~valid_mask_child1] = -float('inf')
                    valid_values_child2_max[~valid_mask_child2] = -float('inf')
                    
                    min_vals_child1 = valid_values_child1_min.min(dim=0)[0]
                    min_vals_child2 = valid_values_child2_min.min(dim=0)[0]
                    
                    max_vals_child1 = valid_values_child1_max.max(dim=0)[0]
                    max_vals_child2 = valid_values_child2_max.max(dim=0)[0]
                    
                    min_vals_child1[torch.isinf(min_vals_child1)] = 0
                    min_vals_child2[torch.isinf(min_vals_child2)] = 0
                    max_vals_child1[torch.isinf(max_vals_child1)] = 0
                    max_vals_child2[torch.isinf(max_vals_child2)] = 0
    
                    
                    # Update and normalize the child similarities:
                    """
                    child_similarity = torch.cat(
                        [
                            abs(first_both_one_count - max_vals_child1[0]),                
                            abs(first_diff_count - min_vals_child1[2]),
                            abs(first_no_intersection_count - min_vals_child1[4]),
    
                            abs(second_both_one_count - max_vals_child2[1]),
                            abs(second_diff_count - min_vals_child2[3]),
                            abs(second_no_intersection_count - min_vals_child2[5]),
                        ],
                        dim=1
                    ).float()  # shape: [N, 6] 
                    """
    
                    child_similarity = torch.clamp(
                        torch.cat(
                            [
                                max_vals_child1[0] - first_both_one_count,
                                first_diff_count - min_vals_child1[1],
                                first_no_intersection_count - min_vals_child1[2],
                    
                                max_vals_child2[3] - second_both_one_count,
                                second_diff_count - min_vals_child2[4],
                                second_no_intersection_count - min_vals_child2[5],
                            ],
                            dim=1
                        ),
                        min=0
                    ).float()
                    
                    root = max(G.nodes())
                    
                    root_time = torch.tensor(G.nodes[root]['time_only_valid_nodes'], dtype=torch.float32)
                    total_tree_length = torch.tensor(G.nodes[root]['tree_length'], dtype=torch.float32)
                    
                    root_seq = x[root].view(max_number_of_snps, len_alphabet)
                    
                    total_number_of_snp_positions = ((root_seq.sum(dim=1) > 1).sum()).float()
                    total_number_of_snps = (root_seq.sum() - max_number_of_snps).float()
                    
                    recipient_time = torch.tensor(G.nodes[recipient_parent_id]['time_only_valid_nodes'], dtype=torch.float32)
                    child1_time = torch.tensor(G.nodes[child1]['time_only_valid_nodes'], dtype=torch.float32)
                    child2_time = torch.tensor(G.nodes[child2]['time_only_valid_nodes'], dtype=torch.float32)
                    min_child_time = min(child1_time, child2_time)
                    max_child_time = max(child1_time, child2_time)
    
                    graph_information = torch.cat(
                        [
                            recipient_time.unsqueeze(0),
                            child1_time.unsqueeze(0),
                            child2_time.unsqueeze(0),
                            min_child_time.unsqueeze(0),
                            max_child_time.unsqueeze(0),
                            root_time.unsqueeze(0),
                            total_tree_length.unsqueeze(0),
                            total_number_of_snp_positions.unsqueeze(0),
                            total_number_of_snps.unsqueeze(0),
                            min_vals_child1,
                            min_vals_child2,
                            max_vals_child1,
                            max_vals_child2
                        ],
                        dim=0
                    )
                                            
                    # ------------------------------------------------------
                    # Build target tensor
                    # ------------------------------------------------------
            
                    event_label = torch.zeros(N, dtype=torch.long)
                    event_label[donor_child_id] = event_class
    
                    # ------------------------------------------------------
                    # Distance computation
                    # ------------------------------------------------------
    
                    # Recipient
                    time_up_rec_np, time_down_rec_np, node_up_rec_np, node_down_rec_np = \
                        distances_from_node_directional(G, recipient_parent_id)
                    
                    # Donor
                    time_up_don_np, time_down_don_np, node_up_don_np, node_down_don_np = \
                        distances_from_node_directional(G, donor_child_id)
                    
                    # Convert to torch
                    time_up_rec = torch.from_numpy(time_up_rec_np).float()
                    time_down_rec = torch.from_numpy(time_down_rec_np).float()
                    node_up_rec = torch.from_numpy(node_up_rec_np).float()
                    node_down_rec = torch.from_numpy(node_down_rec_np).float()
                    
                    time_up_don = torch.from_numpy(time_up_don_np).float()
                    time_down_don = torch.from_numpy(time_down_don_np).float()
                    node_up_don = torch.from_numpy(node_up_don_np).float()
                    node_down_don = torch.from_numpy(node_down_don_np).float()
    
                    time_dist_from_recipient = torch.cat([time_up_rec.unsqueeze(1), time_down_rec.unsqueeze(1)], dim=1)
                    node_count_dist_from_recipient = torch.cat([node_up_rec.unsqueeze(1), node_down_rec.unsqueeze(1)], dim=1)
                    time_dist_from_donor = torch.cat([time_up_don.unsqueeze(1), time_down_don.unsqueeze(1)], dim=1)
                    node_count_dist_from_donor = torch.cat([node_up_don.unsqueeze(1), node_down_don.unsqueeze(1)], dim=1)
    
                    ## Hard condition for being a possible donor (i.e. has to be further back in time)
                    parent_time = torch.zeros_like(time)
                    valid_parent_mask = parent >= 0
                    parent_time[valid_parent_mask] = time_only_valid_nodes_from_parent[parent[valid_parent_mask]]
                    possible_donor = (parent_time >= min_child_time).float()
                    
                    # ----------------------------------------------------------
                    # Internal node features
                    # ----------------------------------------------------------

                    """
                    internal_node_data = torch.cat(
                        [
                            sum_x,
                            tree_length,
                            num_leaves_below,
                            num_leaves_below_gene_present,
                            gene_present_below_node,
                    
                            # Recipient directional distances
                            time_up_rec.unsqueeze(1),
                            time_down_rec.unsqueeze(1),
                            node_up_rec.unsqueeze(1),
                            node_down_rec.unsqueeze(1),
                    
                            child_similarity.float(),
                            
                            time_only_valid_nodes,
                            time_only_valid_nodes_from_parent,
                            
                            valid_node,
                            possible_donor
                        ],
                        dim=1
                    )
                    """
                    internal_node_data = torch.cat(
                        [
                            sum_x,
                            tree_length,
                            num_leaves_below,
                            num_leaves_below_gene_present,
                    
                            # Recipient directional distances
                            time_up_rec.unsqueeze(1),
                            time_down_rec.unsqueeze(1),
                            time_up_rec.unsqueeze(1) + time_down_rec.unsqueeze(1),
                            node_up_rec.unsqueeze(1),
                            node_down_rec.unsqueeze(1),
                            node_up_rec.unsqueeze(1) + node_down_rec.unsqueeze(1),
                    
                            child_similarity.float() / max_number_of_snps,
                    
                            time_only_valid_nodes,
                            time_only_valid_nodes_from_parent
                        ],
                        dim=1
                    )
                    
                    possible_donor_mask = (
                        gene_present_below_node.bool() &
                        valid_node.bool() &
                        possible_donor.bool()
                    ).float()
                    
                    internal_node_data = torch.log1p(internal_node_data)
                    
                    eps = 1e-8
                    mask = possible_donor_mask.bool()
                    
                    # Werte außerhalb der Donoren ignorieren
                    masked_data_min = internal_node_data.masked_fill(~mask, float("inf"))
                    masked_data_max = internal_node_data.masked_fill(~mask, float("-inf"))
                    
                    min_vals = masked_data_min.min(dim=0, keepdim=True)[0]
                    max_vals = masked_data_max.max(dim=0, keepdim=True)[0]
                    
                    # Falls es kein gültiges Element gab
                    min_vals[min_vals == float("inf")] = 0.0
                    max_vals[max_vals == float("-inf")] = 1.0
                    
                    # Min-Max Normalisierung
                    internal_node_data = (internal_node_data - min_vals) / (max_vals - min_vals + eps)
                    
                    internal_node_data = torch.cat(
                        [
                            internal_node_data,
                            recipient_topology,
                            possible_donor_mask
                        ],
                        dim=1
                    )
    
                    # ------------------------------------------------------
                    # Append graph sample
                    # ------------------------------------------------------
            
                    data.append(
                        Data(
                            x=x,
                            edge_index=edge_index,
                            internal_node_data=internal_node_data,
                            level=level,
                            event_label=event_label,
                            time_dist_from_recipient=time_dist_from_recipient,
                            node_count_dist_from_recipient=node_count_dist_from_recipient,
                            time_dist_from_donor=time_dist_from_donor,
                            node_count_dist_from_donor=node_count_dist_from_donor,
                            child_similarity=child_similarity,
                            graph_information = graph_information,
                            valid_node = valid_node,
                            possible_donor_mask = possible_donor_mask,
                        )
                    )
            
        # ==============================================================
        # CASE 2: Not a real recipient → negative sample
        # ==============================================================
        
        elif n > (max(nodes)+1)/2: # We dont regard negative samples in the leaves as they dont make sense.
        
            recipient_parent_id = n
        
            # Alle Knoten Klasse 0
            event_label = torch.zeros(N, dtype=torch.long)

            # ------------------------------------------------------
            # Calculate similarities between children of recipient parent node and all other nodes:
            # ------------------------------------------------------
            x_first_child = x[children_dict[recipient_parent_id][0]]
            x_second_child = x[children_dict[recipient_parent_id][1]]

            x_bin = (x > 0).float()

            # Expand children for broadcasting
            x_first = x_first_child.unsqueeze(0)      # [1, F]
            x_second = x_second_child.unsqueeze(0)    # [1, F]
            
            # ----------------------------------------------------------
            # Compare FIRST child against all nodes
            # ----------------------------------------------------------
            
            # Both == 1
            first_both_one = (x_bin == 1) & (x_first == 1)
            first_both_one_count = first_both_one.sum(dim=1, keepdim=True)  # [N, 1]
            
            # Different
            first_diff = (x_bin != x_first)
            first_diff_count = first_diff.sum(dim=1, keepdim=True)          # [N, 1]
            
            # ----------------------------------------------------------
            # Compare SECOND child against all nodes
            # ----------------------------------------------------------
            
            second_both_one = (x_bin == 1) & (x_second == 1)
            second_both_one_count = second_both_one.sum(dim=1, keepdim=True)
            
            second_diff = (x_bin != x_second)
            second_diff_count = second_diff.sum(dim=1, keepdim=True)

            # ----------------------------------------------------------
            # Intersection-based comparison (set overlap per SNP)
            # ----------------------------------------------------------
            
            # reshape to [N, max_number_of_snps, len_alphabet]
            x_reshaped = x.view(N, max_number_of_snps, len_alphabet)
            first_child_reshaped = x[children_dict[recipient_parent_id][0]].view(1, max_number_of_snps, len_alphabet)
            second_child_reshaped = x[children_dict[recipient_parent_id][1]].view(1, max_number_of_snps, len_alphabet)
            
            x_bin_reshaped = (x_reshaped > 0).float()
            first_bin = (first_child_reshaped > 0).float()
            second_bin = (second_child_reshaped > 0).float()
            
            # Intersection per SNP (any overlap in the 5-dim alphabet)
            first_intersection = (x_bin_reshaped * first_bin).sum(dim=2) > 0
            second_intersection = (x_bin_reshaped * second_bin).sum(dim=2) > 0
            
            # Count SNP positions with NO intersection
            first_no_intersection_count = (~first_intersection).sum(dim=1, keepdim=True)
            second_no_intersection_count = (~second_intersection).sum(dim=1, keepdim=True)

            
    
            child_similarity = torch.cat(
                [
                    first_both_one_count,
                    second_both_one_count,
                    first_diff_count,
                    second_diff_count,
                    first_no_intersection_count,
                    second_no_intersection_count
                ],
                dim=1
            )  # shape: [N, 6]
            
            # ----------------------------------------------------------
            # Distance computation
            # ----------------------------------------------------------
        
            time_dist_from_recipient_np, node_count_dist_from_recipient_np = \
                distances_from_node(G, recipient_parent_id)
        
            time_dist_from_recipient = torch.from_numpy(
                time_dist_from_recipient_np).float()
        
            node_count_dist_from_recipient = torch.from_numpy(
                node_count_dist_from_recipient_np).float()
        
            time_dist_from_donor = torch.zeros(N)
            node_count_dist_from_donor = torch.zeros(N)
        
            # ----------------------------------------------------------
            # Internal node features
            # ----------------------------------------------------------
        
            internal_node_data = torch.cat(
                [
                    num_leaves_below,
                    num_leaves_below_gene_present,
                    gene_present_below_node,
                    time,
                    time_dist_from_recipient.unsqueeze(1),
                    node_count_dist_from_recipient.unsqueeze(1),
                    child_similarity.float(),
                    time_only_valid_nodes,
                    valid_node
                ],
                dim=1
            )
        
            # ----------------------------------------------------------
            # Append graph sample
            # ----------------------------------------------------------
        
            data.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    internal_node_data=internal_node_data,
                    level=level,
                    event_label=event_label,
                    time_dist_from_recipient=time_dist_from_recipient,
                    node_count_dist_from_recipient=node_count_dist_from_recipient,
                    time_dist_from_donor=time_dist_from_donor,
                    node_count_dist_from_donor=node_count_dist_from_donor,
                    child_similarity=child_similarity
                )
            )
            
    return data

    """
    for n in selected_indices:
    
        real_recipient = bool(is_real_recipient[n].item())
    
        # ==============================================================
        # CASE 1: Real recipient node → one sample PER event
        # ==============================================================
        if real_recipient and n in recipient_events:
    
            for event in recipient_events[n]:

                recipient_parent_id = event[0][0]
                recipient_child_id = event[0][1]
                donor_child_id = event[1][1]
                real_event = 1.0
                
                if is_ancestor(children_dict[recipient_parent_id][0], recipient_child_id, tin, tout): 
                    # Hgt from recipient_parent_id to the child node with the smaller id.
                    recipient_child = torch.tensor([0], dtype=torch.float32)
                elif is_ancestor(children_dict[recipient_parent_id][1], recipient_child_id, tin, tout): 
                    # Hgt from recipient_parent_id to the child node with the bigger id.
                    recipient_child = torch.tensor([1], dtype=torch.float32)
                else:
                    print("Error with left/right child.")

                # ------------------------------------------------------
                # Distance computation (only once per event)
                # ------------------------------------------------------

                time_dist_from_recipient_np, node_count_dist_from_recipient_np = distances_from_node(G, recipient_parent_id)
                time_dist_from_donor_np, node_count_dist_from_donor_np = distances_from_node(G, donor_child_id)
    
                time_dist_from_recipient = torch.from_numpy(time_dist_from_recipient_np).float()
                node_count_dist_from_recipient = torch.from_numpy(node_count_dist_from_recipient_np).float()

                time_dist_from_donor = torch.from_numpy(time_dist_from_donor_np).float()
                node_count_dist_from_donor = torch.from_numpy(node_count_dist_from_donor_np).float()
    
                # ------------------------------------------------------
                # Donor child indicator
                # ------------------------------------------------------

                recipient_parent = (nodes_tensor == recipient_parent_id).float()
                donor_child = (nodes_tensor == donor_child_id).float()
                
                # ------------------------------------------------------
                # Internal node features
                # ------------------------------------------------------
    
                internal_node_data = torch.cat(
                    [
                        num_leaves_below,
                        num_leaves_below_gene_present,
                        gene_present_below_node,
                        time,
                        time_dist_from_recipient.unsqueeze(1),
                        node_count_dist_from_recipient.unsqueeze(1)
                    ],
                    dim=1
                )
    
                # ------------------------------------------------------
                # Append graph sample
                # ------------------------------------------------------
    
                data.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        internal_node_data=internal_node_data,
                        level=level,
                        donor_child=donor_child,
                        recipient_parent_id = recipient_parent_id,
                        recipient_child=recipient_child,
                        time_dist_from_recipient = time_dist_from_recipient,
                        node_count_dist_from_recipient = node_count_dist_from_recipient,
                        real_event=torch.tensor([real_event], dtype=torch.float32),
                        has_hgt_later_in_time=has_hgt_later_in_time,
                        time_dist_from_donor = time_dist_from_donor,
                        node_count_dist_from_donor = node_count_dist_from_donor
                    )
                )
    
        # ==============================================================
        # CASE 2: Not a real recipient → single negative sample
        # ==============================================================
    
        else:

            recipient_parent_id = n
            donor_child = torch.zeros(N, dtype=torch.float32)
            recipient_child = torch.tensor([0.5], dtype=torch.float32)
            real_event = 0.0
            
            # ----------------------------------------------------------
            # Distance computation
            # ----------------------------------------------------------
                
            time_dist_from_recipient_np, node_count_dist_from_recipient_np = distances_from_node(G, recipient_parent_id)

            time_dist_from_recipient = torch.from_numpy(time_dist_from_recipient_np).float()
            node_count_dist_from_recipient = torch.from_numpy(node_count_dist_from_recipient_np).float()
            
            time_dist_from_donor = torch.zeros(N, dtype=torch.float32)
            node_count_dist_from_donor = torch.zeros(N, dtype=torch.float32)
            
    
            # ----------------------------------------------------------
            # Internal node features
            # ----------------------------------------------------------
    
            internal_node_data = torch.cat(
                [
                    num_leaves_below,
                    num_leaves_below_gene_present,
                    gene_present_below_node,
                    time,
                    time_dist_from_recipient.unsqueeze(1),
                    node_count_dist_from_recipient.unsqueeze(1)
                ],
                dim=1
            )
    
            # ----------------------------------------------------------
            # Append graph sample
            # ----------------------------------------------------------
    
            data.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    internal_node_data=internal_node_data,
                    level=level,
                    donor_child=donor_child,
                    recipient_parent_id = recipient_parent_id,
                    recipient_child=recipient_child,
                    time_dist_from_recipient = time_dist_from_recipient,
                    node_count_dist_from_recipient = node_count_dist_from_recipient,
                    real_event=torch.tensor([real_event], dtype=torch.float32),
                    has_hgt_later_in_time=has_hgt_later_in_time,
                    time_dist_from_donor = time_dist_from_donor,
                    node_count_dist_from_donor = node_count_dist_from_donor
                )
            )
    
    return data
    """

def distances_from_node_directional(G, u):
    """
    Compute directional temporal and node-count distances from node u to all nodes in a binary tree.
    Returns two numpy arrays where index i corresponds to node i.

    Parameters:
    - G: NetworkX DiGraph representing a tree
         Each node has attributes:
           - "time": numeric
           - "gene_present_below_node": bool
    - u: source node (integer node id)

    Returns four arrays:
        time_up, time_down,
        node_up, node_down
    """

    N = max(G.nodes()) + 1

    time_up = np.zeros(N, dtype=float)
    time_down = np.zeros(N, dtype=float)

    node_up = np.zeros(N, dtype=int)
    node_down = np.zeros(N, dtype=int)

    parent = {v: next(G.predecessors(v), None) for v in G.nodes()}

    valid_node = {
        v: len(list(G.successors(v))) > 0 and
           all(G.nodes[c]["gene_present_below_node"] for c in G.successors(v))
        for v in G.nodes()
    }

    children = defaultdict(list)
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)

    visited = np.zeros(N, dtype=bool)
    visited[u] = True
    queue = deque([u])

    while queue:
        x = queue.popleft()

        # --- nach unten ---
        for c in children.get(x, []):
            if not visited[c]:
                delta_t = abs(G.nodes[c]["time"] - G.nodes[x]["time"])

                time_up[c] = time_up[x]
                time_down[c] = time_down[x] + delta_t

                node_up[c] = node_up[x]
                node_down[c] = node_down[x] + valid_node[c]

                visited[c] = True
                queue.append(c)

        # --- nach oben ---
        p = parent[x]
        if p is not None and not visited[p]:
            delta_t = abs(G.nodes[p]["time"] - G.nodes[x]["time"])

            time_up[p] = time_up[x] + delta_t
            time_down[p] = time_down[x]

            node_up[p] = node_up[x] + valid_node[p]
            node_down[p] = node_down[x]

            visited[p] = True
            queue.append(p)

    return time_up, time_down, node_up, node_down


def distances_from_node(G, u):
    """
    Compute temporal and node-count distances from node u to all nodes in a binary tree.
    Returns two numpy arrays where index i corresponds to node i.

    Parameters:
    - G: NetworkX DiGraph representing a tree
         Each node has attributes:
           - "time": numeric
           - "gene_present_below_node": bool
    - u: source node (integer node id)

    Returns:
    - time_dist: np.array of shape (N,), temporal distances to u
    - node_count_dist: np.array of shape (N,), count of valid nodes along path to u
    """

    N = max(G.nodes()) + 1  # assumes nodes are 0-indexed integers
    time_dist = np.zeros(N, dtype=float)
    node_count_dist = np.zeros(N, dtype=int)

    # parent dictionary
    parent = {v: next(G.predecessors(v), None) for v in G.nodes()}
    # valid node mask
    valid_node = {
        v: len(list(G.successors(v))) > 0 and 
           all(G.nodes[c]["gene_present_below_node"] for c in G.successors(v))
        for v in G.nodes()
    }
    # child adjacency
    children = defaultdict(list)
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)

    time_dist[u] = 0
    #node_count_dist[u] = valid_node[u]
    queue = deque([u])

    visited = np.zeros(N, dtype=bool)
    visited[u] = True

    while queue:
        x = queue.popleft()
        # nach unten zu Kindern
        for c in children.get(x, []):
            if not visited[c]:
                time_dist[c] = time_dist[x] + abs(G.nodes[c]["time"] - G.nodes[x]["time"])
                node_count_dist[c] = node_count_dist[x] + valid_node[c]
                visited[c] = True
                queue.append(c)
        # nach oben zu Eltern
        p = parent[x]
        if p is not None and not visited[p]:
            time_dist[p] = time_dist[x] + abs(G.nodes[p]["time"] - G.nodes[x]["time"])
            node_count_dist[p] = node_count_dist[x] + valid_node[p]
            visited[p] = True
            queue.append(p)

    return time_dist, node_count_dist

def compute_dfs_intervals(nodes, edges):

    children = defaultdict(list)
    has_parent = set()

    for parent, child in edges:
        children[parent].append(child)
        has_parent.add(child)

    # Wurzel bestimmen (kein eingehender Edge)
    roots = [n for n in nodes if n not in has_parent]

    tin = {}
    tout = {}
    time = 0

    def dfs(node):
        nonlocal time
        tin[node] = time
        time += 1

        for child in children[node]:
            dfs(child)

        tout[node] = time
        time += 1

    for r in roots:
        dfs(r)

    return tin, tout

def is_ancestor(u, v, tin, tout):
    return tin[u] <= tin[v] and tout[v] <= tout[u]


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
    
        if dim_size is not None:
            num_nodes = dim_size
        else:
            num_nodes = int(index.max().item()) + 1
    
        device = inputs.device
    
        # ---- CHILDREN ----
        children = torch.zeros(num_nodes, 2, self.in_dim, device=device)
    
        slot = torch.zeros_like(index)
        slot[1:] = (index[1:] == index[:-1]).long()
    
        children[index, slot] = inputs   # shape: [num_nodes, 2, in_dim]
    
        # ---- MIN / MAX über Kinder ----
        child_min = torch.min(children[:, 0, :], children[:, 1, :])
        child_max = torch.max(children[:, 0, :], children[:, 1, :])
    
        # reshape children
        children_flat = children.reshape(num_nodes, 2 * self.in_dim)
    
        # ---- PARENT ----
        parent = torch.zeros(num_nodes, self.in_dim, device=device)
    
        src, dst = self._edge_index  # child → parent
        parent[src] = self.x[dst]
    
        # ---- CONCAT ----
        return torch.cat(
            [
                children_flat,
                child_min,
                child_max,
                parent
            ],
            dim=-1
        )
        
    def update(self, aggr_out, x):
        """
        aggr_out: (num_nodes, 2*in_dim) concatenated children features
        x:        (num_nodes, in_dim)   parent features

        Returns:
            Fused parent representation → out_dim
        """
        fused = torch.cat([x, aggr_out], dim=-1)
        return fused

    def forward(self, x, edge_index):
        """
        x: (num_nodes, in_dim)
        edge_index: (2, num_edges), where edges point child -> parent

        Returns:
            Updated node embeddings (num_nodes, out_dim)
        """
        self._edge_index = edge_index
        self.x = x
        return self.propagate(edge_index, x=x)


class ParentChildFusionLayer_RecipientFinder(MessagePassing):
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



def compute_tree_length(G: nx.DiGraph) -> nx.DiGraph:
    leaves = [node for node in G.nodes() if G.successors(node) == 0]
    

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


class ParentChildFusionLayer_expected_values(MessagePassing):
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
