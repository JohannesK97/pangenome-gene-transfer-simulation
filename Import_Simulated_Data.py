import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DirGNNConv, GraphConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random, h5py, pickle, glob, os
import networkx as nx
from copy import deepcopy
from pathlib import Path
import numpy as np

def load_file(file):

    gene_length = 1000
    #nucleotide_mutation_rate = 0.1
    
    with h5py.File(file, "r") as f:
            grp = f["results"]
            # Load graph_properties (pickle stored in dataset)
            graph_properties = pickle.loads(grp["graph_properties"][()])
    
            # Unpack graph properties
            nodes = torch.tensor(graph_properties[0])                # [num_nodes]
            edges = torch.tensor(graph_properties[1], dtype=torch.long)  # [2, num_edges]
            coords = torch.tensor(graph_properties[2].T)             # [2, num_nodes]
    
            # Load datasets instead of attrs
            gene_absence_presence_matrix = grp["gene_absence_presence_matrix"][()]
            #fitch_scores = grp["fitch_score"][()]
            #children_gene_nodes_loss_events = grp["children_gene_nodes_loss_events"][()]
    
            # Load HGT events (simplified)
            hgt_events = {}
            hgt_grp_simpl = grp["nodes_hgt_events_simplified"]
            for site_id in hgt_grp_simpl.keys():
                hgt_events[int(site_id)] = hgt_grp_simpl[site_id][()]
    
            x_node_features = coords.float().T
            
            # Erstelle einen gerichteten Graphen
            G = nx.DiGraph()
            
            # Füge Knoten hinzu (optional mit Koordinaten als Attribut)
            for i, node_id in enumerate(nodes.tolist()):
                G.add_node(node_id, 
                           core_distance = coords[:, i].tolist()[0]* gene_length,
                           allele_distance = coords[:, i].tolist()[1]* gene_length, 
                           allele_distance_only_new = coords[:, i].tolist()[2]* gene_length, 
                           allele_distances_both_children_polymorph = coords[:, i].tolist()[3]* gene_length,
                           true_allele_distance = coords[:, i].tolist()[4]* gene_length, 
                           node_time = coords[:, i].tolist()[5])
            # Füge Kanten hinzu
            edge_list = edges.tolist()
            for src, dst in zip(edge_list[0], edge_list[1]):
                G.add_edge(src, dst)
    
            H = deepcopy(G)
                
            for node in G.nodes():
                children = list(G.successors(node))
                
                if children:
                    
                    allele_sum = sum(G.nodes[child]['allele_distance'] for child in children)

                    child1, child2 = children
                    n = snp_percentage_of_gen_to_absolute_number(G.nodes[child1]['true_allele_distance'] / gene_length, gene_length) 
                    m = snp_percentage_of_gen_to_absolute_number(G.nodes[child2]['true_allele_distance'] / gene_length, gene_length)
                                       
                    # Erwartete Anzahl verschiedener SNPs
                    if not (n == np.inf and m == np.inf):
                        expected_snps = n + m - (n * m) / gene_length
                    else:
                        expected_snps = gen_length        
                    true_allele_sum = absolute_number_to_snp_percentage_of_gen(expected_snps, gene_length) * gene_length

                    n = snp_percentage_of_gen_to_absolute_number(G.nodes[child1]['core_distance'] / gene_length, gene_length) 
                    m = snp_percentage_of_gen_to_absolute_number(G.nodes[child2]['core_distance'] / gene_length, gene_length)
                                       
                    # Erwartete Anzahl verschiedener SNPs
                    if not (n == np.inf and m == np.inf):
                        expected_snps = n + m - (n * m) / gene_length
                    else:
                        expected_snps = gen_length        
                    core_sum = absolute_number_to_snp_percentage_of_gen(expected_snps, gene_length) * gene_length

                    H.nodes[node]['core_distance_convolution'] = G.nodes[node]['core_distance'] - core_sum
                    H.nodes[node]['allele_distance_convolution'] = G.nodes[node]['allele_distance'] - allele_sum + G.nodes[node]['allele_distances_both_children_polymorph']
                    H.nodes[node]['true_allele_convolution'] = G.nodes[node]['true_allele_distance'] - true_allele_sum
                else:
                    H.nodes[node]['core_distance_convolution'] = G.nodes[node]['core_distance']
                    H.nodes[node]['allele_distance_convolution'] = G.nodes[node]['allele_distance']
                    H.nodes[node]['true_allele_convolution'] = G.nodes[node]['true_allele_distance']
    
            # Schritt 1: Leaf-Nodes bestimmen (Blätter = Knoten ohne Nachfolger)
            leaves = [n for n in H.nodes if H.out_degree(n) == 0]
                      
            # Schritt 3: Werte in H eintragen
            for node in H.nodes():
                H.nodes[node]["leaf_count"] = count_leaves(node, H, leaves, gene_absence_presence_matrix, gene_presence_matters = False)
                H.nodes[node]["leaf_count_presence_matters"] = count_leaves(node, H, leaves, gene_absence_presence_matrix, gene_presence_matters = True)
                H.nodes[node]["node_count"] = count_nodes(node, H, leaves, gene_absence_presence_matrix, gene_presence_matters = False)
                H.nodes[node]["node_count_presence_matters"] = count_nodes(node, H, leaves, gene_absence_presence_matrix, gene_presence_matters = True)
    
            node_features = []
            for node in list(H.nodes):
                core = G.nodes[node].get("core_distance", 0.0)
                allele = G.nodes[node].get("allele_distance", 0.0)
                core_convolution = H.nodes[node].get("core_distance_convolution", 0.0)
                allele_convolution = H.nodes[node].get("allele_distance_convolution", 0.0)
                leaf_count = H.nodes[node].get("leaf_count", 0)
                leaf_count_presence_matters = H.nodes[node].get("leaf_count_presence_matters", 0)
                node_count = H.nodes[node].get("node_count", 0)
                node_count_presence_matters = H.nodes[node].get("node_count_presence_matters", 0)
                allele_distance_only_new = G.nodes[node].get("allele_distance_only_new", 0.0)
                allele_distances_both_children_polymorph = G.nodes[node].get("allele_distances_both_children_polymorph", 0.0)
                true_allele_distance = G.nodes[node].get("true_allele_distance", 0.0)
                true_allele_convolution = H.nodes[node].get("true_allele_convolution", 0.0)
                node_time = H.nodes[node].get("node_time", 0)
                #node_features.append([core_convolution, allele_distance_only_new])
                #node_features.append([core, allele, allele_distance_only_new, core_convolution, allele_convolution, allele_distances_both_children_polymorph, true_allele_distance, true_allele_convolution, node_time, leaf_count, leaf_count_presence_matters, node_count, node_count_presence_matters])
                node_features.append([core, allele, allele_distance_only_new, core_convolution, allele_convolution, allele_distances_both_children_polymorph, node_time, leaf_count, leaf_count_presence_matters, node_count, node_count_presence_matters])
                
            
            coords_modified = torch.tensor(node_features, dtype=torch.float32).T
    
            # Node features: 
            x_node_features_modified = coords_modified.float().T  
        
            # Collect all recipient_parent_nodes from all sites
            recipient_parent_nodes = set()
            for site_id in hgt_grp_simpl.keys():
                arr = hgt_grp_simpl[site_id][()]  # load dataset as numpy structured array
                recipient_parent_nodes.update(arr["recipient_parent_node"].tolist())
            
            # Build theta_gains: 1 if node is in recipient_parent_nodes, else 0
            theta_gains = torch.tensor(
                [1 if node in recipient_parent_nodes else 0 for node in graph_properties[0]],
                dtype=torch.long
            )

            level = {n: 0 for n in H.nodes}  # Leaves haben Level 0
            
            # 3. Topologische Sortierung (damit Kinder vor Eltern behandelt werden)
            for node in reversed(list(nx.topological_sort(H))):
                successors = list(H.successors(node))
                if successors:
                    level[node] = 1 + max(level[s] for s in successors)
            print(level)
            
            # 4. Level als Attribut setzen
            nx.set_node_attributes(H, level, "level")
        
            data = Data(
                x = x_node_features_modified,       # Node Features [num_nodes, 2]
                edge_index = edges,        # Edge Index [2, num_edges]
                y = theta_gains,            # Labels [num_nodes]
                file = file,
                H = H,
                recipient_parent_nodes = recipient_parent_nodes,
                gene_absence_presence_matrix = gene_absence_presence_matrix
            )

    return data

# Schritt 2: Rekursiv/Subgraph-basiert Anzahl der Leaves pro Knoten zählen
def count_leaves(node, G, leaves, gene_absence_presence_matrix, gene_presence_matters = False):
    # wenn Blatt → 1
    if node in leaves:
        if not gene_presence_matters or gene_absence_presence_matrix[node] == 1:
            return 1
        else:
            return 0
    # sonst Summe der Leaves aller Kinder
    return sum(count_leaves(child, G, leaves, gene_absence_presence_matrix, gene_presence_matters) for child in G.successors(node))

def count_nodes(node, G, leaves, gene_absence_presence_matrix, gene_presence_matters = False):
    # wenn Blatt → 1
    if node in leaves:
        if not gene_presence_matters or gene_absence_presence_matrix[node] == 1:
            return 1
        else:
            return 0
    # sonst Summe der Nodes aller Kinder
    summ = sum(count_nodes(child, G, leaves, gene_absence_presence_matrix, gene_presence_matters) for child in G.successors(node))
    return summ + int(summ > 0)

def snp_percentage_of_gen_to_absolute_number(x, gene_length):
    if x == 1:
        return np.inf
    else:
        y = np.log(1 - x) / np.log(1 - 1/gene_length)
    return y

def absolute_number_to_snp_percentage_of_gen(y, gene_length):
    return 1 - np.exp(y * np.log(1 - 1/gene_length))
