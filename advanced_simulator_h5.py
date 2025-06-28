import msprime
import tskit
import hgt_simulation
import hgt_sim_args
import numpy as np
import random
import torch
import os
import time
import re
import secrets
import copy
import json
import h5py
import uuid
import glob
import shutil

from random import randint
from collections import namedtuple
from collections import defaultdict
from collections import deque
from typing import List, Union
from sbi.utils import BoxUniform
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from concurrent.futures import ProcessPoolExecutor

#@profile
def simulator(
    num_samples: Union[int, None],
    theta: int = 1,
    rho: float = 0.3,
    hgt_rate: float = 0,
    num_genes: int = 1,
    nucleotide_mutation_rate: float = 0.01,
    gene_length: int = 1000,
    pca_dimensions: int = 10,
    ce_from_nwk: Union[str, None] = None,
    infinite_sites_factor: int = 3,
    seed: Union[int, None] = None,
    distance_matrix: Union[np.ndarray, None] = None,
    multidimensional_scaling_dimensions: int = 50,
) -> tskit.TreeSequence:

    start_time = time.time()
    
    """
    if ce_from_nwk is not None and num_samples is not None:
        raise ValueError(
            "A core tree and parameters for simulation were provided. Choose either."
        )
    """
        
    if ce_from_nwk is None and num_samples is None:
        raise ValueError(
            "Neither a core tree or parameters for simulation were provided. Choose either."
        )
    
    if ce_from_nwk is None:
        core_tree = msprime.sim_ancestry(
                samples=num_samples,
                sequence_length=1,
                ploidy=1,
                recombination_rate=0,
                gene_conversion_rate=0,
                gene_conversion_tract_length=1,  # One gene
                random_seed=seed,
            )
    
        ce_from_nwk = core_tree.first().newick()

    if seed is None:
        seed = secrets.randbelow(2**32 - 4) + 2

    random.seed(seed)
    np.random.seed(seed)
    #print("Seed: ", seed)
    
    ### Calculate hgt events:

    args = hgt_sim_args.Args(
        sample_size=num_samples,
        num_sites=num_genes,
        gene_conversion_rate=0,
        recombination_rate=0,
        hgt_rate=hgt_rate,
        ce_from_ts=None,
        ce_from_nwk=ce_from_nwk,
        random_seed=seed,
        #random_seed=84,
    )
    
    ts, hgt_edges = hgt_simulation.run_simulate(args)
            
    ### Place mutations
    
    alleles = ["absent", "present"]
    
    # Place one mutation per site, e.g. genome position
    
    gains_model = msprime.MatrixMutationModel(
        alleles = alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [0, 1],
            [0, 1],
        ],
    )
    
    ts_gains = msprime.sim_mutations(ts, rate=1, model = gains_model, keep = True, random_seed=seed)
    
    k = 1
    while (ts_gains.num_sites < ts_gains.sequence_length):
        ts_gains = msprime.sim_mutations(ts_gains, rate=1, model = gains_model, keep = True, random_seed=seed+k)
        k = k+1
    
    # Remove superfluous mutations
    
    tables = ts_gains.dump_tables()
    
    mutations_by_site = {}
    for mut in tables.mutations:
        if mut.site not in mutations_by_site:
            mutations_by_site[mut.site] = []
        mutations_by_site[mut.site].append(mut)
    
    tables.mutations.clear()
    
    for site, mutations in mutations_by_site.items():
        selected_mutation = random.choice(mutations)
    
        tables.mutations.add_row(
            site=selected_mutation.site,
            node=selected_mutation.node,
            derived_state=selected_mutation.derived_state,
            parent=-1,
            metadata=None,
            time=selected_mutation.time,
        )
    
        # Add sentinel mutations at the leafs:
    
        for leaf_position in range(num_samples):
            tables.mutations.add_row(
                site = selected_mutation.site,
                node = leaf_position,
                derived_state = "absent",
                time = 0.00000000001,
            )
    
    ts_gains = tables.tree_sequence()
    
    
    # Place losses:
    
    losses_model = msprime.MatrixMutationModel(
        alleles = alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [1, 0],
            [1, 0],
        ],
    )
    
    ts_gains_losses = msprime.sim_mutations(ts_gains, rate = rho, model = losses_model, keep = True, random_seed=seed-1)
    
    tables = ts_gains_losses.dump_tables()
    
    # Find the node of the root of the clonal tree:
    clonal_node = ts_gains.first().mrca(*list(range(num_samples)))
    
    print("Clonal node:", clonal_node)
    
    for site, mutations in mutations_by_site.items():
        # Mutation at clonal root:
        tables.mutations.add_row(
            site=site,
            node=clonal_node,
            derived_state="absent",
            parent=-1,
            metadata=None,
            time=tables.nodes.time[clonal_node],
        )
    
    tables.sort()
    ts_gains_losses = tables.tree_sequence()
    
    ### Calculate the gene absence presence matrix:
    
    MutationRecord = namedtuple('MutationRecord', ['site_id', 'mutation_id', 'node', 'is_hgt'])
    
    tables = ts_gains_losses.dump_tables()
    
    tables.mutations.clear() # SIMPLE VERSION!
    
    hgt_parent_nodes = [edge.parent-1 for edge in hgt_edges]
    hgt_children_nodes = [edge.child for edge in hgt_edges]
    hgt_parent_children = defaultdict(list)
    #hgt_children_parent = defaultdict(list)
    
    for parent in hgt_parent_nodes:
        hgt_parent_children[parent].append(parent-1)
    
    
    #for child in hgt_children_nodes:
    #    hgt_children_parent[child].append(child+1)
    
    # Initialize tables for the diversity of each gene
    tables_gene = tskit.TableCollection(sequence_length=gene_length)
    tables_gene.nodes.replace_with(ts_gains_losses.tables.nodes)
    tables_gene.populations.replace_with(ts_gains_losses.tables.populations)
    
    #tables_gene.mutations.clear()
    #tables_gene.edges.clear()
    
    #for i in range(num_samples):
    #    tables_gene.nodes.add_row(time = 0, population = 0, flags = 1) 
    
    tables_gene_list = [copy.deepcopy(tables_gene) for _ in range(num_genes)]
    gene_number_hgt_events_passed = [0 for _ in range(num_genes)]
    gene_trees_list = []
    
    
    for tree in ts_gains_losses.trees():

        for site in tree.sites():
            
            hgt_parent_children_passed = [False] * ts_gains_losses.num_nodes
            mutations = site.mutations
            #site = list(tree.sites())[0]
            
            present_mutation = [m for m in mutations if m.derived_state == "present"][0]
            absent_mutation_nodes = {m.node for m in mutations if m.derived_state == "absent"}

            absent_mutations = defaultdict(list)
            for m in mutations:
                if m.derived_state == "absent" and m.time < present_mutation.time:
                    absent_mutations[m.node].append(m)

            branching_nodes_reached_before = defaultdict(list)
            for node_id in range(ts_gains_losses.num_nodes):
                branching_nodes_reached_before[node_id] = False
        
            print("Present_mutation at node: ", present_mutation.node)
        
            branching_nodes_to_process = deque([(present_mutation.node, False, False, 0)])
            # The second variable describes if a hgt edge was passed the whole way down to the actual node. 
            # The third describes if a hgt edge was passed in the last step.
            # The fourth is the number of hgt edges that were passed.
        
            child_mutations = []
        
            if present_mutation.node < num_samples: # Gain directly above leaf:
                if present_mutation.id == sorted([mut for mut in mutations if mut.node == present_mutation.node], key=lambda m: m.time)[1].id:
                    sentinel_mutation = min([mut for mut in absent_mutations[present_mutation.node]], key=lambda m: m.time)
                    child_mutations.append(MutationRecord(
                        site_id=site.id,
                        mutation_id=sentinel_mutation.id,
                        node=sentinel_mutation.node,
                        is_hgt=False
                    ))
                    tables_gene_list[site.id].edges.add_row(
                            left = 0, right = gene_length, parent = tree.parent(present_mutation.node), child = sentinel_mutation.node
                    )
        
            else: 
                
                # To see, which HGT edges are passed, we have to go through the tree two times. 
                # First, we detect all passed HGT edges, then we calculate the presence of mutations in the leaves.
        
                # First time going through:
                while branching_nodes_to_process:
                    
                    last_branching_node = branching_nodes_to_process.popleft()
                    selected_branch_nodes_to_process = deque([last_branching_node])
                    branching_nodes_reached_before[tree.parent(last_branching_node[0])] = True
                    
                    while selected_branch_nodes_to_process:
                    
                        child_node = selected_branch_nodes_to_process.popleft()    
            
                        # If there is a mutation on the edge, find the earliest one.
                        if not child_node[2] and child_node[0] in absent_mutation_nodes:
                            
                            absent_mutation_after_gain_at_node = absent_mutations[child_node[0]]
                            
                            if not absent_mutation_after_gain_at_node: # empty
                                children = tree.children(child_node[0])
                                if len(children) > 1:
                                    for child in reversed(children):
                                        if not branching_nodes_reached_before[child_node[0]]:
                                            branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                        #else:
                                        #    print("Child passed before", " Child: ", child)
                                else:
                                    for child in reversed(children):
                                        selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                                if hgt_parent_children[child_node[0]]:
                                    selected_branch_nodes_to_process.extendleft([(hgt_parent_children[child_node[0]][0], True, True, child_node[3] + 1)])
                                    hgt_parent_children_passed[hgt_parent_children[child_node[0]][0]] = True # The child of the hgt_edge is marked
                                    #print("HGT at Node: ", hgt_parent_children[child_node[0]][0])
                                
                
                        # If there is no mutation, add child nodes.
                        else:
                            children = tree.children(child_node[0])
                            if len(children) > 1:
                                for child in reversed(children):
                                    #print("Child: ", child)
                                    if not branching_nodes_reached_before[child_node[0]]:
                                        branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                    #else:
                                    #    print("Child passed before", " Child: ", child)
                            else:
                                for child in reversed(children):
                                    #print("Child: ", child)
                                    selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                            if hgt_parent_children[child_node[0]]:
                                selected_branch_nodes_to_process.extendleft([(hgt_parent_children[child_node[0]][0], True, True, child_node[3] + 1)])
                                hgt_parent_children_passed[hgt_parent_children[child_node[0]][0]] = True # The child of the hgt_edge is marked
                                #print("HGT at Node: ", hgt_parent_children[child_node[0]][0])
        
                # Second time going through:
                
                branching_nodes_to_process = deque([(present_mutation.node, False, False, 0)])

                branching_nodes_reached_before = defaultdict(list)
                for node_id in range(ts_gains_losses.num_nodes):
                    branching_nodes_reached_before[node_id] = False
                
                while branching_nodes_to_process:
                    
                    last_branching_node = branching_nodes_to_process.popleft()
                    selected_branch_nodes_to_process = deque([last_branching_node])
                    branching_nodes_reached_before[tree.parent(last_branching_node[0])] = True
                    
                    while selected_branch_nodes_to_process:
                    
                        child_node = selected_branch_nodes_to_process.popleft()
        
                        if not child_node[2] and hgt_parent_children_passed[child_node[0]]:
                            #print("Incoming HGT edge registered at node: ", child_node[0])
                            continue
            
                        # If there is a mutation on the edge, find the earliest one.
                        if not child_node[2] and child_node[0] in absent_mutation_nodes:
                            
                            absent_mutation_after_gain_at_node = absent_mutations[child_node[0]]
                            
                            if absent_mutation_after_gain_at_node: # not empty
                                earliest_mutation = max(
                                    absent_mutations[child_node[0]], 
                                    key=lambda m: m.time
                                )
                
                                if earliest_mutation.time == 0.00000000001:
                                    earliest_mutation.derived_state = "present"
                                    child_mutations.append(MutationRecord(
                                        site_id=site.id,
                                        mutation_id=earliest_mutation.id,
                                        node=earliest_mutation.node,
                                        is_hgt=child_node[1]
                                    ))
                                    tables_gene_list[site.id].edges.add_row(
                                        left = 0, right = gene_length, parent = tree.parent(last_branching_node[0]), child = earliest_mutation.node
                                    )
                                    gene_number_hgt_events_passed[site.id] += child_node[3]
                            else:
                                children = tree.children(child_node[0])
                                if len(children) > 1:
                                    for child in reversed(children):
                                        if not branching_nodes_reached_before[child_node[0]]:
                                            branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                    tables_gene_list[site.id].edges.add_row(
                                            left = 0, right = gene_length, parent = tree.parent(last_branching_node[0]), child = child_node[0]
                                    )
                                    gene_number_hgt_events_passed[site.id] += child_node[3]
                                else:
                                    for child in reversed(children):
                                        selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                                if hgt_parent_children[child_node[0]]:
                                    selected_branch_nodes_to_process.extendleft([(hgt_parent_children[child_node[0]][0], True, True, child_node[3] + 1)])  
                                
                
                        # If there is no mutation, add child nodes.
                        else:
                            children = tree.children(child_node[0])
                            if len(children) > 1:
                                for child in reversed(children):
                                    if not branching_nodes_reached_before[child_node[0]]:
                                        branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                tables_gene_list[site.id].edges.add_row(
                                        left = 0, right = gene_length, parent = tree.parent(last_branching_node[0]), child = child_node[0]
                                )
                                gene_number_hgt_events_passed[site.id] += child_node[3]
                            else:
                                for child in reversed(children):
                                    selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                            if hgt_parent_children[child_node[0]]:
                                selected_branch_nodes_to_process.extendleft([(hgt_parent_children[child_node[0]][0], True, True, child_node[3] + 1)])

            #print("Tree computed")
            #for i in range(len(branching_nodes_reached_before)):
            #    if branching_nodes_reached_before[i]:
            #        print("Node branched before: ", i)
            child_mutations.sort(key=lambda mut: not mut.is_hgt) # Will set is_hgt to False later if there are paths without hgt events to the leaf.
        
            # We have to adress multiple paths to the same destiny, some with hgt and other without it:
            unique_mutations = {}
        
            for mut in child_mutations:
                if mut.node not in unique_mutations:
                    unique_mutations[mut.node] = mut
                else:
                    existing_mut = unique_mutations[mut.node]
                    if not existing_mut.is_hgt or not mut.is_hgt:
                        unique_mutations[mut.node] = mut._replace(is_hgt=False)
            
            child_mutations_filtered = list(unique_mutations.values())

            for mutation in mutations:
                if mutation.time > 0.00000000001:
                    if mutation.derived_state == "absent":
                        metadata_value = bytes([3]) 
                    elif mutation.derived_state == "present":
                        metadata_value = bytes([7])
                    tables.mutations.add_row(
                        site=site.id,
                        node=mutation.node,
                        derived_state=mutation.derived_state,
                        parent=-1,
                        metadata=metadata_value,
                        time=mutation.time,
                    )
            
            for mutation in child_mutations_filtered:
        
                tables.mutations.add_row(
                    site=site.id,
                    node=mutation.node,
                    derived_state="present",
                    parent=-1,
                    metadata=bytes([mutation.is_hgt]),
                    time=0.00000000001,
                )
        

    mts = tables.tree_sequence()
    
    # Simulate the tree for each gene:
    
    """
    nucleotide_mutation = msprime.MatrixMutationModel(
            alleles=["C", "M1", "M2", "M3"],
            root_distribution=[1.0, 0.0, 0.0, 0.0],  # nur C als Wurzel
            transition_matrix=[
                # C     M1     M2     M3
                [ 0.0,  1/3,  1/3,  1/3 ], 
                [ 1/3,  0,    1/3,  1/3 ],  
                [ 1/3,  1/3,  0.0,  1/3 ],  
                [ 1/3,  1/3,  1/3,  0.0 ],  
            ]
        )
    """
    nucleotide_mutation = msprime.MatrixMutationModel(
        alleles=["C", "M"],
        root_distribution=[1.0, 0.0],  # nur C als Wurzel
        transition_matrix=[
            # C     M1     
            [ 0.0,  1 ], 
            [ 1/3,  2/3],  
        ]
    )

    """
    nucleotide_mutation = msprime.InfiniteAlleles()
    """
    
    for i in range(num_genes): 
                
        tables_gene_list[i].sort()
        
        gene_trees_list.append(tables_gene_list[i].tree_sequence())
        
        gene_trees_list[i] = msprime.sim_mutations(gene_trees_list[i], 
                                                   rate = nucleotide_mutation_rate, model = nucleotide_mutation, keep = True, random_seed=seed-1)
    
    # Calculate the different alleles:
    
    alleles_list = []
    
    for i in range(num_genes):
        alleles_list.append([])
        for var in gene_trees_list[i].variants():
            alleles_list[i].append(var.genotypes)
        alleles_list[i] = np.array(alleles_list[i])
    
    
    # Perform a PCA to reduce the number of dimensions while keeping the same distances:
    """
    alleles_list_pca = []


    for A in alleles_list:
        A = A.astype(float)
        
        if A.shape[0] == 0:
            A_reconstructed = np.full((pca_dimensions, num_samples), 0.0)
            alleles_list_pca.append(A_reconstructed)
            continue
        
        # Spalten finden, die nur -1 enthalten
        gene_absent_vector = (A[0] == -1)
        gene_present_vector = ~gene_absent_vector
        
        # Falls alle Spalten -1 sind (also keine gültigen Spalten)
        if gene_present_vector.sum() == 0:
            alleles_list_pca.append(np.full((pca_dimensions, A.shape[1]), -1))
            continue
        # Sonderfall: Nur eine gültige Spalte → kein SVD möglich
        if gene_present_vector.sum() == 1:
            valid_index = np.where(gene_present_vector)[0][0]
            A_valid = A[0, valid_index]  # nur erste Zeile, einzelner Wert
            A_final = np.full((pca_dimensions, A.shape[1]), -1)
            A_final[:, valid_index] = A_valid  # kopiert den Wert in alle Zeilen
            alleles_list_pca.append(A_final)
            continue
        
        # PCA auf nur gültige Spalten
        A_valid = A[:, gene_present_vector]
        
        # Add rows of zeros if there are not enough valid rows:
        n_rows = A_valid.shape[0]
        if n_rows < pca_dimensions:
            pad_rows = pca_dimensions - n_rows
            A_valid = np.vstack([A_valid, np.zeros((pad_rows, A_valid.shape[1]))])
        
        # Anzahl Komponenten begrenzen auf gültige Dimensionen
        n_components = min(pca_dimensions, A_valid.shape[1], A_valid.shape[0])
        
        # PCA auf A_valid.T (also Spalten als Samples)
        pca = PCA(n_components=n_components)
        A_pca = pca.fit_transform(A_valid.T).T
    
        if A_pca.shape[0] < pca_dimensions: # Fill the bottom with zeros if neccessary to always get the same dimension
            A_pca = np.vstack([A_pca, np.zeros((pca_dimensions - A_pca.shape[0], A_pca.shape[1]))])
    
        A_reconstructed = np.full((pca_dimensions, A.shape[1]), 0.0)
        A_reconstructed[:, gene_present_vector] = A_pca
        
        #print(squareform(pdist(A_valid.T, metric='euclidean')))
        #print(squareform(pdist(A_pca.T, metric='euclidean')))
    
        alleles_list_pca.append(A_reconstructed)
    """

    ### Compute the gene presence and absence:
    
    gene_absence_presence_matrix = []
    
    for var in mts.variants():
        gene_absence_presence_matrix.append(var.genotypes)
    gene_absence_presence_matrix = np.array(gene_absence_presence_matrix)

    print("Number of present genes: ", sum(gene_absence_presence_matrix[0]))
    
    fitch_scores = fitch_parsimony_score(mts, gene_absence_presence_matrix)
    
    # Distances in alleles (no pca)

    alleles_list_pca = []
    alleles_list_pca_unscaled = []
    
    for A in alleles_list:

        if A.shape[0] == 0:
            A_reconstructed = np.full((num_samples, multidimensional_scaling_dimensions), -1)
            alleles_list_pca.append(A_reconstructed)
            continue

        gene_absent_vector = (A[0] == -1)
        gene_present_vector = ~gene_absent_vector

        # Berechne die paarweisen Hamming-Distanzen
        euclidean_distances = squareform(pdist(A.T, metric='euclidean')) # Will be transformed into Hamming distances later.
        scaled_euclidean_distances = np.full((num_samples, multidimensional_scaling_dimensions), 0.0)
        
        if gene_present_vector.sum() > 0:
            scaled_euclidean_distances[gene_present_vector, :] = multidimensional_scaling(
                euclidean_distances[np.ix_(gene_present_vector, gene_present_vector)], multidimensional_scaling_dimensions = multidimensional_scaling_dimensions)

        euclidean_distances[gene_absent_vector, : ] = -1
        euclidean_distances[:, gene_absent_vector ] = -1

        alleles_list_pca.append(scaled_euclidean_distances)
        
    ### Calculate the distances between the leaves

    if distance_matrix is None:
        distance_matrix = distance_core(tree_sequence = mts, num_samples = num_samples)
    scaled_distance_matrix = multidimensional_scaling(distance_matrix, multidimensional_scaling_dimensions = multidimensional_scaling_dimensions)


    ### Print the computation time.
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Success: hgt_rate = {hgt_rate}, Total computation time = {elapsed_time:.6f} seconds.")

    return mts, gene_trees_list, gene_absence_presence_matrix, alleles_list_pca, scaled_distance_matrix, fitch_scores, gene_number_hgt_events_passed #, euclidean_distances, alleles_list


def distance_core(tree_sequence: tskit.TreeSequence, num_samples: int) -> np.ndarray:
    """
    Computes a pairwise distance matrix between all samples in a TreeSequence,
    based on the time to their most recent common ancestor (MRCA).

    The distance between sample i and j is defined as:
        d(i, j) = t(MRCA(i, j)) - t(i) - t(j),
    where t(x) is the time (in generations) of node x in the tree.

    Parameters
    ----------
    tree_sequence : tskit.TreeSequence
        A tree sequence object containing one or more trees.
        Only the first tree is used.
    num_samples : int
        The number of samples (usually: len(ts.samples())).

    Returns
    -------
    distance_matrix : np.ndarray
        A symmetric matrix (num_samples x num_samples) containing
        pairwise MRCA-based distances between samples.
    """
    tree = tree_sequence.first()
    distance_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        time_i = tree.get_time(i)
        for j in range(i):
            time_j = tree.get_time(j)
            mrca = tree.mrca(i, j)
            time_mrca = tree.get_time(mrca)

            # Distance is the time between each sample and their MRCA
            distance = (time_mrca - time_i) + (time_mrca - time_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # symmetric

    return distance_matrix

def multidimensional_scaling(D, multidimensional_scaling_dimensions=100):
    """
    Klassisches MDS (cMDS) für eine gegebene Distanzmatrix D.
    Gibt niedrigdimensionale Koordinaten zurück.
    """
    # 1. Quadrat der Abstände
    D2 = D ** 2

    # 2. Zentrierungsmatrix
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n

    # 3. Doubly centered inner product matrix
    B = -0.5 * J @ D2 @ J

    # 4. Eigenzerlegung
    eigvals, eigvecs = np.linalg.eigh(B)

    # 5. Sortieren (absteigend)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 6. Komponenten auswählen
    L = np.diag(np.sqrt(np.maximum(eigvals[:multidimensional_scaling_dimensions], 0)))
    V = eigvecs[:, :multidimensional_scaling_dimensions]

    out = V @ L  # → [n, multidimensional_scaling_dimensions]

    if out.shape[1] < multidimensional_scaling_dimensions:
        padding = np.zeros((out.shape[0], multidimensional_scaling_dimensions - out.shape[1]))
        out = np.hstack([out, padding])
    
    return out

def reconstruct_distance_matrix(coords, gene_absence_presence = None):
    if gene_absence_presence is None:
        return squareform(pdist(coords, metric="euclidean")) # For the core distance
    else: 
        output = squareform(pdist(coords, metric="euclidean")** 2) # For the allel distances. ** 2 transforms it into Hamming distances
        sample_not_present = (gene_absence_presence == 0)
        output[sample_not_present, : ] = -1
        output[ : , sample_not_present] = -1
        return output


def fitch_parsimony_score(tree_sequence: tskit.trees.TreeSequence, genotypes: np.ndarray) -> int:
    """
    Berechnet den Fitch-Parsimony-Score für einen einzelnen tskit.Tree
    und einen Vektor von binären Genotypen (0/1) für jeden Leaf-Node.
    
    Parameter:
    - tree: tskit.Tree Objekt
    - genotypes: np.ndarray der Länge tree.num_samples (z. B. 0 = Gen fehlt, 1 = Gen vorhanden)
    
    Rückgabe:
    - parsimony_score: minimale Anzahl der Zustandsänderungen im Baum
    """

    scores = list()
    site_id = 0
    
    if len(genotypes) > 0:
        num_samples = len(genotypes[0])
    else:
        raise ValueError("Empty genotypes matrix")
    
    for tree in tree_sequence.trees():
        # Map: node -> Menge möglicher Zustände (initial nur Blätter gesetzt)
        states = {}
    
        # Bottom-up Traversierung (Postorder)
        score = 0
        for node in tree.postorder():
            if node < num_samples and tree.is_leaf(node):
                # Zustand des Blattes
                g = genotypes[site_id][node]
                states[node] = {g}
            elif tree.is_leaf(node):
                states[node] = {0,1}
            else:
                # Hole Zustände der Kinder
                child_states = [states[child] for child in tree.children(node)]
    
                # Schnittmenge berechnen
                intersection = child_states[0].intersection(*child_states[1:])
                if intersection:
                    states[node] = intersection
                else:
                    # Kein gemeinsamer Zustand → Score +1, Union verwenden
                    union = set.union(*child_states)
                    states[node] = union
                    score += 1
                
        scores.append(score)
        
        site_id += 1

    return scores


def load_simulation_results_h5(output_dir):
    hgt_rates = []
    rhos = []
    fitch_scores_list = []
    gene_number_hgt_events_passed_list = []
    gene_presence_absence_matrices = []
    distance_matrices = []
    alleles_list_pca_all_runs = []

    # alle h5 Dateien finden
    files = glob.glob(os.path.join(output_dir, "*.h5"))
    
    for file in files:
        try:
            with h5py.File(file, "r") as f:
                # in jeder Datei steckt eine Gruppe namens 'results'
                grp = f["results"]
                
                hgt_rate = grp.attrs["hgt_rate"]
                rho = grp.attrs["rho"]
                fitch_score = grp.attrs["fitch_score"]
                gene_number_hgt_events_passed = grp.attrs["gene_number_hgt_events_passed"]
                
                matrix = grp["matrix"][:]
                
                distance_matrix_raw = grp["distance_matrix"][:]
                reconstructed_distance_matrix = reconstruct_distance_matrix(distance_matrix_raw, gene_absence_presence=None)
                
                # alle alleles_list_pca_X datasets einsammeln
                alleles_list_pca_matrices = []
                pca_keys = [key for key in grp.keys() if key.startswith("alleles_list_pca_")]
                pca_keys.sort()
                for key in pca_keys:
                    alleles_pca_raw = grp[key][:]
                    idx = int(key.split("_")[-1])
                    alleles_pca_reconstructed = reconstruct_distance_matrix(
                        alleles_pca_raw,
                        gene_absence_presence=matrix[idx]
                    )
                    alleles_list_pca_matrices.append(alleles_pca_reconstructed)

                # Listen wie gewohnt befüllen
                hgt_rates.append(hgt_rate)
                rhos.append(rho)
                fitch_scores_list.append(fitch_score)
                gene_number_hgt_events_passed_list.append(gene_number_hgt_events_passed)
                gene_presence_absence_matrices.append(matrix)
                distance_matrices.append(reconstructed_distance_matrix)
                alleles_list_pca_all_runs.append(alleles_list_pca_matrices)
        
        except Exception as e:
            print(f"Fehler bei Datei {file}: {e}")

    return (
        hgt_rates,
        rhos,
        fitch_scores_list,
        gene_number_hgt_events_passed_list,
        gene_presence_absence_matrices,
        alleles_list_pca_all_runs,
        distance_matrices
    )



def simulate_and_store(theta, rho, num_samples, num_genes, hgt_rate, ce_from_nwk, distance_matrix_core, output_dir):
    # run_id für eindeutige Dateinamen
    run_id = str(uuid.uuid4())
    
    mts, gene_trees_list, gene_absence_presence_matrix, alleles_list_pca, distance_matrix, fitch_scores, gene_number_hgt_events_passed = simulator(
        theta = theta, rho = rho, num_samples = num_samples, num_genes = num_genes, 
        hgt_rate = hgt_rate, ce_from_nwk = ce_from_nwk, distance_matrix = distance_matrix_core)

    if gene_absence_presence_matrix.sum() > 0:
        # Dateiname pro Prozess eindeutig
        output_file = f"{output_dir}/simulation_results_{run_id}.h5"
    
        with h5py.File(output_file, "w") as f:
            grp = f.create_group("results")
            grp.attrs["hgt_rate"] = hgt_rate
            grp.attrs["rho"] = rho
            grp.attrs["fitch_score"] = fitch_scores
            grp.attrs["gene_number_hgt_events_passed"] = gene_number_hgt_events_passed
        
            grp.create_dataset("matrix", data=gene_absence_presence_matrix)
            grp.create_dataset("distance_matrix", data=np.round(distance_matrix, 3))
            
            for i, matrix in enumerate(alleles_list_pca):
                grp.create_dataset(f"alleles_list_pca_{i}", data=np.round(matrix, 3))

    return output_dir

def wrapper(args):
    return simulate_and_store(*args)

def run_simulation(same_core_tree, num_simulations, output_dir, theta, hgt_rate_samples, rho_samples, num_samples, num_genes):
    if same_core_tree:
        core_tree = msprime.sim_ancestry(
            samples=num_samples,
            sequence_length=1,
            ploidy=1,
            recombination_rate=0,
            gene_conversion_rate=0,
            gene_conversion_tract_length=1,
        )
        ce_from_nwk = core_tree.first().newick()
        
        args = hgt_sim_args.Args(
            sample_size=num_samples,
            num_sites=num_genes,
            gene_conversion_rate=0,
            recombination_rate=0,
            hgt_rate=0,
            ce_from_ts=None,
            ce_from_nwk=ce_from_nwk,
            random_seed=secrets.randbelow(2**32 - 4) + 2,
        )
        ts, hgt_edges = hgt_simulation.run_simulate(args)
        distance_matrix_core = distance_core(tree_sequence = ts, num_samples = num_samples)
    else:
        ce_from_nwk = None
        distance_matrix_core = None

    chunk_size = 1000
    for start in range(0, num_simulations, chunk_size):
        end = min(start + chunk_size, num_simulations)
        args_list = [
            (theta, rho_samples[idx].item(), num_samples, num_genes, hgt_rate_samples[idx].item(), ce_from_nwk, distance_matrix_core, output_dir)
            for idx in range(start, end)
        ]
        with ProcessPoolExecutor(max_workers=8) as executor:
            list(executor.map(wrapper, args_list))
            
if __name__ == '__main__':
    
    num_simulations = 35000
    same_core_tree = False

    num_samples = 1000
    num_genes = 1

    ### Define random rates:

    theta = 1
    
    hgt_rate_max = 20 # Maximum hgt rate
    hgt_rate_min = 0 # Minimum hgt rate

    rho_max = 3 # Maximum gene loss rate
    rho_min = 0 # Minimum gene loss rate

    prior_hgt_rate = BoxUniform(low=hgt_rate_min * torch.ones(1), high=hgt_rate_max * torch.ones(1))
    prior_rho = BoxUniform(low=rho_min * torch.ones(1), high=rho_max * torch.ones(1))
    
    hgt_rate_samples = prior_hgt_rate.sample((num_simulations,))
    rho_samples = prior_rho.sample((num_simulations,))
    
    output_dir = r"C:\Users\uhewm\Desktop\SBI_Data\simulation_chunks"
    
    # wenn Ordner existiert, komplett löschen
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # dann neu anlegen
    os.makedirs(output_dir, exist_ok=True)

    run_simulation(same_core_tree, num_simulations, output_dir, theta, hgt_rate_samples, rho_samples, num_samples, num_genes)
    #simulator(num_samples = 1000, rho = 0, hgt_rate = 3, num_genes = 10)
