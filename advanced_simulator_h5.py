import msprime
import tskit
import hgt_simulation
import hgt_sim_args
import numpy as np
import networkx as nx
import random
import torch
import os
import re
import secrets
import copy
import json
import h5py
import uuid
import glob
import shutil
import pickle
import time

from random import randint
from collections import namedtuple
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from sbi.utils import BoxUniform
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from numba import njit
from typing import Union, Sequence, List, Tuple
from networkx.readwrite import json_graph

from concurrent.futures import ProcessPoolExecutor



@dataclass
class HGTransfer:
    recipient_parent_node: int
    recipient_child_node: int
    leaf: int
    donor_parent_node: int
    donor_child_node: int


#@profile
def simulator(
    num_samples: Union[int, None],
    theta: int = 1,
    rho: float = 0.3,
    hgt_rate: float = 0,
    num_genes: int = 1,
    nucleotide_mutation_rate: Union[float, None] = None,
    gene_length: int = 1000,
    pca_dimensions: int = 10,
    ce_from_nwk: Union[str, None] = None,
    seed: Union[int, None] = None,
    distance_matrix: Union[np.ndarray, None] = None,
    multidimensional_scaling_dimensions: int = 100,
    block_clustering_threshold: Union[float, Sequence[float]] = np.arange(0, 1.0001, 0.025),
    clonal_root_mutation: Union[bool, None] = None,
    hgt_difference_removal_threshold: float = 0,
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

    random.seed(seed)
    np.random.seed(seed)
    #print("Seed: ", seed)
    
    if seed is None:
        seed = secrets.randbelow(2**32 - 4) + 2

    if nucleotide_mutation_rate is None:
        #nucleotide_mutation_rate = np.random.uniform(0.033, 0.066)
        nucleotide_mutation_rate = 0.1

    if num_samples < multidimensional_scaling_dimensions:
        multidimensional_scaling_dimensions = num_samples


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
    
    # Place one mutation per site, e.g. gen
    
    gains_model = msprime.MatrixMutationModel(
        alleles = alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [0, 1],
            [0, 1],
        ],
    )

    k = 0
    tables = ts.tables
    ts_gains = tables.tree_sequence()
    while True:
        ts_gains = msprime.sim_mutations(ts_gains,rate=1,model=gains_model,keep=True,random_seed=seed+k,end_time=core_tree.node(core_tree.first().root).time)
        
        tables = ts_gains.dump_tables()
        tables.mutations.clear()  
        for tree in ts_gains.trees():
            for site in tree.sites():
                for mut in site.mutations:
                    # Alle Blätter unterhalb des Mutations-Knotens holen
                    leaf_nodes = list(tree.leaves(mut.node))
        
                    # Prüfen, ob eines der Blätter ein "echtes Sample" ist
                    if any(leaf < num_samples for leaf in leaf_nodes):
                        # Mutation behalten
                        tables.mutations.add_row(
                            site=mut.site,
                            node=mut.node,
                            derived_state=mut.derived_state,
                            metadata=mut.metadata,
                            time=mut.time
                        )
        
        ts_gains = tables.tree_sequence()

        # Prüfen: jede Site mindestens eine Mutation?
        all_sites_ok = True
        for site_id in range(num_genes):
            site_mutations = [m for m in ts_gains.tables.mutations if m.site == site_id]
            if len(site_mutations) == 0:
                all_sites_ok = False
                break
        
        if all_sites_ok:
            break 
    
        k += 1

    # Remove superfluous mutations
    
    tables = ts_gains.dump_tables()
    
    mutations_by_site = {}
    for mut in tables.mutations:
        if mut.site not in mutations_by_site:
            mutations_by_site[mut.site] = []
        mutations_by_site[mut.site].append(mut)
    
    tables.mutations.clear()

    clonal_root_node = ts_gains.first().mrca(*list(range(num_samples)))
    
    print("Clonal node:", clonal_root_node)

    if clonal_root_mutation == None:
        clonal_root_mutation = random.random() < 0.5

    present_mutations = [[] for i in range(num_genes)]
    for site, mutations in mutations_by_site.items():

        if clonal_root_mutation == True:
            tables.mutations.add_row(
                site=site,
                node=clonal_root_node,
                derived_state="present",
                parent=-1,
                metadata=None,
                time=tables.nodes.time[clonal_root_node],
            )
            
        else:
            selected_mutation = random.choice(mutations)
            tables.mutations.add_row(
                site=selected_mutation.site,
                node=selected_mutation.node,
                derived_state=selected_mutation.derived_state,
                parent=-1,
                metadata=None,
                time=selected_mutation.time,
            )
            present_mutations[site].append(selected_mutation)

            # Mutation at clonal root:
            tables.mutations.add_row(
                site=site,
                node=clonal_root_node,
                derived_state="absent",
                parent=-1,
                metadata=None,
                time=tables.nodes.time[clonal_root_node],
            )
    
        # Add sentinel mutations at the leafs:
    
        for leaf_position in range(num_samples):
            tables.mutations.add_row(
                site = site,
                node = leaf_position,
                derived_state = "absent",
                time = 0.00000000001,
            )

    tables.sort()
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

    ### Calculate the gene absence presence matrix:
    
    MutationRecord = namedtuple('MutationRecord', ['site_id', 'mutation_id', 'node', 'is_hgt'])
    
    tables = ts_gains_losses.dump_tables()
    
    tables.mutations.clear()
    
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
    gene_number_loss_events = [0 for _ in range(num_genes)]
    gene_nodes_loss_events = [[] for _ in range(num_genes)]
    gene_trees_list = []
    hgt_parent_children_passed_list = []
    
    for tree in ts_gains_losses.trees():

        for site in tree.sites():
            
            hgt_parent_children_passed = [False] * ts_gains_losses.num_nodes
            mutations = site.mutations

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
                    if not tree.parent(present_mutation.node) == -1: # Can occur if its a clonal root mutation
                        tables_gene_list[site.id].edges.add_row(
                                left = 0, right = gene_length, parent = tree.parent(present_mutation.node), child = sentinel_mutation.node
                        )
                else:
                    gene_number_loss_events[site.id] += 1
                    gene_nodes_loss_events[site.id].append(present_mutation.node)
        
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

                hgt_parent_children_passed_list.append(hgt_parent_children_passed)
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
                                    gene_number_loss_events[site.id] += 1
                                    gene_nodes_loss_events[site.id].append(earliest_mutation.node)
                            else:
                                children = tree.children(child_node[0])
                                if len(children) > 1:
                                    for child in reversed(children):
                                        if not branching_nodes_reached_before[child_node[0]]:
                                            branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                    if not tree.parent(last_branching_node[0]) == -1:
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
                                if not tree.parent(last_branching_node[0]) == -1:
                                    tables_gene_list[site.id].edges.add_row(
                                            left = 0, right = gene_length, parent = tree.parent(last_branching_node[0]), child = child_node[0]
                                    )
                                gene_number_hgt_events_passed[site.id] += child_node[3]
                            else:
                                for child in reversed(children):
                                    selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                            if hgt_parent_children[child_node[0]]:
                                selected_branch_nodes_to_process.extendleft([(hgt_parent_children[child_node[0]][0], True, True, child_node[3] + 1)])

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

    for i in range(num_genes): 

        tables_gene_list[i].sort()
        
        gene_trees_list.append(tables_gene_list[i].tree_sequence())

    ### Compute the gene presence and absence:
    
    gene_absence_presence_matrix = []
    
    for var in mts.variants():
        gene_absence_presence_matrix.append(var.genotypes)
    gene_absence_presence_matrix = np.array(gene_absence_presence_matrix)

    print("Number of present genes: ", sum(gene_absence_presence_matrix[0]))
    
    fitch_scores = fitch_parsimony_score(mts, gene_absence_presence_matrix)

    ### Calculate the HGT events:

    gene_number_hgt_events_passed = [0 for _ in range(num_genes)]
    nodes_hgt_events = [[] for _ in range(num_genes)]
    hgt_trees = [[] for _ in range(num_genes)]
    
    for tree in mts.trees():
        for site in tree.sites():
            reached_nodes_from_leaves = defaultdict(list)
            for node_id in range(mts.num_nodes):
                reached_nodes_from_leaves[node_id] = False
            
            stack = []
            for node_id in range(mts.num_samples):
                if gene_absence_presence_matrix[site.id][node_id]:
                    stack.append((node_id, node_id))
                    
            while stack:
                node, leaf_node = stack.pop()
                parent = tree.parent(node)
                if not hgt_parent_children_passed[node] and not reached_nodes_from_leaves[node] and node < clonal_root_node:
                    stack.append((parent, leaf_node))
                    reached_nodes_from_leaves[node] = True
                elif hgt_parent_children_passed[node] and not reached_nodes_from_leaves[node] and node < clonal_root_node:
                    #print("HGT event spotted at node: ", node)
                    nodes_hgt_events[site.id].append((node, leaf_node, None))
                    gene_number_hgt_events_passed[site.id] += 1
                    reached_nodes_from_leaves[node] = True
    
            tables = gene_trees_list[site.id].tables
            for subtree in gene_trees_list[site.id].trees():
                hgt_events_with_origins = []
                for node, leaf_node, _ in nodes_hgt_events[site.id]:
                    temp_time = 0
                    parent = leaf_node
                    time_of_node = gene_trees_list[site.id].node(node).time
                    while temp_time < time_of_node:
                        child = parent
                        parent = subtree.parent(parent)
                        temp_time = gene_trees_list[site.id].node(parent).time
    
                    mask = [
                        not (edge.parent == parent and edge.child == child)
                        for edge in tables.edges
                    ]
                    
                    tables.edges.set_columns(
                        left=[e.left for i, e in enumerate(tables.edges) if mask[i]],
                        right=[e.right for i, e in enumerate(tables.edges) if mask[i]],
                        parent=[e.parent for i, e in enumerate(tables.edges) if mask[i]],
                        child=[e.child for i, e in enumerate(tables.edges) if mask[i]],
                    )
                    
                    tables.edges.add_row(left = 0, right = gene_length, parent=node, child=child)
                    tables.edges.add_row(left = 0, right = gene_length, parent=parent, child=node)

                    """
                    # Find correct HGT origin:
                    hgt_origin = parent
                    while True:
                        child1, child2 = tuple(subtree.children(hgt_origin))
                        child1_leaves = [leaf for leaf in subtree.leaves(child1) if leaf < num_samples]
                        child2_leaves = [leaf for leaf in subtree.leaves(child2) if leaf < num_samples]
                    
                        if (len(child1_leaves) > 0 and len(child2_leaves) > 0   and child1_leaves != child2_leaves):
                            break
                        else:
                            hgt_origin = subtree.parent(hgt_origin)


                    hgt_events_with_origins.append((node, leaf_node, hgt_origin))
                    """
                    hgt_events_with_origins.append((node, leaf_node, parent))
                    
                nodes_hgt_events[site.id] = hgt_events_with_origins

            tables.sort()
            gene_trees_list[site.id] = tables.tree_sequence()
    
            valid_nodes_hgt_events = []
            for subtree in gene_trees_list[site.id].trees():
                for node, leaf_node, hgt_origin in nodes_hgt_events[site.id]:
    
                    tables = mts.tables
                    
                    temp_time = 0
                    parent = leaf_node
                    time_of_node = gene_trees_list[site.id].node(node).time
    
                    mask = [
                        not (edge.parent == tree.parent(node) and edge.child == node)
                        for edge in tables.edges
                    ]
    
                    tables.edges.set_columns(
                        left=[e.left for i, e in enumerate(tables.edges) if mask[i]],
                        right=[e.right for i, e in enumerate(tables.edges) if mask[i]],
                        parent=[e.parent for i, e in enumerate(tables.edges) if mask[i]],
                        child=[e.child for i, e in enumerate(tables.edges) if mask[i]],
                    )
                    
                    tables.edges.add_row(left = site.id, right = site.id+1, parent=subtree.parent(node), child=node)
    
                    tables.sort()
                    tables.mutations.clear()
    
                    hgt_tree = tables.tree_sequence()
    
                    hgt_trees[site.id].append((hgt_tree, (node, leaf_node)))
                    
                    keep = False
                    for leaf in range(num_samples):
                        if gene_absence_presence_matrix[site.id][leaf] == 1 and abs(
                            hgt_tree.node(hgt_tree.at(hgt_tree.sites()[site.id].position).mrca(node, leaf)).time - mts.node(tree.mrca(node, leaf)).time) > hgt_difference_removal_threshold:
                            keep = True
                            break
                    if keep:
                        valid_nodes_hgt_events.append((node, leaf_node, hgt_origin))
                    else:
                        print("Removed HGT edge")
            nodes_hgt_events[site.id] = valid_nodes_hgt_events
    
            """
            # Add the present mutation(s) as an HGT node, if it lies on the core tree:
            for mut in present_mutations[site.id]:
                reachable_samples_from_gaining_mutations = set(tree.samples(mut.node))
                if any(reachable_samples_from_gaining_mutations):
                    nodes_hgt_events[site.id].append((mut.node, next(iter(reachable_samples_from_gaining_mutations))))
                    print("Added HGT node")
            """
      
                

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
            # C     M     
            [ 0.0,  1 ], 
            [ 1/3,  2/3],  
        ]
    )

    
    nucleotide_mutation = msprime.InfiniteAlleles()
    
    
    for i in range(num_genes): 
        
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

        
    ### Calculate the distances between the leaves

    if distance_matrix is None:
        distance_matrix = distance_core(tree_sequence = mts, num_samples = num_samples)
    scaled_distance_matrix = multidimensional_scaling(distance_matrix, multidimensional_scaling_dimensions = multidimensional_scaling_dimensions)
    
    # Distances in alleles (no pca)

    alleles_list_pca = []
    core_allel_distance_list = []
    core_allel_distance_number_of_clusters = []
    
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

        distance_matrix_valid = distance_matrix[np.ix_(gene_present_vector, gene_present_vector)]
        euclidean_distances_valid = euclidean_distances[np.ix_(gene_present_vector, gene_present_vector)]
        euclidean_distances_valid = euclidean_distances_valid ** 2 # Return to Hamming distances.
        
        #distance_matrix_valid = distance_matrix_valid / np.max(distance_matrix_valid)
        if gene_present_vector.sum() > 0 and np.max(euclidean_distances_valid) > 0:
            euclidean_distances_valid = euclidean_distances_valid / np.mean(euclidean_distances_valid) * np.mean(distance_matrix_valid)

        core_allel_distance_valid = distance_matrix_valid - euclidean_distances_valid

        core_allel_distance_list.append(core_allel_distance_valid)

        core_allel_distance_cluster_results = count_blocks_sym(core_allel_distance_valid, block_clustering_threshold)
        
        core_allel_distance_number_of_clusters.append(core_allel_distance_cluster_results)

    ### Construct the graph:
    
    tree = mts.first()
    children = defaultdict(list)
    for leaf in range(num_samples):
        node = leaf
        while node < clonal_root_node and len(children[node]) < 2:
            parent = tree.parent(node)
            children[parent].append(node)
            node = parent
            if parent == tskit.NULL:
                break
                  
    for node in list(children):
        if len(children[node]) == 1:
            parent = tree.parent(node)
            if parent == tskit.NULL:
                continue  # kein Parent → überspringen
            child = children[node][0]  # das eine Kind
    
            # 1. altes Kind durch neues Kind ersetzen
            if node in children[parent]:
                children[parent].remove(node)
            children[parent].append(child)
    
            # 2. entferne den Zwischenknoten
            del children[node]

    parents = defaultdict(int)
    for parent, child_list in children.items():
        for child in child_list:
            parents[child] = parent

    node_to_leaf = defaultdict(list)
    for child in sorted(list(parents)):
        if child < num_samples:
            node_to_leaf[child] = [child]
        node_to_leaf[parents[child]].extend(node_to_leaf[child])

    Graphs = []
    graph_properties = []
    #parental_nodes_hgt_events = [[] for _ in range(num_genes)]
    #parental_nodes_hgt_events_corrected = [[] for _ in range(num_genes)]
    nodes_hgt_events_simplified = [[] for _ in range(num_genes)]
    children_gene_nodes_loss_events = [[] for _ in range(num_genes)]

    HGTEvent = namedtuple("HGTEvent", ["recipient", "leaf", "donor"])

    for tree in mts.trees():
        for site in tree.sites():
            
            G = nx.DiGraph()
            # Alle Knoten sammeln
            all_nodes = set(children.keys())
            for child_list in children.values():
                all_nodes.update(child_list)
            
            # Sortierte Nodes hinzufügen
            for node in sorted(all_nodes):
                G.add_node(node, core_distance=0, allele_distance=0, true_allele_distance = 0, time = tree.get_time(node))
                
            for parent, child_list in children.items():
                for child in child_list:
                    #G.add_edge(child, parent)
                    G.add_edge(parent, child)

            # Add clonal distances:
            subtree_has_gene = {}
            for node in range(clonal_root_node):
                if node < num_samples:
                    if gene_absence_presence_matrix[site.id][node] == 1:
                        subtree_has_gene[node] = True
                    else:
                        subtree_has_gene[node] = False
                else:
                    child_list = children[node]
                    if any(subtree_has_gene[child] for child in child_list):
                        subtree_has_gene[node] = True
                    else:
                        subtree_has_gene[node] = False

            for node in sorted(range(num_samples, clonal_root_node + 1), key=lambda n: tree.get_time(n)):
                if node in G.nodes:
                    c0, c1 = children[node]  # Entpacke einmal
                    c0_has, c1_has = subtree_has_gene[c0], subtree_has_gene[c1]
                
                    # Falls beide Subtrees kein Gen enthalten, können wir überspringen
                    if not (c0_has or c1_has):
                        continue
                
                    #if c0_has and c1_has:
                    #    node_time = tree.get_time(node)
                    #    core_distance = (
                    #        2 * node_time
                    #        - tree.get_time(c0) - tree.get_time(c1)
                    #        + G.nodes[c0]["core_distance"]
                    #        + G.nodes[c1]["core_distance"]
                    #    )
                    if c0_has and c1_has:
                        # Finde tiefste Kinderknoten, bei denen beide Subtrees True sind
                        if c0 >= num_samples:
                            d0, d1 = children[c0]
                            while (not subtree_has_gene[d0] and subtree_has_gene[d1]) or (subtree_has_gene[d0] and not subtree_has_gene[d1]):
                                if subtree_has_gene[d0]:
                                    c0 = d0
                                    if d0 < num_samples:
                                        break
                                elif subtree_has_gene[d1]:
                                    c0 = d1
                                    if d1 < num_samples:
                                        break
                                d0, d1 = children[c0]
                                        
                        if c1 >= num_samples:
                            d0, d1 = children[c1]
                            while (not subtree_has_gene[d0] and subtree_has_gene[d1]) or (subtree_has_gene[d0] and not subtree_has_gene[d1]):
                                if subtree_has_gene[d0]:
                                    c1 = d0
                                    if d0 < num_samples:
                                        break
                                elif subtree_has_gene[d1]:
                                    c1 = d1
                                    if d1 < num_samples:
                                        break
                                d0, d1 = children[c1]
             
                        node_time = tree.get_time(node)
                        core_distance = (
                            2 * node_time
                            - tree.get_time(c0) - tree.get_time(c1)
                            + G.nodes[c0]["core_distance"]
                            + G.nodes[c1]["core_distance"]
                        )
                    elif c0_has:
                        core_distance = G.nodes[c0]["core_distance"]
                    else:  # c1_has
                        core_distance = G.nodes[c1]["core_distance"]

                    G.nodes[node]["core_distance"] = core_distance
                    
            for node in range(num_samples, clonal_root_node + 1):
                if node in G.nodes:
                    G.nodes[node]["core_distance"] = (1 - (1 - 1/gene_length)**(G.nodes[node]["core_distance"] * nucleotide_mutation_rate * gene_length))

            # Multiply to get the expected amount of mutations:
            """
            for node in range(num_samples, clonal_root_node + 1):
                if node in G.nodes:
                    G.nodes[node]["core_distance"] = G.nodes[node]["core_distance"] * gene_length * nucleotide_mutation_rate
            """
            
            #### Add allele distances:
            
            # Calculate for every node the number of SNPs below it.
            allele_distances = defaultdict(list)
            allele_distances_both_children_polymorph = defaultdict(list)
            gene_present_bool = gene_absence_presence_matrix[site.id] == 1
            
            for node, leaves in node_to_leaf.items():
                if alleles_list[site.id].ndim == 1:
                   alleles_list[site.id] = np.zeros((1, num_samples), dtype=int)
                leaves_arr = np.array(leaves)
                valid_cols = leaves_arr[gene_present_bool[leaves_arr]]
                if valid_cols.size == 0:
                    distance = 0
                else:
                    subset = alleles_list[site.id][:, valid_cols]
                    distance = np.count_nonzero(np.ptp(subset, axis=1)) # ptp = max - min
                allele_distances[tuple(leaves)] = distance

            # Calculate for every node the number of SNPs below it which were not already SNPs in at least one of the children (new SNPs).
            allele_distances_only_new = defaultdict(list)
            gene_present_bool = gene_absence_presence_matrix[site.id] == 1
            
            for node, leaves in node_to_leaf.items():
                if alleles_list[site.id].ndim == 1:
                    alleles_list[site.id] = np.zeros((1, num_samples), dtype=int)
            
                leaves_arr = np.array(leaves)
                valid_cols = leaves_arr[gene_present_bool[leaves_arr]]
            
                if valid_cols.size == 0:
                    new_distance = 0
                    shared_distance = 0                    
                else:
                    subset = alleles_list[site.id][:, valid_cols]
                    
                    # Polymorphe Positionen im aktuellen Knoten
                    polymorph_mask = np.ptp(subset, axis=1) > 0   # shape = [num_sites]
                    
                    # Kinder-Masken sammeln
                    child_polymorph_mask_list = []
                    for child in G.successors(node): 
                        child_leaves = np.array(node_to_leaf[child])
                        child_valid = child_leaves[gene_present_bool[child_leaves]]
                        child_subset = alleles_list[site.id][:, child_valid]
                    
                        if child_valid.size > 0:
                            child_mask = np.ptp(child_subset, axis=1) > 0
                        else:
                            child_mask = np.zeros(subset.shape[0], dtype=bool)
                    
                        child_polymorph_mask_list.append(child_mask)
                    
                    if child_polymorph_mask_list:
                        # OR über alle Kinder -> Position polymorph in mind. einem Kind
                        child_any_mask = np.logical_or.reduce(child_polymorph_mask_list)
                    
                        # AND über alle Kinder -> Position polymorph in allen Kindern
                        if len(child_polymorph_mask_list) > 1:
                            child_all_mask = np.logical_and.reduce(child_polymorph_mask_list)
                        else:
                            # nur ein Kind => all = that one
                            #child_all_mask = child_polymorph_mask_list[0]
                            child_all_mask = np.zeros_like(polymorph_mask)
                    else:
                        child_any_mask = np.zeros_like(polymorph_mask)
                        child_all_mask = np.zeros_like(polymorph_mask)

                    # Nur Positionen, die polymorph sind UND in allen Kindern noch monomorph
                    new_polymorph_mask = polymorph_mask & (~child_any_mask)
                    new_distance = np.count_nonzero(new_polymorph_mask)
                    
                    # Positionen, die in allen Kindern polymorph sind
                    shared_polymorph_mask = polymorph_mask & child_all_mask
                    shared_distance = np.count_nonzero(shared_polymorph_mask)
                    
                    
                allele_distances_only_new[tuple(leaves)] = new_distance
                allele_distances_both_children_polymorph[tuple(leaves)] = shared_distance


            """
            for node in list(parents) + [clonal_root_node]:
                G.nodes[node]["allele_distance"] = allele_distances[tuple(node_to_leaf[node])]
            """

            for node in list(parents) + [clonal_root_node]:
                G.nodes[node]["allele_distance"] = allele_distances[tuple(node_to_leaf[node])] / gene_length
                G.nodes[node]["allele_distance_only_new"] = allele_distances_only_new[tuple(node_to_leaf[node])] / gene_length
                G.nodes[node]["allele_distances_both_children_polymorph"] = allele_distances_both_children_polymorph[tuple(node_to_leaf[node])] / gene_length

            for gen_tree in gene_trees_list[site.id].trees():
                for node in list(parents) + [clonal_root_node]:
                    
                    # Calculate the expected number of alleles, which also can be seen as the true allele distance:
                    leaves = [leaf for leaf in node_to_leaf[node] if gene_absence_presence_matrix[site.id][leaf] == 1]
                    if len(leaves) <= 1:
                        G.nodes[node]["true_allele_distance"] = 0
                    else:
                        m = leaves[0]
                        for u in leaves[1:]:
                            m = gen_tree.mrca(m, u) # The most recent common ancestor of all leaves from the node
                        
                        visited_edges = set()
                        total_length = 0.0 
                        for leaf in leaves:
                            u = leaf
                            while u != tskit.NULL and u < m:
                                p = gen_tree.parent(u)
                                if p == tskit.NULL:
                                    break
                                edge = (p, u)
                                if edge not in visited_edges:
                                    visited_edges.add(edge)
                                    total_length += gen_tree.time(p) - gen_tree.time(u)
                                # weiter hoch
                                u = p
              
                        # The expected proportion of SNPS:
                        G.nodes[node]["true_allele_distance"] = (1 - (1 - 1/gene_length)**(total_length * nucleotide_mutation_rate * gene_length))

            # Set the numbers of the nodes in the graph to 0 to num_samples-1 :
            G_nodes_reordering = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
            nx.relabel_nodes(G, G_nodes_reordering, copy=False)

            # Fix the origin of the HGT events:
            corrected_hgt_events = []
            corrected_hgt_events_simplified = []
            for node, leaf, hgt_origin in nodes_hgt_events[site.id]:
                hgt_origin = node + 1
                while True:
                    if hgt_origin in G_nodes_reordering.keys() and [leaf for leaf in tree.leaves(hgt_origin) if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1]:
                        break
                    hgt_child = hgt_origin
                    if not hgt_parent_children_passed_list[site.id][hgt_origin]:
                        hgt_origin = tree.parent(hgt_origin)
                    else:
                        hgt_origin = hgt_origin + 1

                if any([[leaf for leaf in tree.leaves(node) if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1] == [] for node in tree.children(hgt_origin)]):
                    hgt_child = hgt_origin
                while True:
                    if hgt_child != hgt_origin and hgt_child in G_nodes_reordering.keys():
                        break
                    children_hgt = tree.children(hgt_child)
                    if len(children_hgt) == 1:
                        hgt_child = children_hgt[0]
                        counts = [1]
                    else:
                        counts = []
                        for child in children_hgt:
                            child_leaves = [leaf for leaf in tree.leaves(child) if leaf < num_samples]

                            valid_leaves = [leaf for leaf in child_leaves if gene_absence_presence_matrix[site.id][leaf] == 1]
                    
                            counts.append((child, len(valid_leaves)))                      
                            

                        hgt_child = max(counts, key=lambda x: x[1])[0]
              
                #corrected_hgt_events.append((node, leaf, (hgt_origin, hgt_child)))
            #nodes_hgt_events[site.id] = corrected_hgt_events

            # Move nodes with HGT events to the right node.
            #for node, leaf, hgt_origin in nodes_hgt_events[site.id]:
                hgt_recipient_parent = node
                hgt_recipient_child = node
                while hgt_recipient_parent not in G_nodes_reordering.keys():
                    hgt_recipient_parent = tree.parent(hgt_recipient_parent)
                while hgt_recipient_child not in G_nodes_reordering.keys():
                    children_hgt = tree.children(hgt_recipient_child)
                    if len(children_hgt) == 1:
                        hgt_recipient_child = children_hgt[0]
                    else:
                        counts = []
                        for child in children_hgt:
                            child_leaves = [leaf for leaf in tree.leaves(child) if leaf < num_samples]
                            valid_leaves = [leaf for leaf in child_leaves if gene_absence_presence_matrix[site.id][leaf] == 1]
                            counts.append((child, len(valid_leaves)))
                        hgt_recipient_child = max(counts, key=lambda x: x[1])[0]

                if hgt_recipient_parent <= clonal_root_node:
                    event = HGTransfer(
                        recipient_parent_node=hgt_recipient_parent,
                        recipient_child_node=hgt_recipient_child,
                        leaf=leaf,
                        donor_parent_node=hgt_origin,
                        donor_child_node=hgt_child,
                    )
                    corrected_hgt_events.append(event)
                    event_simplified = HGTransfer(
                        recipient_parent_node=G_nodes_reordering[hgt_recipient_parent],
                        recipient_child_node=G_nodes_reordering[hgt_recipient_child],
                        leaf=leaf,
                        donor_parent_node=G_nodes_reordering[hgt_origin],
                        donor_child_node=G_nodes_reordering[hgt_child],
                    )
                    corrected_hgt_events_simplified.append(event_simplified)
                             
            nodes_hgt_events[site.id] = corrected_hgt_events
            nodes_hgt_events_simplified[site.id] = corrected_hgt_events_simplified
            
            for node in gene_nodes_loss_events[site.id]:
                stack = [node]
                while stack:
                    node = stack.pop()
                    if node in G_nodes_reordering.keys():
                        children_gene_nodes_loss_events[site.id].append(node)
                        break
                    else:
                        if node >= num_samples:
                            child_list = tree.children(node)
                            stack.extend(child_list)

            
            #parental_nodes_hgt_events_corrected[site.id] = [G_nodes_reordering[node] for node in nodes_hgt_events[site.id]]
            #parental_nodes_hgt_events[site.id] = [G_nodes_reordering[node] for node in parental_nodes_hgt_events[site.id]]
            children_gene_nodes_loss_events[site.id] = [G_nodes_reordering[node] for node in children_gene_nodes_loss_events[site.id]]


            Graphs.append(G)

            # Process the graph to save it more easily:
            graph_nodes = list(G.nodes)
            graph_edges = list(G.edges)
            graph_edge_index = torch.tensor(np.array(graph_edges).T)

            node_features = []
            for node in graph_nodes:
                core_distance = G.nodes[node].get("core_distance", 0.0)
                allele_distance = G.nodes[node].get("allele_distance", 0.0)
                allele_only_new = G.nodes[node].get("allele_distance_only_new", 0.0)
                allele_distances_both_children_polymorph = G.nodes[node].get("allele_distances_both_children_polymorph", 0.0)
                true_allele_distance = G.nodes[node].get("true_allele_distance", 0.0)
                node_time = G.nodes[node].get("time", 0.0)
                node_features.append([core_distance, allele_distance, allele_only_new, allele_distances_both_children_polymorph, true_allele_distance, node_time])
            node_features = torch.tensor(node_features, dtype=torch.float32)

            graph_properties.append([graph_nodes, graph_edge_index, node_features])

       
    ### Print the computation time.
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Success: hgt_rate = {hgt_rate}, Total computation time = {elapsed_time:.6f} seconds.")

    return {
        "mts": mts,
        "gene_trees_list": gene_trees_list[0] if num_genes == 1 else gene_trees_list,
        "gene_absence_presence_matrix": gene_absence_presence_matrix[0] if num_genes == 1 else gene_absence_presence_matrix,
        "alleles_list_pca": alleles_list_pca[0] if num_genes == 1 else alleles_list_pca,
        "scaled_distance_matrix": scaled_distance_matrix,
        "fitch_scores": fitch_scores[0] if num_genes == 1 else fitch_scores,
        "gene_number_hgt_events_passed": gene_number_hgt_events_passed[0] if num_genes == 1 else gene_number_hgt_events_passed,
        "distance_matrix": distance_matrix,
        "core_allel_distance_number_of_clusters": core_allel_distance_number_of_clusters[0] if num_genes == 1 and core_allel_distance_number_of_clusters else core_allel_distance_number_of_clusters,
        "gene_number_loss_events": gene_number_loss_events[0] if num_genes == 1 else gene_number_loss_events,
        "alleles_list": alleles_list[0] if num_genes == 1 else alleles_list,
        "clonal_root_node": clonal_root_node,
        "graphs": Graphs[0] if num_genes == 1 else Graphs,
        "graph_properties": graph_properties[0] if num_genes == 1 else graph_properties,
        "nodes_hgt_events_simplified": nodes_hgt_events_simplified[0] if num_genes == 1 else nodes_hgt_events_simplified,
        "nodes_hgt_events": nodes_hgt_events[0] if num_genes == 1 else nodes_hgt_events,
        "children_gene_nodes_loss_events": children_gene_nodes_loss_events[0] if num_genes == 1 else children_gene_nodes_loss_events,
        "node_to_leaf": node_to_leaf,
        #"parental_nodes_hgt_events_corrected": parental_nodes_hgt_events_corrected[0] if num_genes == 1 else parental_nodes_hgt_events_corrected,
        "G_nodes_reordering": G_nodes_reordering,
    }

def deepest_both_has(node, children, subtree_has_gene):
    """
    Gehe vom gegebenen Knoten aus nach unten, solange beide Kinder das Gen haben.
    Gibt den 'tiefsten' Knoten zurück, bei dem beide Subtrees True sind.
    """
    # Wenn Blatt, einfach zurück
    if node not in children:
        return node
    
    c0, c1 = children[node]
    if subtree_has_gene[c0] and subtree_has_gene[c1]:
        # Dann können wir noch tiefer gehen
        # Rekursiv jeweils nach unten
        c0_deep = deepest_both_has(c0, children, subtree_has_gene)
        c1_deep = deepest_both_has(c1, children, subtree_has_gene)
        # Hier entscheidest du, ob du beide zurückgeben willst oder einen
        # -> für deinen Fall reicht es, den Knoten selbst zurückzugeben
        return node
    else:
        return node

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

@njit                       # ---------- Kern für *einen* threshold ----------
def _count_blocks_single(mat, idx, threshold):
    n = idx.size
    label_matrix = -np.ones((n, n), dtype=np.int32)
    num_labels   = 0

    for ii in range(n):
        i = idx[ii]
        for jj in range(ii, n):
            j = idx[jj]
            if ii == 0 and jj == 0:
                label_matrix[ii, jj] = 0
                num_labels += 1
            elif ii == 0:
                if abs(mat[i, j] - mat[i, idx[jj-1]]) < threshold:
                    label_matrix[ii, jj] = label_matrix[ii, jj-1]
                else:
                    label_matrix[ii, jj] = num_labels
                    num_labels += 1
            else:
                if abs(mat[i, j] - mat[idx[ii-1], j]) < threshold:
                    label_matrix[ii, jj] = label_matrix[ii-1, jj]
                elif abs(mat[i, j] - mat[i, idx[jj-1]]) < threshold:
                    label_matrix[ii, jj] = label_matrix[ii, jj-1]
                else:
                    label_matrix[ii, jj] = num_labels
                    num_labels += 1
    return label_matrix, num_labels


# ----------------------- Benutzer‑Funktion -----------------------
def count_blocks_sym(mat: np.ndarray,
                     threshold: Union[float, Sequence[float]] = 0.05
                    ) -> List[Tuple[np.ndarray, int]]:
    """
    Führe die Blockzählung für einen oder mehrere threshold‑Werte durch.

    Parameters
    ----------
    mat : np.ndarray (N, N)
        Symmetrische Matrix mit NaNs.
    threshold : float | list[float]
        Einzelner Grenzwert oder Liste davon.

    Returns
    -------
    results : list[(label_matrix, num_labels)]
        Liste in derselben Reihenfolge wie threshold(s).
    """
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], "Matrix muss quadratisch sein."
    thresholds = np.atleast_1d(threshold).astype(np.float64)

    if mat.size == 0:
        return np.zeros_like(thresholds)
    
    # Vorarbeit außerhalb der Schleife
    idx = np.where(~np.isnan(mat[0]))[0]

    #results_matrices = []
    results_number_of_clusters = []
    for th in thresholds:
        labels, k = _count_blocks_single(mat, idx, th)
        #results_matrices.append(labels)
        results_number_of_clusters.append(int(k))
    results_number_of_clusters = np.array(results_number_of_clusters, dtype=int)
    #return results_matrices, results_number_of_clusters
    return results_number_of_clusters


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
                    gene_present_vector = (matrix[idx, :] == 1)
                    print(gene_present_vector)
                    mean_core_distance = np.mean(reconstructed_distance_matrix[np.ix_(gene_present_vector, gene_present_vector)])
                    mean_allel_distance = np.mean(alleles_pca_reconstructed[np.ix_(gene_present_vector, gene_present_vector)])
                    if mean_allel_distance > 0:
                        alleles_pca_reconstructed[np.ix_(gene_present_vector, gene_present_vector)] = alleles_pca_reconstructed[np.ix_(gene_present_vector, gene_present_vector)] / mean_allel_distance * mean_core_distance
                        
                    alleles_list_pca_matrices.append(alleles_pca_reconstructed)

                # Scale the distances depending on the core distances to make the comparison happening later easier.
                gene_present_vector = (matrix == 1)
                mean_core_distance = np.mean(reconstructed_distance_matrix[np.ix_(gene_present_vector, gene_present_vector)])
                mean_allel_distance = np.mean(euclidean_distances[np.ix_(gene_present_vector, gene_present_vector)])
                if mean_allel_distance > 0:
                    euclidean_distances = euclidean_distances / mean_allel_distance #* np.sqrt(mean_core_distance)

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
    
    data = simulator(theta = theta, rho = rho, num_samples = num_samples, num_genes = num_genes, hgt_rate = hgt_rate, ce_from_nwk = ce_from_nwk, distance_matrix = distance_matrix_core)

    graph_properties = data["graph_properties"]
    nodes_hgt_events = data["nodes_hgt_events"]
    children_gene_nodes_loss_events = data["children_gene_nodes_loss_events"]
    gene_absence_presence_matrix = data["gene_absence_presence_matrix"]
    fitch_scores = data["fitch_scores"]
    gene_number_loss_events = data["gene_number_loss_events"]
    gene_number_hgt_events_passed = data["gene_number_hgt_events_passed"]
    nodes_hgt_events_simplified = data["nodes_hgt_events_simplified"]
    
    if gene_absence_presence_matrix.sum() > 1:
        # Dateiname pro Prozess eindeutig
        output_file = f"{output_dir}/simulation_results_{run_id}.h5"

        """
        with h5py.File(output_file, "w") as f:
            grp = f.create_group("results")
            grp.attrs["gene_absence_presence_matrix"] = gene_absence_presence_matrix
            grp.attrs["hgt_rate"] = hgt_rate
            grp.attrs["rho"] = rho
            grp.attrs["fitch_score"] = fitch_scores
            grp.attrs["gene_number_hgt_events_passed"] = gene_number_hgt_events_passed
            grp.attrs["gene_number_loss_events"] = gene_number_loss_events
            grp.attrs["nodes_hgt_events"] = nodes_hgt_events
            grp.attrs["nodes_hgt_events_simplified"] = nodes_hgt_events_simplified 
            grp.attrs["children_gene_nodes_loss_events"] = children_gene_nodes_loss_events
            grp.create_dataset("graph_properties", data=np.void(pickle.dumps(graph_properties)))
            
            for site_id, events in nodes_hgt_events.items():
                # convert list[HGTransfer] -> structured numpy array
                arr = np.array([(ev.recipient_parent_node, ev.recipient_child_node, ev.leaf, ev.donor_parent_node, ev.donor_child_node) for ev in events], dtype=dtype)
    
                # store under key = site_id
                grp.create_dataset(str(site_id), data=arr)

            for site_id, events in nodes_hgt_events_simplified.items():
                # convert list[HGTransfer] -> structured numpy array
                arr = np.array([(ev.recipient_parent_node, ev.recipient_child_node, ev.leaf, ev.donor_parent_node, ev.donor_child_node) for ev in events], dtype=dtype)
    
                # store under key = site_id
                grp.create_dataset(str(site_id), data=arr)
        """

        # Structured dtype for HGTransfer
        dtype = np.dtype([
            ("recipient_parent_node", np.int32),
            ("recipient_child_node", np.int32),
            ("leaf", np.int32),
            ("donor_parent_node", np.int32),
            ("donor_child_node", np.int32),
        ])
        
        with h5py.File(output_file, "w") as f:
            grp = f.create_group("results")
        
            # Simple attributes (nur primitive Typen/Arrays erlaubt!)
            grp.attrs["hgt_rate"] = hgt_rate
            grp.attrs["rho"] = rho
            grp.attrs["gene_number_hgt_events_passed"] = gene_number_hgt_events_passed
            grp.attrs["gene_number_loss_events"] = gene_number_loss_events
        
            # Falls die Matrizen/Arrays numpy-kompatibel sind, geht das:
            grp.create_dataset("gene_absence_presence_matrix", data=gene_absence_presence_matrix)
            grp.create_dataset("fitch_score", data=fitch_scores)
            grp.create_dataset("children_gene_nodes_loss_events", data=children_gene_nodes_loss_events)
        
            # graph_properties als Pickle speichern
            grp.create_dataset("graph_properties", data=np.void(pickle.dumps(graph_properties)))
        
            # Untergruppe für originale Events
            hgt_grp = grp.create_group("nodes_hgt_events")
            if isinstance(nodes_hgt_events, list) and len(nodes_hgt_events) > 0 and isinstance(nodes_hgt_events[0], HGTransfer):
                nodes_hgt_events = [nodes_hgt_events]
                nodes_hgt_events_simplified = [nodes_hgt_events_simplified]
            for site_id, events in enumerate(nodes_hgt_events):
                arr = np.array([
                    (
                        ev.recipient_parent_node,
                        ev.recipient_child_node,
                        ev.leaf,
                        ev.donor_parent_node,
                        ev.donor_child_node
                    )
                    for ev in events
                ], dtype=dtype)
            
                hgt_grp.create_dataset(str(site_id), data=arr)
        
            # Untergruppe für vereinfachte Events
            hgt_grp_simpl = grp.create_group("nodes_hgt_events_simplified")
            for site_id, events in enumerate(nodes_hgt_events_simplified):
                arr = np.array([
                    (
                        ev.recipient_parent_node,
                        ev.recipient_child_node,
                        ev.leaf,
                        ev.donor_parent_node,
                        ev.donor_child_node
                    )
                    for ev in events
                ], dtype=dtype)
            
                hgt_grp_simpl.create_dataset(str(site_id), data=arr)
        
    
    return output_dir     

def simulate_and_store_old(theta, rho, num_samples, num_genes, hgt_rate, ce_from_nwk, distance_matrix_core, output_dir):
    # run_id für eindeutige Dateinamen
    run_id = str(uuid.uuid4())
    
    data = simulator(theta = theta, rho = rho, num_samples = num_samples, num_genes = num_genes, hgt_rate = hgt_rate, ce_from_nwk = ce_from_nwk, distance_matrix = distance_matrix_core)

    gene_absence_presence_matrix = data["gene_absence_presence_matrix"]
    fitch_scores = data["fitch_scores"]
    gene_number_hgt_events_passed = data["gene_number_hgt_events_passed"]
    distance_matrix = data["distance_matrix"]
    core_allel_distance_number_of_clusters = data["core_allel_distance_number_of_clusters"]
    alleles_list_pca = data["alleles_list_pca"]
    gene_number_loss_events = data["gene_number_loss_events"]
    graphs = data["graphs"]
    
    if gene_absence_presence_matrix.sum() > 1:
        # Dateiname pro Prozess eindeutig
        output_file = f"{output_dir}/simulation_results_{run_id}.h5"
    
        with h5py.File(output_file, "w") as f:
            grp = f.create_group("results")
            grp.attrs["hgt_rate"] = hgt_rate
            grp.attrs["rho"] = rho
            grp.attrs["fitch_score"] = fitch_scores
            grp.attrs["gene_number_hgt_events_passed"] = gene_number_hgt_events_passed
            grp.attrs["gene_number_loss_events"] = gene_number_loss_events
        
            grp.create_dataset("matrix", data=gene_absence_presence_matrix)
            #grp.create_dataset("distance_matrix", data=np.round(distance_matrix, 3))
            grp.create_dataset("distance_matrix", data = distance_matrix)
            
            for i, matrix in enumerate(core_allel_distance_number_of_clusters):
                grp.create_dataset(f"core_allel_distance_number_of_clusters_{i}", data = matrix)
            for i, matrix in enumerate(alleles_list_pca):
                grp.create_dataset(f"alleles_list_pca_{i}", data = matrix)

            """
            for i, G in enumerate(graphs):
                # Graph in node-link dict umwandeln
                data_dict = json_graph.node_link_data(G)
                # Als JSON-String serialisieren
                json_str = json.dumps(data_dict)
                # JSON-String als bytes speichern
                grp.create_dataset(f"graph_{i}", data=np.void(json_str.encode('utf-8')))
            """

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

        # The following is neccessary, since the number of the leaves are reordered:
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
        #distance_matrix_core = multidimensional_scaling(distance_matrix_core, multidimensional_scaling_dimensions = multidimensional_scaling_dimensions)
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
    
    num_simulations = 1
    same_core_tree = False

    num_samples = 10
    num_genes = 1

    ### Define random rates:

    theta = 1
    
    hgt_rate_max = 5 # Maximum hgt rate
    hgt_rate_min = 0 # Minimum hgt rate

    rho_max = 3 # Maximum gene loss rate
    rho_min = 0 # Minimum gene loss rate

    prior_hgt_rate = BoxUniform(low=hgt_rate_min * torch.ones(1), high=hgt_rate_max * torch.ones(1))
    prior_rho = BoxUniform(low=rho_min * torch.ones(1), high=rho_max * torch.ones(1))
    
    hgt_rate_samples, _ = torch.sort(prior_hgt_rate.sample((num_simulations,)))
    rho_samples = prior_rho.sample((num_simulations,))

    hgt_rate_samples = torch.linspace(0, hgt_rate_max, num_simulations)
    
    output_dir = r"C:\Users\uhewm\Desktop\ProjectHGT\simulation_chunks"
    
    # wenn Ordner existiert, komplett löschen
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # dann neu anlegen
    os.makedirs(output_dir, exist_ok=True)

    run_simulation(same_core_tree, num_simulations, output_dir, theta, hgt_rate_samples, rho_samples, num_samples, num_genes)
