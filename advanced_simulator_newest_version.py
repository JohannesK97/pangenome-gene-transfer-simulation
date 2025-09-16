import os
import re
import time
import glob
import copy
import json
import uuid
import pickle
import random
import shutil
import secrets
from random import randint
from collections import namedtuple, defaultdict, deque
from dataclasses import dataclass
from typing import Union, Sequence, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import msprime
import tskit
import hgt_simulation
import hgt_sim_args
import numpy as np
import networkx as nx
import torch
import h5py

from sbi.utils import BoxUniform
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from numba import njit
from networkx.readwrite import json_graph


@dataclass
class HGTransfer:
    """Container for horizontal gene transfer events."""
    recipient_parent_node: int
    recipient_child_node: int
    leaf: int
    donor_parent_node: int
    donor_child_node: int


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
    """
    Simulate a tree sequence under a horizontal gene transfer model.

    Parameters
    ----------
    num_samples : int or None
        Number of sampled individuals. Mutually exclusive with `ce_from_nwk`.
    theta : int, default=1
        Mutation rate parameter.
    rho : float, default=0.3
        Recombination rate parameter.
    hgt_rate : float, default=0
        Horizontal gene transfer rate.
    num_genes : int, default=1
        Number of genes simulated.
    nucleotide_mutation_rate : float or None
        Mutation rate per nucleotide.
    gene_length : int, default=1000
        Length of each gene.
    pca_dimensions : int, default=10
        Number of PCA dimensions to compute.
    ce_from_nwk : str or None
        Newick string for core tree (overrides num_samples).
    seed : int or None
        Random seed.
    distance_matrix : np.ndarray or None
        Optional distance matrix for MDS.
    multidimensional_scaling_dimensions : int, default=100
        Number of dimensions for MDS projection.
    block_clustering_threshold : float or Sequence[float]
        Threshold(s) for block clustering.
    clonal_root_mutation : bool or None
        Whether to apply mutations at the clonal root.
    hgt_difference_removal_threshold : float, default=0
        Threshold for removing differences after HGT.

    Returns
    -------
    tskit.TreeSequence
        Simulated tree sequence with optional HGT events.
    """

    start_time = time.time()

    # Ensure either a core tree is provided or num_samples is specified
    if ce_from_nwk is None and num_samples is None:
        raise ValueError(
            "Neither a core tree nor parameters for simulation were provided. Choose either."
        )

    # Generate a core tree if only number of samples is provided
    if ce_from_nwk is None:
        core_tree = msprime.sim_ancestry(
            samples=num_samples,
            sequence_length=1,
            ploidy=1,
            recombination_rate=0,
            gene_conversion_rate=0,
            gene_conversion_tract_length=1,  # Simulate one gene
            random_seed=seed,
        )
        ce_from_nwk = core_tree.first().newick()

    # Initialize random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    if seed is None:
        seed = secrets.randbelow(2**32 - 4) + 2

    # Default nucleotide mutation rate if none provided
    if nucleotide_mutation_rate is None:
        nucleotide_mutation_rate = 0.1

    # Ensure that the number of dimensions for MDS does not exceed the sample size
    if num_samples < multidimensional_scaling_dimensions:
        multidimensional_scaling_dimensions = num_samples

    # ------------------------------------------------------------
    # Calculate horizontal gene transfer (HGT) events
    # ------------------------------------------------------------
    args = hgt_sim_args.Args(
        sample_size=num_samples,
        num_sites=num_genes,
        gene_conversion_rate=0,
        recombination_rate=0,
        hgt_rate=hgt_rate,
        ce_from_ts=None,
        ce_from_nwk=ce_from_nwk,
        random_seed=seed,
    )
    ts, hgt_edges = hgt_simulation.run_simulate(args)

    # ------------------------------------------------------------
    # Place mutations in the simulated tree sequence
    # ------------------------------------------------------------
    alleles = ["absent", "present"]

    gains_model = msprime.MatrixMutationModel(
        alleles=alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [0, 1],
            [0, 1],
        ],
    )

    # Ensure every site has at least one valid mutation
    k = 0
    tables = ts.tables
    ts_gains = tables.tree_sequence()
    while True:
        ts_gains = msprime.sim_mutations(
            ts_gains,
            rate=1,
            model=gains_model,
            keep=True,
            random_seed=seed + k,
            end_time=core_tree.node(core_tree.first().root).time,
        )

        tables = ts_gains.dump_tables()
        tables.mutations.clear()

        # Keep only mutations that affect true sample leaves
        for tree in ts_gains.trees():
            for site in tree.sites():
                for mut in site.mutations:
                    leaf_nodes = list(tree.leaves(mut.node))
                    if any(leaf < num_samples for leaf in leaf_nodes):
                        tables.mutations.add_row(
                            site=mut.site,
                            node=mut.node,
                            derived_state=mut.derived_state,
                            metadata=mut.metadata,
                            time=mut.time,
                        )

        ts_gains = tables.tree_sequence()

        # Verify that each site has at least one mutation
        all_sites_ok = all(
            any(m.site == site_id for m in ts_gains.tables.mutations)
            for site_id in range(num_genes)
        )
        if all_sites_ok:
            break

        k += 1

    # ------------------------------------------------------------
    # Remove superfluous mutations and adjust clonal root mutations
    # ------------------------------------------------------------
    tables = ts_gains.dump_tables()

    mutations_by_site = defaultdict(list)
    for mut in tables.mutations:
        mutations_by_site[mut.site].append(mut)

    tables.mutations.clear()

    clonal_root_node = ts_gains.first().mrca(*list(range(num_samples)))
    print("Clonal node:", clonal_root_node)

    if clonal_root_mutation is None:
        clonal_root_mutation = random.random() < 0.5

    present_mutations = [[] for _ in range(num_genes)]
    for site, mutations in mutations_by_site.items():

        if clonal_root_mutation:
            # Mutation at clonal root: set derived state to "present"
            tables.mutations.add_row(
                site=site,
                node=clonal_root_node,
                derived_state="present",
                parent=-1,
                metadata=None,
                time=tables.nodes.time[clonal_root_node],
            )

        else:
            # Choose one mutation randomly and add it
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

            # Add "absent" mutation at clonal root
            tables.mutations.add_row(
                site=site,
                node=clonal_root_node,
                derived_state="absent",
                parent=-1,
                metadata=None,
                time=tables.nodes.time[clonal_root_node],
            )

        # Add sentinel mutations at the leaves
        for leaf_position in range(num_samples):
            tables.mutations.add_row(
                site=site,
                node=leaf_position,
                derived_state="absent",
                time=1e-11,
            )

    tables.sort()
    ts_gains = tables.tree_sequence()

    # ------------------------------------------------------------
    # Place losses and calculate gene presence/absence matrices
    # ------------------------------------------------------------
    
    # Helper namedtuple to record child mutation info
    MutationRecord = namedtuple("MutationRecord", ["site_id", "mutation_id", "node", "is_hgt"])
    
    # Place losses (mutations representing loss of gene)
    losses_model = msprime.MatrixMutationModel(
        alleles=alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [1, 0],
            [1, 0],
        ],
    )
    
    # Apply loss mutations across the tree sequence
    ts_gains_losses = msprime.sim_mutations(
        ts_gains, rate=rho, model=losses_model, keep=True, random_seed=seed - 1
    )
    
    # Create containers for per-site processing
    tables = ts_gains_losses.dump_tables()
    
    # We'll rebuild the mutations table from selected events
    tables.mutations.clear()
    
    # Prepare HGT mapping: parent -> list(child_parent_index)
    # Note: hgt_edges appear to use parent/child indices with specific offsets in upstream code
    hgt_parent_nodes = [edge.parent - 1 for edge in hgt_edges]
    hgt_children_nodes = [edge.child for edge in hgt_edges]
    hgt_parent_children = defaultdict(list)
    for parent in hgt_parent_nodes:
        hgt_parent_children[parent].append(parent - 1)
    
    # Prepare a template TableCollection for each gene, copying nodes and populations
    tables_gene = tskit.TableCollection(sequence_length=gene_length)
    tables_gene.nodes.replace_with(ts_gains_losses.tables.nodes)
    tables_gene.populations.replace_with(ts_gains_losses.tables.populations)
    
    # Make per-gene copies and initialize counters
    tables_gene_list = [copy.deepcopy(tables_gene) for _ in range(num_genes)]
    gene_number_hgt_events_passed = [0 for _ in range(num_genes)]
    gene_number_loss_events = [0 for _ in range(num_genes)]
    gene_nodes_loss_events = [[] for _ in range(num_genes)]
    gene_trees_list = []
    hgt_parent_children_passed_list = []
    
    # Iterate over local trees and their sites to reconstruct per-gene trees
    for tree in ts_gains_losses.trees():
        for site in tree.sites():
    
            # Track which HGT parent->child flags were passed for this site
            hgt_parent_children_passed = [False] * ts_gains_losses.num_nodes
    
            # All mutations on the current site
            mutations = site.mutations
    
            # Identify the 'present' mutation (gain) and collect 'absent' nodes
            present_mutation = [m for m in mutations if m.derived_state == "present"][0]
            absent_mutation_nodes = {m.node for m in mutations if m.derived_state == "absent"}
    
            # Absent mutations that occurred before the present (gain) event
            absent_mutations = defaultdict(list)
            for m in mutations:
                if m.derived_state == "absent" and m.time < present_mutation.time:
                    absent_mutations[m.node].append(m)
    
            # Track nodes that have already been reached in branching traversal
            branching_nodes_reached_before = defaultdict(lambda: False)
    
            print("Present_mutation at node:", present_mutation.node)
    
            # BFS/DFS-like traversal starting from the gain node
            # Each queue entry: (node_id, passed_hgt_anywhere, passed_hgt_last_step, num_hgt_passed)
            branching_nodes_to_process = deque([(present_mutation.node, False, False, 0)])
            child_mutations = []
    
            # If the gain happened directly above a leaf
            if present_mutation.node < num_samples:
                # Check whether this is actually a sentinel scenario (second mutation for that leaf)
                mutations_on_node = sorted([mut for mut in mutations if mut.node == present_mutation.node], key=lambda m: m.time)
                if present_mutation.id == mutations_on_node[1].id:
                    sentinel_mutation = min(absent_mutations[present_mutation.node], key=lambda m: m.time)
                    child_mutations.append(
                        MutationRecord(site_id=site.id, mutation_id=sentinel_mutation.id, node=sentinel_mutation.node, is_hgt=False)
                    )
                    parent_of_leaf = tree.parent(present_mutation.node)
                    _add_edge_safe(tables_gene_list[site.id], 0, gene_length, parent_of_leaf, sentinel_mutation.node)
                else:
                    gene_number_loss_events[site.id] += 1
                    gene_nodes_loss_events[site.id].append(present_mutation.node)
    
            else:
                # First pass: determine which HGT edges are traversed on any path
                while branching_nodes_to_process:
                    last_branching_node = branching_nodes_to_process.popleft()
                    selected_branch_nodes_to_process = deque([last_branching_node])
                    parent_of_last = tree.parent(last_branching_node[0])
                    branching_nodes_reached_before[parent_of_last] = True
    
                    while selected_branch_nodes_to_process:
                        child_node = selected_branch_nodes_to_process.popleft()
    
                        # If there is an absent mutation on this node and it occurs before gain
                        if (not child_node[2]) and (child_node[0] in absent_mutation_nodes):
                            absent_after_gain = absent_mutations[child_node[0]]
    
                            if not absent_after_gain:  # empty
                                children = tree.children(child_node[0])
                                if len(children) > 1:
                                    for child in reversed(children):
                                        if not branching_nodes_reached_before[child_node[0]]:
                                            branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                else:
                                    for child in reversed(children):
                                        selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
    
                                if hgt_parent_children[child_node[0]]:
                                    hgt_child = hgt_parent_children[child_node[0]][0]
                                    selected_branch_nodes_to_process.extendleft([(hgt_child, True, True, child_node[3] + 1)])
                                    hgt_parent_children_passed[hgt_child] = True
    
                        else:
                            # No mutation on edge -> propagate down
                            children = tree.children(child_node[0])
                            if len(children) > 1:
                                for child in reversed(children):
                                    if not branching_nodes_reached_before[child_node[0]]:
                                        branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                            else:
                                for child in reversed(children):
                                    selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
    
                            if hgt_parent_children[child_node[0]]:
                                hgt_child = hgt_parent_children[child_node[0]][0]
                                selected_branch_nodes_to_process.extendleft([(hgt_child, True, True, child_node[3] + 1)])
                                hgt_parent_children_passed[hgt_child] = True
    
                hgt_parent_children_passed_list.append(hgt_parent_children_passed)
    
                # Second pass: compute final presence/loss decisions and record edges
                branching_nodes_to_process = deque([(present_mutation.node, False, False, 0)])
                branching_nodes_reached_before = defaultdict(lambda: False)
    
                while branching_nodes_to_process:
                    last_branching_node = branching_nodes_to_process.popleft()
                    selected_branch_nodes_to_process = deque([last_branching_node])
                    parent_of_last = tree.parent(last_branching_node[0])
                    branching_nodes_reached_before[parent_of_last] = True
    
                    while selected_branch_nodes_to_process:
                        child_node = selected_branch_nodes_to_process.popleft()
    
                        # Skip traversal paths that passed an HGT parent-child previously
                        if (not child_node[2]) and hgt_parent_children_passed[child_node[0]]:
                            continue
    
                        # If an absent mutation occurred before gain at this node
                        if (not child_node[2]) and (child_node[0] in absent_mutation_nodes):
                            absent_after_gain = absent_mutations[child_node[0]]
    
                            if absent_after_gain:
                                earliest_mutation = max(absent_after_gain, key=lambda m: m.time)
    
                                if earliest_mutation.time == 1e-11:
                                    # Convert sentinel absent mutation to present (leaf-level presence)
                                    earliest_mutation.derived_state = "present"
                                    child_mutations.append(
                                        MutationRecord(site_id=site.id, mutation_id=earliest_mutation.id, node=earliest_mutation.node, is_hgt=child_node[1])
                                    )
                                    _add_edge_safe(
                                        tables_gene_list[site.id], 0, gene_length, tree.parent(last_branching_node[0]), earliest_mutation.node
                                    )
                                    gene_number_hgt_events_passed[site.id] += child_node[3]
                                else:
                                    gene_number_loss_events[site.id] += 1
                                    gene_nodes_loss_events[site.id].append(earliest_mutation.node)
    
                            else:
                                # No absent mutation on this path: propagate down
                                children = tree.children(child_node[0])
                                if len(children) > 1:
                                    for child in reversed(children):
                                        if not branching_nodes_reached_before[child_node[0]]:
                                            branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                    if tree.parent(last_branching_node[0]) != -1:
                                        _add_edge_safe(
                                            tables_gene_list[site.id], 0, gene_length, tree.parent(last_branching_node[0]), child_node[0]
                                        )
                                    gene_number_hgt_events_passed[site.id] += child_node[3]
                                else:
                                    for child in reversed(children):
                                        selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
    
                                if hgt_parent_children[child_node[0]]:
                                    hgt_child = hgt_parent_children[child_node[0]][0]
                                    selected_branch_nodes_to_process.extendleft([(hgt_child, True, True, child_node[3] + 1)])
    
                        else:
                            # No mutation at this step; continue traversing
                            children = tree.children(child_node[0])
                            if len(children) > 1:
                                for child in reversed(children):
                                    if not branching_nodes_reached_before[child_node[0]]:
                                        branching_nodes_to_process.extendleft([(child, child_node[1], False, 0)])
                                if tree.parent(last_branching_node[0]) != -1:
                                    _add_edge_safe(
                                        tables_gene_list[site.id], 0, gene_length, tree.parent(last_branching_node[0]), child_node[0]
                                    )
                                gene_number_hgt_events_passed[site.id] += child_node[3]
                            else:
                                for child in reversed(children):
                                    selected_branch_nodes_to_process.extendleft([(child, child_node[1], False, child_node[3])])
                            if hgt_parent_children[child_node[0]]:
                                hgt_child = hgt_parent_children[child_node[0]][0]
                                selected_branch_nodes_to_process.extendleft([(hgt_child, True, True, child_node[3] + 1)])
    
            # Sort child mutations so that non-HGT paths get priority when resolving conflicts
            child_mutations.sort(key=lambda mut: not mut.is_hgt)
    
            # Resolve multiple paths to the same leaf: prefer non-HGT if available
            unique_mutations = {}
            for mut in child_mutations:
                if mut.node not in unique_mutations:
                    unique_mutations[mut.node] = mut
                else:
                    existing_mut = unique_mutations[mut.node]
                    if (not existing_mut.is_hgt) or (not mut.is_hgt):
                        unique_mutations[mut.node] = mut._replace(is_hgt=False)
    
            child_mutations_filtered = list(unique_mutations.values())
    
            # Reconstruct the mutations table: keep all non-sentinel mutations and add child-level presents
            for mutation in mutations:
                if mutation.time > 1e-11:
                    if mutation.derived_state == "absent":
                        metadata_value = bytes([3])
                    elif mutation.derived_state == "present":
                        metadata_value = bytes([7])
                    else:
                        metadata_value = None
    
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
                    time=1e-11,
                )
    
    # Finalize tree sequence after losses and site-wise reconstruction
    mts = tables.tree_sequence()
    
    # ------------------------------------------------------------
    # Finalize per-gene tree collections and compute presence/absence
    # ------------------------------------------------------------
    
    # Sort and materialize per-gene tree sequences
    for i in range(num_genes):
        tables_gene_list[i].sort()
        gene_trees_list.append(tables_gene_list[i].tree_sequence())
    
    # Compute the gene presence/absence matrix from the master tree sequence
    gene_absence_presence_matrix = np.array([var.genotypes for var in mts.variants()])
    
    print("Number of present genes:", int(np.sum(gene_absence_presence_matrix)))
    
    # Compute Fitch parsimony scores (user-provided function expected in namespace)
    fitch_scores = fitch_parsimony_score(mts, gene_absence_presence_matrix)
    
    # ------------------------------------------------------------
    # Identify HGT events per gene by traversing the master tree and
    # mapping events into each gene-specific tree sequence.
    # ------------------------------------------------------------
    
    gene_number_hgt_events_passed = [0 for _ in range(num_genes)]
    nodes_hgt_events = [[] for _ in range(num_genes)]
    nodes_hgt_events_simplified = [[] for _ in range(num_genes)]
    children_gene_nodes_loss_events = [[] for _ in range(num_genes)]
    hgt_trees = [[] for _ in range(num_genes)]
    
    # Iterate through master tree sequence to find candidate HGT nodes
    for tree in mts.trees():
        for site in tree.sites():
            reached_nodes_from_leaves = defaultdict(lambda: False)
    
            # Seed stack with sampled leaves that carry the gene (presence)
            stack = [(node_id, node_id) for node_id in range(mts.num_samples) if gene_absence_presence_matrix[site.id][node_id]]
    
            # Traverse upwards: if an HGT flag was passed on a node, record an event
            while stack:
                node, leaf_node = stack.pop()
                parent = tree.parent(node)
                if node < clonal_root_node and not reached_nodes_from_leaves[node]:
                    if not hgt_parent_children_passed[node]:
                        stack.append((parent, leaf_node))
                        reached_nodes_from_leaves[node] = True
                    else:
                        # HGT encountered on path from leaf to root
                        nodes_hgt_events[site.id].append((node, leaf_node, None))
                        gene_number_hgt_events_passed[site.id] += 1
                        reached_nodes_from_leaves[node] = True
    
            # For each gene-local tree, map the HGT node into the gene tree
            tables = gene_trees_list[site.id].tables
            for subtree in gene_trees_list[site.id].trees():
                hgt_events_with_origins = []
                for node, leaf_node, _ in nodes_hgt_events[site.id]:
                    temp_time = 0
                    parent = leaf_node
                    time_of_node = gene_trees_list[site.id].node(node).time
    
                    # Walk upwards in the gene subtree until we cross or reach the time of the HGT node
                    while temp_time < time_of_node:
                        child = parent
                        parent = subtree.parent(parent)
                        temp_time = gene_trees_list[site.id].node(parent).time
    
                    # Remove the edge (parent->child) in the gene table if present
                    mask = [not (edge.parent == parent and edge.child == child) for edge in tables.edges]
                    tables.edges.set_columns(
                        left=[e.left for i, e in enumerate(tables.edges) if mask[i]],
                        right=[e.right for i, e in enumerate(tables.edges) if mask[i]],
                        parent=[e.parent for i, e in enumerate(tables.edges) if mask[i]],
                        child=[e.child for i, e in enumerate(tables.edges) if mask[i]],
                    )
    
                    # Insert the two edges that represent the HGT insertion (node <-> child and parent <-> node)
                    tables.edges.add_row(left=0, right=gene_length, parent=node, child=child)
                    tables.edges.add_row(left=0, right=gene_length, parent=parent, child=node)
    
                    # Record origin as the parent for later validation
                    hgt_events_with_origins.append((node, leaf_node, parent))
    
                nodes_hgt_events[site.id] = hgt_events_with_origins
            
            tables.sort()
            gene_trees_list[site.id] = tables.tree_sequence()
    
            # Rebuild per-gene tree sequence and validate HGT events against master tree
            for idx_subtree, _ in enumerate(gene_trees_list[site.id].trees()):
                tables = gene_trees_list[site.id].tables
                tables.sort()
                gene_trees_list[site.id] = tables.tree_sequence()
    
                valid_nodes_hgt_events = []
                for subtree in gene_trees_list[site.id].trees():
                    for node, leaf_node, hgt_origin in nodes_hgt_events[site.id]:
                        tables_master = mts.tables
    
                        # Remove any existing edge connecting core tree parent->node in master tables
                        mask = [not (edge.parent == tree.parent(node) and edge.child == node) for edge in tables_master.edges]
                        tables_master.edges.set_columns(
                            left=[e.left for i, e in enumerate(tables_master.edges) if mask[i]],
                            right=[e.right for i, e in enumerate(tables_master.edges) if mask[i]],
                            parent=[e.parent for i, e in enumerate(tables_master.edges) if mask[i]],
                            child=[e.child for i, e in enumerate(tables_master.edges) if mask[i]],
                        )
    
                        # Add the HGT edge for the master table limited to the current site interval
                        tables_master.edges.add_row(left=site.id, right=site.id + 1, parent=subtree.parent(node), child=node)
    
                        tables_master.sort()
                        tables_master.mutations.clear()
    
                        hgt_tree = tables_master.tree_sequence()
                        hgt_trees[site.id].append((hgt_tree, (node, leaf_node)))
    
                        # Validate whether the HGT-induced topology differs sufficiently
                        keep = False
                        for leaf in range(num_samples):
                            if gene_absence_presence_matrix[site.id][leaf] == 1:
                                # Compute time difference of MRCA between gene-specific and master tree
                                pos = hgt_tree.sites()[site.id].position
                                try:
                                    hgt_mrca_time = hgt_tree.node(hgt_tree.at(pos).mrca(node, leaf)).time
                                    master_mrca_time = mts.node(tree.mrca(node, leaf)).time
                                    if abs(hgt_mrca_time - master_mrca_time) > hgt_difference_removal_threshold:
                                        keep = True
                                        break
                                except Exception:
                                    # If MRCA computation fails for any reason, conservatively keep the event
                                    keep = True
                                    break
    
                        if keep:
                            valid_nodes_hgt_events.append((node, leaf_node, hgt_origin))
                        else:
                            print("Removed HGT edge")
    
                nodes_hgt_events[site.id] = valid_nodes_hgt_events
    
    # ------------------------------------------------------------
    # Simulate nucleotide-level mutations for each gene tree
    # ------------------------------------------------------------
    
    # Define a simple nucleotide mutation model or use infinite alleles
    nucleotide_mutation = msprime.InfiniteAlleles()
    
    # Place nucleotide mutations on each per-gene tree
    for i in range(num_genes):
        gene_trees_list[i] = msprime.sim_mutations(
            gene_trees_list[i], rate=nucleotide_mutation_rate, model=nucleotide_mutation, keep=True, random_seed=seed - 1
        )
    
    # Extract allele matrices for each gene
    alleles_list = []
    for i in range(num_genes):
        alleles_list.append(np.array([var.genotypes for var in gene_trees_list[i].variants()]))

    # ------------------------------------------------------------
    # Process every HGT and save it correctly
    # ------------------------------------------------------------
    
    for tree in mts.trees():
        for site in tree.sites():
            children, parents, node_to_leaf = build_tree_mappings(tree, num_samples, clonal_root_node)
            all_nodes = set(children.keys())
            for child_list in children.values():
                all_nodes.update(child_list)
            # build reordering dictionary (sorted order)
            G_nodes_reordering = {old_label: new_label for new_label, old_label in enumerate(sorted(all_nodes))}
            
            corrected_hgt_events = []
            corrected_hgt_events_simplified = []
            
            # Process all candidate HGT events for this site
            for node, leaf, hgt_origin in nodes_hgt_events[site.id]:
            
                # Step 1: Find the donor origin node
                hgt_origin = node + 1
                while True:
                    # Stop if we reach a valid donor node
                    has_valid_leaf = [
                        leaf for leaf in tree.leaves(hgt_origin)
                        if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1
                    ]
                    if hgt_origin in G_nodes_reordering and has_valid_leaf:
                        break
            
                    # Otherwise, move upward or to the next node
                    hgt_child = hgt_origin
                    if not hgt_parent_children_passed_list[site.id][hgt_origin]:
                        hgt_origin = tree.parent(hgt_origin)
                    else:
                        hgt_origin = hgt_origin + 1
            
                # Step 2: Identify donor child node
                if any([
                    [leaf for leaf in tree.leaves(child) if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1] == []
                    for child in tree.children(hgt_origin)
                ]):
                    hgt_child = hgt_origin
            
                while True:
                    # Stop if we reached a valid donor child node
                    if hgt_child != hgt_origin and hgt_child in G_nodes_reordering:
                        break
            
                    children_hgt = tree.children(hgt_child)
            
                    # Case 1: Unary child → just move down
                    if len(children_hgt) == 1:
                        hgt_child = children_hgt[0]
                        counts = [1]  # dummy value for consistency
            
                    # Case 2: Multiple children → select the child with the most valid leaves
                    else:
                        counts = []
                        for child in children_hgt:
                            child_leaves = [leaf for leaf in tree.leaves(child) if leaf < num_samples]
                            valid_leaves = [
                                leaf for leaf in child_leaves
                                if gene_absence_presence_matrix[site.id][leaf] == 1
                            ]
                            counts.append((child, len(valid_leaves)))
            
                        # Choose the child with the maximum number of valid leaves
                        hgt_child = max(counts, key=lambda x: x[1])[0]
            
                # Step 3: Determine recipient nodes (parent and child)
                hgt_recipient_parent = node
                hgt_recipient_child = node
            
                # Move upward until a valid parent is found
                while True:
                    invalid_parent = hgt_recipient_parent not in G_nodes_reordering

                    parent_leaves = [
                        leaf for leaf in tree.leaves(hgt_recipient_parent)
                        if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1
                    ]
                    node_leaves = [
                        leaf for leaf in tree.leaves(node)
                        if leaf < num_samples and gene_absence_presence_matrix[0][leaf] == 1
                    ]
                    same_leaves = (parent_leaves == node_leaves)
                
                    if not (invalid_parent or same_leaves):
                        break
                
                    # Move one level up
                    hgt_recipient_parent = tree.parent(hgt_recipient_parent)

                # Move downward until a valid child is found
                while hgt_recipient_child not in G_nodes_reordering:
                    children_hgt = tree.children(hgt_recipient_child)
            
                    if len(children_hgt) == 1:
                        hgt_recipient_child = children_hgt[0]
                    else:
                        counts = []
                        for child in children_hgt:
                            child_leaves = [leaf for leaf in tree.leaves(child) if leaf < num_samples]
                            valid_leaves = [
                                leaf for leaf in child_leaves
                                if gene_absence_presence_matrix[site.id][leaf] == 1
                            ]
                            counts.append((child, len(valid_leaves)))
                        hgt_recipient_child = max(counts, key=lambda x: x[1])[0]
            
                # Step 4: Store the event if it is valid
                if hgt_recipient_parent <= clonal_root_node:
                    event = HGTransfer(
                        recipient_parent_node=hgt_recipient_parent,
                        recipient_child_node=hgt_recipient_child,
                        leaf=leaf,
                        donor_parent_node=hgt_origin,
                        donor_child_node=hgt_child,
                    )
                    corrected_hgt_events.append(event)
            
                    # Store simplified version (with relabeled nodes)
                    event_simplified = HGTransfer(
                        recipient_parent_node=G_nodes_reordering[hgt_recipient_parent],
                        recipient_child_node=G_nodes_reordering[hgt_recipient_child],
                        leaf=leaf,
                        donor_parent_node=G_nodes_reordering[hgt_origin],
                        donor_child_node=G_nodes_reordering[hgt_child],
                    )
                    corrected_hgt_events_simplified.append(event_simplified)
            
            # Update event lists for this site
            nodes_hgt_events[site.id] = corrected_hgt_events
            nodes_hgt_events_simplified[site.id] = corrected_hgt_events_simplified

            # collect nodes corresponding to loss events
            for node in gene_nodes_loss_events[site.id]:
                stack_loss = [node]
                while stack_loss:
                    n = stack_loss.pop()
                    if n in G_nodes_reordering.keys():
                        children_gene_nodes_loss_events[site.id].append(n)
                        break
                    else:
                        if n >= num_samples:
                            stack_loss.extend(tree.children(n))

            children_gene_nodes_loss_events[site.id] = [G_nodes_reordering[node] for node in children_gene_nodes_loss_events[site.id] if node in G_nodes_reordering]


    
    # ------------------------------------------------------------
    # From per-gene trees -> allele matrices, distances and graphs
    # ------------------------------------------------------------
    # At this point we assume:
    # - gene_trees_list (list of per-gene TreeSequences) exists
    # - alleles_list (list of allele matrices for each gene) exists
    # - mts (master tree sequence) exists
    # - fitch_scores computed earlier
    # - many helper lists exist such as gene_number_hgt_events_passed, gene_nodes_loss_events, etc.
    #
    # Now compute scaled distances (MDS / allele embeddings) and build graphs.
    # All heavy-lifting is delegated to the helper functions defined below.
    # ------------------------------------------------------------

    # Compute core distance embedding and per-gene allele embeddings/clusters
    (
        distance_matrix,
        scaled_distance_matrix,
        alleles_list_pca,
        core_allel_distance_list,
    ) = compute_scaled_distances(
        mts=mts,
        num_samples=num_samples,
        distance_matrix=distance_matrix,
        multidimensional_scaling_dimensions=multidimensional_scaling_dimensions,
        alleles_list=alleles_list,
        gene_length=gene_length,
    )

    # Build the directed graphs and extract graph properties & HGT/loss event mappings
    (
        Graphs,
        graph_properties,
    ) = build_graphs_from_mts(
        mts=mts,
        gene_trees_list=gene_trees_list,
        gene_absence_presence_matrix=gene_absence_presence_matrix,
        alleles_list=alleles_list,
        num_samples=num_samples,
        clonal_root_node=clonal_root_node,
        gene_length=gene_length,
        nucleotide_mutation_rate=nucleotide_mutation_rate,
    )

    # ------------------------------------------------------------
    # Finish timing, print summary and return results
    # ------------------------------------------------------------
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Success: hgt_rate = {hgt_rate}, Total computation time = {elapsed_time:.6f} seconds.")

    # Prepare return values with the original single-gene convenience behavior
    return {
        "mts": mts,
        "gene_trees_list": gene_trees_list[0] if num_genes == 1 else gene_trees_list,
        "gene_absence_presence_matrix": (gene_absence_presence_matrix[0]
                                         if (num_genes == 1 and gene_absence_presence_matrix.ndim > 1)
                                         else (gene_absence_presence_matrix if num_genes > 1 else gene_absence_presence_matrix)),
        "alleles_list_pca": alleles_list_pca[0] if num_genes == 1 else alleles_list_pca,
        "scaled_distance_matrix": scaled_distance_matrix,
        "fitch_scores": fitch_scores[0] if (num_genes == 1 and hasattr(fitch_scores, "__len__")) else fitch_scores,
        "gene_number_hgt_events_passed": (gene_number_hgt_events_passed[0] if num_genes == 1 else gene_number_hgt_events_passed),
        "distance_matrix": distance_matrix,
        "gene_number_loss_events": gene_number_loss_events[0] if num_genes == 1 else gene_number_loss_events,
        "alleles_list": alleles_list[0] if num_genes == 1 else alleles_list,
        "clonal_root_node": clonal_root_node,
        "graphs": Graphs[0] if num_genes == 1 else Graphs,
        "graph_properties": graph_properties[0] if num_genes == 1 else graph_properties,
        "nodes_hgt_events_simplified": nodes_hgt_events_simplified[0] if num_genes == 1 else nodes_hgt_events_simplified,
        "nodes_hgt_events": nodes_hgt_events[0] if num_genes == 1 else nodes_hgt_events,
        "children_gene_nodes_loss_events": (children_gene_nodes_loss_events[0] if num_genes == 1 else children_gene_nodes_loss_events),
        "node_to_leaf": node_to_leaf,
        "G_nodes_reordering": G_nodes_reordering,
    }



# ------------------------------------------------------------
# Helper functions to modularize final graph construction
# ------------------------------------------------------------

def compute_scaled_distances(
    mts: tskit.TreeSequence,
    num_samples: int,
    distance_matrix: Union[np.ndarray, None],
    multidimensional_scaling_dimensions: Union[int, None],
    alleles_list: List[np.ndarray],
    gene_length: int,
):
    """
    Compute (or reuse) a distance matrix over leaves and produce scaled
    representations for allelic distances per gene.

    Returns
    -------
    distance_matrix: np.ndarray
        The (possibly computed) core distance matrix.
    scaled_distance_matrix: np.ndarray
        Low-dimensional embedding of the core distance matrix.
    alleles_list_pca: List[np.ndarray]
        List of scaled allele-derived embeddings per gene.
    core_allel_distance_list: List[np.ndarray]
        Per-gene distance difference matrices (core - allele).
    core_allel_distance_number_of_clusters: List
        Per-gene clustering results from `count_blocks_sym`.
    """
    # Compute core distance matrix if not provided
    if distance_matrix is None:
        distance_matrix = distance_core(tree_sequence=mts, num_samples=num_samples)

    if multidimensional_scaling_dimensions is None:
        multidimensional_scaling_dimensions = num_samples

    # Possible multidimensional scaling to save storage.
    scaled_distance_matrix = multidimensional_scaling(
        distance_matrix, multidimensional_scaling_dimensions=multidimensional_scaling_dimensions
    )

    alleles_list_pca = []
    core_allel_distance_list = []
    core_allel_distance_number_of_clusters = []

    for A in alleles_list:
        if A.shape[0] == 0:
            A_reconstructed = np.full((num_samples, multidimensional_scaling_dimensions), -1)
            alleles_list_pca.append(A_reconstructed)
            core_allel_distance_list.append(np.array([]))
            core_allel_distance_number_of_clusters.append([])
            continue

        gene_absent_vector = (A[0] == -1)
        gene_present_vector = ~gene_absent_vector

        euclidean_distances = squareform(pdist(A.T, metric="euclidean"))
        scaled_euclidean_distances = np.full((num_samples, multidimensional_scaling_dimensions), 0.0)

        if gene_present_vector.sum() > 0:
            scaled_euclidean_distances[gene_present_vector, :] = multidimensional_scaling(
                euclidean_distances[np.ix_(gene_present_vector, gene_present_vector)],
                multidimensional_scaling_dimensions=multidimensional_scaling_dimensions,
            )

        euclidean_distances[gene_absent_vector, :] = -1
        euclidean_distances[:, gene_absent_vector] = -1

        alleles_list_pca.append(scaled_euclidean_distances)

        distance_matrix_valid = distance_matrix[np.ix_(gene_present_vector, gene_present_vector)]
        euclidean_distances_valid = euclidean_distances[np.ix_(gene_present_vector, gene_present_vector)]
        euclidean_distances_valid = euclidean_distances_valid ** 2

        if gene_present_vector.sum() > 0 and np.max(euclidean_distances_valid) > 0:
            euclidean_distances_valid = euclidean_distances_valid / np.mean(euclidean_distances_valid) * np.mean(
                distance_matrix_valid
            )

        core_allel_distance_valid = distance_matrix_valid - euclidean_distances_valid
        core_allel_distance_list.append(core_allel_distance_valid)

    return (
        distance_matrix,
        scaled_distance_matrix,
        alleles_list_pca,
        core_allel_distance_list,
        #core_allel_distance_number_of_clusters,
    )


def build_graphs_from_mts(
    mts: tskit.TreeSequence,
    gene_trees_list: List[tskit.TreeSequence],
    gene_absence_presence_matrix: np.ndarray,
    alleles_list: List[np.ndarray],
    num_samples: int,
    clonal_root_node: int,
    gene_length: int,
    nucleotide_mutation_rate: float,
) -> Tuple[List[nx.DiGraph], List]:
    
    """
    Build gene-specific graphs from the main tree sequence (MTS).

    Parameters
    ----------
    mts : tskit.TreeSequence
        Main tree sequence containing the clonal genealogy.
    gene_trees_list : List[tskit.TreeSequence]
        List of gene tree sequences, one per gene.
    gene_absence_presence_matrix : np.ndarray
        Binary matrix indicating presence/absence of each gene in each sample.
    alleles_list : List[np.ndarray]
        List of allele vectors per gene.
    num_samples : int
        Number of samples in the dataset.
    clonal_root_node : int
        Root node of the clonal genealogy.
    gene_length : int
        Length of each gene (used for mutation modeling).
    nucleotide_mutation_rate : float
        Mutation rate per nucleotide.

    Returns
    -------
    Tuple[List[nx.DiGraph], List]
        - A list of directed graphs (one per gene).
        - A list of graph properties (metadata for each graph).
    """

    Graphs = []
    graph_properties = []
    nodes_hgt_events_simplified = [[] for _ in range(len(gene_trees_list))]
    children_gene_nodes_loss_events = [[] for _ in range(len(gene_trees_list))]

    for tree in mts.trees():
        for site in tree.sites():
            #site.id = site.id
            children, parents, node_to_leaf = build_tree_mappings(tree, num_samples, clonal_root_node)
                
            G = nx.DiGraph()

            # collect nodes
            all_nodes = set(children.keys())
            for child_list in children.values():
                all_nodes.update(child_list)

            for node in sorted(all_nodes):
                G.add_node(node, core_distance=0, allele_distance=0, true_allele_distance=0, time=tree.get_time(node))

            for parent, child_list in children.items():
                for child in child_list:
                    G.add_edge(parent, child)

            # compute subtree_has_gene for this site
            subtree_has_gene = {}
            for node in range(clonal_root_node):
                if node < num_samples:
                    subtree_has_gene[node] = bool(gene_absence_presence_matrix[site.id][node])
                else:
                    child_list = children[node]
                    subtree_has_gene[node] = any(subtree_has_gene[child] for child in child_list)

            # compute core_distance per internal node
            for node in sorted(range(num_samples, clonal_root_node + 1), key=lambda n: tree.get_time(n)):
                if node in G.nodes:
                    c0, c1 = children[node]
                    c0_has, c1_has = subtree_has_gene[c0], subtree_has_gene[c1]

                    if not (c0_has or c1_has):
                        continue

                    if c0_has and c1_has:
                        if c0 >= num_samples:
                            d0, d1 = children[c0]
                            while (not subtree_has_gene[d0] and subtree_has_gene[d1]) or (
                                subtree_has_gene[d0] and not subtree_has_gene[d1]
                            ):
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
                            while (not subtree_has_gene[d0] and subtree_has_gene[d1]) or (
                                subtree_has_gene[d0] and not subtree_has_gene[d1]
                            ):
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
                            - tree.get_time(c0)
                            - tree.get_time(c1)
                            + G.nodes[c0]["core_distance"]
                            + G.nodes[c1]["core_distance"]
                        )
                    elif c0_has:
                        core_distance = G.nodes[c0]["core_distance"]
                    else:
                        core_distance = G.nodes[c1]["core_distance"]

                    G.nodes[node]["core_distance"] = core_distance

            # transform core distances to expected mutation counts
            for node in range(num_samples, clonal_root_node + 1):
                if node in G.nodes:
                    G.nodes[node]["core_distance"] = (
                        1 - (1 - 1 / gene_length) ** (G.nodes[node]["core_distance"] * nucleotide_mutation_rate * gene_length)
                    )

            # Allele-based distances
            allele_distances = defaultdict(int)
            allele_distances_only_new = defaultdict(int)
            allele_distances_both_children_polymorph = defaultdict(int)
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
                    distance = np.count_nonzero(np.ptp(subset, axis=1))
                allele_distances[tuple(leaves)] = distance

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
                    polymorph_mask = np.ptp(subset, axis=1) > 0

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
                        child_any_mask = np.logical_or.reduce(child_polymorph_mask_list)
                        if len(child_polymorph_mask_list) > 1:
                            child_all_mask = np.logical_and.reduce(child_polymorph_mask_list)
                        else:
                            child_all_mask = np.zeros_like(polymorph_mask)
                    else:
                        child_any_mask = np.zeros_like(polymorph_mask)
                        child_all_mask = np.zeros_like(polymorph_mask)

                    new_polymorph_mask = polymorph_mask & (~child_any_mask)
                    new_distance = np.count_nonzero(new_polymorph_mask)

                    shared_polymorph_mask = polymorph_mask & child_all_mask
                    shared_distance = np.count_nonzero(shared_polymorph_mask)

                allele_distances_only_new[tuple(leaves)] = new_distance
                allele_distances_both_children_polymorph[tuple(leaves)] = shared_distance

            for node in list(parents) + [clonal_root_node]:
                G.nodes[node]["allele_distance"] = allele_distances[tuple(node_to_leaf[node])] / gene_length
                G.nodes[node]["allele_distance_only_new"] = allele_distances_only_new[tuple(node_to_leaf[node])] / gene_length
                G.nodes[node]["allele_distances_both_children_polymorph"] = (
                    allele_distances_both_children_polymorph[tuple(node_to_leaf[node])] / gene_length
                )

            # Compute true allele distances by walking gene-specific tree
            for gen_tree in gene_trees_list[site.id].trees():
                for node in list(parents) + [clonal_root_node]:
                    leaves = [leaf for leaf in node_to_leaf[node] if gene_absence_presence_matrix[site.id][leaf] == 1]
                    if len(leaves) <= 1:
                        G.nodes[node]["true_allele_distance"] = 0
                    else:
                        m = leaves[0]
                        for u in leaves[1:]:
                            m = gen_tree.mrca(m, u)

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
                                u = p

                        G.nodes[node]["true_allele_distance"] = (
                            1 - (1 - 1 / gene_length) ** (total_length * nucleotide_mutation_rate * gene_length)
                        )

            # Reorder nodes to contiguous indices and collect properties
            G_nodes_reordering = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
            nx.relabel_nodes(G, G_nodes_reordering, copy=False)

            # Append and store graph features
            Graphs.append(G)
            graph_nodes = list(G.nodes)
            graph_edges = list(G.edges)
            graph_edge_index = torch.tensor(np.array(graph_edges).T)

            node_features = []
            for node in graph_nodes:
                core_distance = G.nodes[node].get("core_distance", 0.0)
                allele_distance = G.nodes[node].get("allele_distance", 0.0)
                allele_only_new = G.nodes[node].get("allele_distance_only_new", 0.0)
                allele_distances_both_children_polymorph = G.nodes[node].get(
                    "allele_distances_both_children_polymorph", 0.0
                )
                true_allele_distance = G.nodes[node].get("true_allele_distance", 0.0)
                node_time = G.nodes[node].get("time", 0.0)
                node_features.append(
                    [core_distance, allele_distance, allele_only_new, allele_distances_both_children_polymorph, true_allele_distance, node_time]
                )
            node_features = torch.tensor(node_features, dtype=torch.float32)

            graph_properties.append([graph_nodes, graph_edge_index, node_features])

    # Return many of the intermediate structures for downstream use
    return (
        Graphs,
        graph_properties,
        #nodes_hgt_events_simplified,
        #nodes_hgt_events,
        #children_gene_nodes_loss_events,
        #node_to_leaf,
        #G_nodes_reordering,
    )


def _add_edge_safe(tables, left, right, parent, child):
    """Add an edge to `tables` if parent is valid (not -1).

    This helper avoids repeatedly checking parent == -1 throughout the
    traversal logic and centralizes edge insertion behavior.
    """
    if parent != -1:
        tables.edges.add_row(left=left, right=right, parent=parent, child=child)


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
    - genotypes: np.ndarray der Länge tree.num_samples (0 = Gen fehlt, 1 = Gen vorhanden)
    
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

def build_tree_mappings(tree: tskit.Tree, num_samples: int, clonal_root_node: int):
    """
    Construct parent-child relationships, remove unary nodes,
    and compute leaf mappings for a given tskit tree.
    
    Parameters
    ----------
    tree : tskit.Tree
        The tree object from a tree sequence.
    num_samples : int
        Number of sample nodes (tips).
    clonal_root_node : int
        The clonal root node ID (above which we stop traversing).
    
    Returns
    -------
    children : dict[int, list[int]]
        Mapping from parent node → list of children.
    parents : dict[int, int]
        Mapping from child node → parent node.
    node_to_leaf : dict[int, list[int]]
        Mapping from each node → list of descendant leaves.
    """
    children = defaultdict(list)
    
    # Build children dict bottom-up
    for leaf in range(num_samples):
        node = leaf
        while node < clonal_root_node and len(children[node]) < 2:
            parent = tree.parent(node)
            if parent == tskit.NULL:
                break
            children[parent].append(node)
            node = parent

    # Remove unary nodes by splicing
    for node in list(children):
        if len(children[node]) == 1:
            parent = tree.parent(node)
            if parent == tskit.NULL:
                continue
            child = children[node][0]
            if node in children[parent]:
                children[parent].remove(node)
            children[parent].append(child)
            del children[node]

    # Build parents mapping
    parents = defaultdict(int)
    for parent, child_list in children.items():
        for child in child_list:
            parents[child] = parent

    # Build leaf mapping
    node_to_leaf = defaultdict(list)
    for child in sorted(parents.keys()):
        if child < num_samples:  # it's a leaf
            node_to_leaf[child] = [child]
        node_to_leaf[parents[child]].extend(node_to_leaf[child])

    return children, parents, node_to_leaf


def simulate_and_store(num_samples, theta = 0, rho = 0, num_genes = 1, hgt_rate = 0, ce_from_nwk = None, distance_matrix_core = None, output_file = "simulation_result.h5"):
    # run_id für eindeutige Dateinamen
    output_file = os.path.join(os.getcwd(), output_file)
    
    data = simulator(theta = theta, rho = rho, num_samples = num_samples, num_genes = num_genes, hgt_rate = hgt_rate, ce_from_nwk = ce_from_nwk, distance_matrix = distance_matrix_core)

    graph_properties = data["graph_properties"]
    nodes_hgt_events = data["nodes_hgt_events"]
    children_gene_nodes_loss_events = data["children_gene_nodes_loss_events"]
    gene_absence_presence_matrix = data["gene_absence_presence_matrix"]
    fitch_scores = data["fitch_scores"]
    gene_number_loss_events = data["gene_number_loss_events"]
    gene_number_hgt_events_passed = data["gene_number_hgt_events_passed"]
    nodes_hgt_events_simplified = data["nodes_hgt_events_simplified"]

    print(nodes_hgt_events)
    
    if gene_absence_presence_matrix.sum() > -1:
        
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
    
    return data