"""
Adapted from tests/test_mutations.py
"""

from typing import List, Any
import tskit
import msprime
from msprime import _msprime
import numpy as np
import functools
import dataclasses

from random import randint
from collections import defaultdict
from bisect import bisect_left, bisect_right

import time

########################################################################################
# This part has been extracted from the [msprime GitHub repository](https://github.com/tskit-dev/msprime/blob/main/tests/test_mutations.py).
# For licensing information, please refer to the [msprime LICENSE](https://github.com/tskit-dev/msprime/blob/main/LICENSE).


@dataclasses.dataclass
class Mutation:
    node: int
    derived_state: str
    parent: int
    metadata: bytes
    time: float
    new: bool
    id: int  # noqa: A003

    def __str__(self):
        if self.parent is None:
            parent_id = None
        else:
            parent_id = self.parent.id
        s = f"\t{self.id}\t\tnode: {self.node}\tparent: {parent_id}"
        s += f"\ttime: {self.time}\t{self.derived_state}\t{self.metadata}"
        s += f"\t(new: {self.new})"
        return s


@dataclasses.dataclass
class Site:
    position: float
    ancestral_state: str
    metadata: bytes
    mutations: list[Mutation]
    new: bool

    def __str__(self):
        s = f"Position: {self.position}\t{self.ancestral_state}"
        s += f"\t{self.metadata}\t{self.new}\n"
        for mut in self.mutations:
            s += mut.__str__()
        return s

    def add_mutation(
        self,
        node,
        time,
        new,
        derived_state=None,
        metadata=b"",
        id=tskit.NULL,  # noqa: A002
    ):
        mutation = Mutation(
            node=node,
            derived_state=derived_state,
            parent=None,
            metadata=metadata,
            time=time,
            new=new,
            id=id,
        )
        self.mutations.append(mutation)


def cmp_mutation(a, b):
    # Sort mutations by decreasing time and increasing parent,
    # but preserving order of any kept mutations (assumed to be
    # in order already). Kept mutations are given an id that is
    # their order in the initial tables, and new mutations have id -1.
    out = a.id * (not a.new) - b.id * (not b.new)
    if out == 0:
        out = b.time - a.time
    return out


class PythonMutationModel:
    # Base class for mutation models, which must define these methods:

    def root_allele(self, rng):
        pass

    def transition_allele(self, rng, current_allele):
        pass


@dataclasses.dataclass
class PythonMutationMatrixModel(PythonMutationModel):
    alleles: list[bytes]
    root_distribution: Any
    transition_matrix: Any

    def choose_allele(self, rng, distribution):
        u = rng.flat(0, 1)
        j = 0
        while u > distribution[j]:
            u -= distribution[j]
            j += 1
        return self.alleles[j]

    def root_allele(self, rng):
        return self.choose_allele(rng, self.root_distribution)

    def transition_allele(self, rng, current_allele):
        j = self.alleles.index(current_allele)
        return self.choose_allele(rng, self.transition_matrix[j])


########################################################################################


class HGTMutationGenerator:
    def __init__(self, rate_map, model):
        """
        Defaults to all 0->1 mutations.
        """
        self.rate_map = rate_map
        self.model = model
        self.sites = {}
        self.edges_to_remove = []

        self.bin_null_mask = 0b00
        self.bin_sentinel_mask = 0b01
        self.bin_hgt_mask = 0b10

    def print_state(self):
        positions = sorted(self.sites.keys())
        for pos in positions:
            print(self.sites[pos])

    def add_site(self, position, new, ancestral_state=None, metadata=b""):
        assert position not in self.sites
        site = Site(
            position=position,
            ancestral_state=ancestral_state,
            metadata=metadata,
            mutations=[],
            new=new,
        )
        self.sites[position] = site
        return site

    def initialise_sites(self, tables):
        mutation_rows = iter(tables.mutations)
        mutation_row = next(mutation_rows, None)
        j = 0
        for site_id, site_row in enumerate(tables.sites):
            site = self.add_site(
                position=site_row.position,
                new=False,
                ancestral_state=site_row.ancestral_state,
                metadata=site_row.metadata,
            )
            while mutation_row is not None and mutation_row.site == site_id:
                site.add_mutation(
                    node=mutation_row.node,
                    time=mutation_row.time,
                    new=False,
                    derived_state=mutation_row.derived_state,
                    metadata=mutation_row.metadata,
                    id=j,
                )
                j += 1
                mutation_row = next(mutation_rows, None)

    def populate_tables(self, tables):
        positions = sorted(self.sites.keys())
        site_id = 0
        for pos in positions:
            site = self.sites[pos]
            num_mutations = 0
            for mutation in site.mutations:
                if mutation.parent is None:
                    parent_id = tskit.NULL
                else:
                    parent_id = mutation.parent.id
                    assert parent_id >= 0
                mutation_id = tables.mutations.add_row(
                    site_id,
                    mutation.node,
                    mutation.derived_state,
                    parent=parent_id,
                    metadata=mutation.metadata,
                    time=mutation.time,
                )
                assert mutation_id > parent_id
                mutation.id = mutation_id
                num_mutations += 1

            if (not site.new) or num_mutations > 0:
                sid = tables.sites.add_row(site.position, site.ancestral_state, site.metadata)
                assert sid == site_id
                site_id += 1

    #@profile
    def place_mutations(self, tables, edges, root_nodes, leaf_node_ids, discrete_genome=False):
        """
        edges: List of edges including hgt edges
        """
        # Insert a sentinel into the map for convenience.
        map_position = np.hstack([self.rate_map.position, [tables.sequence_length]])
        node_times = tables.nodes.time
        for edge in edges:
            branch_start = node_times[edge.child]
            branch_end = node_times[edge.parent]
            branch_length = branch_end - branch_start
            index = np.searchsorted(map_position, edge.left)
            if map_position[index] > edge.left:
                index -= 1
            left = edge.left
            right = 0
            while right != edge.right:
                right = min(edge.right, map_position[index + 1])
                site_left = np.ceil(left) if discrete_genome else left
                site_right = np.ceil(right) if discrete_genome else right
                assert site_left <= site_right
                assert map_position[index] <= left
                assert right <= map_position[index + 1]
                assert right <= edge.right
                # Generate the mutations.
                rate = self.rate_map.rate[index]
                mu = rate * (site_right - site_left) * branch_length
                for _ in range(self.rng.poisson(mu)[0]):
                    position = self.rng.flat(site_left, site_right)[0]
                    if discrete_genome:
                        position = np.floor(position)
                    assert edge.left <= position
                    assert position < edge.right
                    if position not in self.sites:
                        self.add_site(position=position, new=True)
                    site = self.sites[position]
                    time = self.rng.flat(branch_start, branch_end)[0]
                    site.add_mutation(
                        node=edge.child,
                        time=time,
                        new=True,
                        metadata=self.bin_null_mask.to_bytes(1),
                        #derived_state="absent",   # CHANGE!!!
                    )
                index += 1
                left = right
        
        
        # Add a sentinel mutation at directly above the leafs
        
        leaf_node_ids = [i for i, f in enumerate(tables.nodes.flags) if f == 1]
        for pos, site in self.sites.items():
            for leaf in leaf_node_ids:
                site.add_mutation(
                    node=leaf,
                    time=0.00000000001,
                    new=True,
                    metadata=self.bin_sentinel_mask.to_bytes(1),
                )
            # NEW:!!
            root_node = max([e.parent for e in edges if e.left <= pos and pos < e.right])
            site.add_mutation(
                node=root_node,
                time=node_times[root_node],
                new=True,
                metadata=self.bin_sentinel_mask.to_bytes(1),
                derived_state=site.ancestral_state,
            )

    def place_one_mutation(self, tables, edges, discrete_genome=False):
        """
        edges: List of edges including hgt edges
        """
        # Insert a sentinel into the map for convenience.
        map_position = np.hstack([self.rate_map.position, [tables.sequence_length]])
        node_times = tables.nodes.time
        
        for pos, site in self.sites.items():  # Iteriere über die Sites

            egdes_site = []
            for edge in edges:
                if (edge.left <= pos and edge.right >= pos + 1):
                    egdes_site.append(edge)
            
            # Berechne die Wahrscheinlichkeiten basierend auf den Branch-Längen
            branch_lengths = []
            for edge in egdes_site:
                branch_start = node_times[edge.child]
                branch_end = node_times[edge.parent]
                branch_length = branch_end - branch_start
                branch_lengths.append(branch_length)
        
            # Normiere die Branch-Längen, um Wahrscheinlichkeiten zu erhalten
            total_length = sum(branch_lengths)
            probabilities = [length / total_length for length in branch_lengths]
        
            # Wähle einen zufälligen Ast basierend auf den Wahrscheinlichkeiten
            chosen_index = np.random.choice(len(egdes_site), p=probabilities)
            chosen_edge = egdes_site[chosen_index]
        
            # Zeit für die Mutation innerhalb des gewählten Branches zufällig auswählen
            branch_start = node_times[chosen_edge.child]
            branch_end = node_times[chosen_edge.parent]
            mutation_time = self.rng.flat(branch_start, branch_end)[0]
        
            # Füge die Mutation hinzu
            site.add_mutation(
                node=chosen_edge.child,
                time=mutation_time,
                new=True,
                metadata=self.bin_null_mask.to_bytes(1),
                derived_state="present", # CHANGE!!
            )

        """
        # Add a sentinel mutation directly above the leafs
        leaf_node_ids = [i for i, f in enumerate(tables.nodes.flags) if f == 1]
        for pos, site in self.sites.items():
            for leaf in leaf_node_ids:
                site.add_mutation(
                    node=leaf,
                    time=0.00000000001,
                    new=True,
                    metadata=self.bin_sentinel_mask.to_bytes(1),
                )
        """

    def find_bottom_mut(self, node_id, tree_parent, was_hgt):
        if node_id in self.bottom_mut:
            # Found mutation directly above current one
            return [(self.bottom_mut[node_id], was_hgt)]
        else:
            # Not directly above, going to traverse parent edge(s)
            parent_ids = tree_parent[node_id]
            if not parent_ids:
                # Reached root
                return []

            traversal_match = []
            for parent_id, is_hgt in parent_ids:
                is_hgt = was_hgt or is_hgt
                traversal_match.extend(self.find_bottom_mut(parent_id, tree_parent, is_hgt))
            return traversal_match

    #@profile
    def choose_alleles(self, tree_parent, site, mutation_id_offset):
        if site.new:
            site.ancestral_state = self.model.root_allele(self.rng)
        # sort mutations by (increasing id if both are not null,
        #  decreasing time, increasing insertion order)
        site.mutations.sort(key=functools.cmp_to_key(cmp_mutation))

        site.mutations.sort(key=lambda mutation: mutation.time, reverse=True)   # CHANGE!!
        
        self.bottom_mut = {}
        for mut in site.mutations:
            # Traverse up the tree to find the parent mutation(s)
            # bottom_mutation[u] is the index in mutations of the most recent
            #    mutation seen on the edge above u so far, if any
            parent_muts = self.find_bottom_mut(mut.node, tree_parent, False)
            # print("Parent muts", parent_muts)

            if not parent_muts:
                # Root  / No previous mutations
                pa = site.ancestral_state
                assert mut.parent is None
            else:
                if len(parent_muts) == 1:
                    parent_mut, _ = parent_muts[0]
                else:
                    # HGT Case, multiple parent mutations
                    # only works for gene gain / loss mutation model
                    # Ensure that non-hgt mutations are first
                    parent_muts = sorted(parent_muts, key=lambda m: m[1], reverse=True)
                    present_parent = [p for p in parent_muts if p[0].derived_state == "present"]
                    if present_parent:
                        # Case were a gene gain happend on at least one of the parents
                        parent_mut, is_hgt = present_parent[0]
                    else:
                        # Select "regular" (non-hgt) parent mutation.
                        parent_mut, is_hgt = parent_muts[0]

                    if is_hgt:
                        # Add hgt flag
                        metadata = (
                            int.from_bytes(mut.metadata) if mut.metadata else self.bin_null_mask
                        )
                        mut.metadata = (metadata | self.bin_hgt_mask).to_bytes(1)

                mut.parent = parent_mut
                assert mut.time <= parent_mut.time, "Parent after child mutation."
                if mut.new:
                    pa = parent_mut.derived_state

            if mut.new:
                if int.from_bytes(mut.metadata) & self.bin_sentinel_mask:
                    mut.derived_state = pa
                else:
                    da = self.model.transition_allele(self.rng, pa)
                    mut.derived_state = da

            self.bottom_mut[mut.node] = mut

    def follow_edge(self, bp, node, node_to_children):
        #child_edges = [e for e in edges if e.parent == node if e.left <= bp[0] and bp[1] <= e.right]
        #child_edges = [e for e in edges if e.parent == node]
        child_edges = node_to_children[node]
        for e in child_edges:
            is_hgt = bool(int.from_bytes(e.metadata) & self.bin_hgt_mask)

            if not self.tree_parent[bp][e.child]:
                self.tree_parent[bp][e.child] = {(e.parent, is_hgt)}
            elif (e.parent, False) in self.tree_parent[bp][e.child] and is_hgt:
                # Is hgt, but not added as parent yet, overwriting the old (non hgt) entry
                self.tree_parent[bp][e.child].remove((e.parent, False))
                self.tree_parent[bp][e.child].add((e.parent, True))
            else:
                self.tree_parent[bp][e.child].add((e.parent, is_hgt))

            self.follow_edge(bp, e.child, node_to_children)

    @profile
    def apply_mutations(self, tables, edges, root_nodes, leaf_node_ids, number_of_sites):
        apply_start_time = time.time()

        bp_left, bp_right = zip(*((int(e.left), int(e.right)) for e in edges))
        breakpoints = sorted(list(set(bp_left + bp_right)))
        
        site_edges = [[] for _ in range(number_of_sites+1)]
        for e in edges:
            for site_breakpoint in breakpoints[bisect_left(breakpoints, e.left):(bisect_right(breakpoints, e.right)-1)]:
                site_edges[site_breakpoint].append(e)

        self.tree_parent = {}
        for bp in zip(breakpoints, breakpoints[1:]): # Neigboring sites with the same tree are treated together.
            self.tree_parent[bp] = [set() for _ in range(tables.nodes.num_rows)]
            node_to_children = defaultdict(list)
            for edge in site_edges[bp[0]]:
                node_to_children[edge.parent].append(edge)

            root_node = max([e.parent for e in site_edges[bp[0]]])
            
            self.follow_edge(bp, root_node, node_to_children)

        for pos, site in self.sites.items():
            assert pos == site.position
            # the responsible
            k = [k for k in self.tree_parent.keys() if k[0] <= pos < k[1]][0]  # CHANGE!!!!!!!!!!!!!!!!!!
            self.choose_alleles(self.tree_parent[k], site, None)
        
        """ # OLD
        # Build tree_parent list for every segment interval
        bp_left, bp_right = zip(*((e.left, e.right) for e in edges))
        breakpoints = sorted(list(set(bp_left + bp_right)))
        #print(breakpoints)
        self.tree_parent = {}
        for bp in zip(breakpoints, breakpoints[1:]): # Neigboring sites with the same tree are treated together.
            self.tree_parent[bp] = [set() for _ in range(tables.nodes.num_rows)]
            site_edges = [e for e in edges if e.left <= bp[0] and bp[1] <= e.right]
            node_to_children = defaultdict(list)
            for edge in site_edges:
                node_to_children[edge.parent].append(edge)
            ### CHANGE:
            # for root_node in root_nodes:
                # Index= Child ID, Value: Set of parents.
                
            #    self.follow_edge(bp, root_node, edges)
            root_node = max([e.parent for e in edges if e.left <= bp[0] and bp[1] <= e.right])
            
            self.follow_edge(bp, root_node, node_to_children)

        for pos, site in self.sites.items():
            assert pos == site.position
            # the responsible
            k = [k for k in self.tree_parent.keys() if k[0] <= pos < k[1]][0]  # CHANGE!!!!!!!!!!!!!!!!!!
            self.choose_alleles(self.tree_parent[k], site, None)
        """
        
        apply_end_time = time.time()
        # print("Apply time:", apply_end_time-apply_start_time)

    def rectify_hgt_edges(self, tables, edges):
        edges = list(e for e in edges if not int.from_bytes(e.metadata) & self.bin_hgt_mask)
        return sorted(edges, key=lambda e: (tables.nodes[e.parent].time, e.child, e.left))

    @profile
    def generate(
        self,
        tables,
        edges,
        root_nodes,
        leaf_node_ids,
        number_of_sites,
        seed,
        one_mutation=False,
        keep=False,
        discrete_genome=False,
    ):

        self.rng = _msprime.RandomGenerator(seed)
        if keep:
            self.initialise_sites(tables)
            
        tables.sites.clear()
        tables.mutations.clear()
        if one_mutation:
            self.place_one_mutation(tables, edges, discrete_genome=discrete_genome)
        else:
            self.place_mutations(tables, edges, root_nodes, leaf_node_ids, discrete_genome=discrete_genome)
            self.apply_mutations(tables, edges, root_nodes, leaf_node_ids, number_of_sites)


        #for pos, site in self.sites.items():
        #    print(site.mutations)

        #self.apply_mutations(tables, edges, root_nodes)

        #for pos, site in self.sites.items():
        #    print(site.mutations)
            
        self.populate_tables(tables)

        edges = self.rectify_hgt_edges(tables, edges)
        tables.edges.clear()

        for e in edges:
            tables.edges.add_row(left=e.left, right=e.right, parent=e.parent, child=e.child)
        ts = tables.tree_sequence()
        return ts


def sim_mutations(
    ts: tskit.TreeSequence,
    hgt_edges: List[tskit.Edge],
    event_rate: float,
    model: PythonMutationMatrixModel,
    one_mutation = True
):
    tables = ts.dump_tables()

    gene_count = tables.sequence_length

    gain_loss_model = PythonMutationMatrixModel(
        alleles=model.alleles,
        root_distribution=model.root_distribution,
        transition_matrix=model.transition_matrix,
    )

    rate_map = msprime.RateMap(position=[0, gene_count], rate=[event_rate])

    edges = list(tables.edges)
    edges.extend(hgt_edges)

    child_ids = {e.child for e in edges}
    root_nodes = list({e.parent for e in edges if e.parent not in child_ids})
    leaf_node_ids = [i for i, f in enumerate(tables.nodes.flags) if f == 1]

    number_of_sites = int(ts.sequence_length)
    
    hgt_generator = HGTMutationGenerator(rate_map=rate_map, model=gain_loss_model)

    ts = hgt_generator.generate(
        tables,
        edges,
        root_nodes,
        leaf_node_ids,
        number_of_sites,
        randint(0, 4294967295),
        one_mutation,
        keep=True,
        discrete_genome=True,
    )
    return ts









