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
from collections import deque
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

        for pos, site in self.sites.items():
            for mutation in site.mutations:
                self.mutation_edge[int(pos)][mutation.node].append(mutation)
            

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
                    #assert parent_id >= 0
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
    def place_mutations(self, tables, edges, discrete_genome=False):
        """
        Place losses in the tree
        
        edges: List of edges including hgt edges
        """
        # Insert a sentinel into the map for convenience.


        total_length = self.edges_lengths.sum()
        
        number_of_mutations = np.random.poisson(self.rho * total_length, size=1)
        
        probabilities = self.edges_lengths / total_length

        mutation_indices = np.random.choice(len(edges), size=number_of_mutations, p=probabilities, replace=True)

        for i in mutation_indices:
            edge_with_mutation = edges[i]
                    
            branch_start = self.node_times[edge_with_mutation.child]
            branch_end = self.node_times[edge_with_mutation.parent]
            time = self.rng.flat(branch_start, branch_end)[0]

            probabilities_site_position = [self.site_edges_total_lengths[pos] for pos in range(int(edge_with_mutation.left),int(edge_with_mutation.right))]
            probabilities_site_position = np.array(probabilities_site_position)
            probabilities_site_position = probabilities_site_position/probabilities_site_position.sum()
            
            site_position = np.random.choice(range(int(edge_with_mutation.left),int(edge_with_mutation.right)), size=1, p=probabilities_site_position)
            site = self.sites[site_position[0]]
            
            site.add_mutation(
                node=edge_with_mutation.child,
                time=time,
                new=True,
                metadata=self.bin_null_mask.to_bytes(1),
                derived_state="absent",   # CHANGE!!!
            )
            mutation = Mutation(
                node=edge_with_mutation.child,
                derived_state="absent",
                parent=None,
                metadata=self.bin_null_mask.to_bytes(1),
                time=time,
                new=True,
                id=tskit.NULL,
            )
            self.mutation_edge[int(site.position)][edge_with_mutation.child].append(mutation)

        # Add a sentinel mutation at directly above the leafs
        
        self.leaf_node_ids = [i for i, f in enumerate(tables.nodes.flags) if f == 1]
        for pos, site in self.sites.items():
            for leaf in self.leaf_node_ids:
                site.add_mutation(
                    node=leaf,
                    time=0.00000000001,
                    new=True,
                    metadata=self.bin_sentinel_mask.to_bytes(1),
                )
            k = bisect_right(self.breakpoints, pos) - 1
            site.add_mutation(
                #node=root_node,
                node=self.root_nodes[self.breakpoints[k]],
                time=self.node_times[self.root_nodes[self.breakpoints[k]]],
                new=True,
                metadata=self.bin_sentinel_mask.to_bytes(1),
                derived_state=site.ancestral_state,
            )
            mutation = Mutation(
                node=self.root_nodes[self.breakpoints[k]],
                derived_state=site.ancestral_state,
                parent=None,
                metadata=self.bin_sentinel_mask.to_bytes(1),
                time=self.node_times[self.root_nodes[self.breakpoints[k]]],
                new=True,
                id=tskit.NULL,
            )
            self.mutation_edge[int(site.position)][self.root_nodes[self.breakpoints[k]]].append(mutation)

    #@profile
    def place_one_mutation(self, tables, edges, discrete_genome=False):
        """

        Place gains (one mutation per site)
        
        edges: List of edges including hgt edges
        """


        for pos, site in self.sites.items():
            k = bisect_right(self.breakpoints, pos) - 1
            pos = self.breakpoints[k]
                   
            probabilities = np.array(self.site_edges_lengths[int(pos)])
            probabilities = probabilities/probabilities.sum()

            mutation_indice = np.random.choice(len(probabilities), size=1, p=probabilities)[0]
            edge_with_mutation = self.site_edges[int(pos)][mutation_indice]

            branch_start = self.node_times[edge_with_mutation.child]
            branch_end = self.node_times[edge_with_mutation.parent]
            time = self.rng.flat(branch_start, branch_end)[0]

            site.add_mutation(
                node=edge_with_mutation.child,
                time=time,
                new=True,
                metadata=self.bin_null_mask.to_bytes(1),
                derived_state="present", # CHANGE!!
            )
            mutation = Mutation(
                node=edge_with_mutation.child,
                derived_state="present",
                parent=None,
                metadata=self.bin_null_mask.to_bytes(1),
                time=time,
                new=True,
                id=tskit.NULL,
            )
            self.mutation_edge[int(site.position)][edge_with_mutation.child].append(mutation)

    #@profile
    def find_ancestor_mutations(self, site, leaf_mut_node):
        # Determine the left breakpoint of the site.
        k = bisect_right(self.breakpoints, int(site.position)) - 1
        bp = self.breakpoints[k]
        
        # Initialize results and a queue for traversal.
        parent_mutations = []
        nodes_to_process = deque([leaf_mut_node])
        
        visited = set()  # Prevents duplicate processing of nodes.
    
        while nodes_to_process:
            child_node = nodes_to_process.popleft()
            
            # Process the node only if it has not been visited yet.
            if child_node in visited:
                continue
            visited.add(child_node)
            
            # If there is a mutation on the edge, find the earliest one.
            if self.mutation_edge[int(site.position)][child_node]:
                earliest_mutation = min(
                    self.mutation_edge[int(site.position)][child_node], 
                    key=lambda m: m.time
                )
                parent_mutations.append(earliest_mutation)
                
                # Check for an HGT event and add parent nodes.
                if len(self.node_to_parent[bp][child_node]) > 1:
                    hgt_parent = min([
                        e.parent for e in self.node_to_parent[bp][child_node]
                    ])
                    #print("TWO PARENTS", hgt_parent, self.node_to_parent[bp][child_node])
                    nodes_to_process.append(hgt_parent)
    
            # If there is no mutation, add parent nodes.
            else:
                parent_nodes = [e.parent for e in self.node_to_parent[bp][child_node]]
                nodes_to_process.extend(parent_nodes)
        
        return parent_mutations


    """
    def find_ancestor_mutations(self, site, leaf_mut_node):

        #print(self.mutation_edge[int(site.position)])
        
        k = bisect_right(self.breakpoints, int(site.position)) - 1
        bp = self.breakpoints[k]   # The left breakpoint of the site.
        
        parent_mutations = []
        nodes_without_mutation = set()

        nodes_without_mutation.add(leaf_mut_node)
        
        while nodes_without_mutation:
            #print(int(site.position), nodes_without_mutation)
            for child_node in set(nodes_without_mutation):
                
                # There is at least a mutation on the edge.
                if self.mutation_edge[int(site.position)][child_node]: 
                    parent_mutations.append(min(self.mutation_edge[int(site.position)][child_node], key=lambda m: m.time))
                    if (len(self.node_to_parent[bp][child_node]) > 1): # HGT event directly before mutation
                        nodes_without_mutation.add(min([e.parent for e in self.node_to_parent[bp][child_node]]))


                # There is no mutation on the edge.
                else:
                    for e in self.node_to_parent[bp][child_node]:
                        nodes_without_mutation.add(e.parent)

                nodes_without_mutation.remove(child_node)
                
                
        return parent_mutations
    """     

    #@profile
    def choose_alleles(self):
        
        for pos, site in self.sites.items():
            if site.new:
                site.ancestral_state = self.model.root_allele(self.rng)
            # sort mutations by (increasing id if both are not null,
            #  decreasing time, increasing insertion order)
            site.mutations.sort(key=functools.cmp_to_key(cmp_mutation))
    
            site.mutations.sort(key=lambda mutation: mutation.time, reverse=True)   # CHANGE!!
    
            k = bisect_right(self.breakpoints, pos) - 1

            self.bottom_mut = {}
            for leaf_mut in site.mutations:
                if (leaf_mut.time == 0.00000000001 and leaf_mut.metadata == self.bin_sentinel_mask.to_bytes(1)): # Sentinel mutations on the leaves
                    # Traverse up the tree to find the parent mutation(s)
                    # bottom_mutation[u] is the index in mutations of the most recent
                    #    mutation seen on the edge above u so far, if any
                    
                    parent_muts = self.find_ancestor_mutations(site, leaf_mut.node)
                    #print("pos:", pos, "leaf_mut.node", leaf_mut.node, parent_muts)
    
                    parent_muts = sorted(parent_muts, key=lambda m: m.time)
                    present_parent = [p for p in parent_muts if p.derived_state == "present"]
                    if present_parent:
                        leaf_mut.parent = min(present_parent, key=lambda m: m.time)
                        leaf_mut.derived_state = "present"
                    else:
                        leaf_mut.parent = min(parent_muts, key=lambda m: m.time)
                        leaf_mut.derived_state = "absent"


    def rectify_hgt_edges(self, tables, edges):
        edges = list(e for e in edges if not int.from_bytes(e.metadata) & self.bin_hgt_mask)
        return sorted(edges, key=lambda e: (tables.nodes[e.parent].time, e.child, e.left))

    #@profile
    def generate(
        self,
        tables,
        edges,
        rho,
        #leaf_node_ids,
        number_of_sites,
        seed,
        one_mutation=False,
        keep=False,
        discrete_genome=False,
    ):
        
        bp_left, bp_right = zip(*((int(e.left), int(e.right)) for e in edges))
        
        self.rho = rho
        
        self.breakpoints = sorted(list(set(bp_left + bp_right)))
        self.site_edges = [[] for _ in range(number_of_sites+1)]
        self.site_edges_lengths = [[] for _ in range(number_of_sites+1)]
        self.site_edges_total_lengths = [[] for _ in range(number_of_sites+1)]
        self.node_times = tables.nodes.time
        self.mutation_edge = [[] for _ in range(number_of_sites+1)]
        self.root_nodes = [None for _ in range(number_of_sites+1)]
        self.node_to_parent = [[] for _ in range(number_of_sites+1)]
        
        for e in edges:
            for site_breakpoint in self.breakpoints[bisect_left(self.breakpoints, e.left):(bisect_right(self.breakpoints, e.right)-1)]:
                self.site_edges[site_breakpoint].append(e)
                self.site_edges_lengths[site_breakpoint].append(self.node_times[e.parent]-self.node_times[e.child])
              
        self.edges_lengths = np.array([(self.node_times[e.parent]-self.node_times[e.child])*(e.right-e.left) for e in edges], dtype=np.float64)

        for pos in range(number_of_sites+1):
            k = bisect_right(self.breakpoints, pos) - 1
            self.site_edges_total_lengths[pos] = np.array(self.site_edges_lengths[self.breakpoints[k]]).sum()
            self.mutation_edge[pos]=defaultdict(list)

        
        for bp in zip(self.breakpoints, self.breakpoints[1:]): # Neigboring sites with the same tree are treated together.
            self.node_to_parent[bp[0]] = defaultdict(list)
            for edge in self.site_edges[bp[0]]:
                self.node_to_parent[bp[0]][edge.child].append(edge)

            self.root_nodes[bp[0]] = max([e.parent for e in self.site_edges[bp[0]]])

        
        self.rng = _msprime.RandomGenerator(seed)
        if keep:
            self.initialise_sites(tables)

        self.mutations = [[] for _ in range(number_of_sites+1)]
            
        tables.sites.clear()
        tables.mutations.clear()

      # Workflow:
        
        
        if one_mutation:
            self.place_one_mutation(tables, edges, discrete_genome=discrete_genome)
        else:
            self.place_mutations(tables, edges, discrete_genome=discrete_genome)
            self.choose_alleles()
            
        self.populate_tables(tables)

        """
        edges = self.rectify_hgt_edges(tables, edges)
        tables.edges.clear()

        for e in edges:
            tables.edges.add_row(left=e.left, right=e.right, parent=e.parent, child=e.child)
        """
        ts = tables.tree_sequence()
        return ts


def sim_mutations(
    ts: tskit.TreeSequence,
    hgt_edges: List[tskit.Edge],
    rho: float,
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
    
    number_of_sites = int(ts.sequence_length)
    
    hgt_generator = HGTMutationGenerator(rate_map=rate_map, model=gain_loss_model)

    ts = hgt_generator.generate(
        tables,
        edges,
        rho,
        number_of_sites,
        randint(0, 4294967295),
        one_mutation,
        keep=True,
        discrete_genome=True,
    )
    return ts




