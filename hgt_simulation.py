"""
Python version of the simulation algorithm.
Based on https://github.com/tskit-dev/msprime/blob/main/algorithms.py
Modified to support HGT and Tree Fixation.
"""

import argparse
import heapq
import itertools
import logging
import math
import random

import bintrees
import daiquiri
import numpy as np
import tskit
import msprime

import sys

logger = daiquiri.getLogger()


class FenwickTree:
    """
    A Fenwick Tree to represent cumulative frequency tables over
    integers. Each index from 1 to max_index initially has a
    zero frequency.

    This is an implementation of the Fenwick tree (also known as a Binary
    Indexed Tree) based on "A new data structure for cumulative frequency
    tables", Software Practice and Experience, Vol 24, No 3, pp 327 336 Mar
    1994. This implementation supports any non-negative frequencies, and the
    search procedure always returns the smallest index such that its cumulative
    frequency <= f. This search procedure is a slightly modified version of
    that presented in Tech Report 110, "A new data structure for cumulative
    frequency tables: an improved frequency-to-symbol algorithm." available at
    https://www.cs.auckland.ac.nz/~peter-f/FTPfiles/TechRep110.ps
    """

    def __init__(self, max_index):
        assert max_index > 0
        self.__max_index = max_index
        self.__tree = [0 for j in range(max_index + 1)]
        self.__value = [0 for j in range(max_index + 1)]
        # Compute the binary logarithm of max_index
        u = self.__max_index
        while u != 0:
            self.__log_max_index = u
            u -= u & -u

    def get_total(self):
        """
        Returns the total cumulative frequency over all indexes.
        """
        return self.get_cumulative_sum(self.__max_index)

    def increment(self, index, v):
        """
        Increments the frequency of the specified index by the specified
        value.
        """
        assert 0 < index <= self.__max_index
        self.__value[index] += v
        j = index
        while j <= self.__max_index:
            self.__tree[j] += v
            j += j & -j

    def set_value(self, index, v):
        """
        Sets the frequency at the specified index to the specified value.
        """
        f = self.get_value(index)
        self.increment(index, v - f)

    def get_cumulative_sum(self, index):
        """
        Returns the cumulative frequency of the specified index.
        """
        assert 0 < index <= self.__max_index
        j = index
        s = 0
        while j > 0:
            s += self.__tree[j]
            j -= j & -j
        return s

    def get_value(self, index):
        """
        Returns the frequency of the specified index.
        """
        return self.__value[index]

    def find(self, v):
        """
        Returns the smallest index with cumulative sum >= v.
        """
        j = 0
        s = v
        half = self.__log_max_index
        while half > 0:
            # Skip non-existant entries
            while j + half > self.__max_index:
                half >>= 1
            k = j + half
            if s > self.__tree[k]:
                j = k
                s -= self.__tree[j]
            half >>= 1
        return j + 1


class Segment:
    """
    A class representing a single segment. Each segment has a left
    and right, denoting the loci over which it spans, a node and a
    next, giving the next in the chain.
    """

    def __init__(self, index):
        self.left = None
        self.right = None
        self.node = None
        self.prev = None
        self.next = None
        self.population = None
        self.label = 0
        self.index = index
        self.hull = None
        self.origin = set()

    def __repr__(self):
        return repr((self.left, self.right, self.node, self.origin))

    @staticmethod
    def show_chain(seg):
        s = ""
        while seg is not None:
            s += f"[{seg.left}, {seg.right}: {seg.node}, {seg.origin}], "
            seg = seg.next
        return s[:-2]

    def __lt__(self, other):
        return (self.left, self.right, self.population, self.node) < (
            other.left,
            other.right,
            other.population,
            self.node,
        )

    def get_hull(self):
        seg = self
        assert seg is not None
        while seg.prev is not None:
            seg = seg.prev
        hull = seg.hull
        return hull

    def get_left_index(self):
        seg = self
        while seg is not None:
            index = seg.index
            seg = seg.prev

        return index


class Population:
    """
    Class representing a population in the simulation.
    """

    def __init__(self, id_, num_labels=1, max_segments=100, model="hudson"):
        self.id = id_
        self.start_time = 0
        self.start_size = 1.0
        self.growth_rate = 0
        # Keep a list of each label.
        # We'd like to use AVLTrees here for P but the API doesn't quite
        # do what we need. Lists are inefficient here and should not be
        # used in a real implementation.
        self._ancestors = [[] for _ in range(num_labels)]

        # ADDITIONAL STATES FOR SMC(k)
        # this has to be done for each label
        # track hulls based on left
        self.hulls_left = [OrderStatisticsTree() for _ in range(num_labels)]
        self.coal_mass_index = [FenwickTree(max_segments) for j in range(num_labels)]
        # track rank of hulls right
        self.hulls_right = [OrderStatisticsTree() for _ in range(num_labels)]

        if model == "smc_k":
            self.get_common_ancestor_waiting_time = self.get_common_ancestor_waiting_time_smc_k()
        else:
            self.get_common_ancestor_waiting_time = self.get_common_ancestor_waiting_time_hudson()

    def print_state(self):
        print("Population ", self.id)
        print("\tstart_size = ", self.start_size)
        print("\tgrowth_rate = ", self.growth_rate)
        print("\tAncestors: ", len(self._ancestors))
        for label, ancestors in enumerate(self._ancestors):
            print("\tLabel = ", label)
            for u in ancestors:
                print("\t\t" + Segment.show_chain(u))

    def set_growth_rate(self, growth_rate, time):
        # TODO This doesn't work because we need to know what the time
        # is so we can set the start size accordingly. Need to look at
        # ms's model carefully to see what it actually does here.
        new_size = self.get_size(time)
        self.start_size = new_size
        self.start_time = time
        self.growth_rate = growth_rate

    def set_start_size(self, start_size):
        self.start_size = start_size
        self.growth_rate = 0

    def get_num_ancestors(self, label=None):
        if label is None:
            return sum(len(label_ancestors) for label_ancestors in self._ancestors)
        else:
            return len(self._ancestors[label])

    def get_num_pairs(self, label=None):
        # can be improved by updating values in self.num_pairs
        if label is None:
            return sum(mass_index.get_total() for mass_index in self.coal_mass_index)
        else:
            return self.coal_mass_index[label].get_total()

    def get_size(self, t):
        """
        Returns the size of this population at time t.
        """
        dt = t - self.start_time
        return self.start_size * math.exp(-self.growth_rate * dt)

    def _get_common_ancestor_waiting_time(self, np, t):
        """
        Returns the random waiting time until a common ancestor event
        occurs within this population.
        """
        ret = sys.float_info.max
        u = random.expovariate(2 * np)
        if self.growth_rate == 0:
            ret = self.start_size * u
        else:
            dt = t - self.start_time
            z = 1 + self.growth_rate * self.start_size * math.exp(-self.growth_rate * dt) * u
            if z > 0:
                ret = math.log(z) / self.growth_rate
        return ret

    def get_common_ancestor_waiting_time_hudson(self):
        def _get_common_ancestor_waiting_time_hudson(t):
            k = self.get_num_ancestors()
            ret = sys.float_info.max
            if k > 1:
                np = k * (k - 1) / 2
                ret = self._get_common_ancestor_waiting_time(np, t)
            return ret

        return _get_common_ancestor_waiting_time_hudson

    def get_common_ancestor_waiting_time_smc_k(self):
        def _get_common_ancestor_waiting_time_smc_k(t):
            np = self.get_num_pairs()
            ret = sys.float_info.max
            if np > 0:
                ret = self._get_common_ancestor_waiting_time(np, t)
            return ret

        return _get_common_ancestor_waiting_time_smc_k

    def get_ind_range(self, t):
        """Returns ind labels at time t"""
        first_ind = np.sum([self.get_size(t_prev) for t_prev in range(0, int(t))])
        last_ind = first_ind + self.get_size(t)

        return range(int(first_ind), int(last_ind) + 1)

    def increment_avl(self, ost, coal_mass, hull, increment):
        right = hull.right
        curr_hull = hull
        curr_hull, _ = ost.succ_key(curr_hull)
        while curr_hull is not None:
            if right > curr_hull.left:
                ost.avl[curr_hull] += increment
                coal_mass.increment(curr_hull.index, increment)
            else:
                break
            curr_hull, _ = ost.succ_key(curr_hull)

    def reset_hull_right(self, label, hull, old_right, new_right):
        # when resetting the hull.right of a pre-existing hull we need to
        # decrement count of all lineages starting off between hull.left and bp
        # FIX: logic is almost identical to increment_avl()!!!
        ost = self.hulls_left[label]
        curr_hull = Hull(-1)
        curr_hull.left = new_right
        curr_hull.right = math.inf
        curr_hull.insertion_order = 0
        floor = ost.floor_key(curr_hull)
        curr_hull = floor
        while curr_hull is not None:
            if curr_hull.left >= old_right:
                break
            if curr_hull.left >= new_right:
                ost.avl[curr_hull] -= 1
                self.coal_mass_index[label].increment(curr_hull.index, -1)
            curr_hull, _ = ost.succ_key(curr_hull)
        hull.right = new_right

        # adjust rank of hull.right
        ost = self.hulls_right[label]
        floor = ost.floor_key(HullEnd(old_right))
        assert floor.x == old_right
        ost.pop(floor)
        insertion_order = 0
        hull_end = HullEnd(new_right)
        floor = ost.floor_key(hull_end)
        if floor is not None:
            if floor.x == hull_end.x:
                insertion_order = floor.insertion_order + 1
        hull_end.insertion_order = insertion_order
        ost[hull_end] = 0

    def remove_hull(self, label, hull):
        ost = self.hulls_left[label]
        coal_mass_index = self.coal_mass_index[label]
        self.increment_avl(ost, coal_mass_index, hull, -1)
        # adjust insertion order
        curr_hull, _ = ost.succ_key(hull)
        count, left_rank = ost.pop(hull)
        while curr_hull is not None:
            if curr_hull.left == hull.left:
                curr_hull.insertion_order -= 1
            else:
                break
            curr_hull, _ = ost.succ_key(curr_hull)
        ost = self.hulls_right[label]
        floor = ost.floor_key(HullEnd(hull.right))
        assert floor.x == hull.right
        _, right_rank = ost.pop(floor)
        hull.insertion_order = math.inf
        self.coal_mass_index[label].set_value(hull.index, 0)

    def remove(self, index, label=0):
        """
        Removes and returns the individual at the specified index.
        """
        return self._ancestors[label].pop(index)

    def remove_individual(self, individual, label=0):
        """
        Removes the given individual from its population.
        """
        return self._ancestors[label].remove(individual)

    def add_hull(self, label, hull):
        # logic left end
        ost_left = self.hulls_left[label]
        ost_right = self.hulls_right[label]
        insertion_order = 0
        num_starting_after_left = 0
        num_ending_before_left = 0

        floor = ost_left.floor_key(hull)
        if floor is not None:
            if floor.left == hull.left:
                insertion_order = floor.insertion_order + 1
            num_starting_after_left = ost_left.get_rank(floor) + 1
        hull.insertion_order = insertion_order

        floor = ost_right.floor_key(HullEnd(hull.left))
        if floor is not None:
            num_ending_before_left = ost_right.get_rank(floor) + 1
        count = num_starting_after_left - num_ending_before_left
        ost_left[hull] = count
        self.coal_mass_index[label].set_value(hull.index, count)

        # logic right end
        insertion_order = 0
        hull_end = HullEnd(hull.right)
        floor = ost_right.floor_key(hull_end)
        if floor is not None:
            if floor.x == hull.right:
                insertion_order = floor.insertion_order + 1
        hull_end.insertion_order = insertion_order
        ost_right[hull_end] = 0
        # self.num_pairs[label] += count - correction
        # Adjust counts for existing hulls in the avl tree
        coal_mass_index = self.coal_mass_index[label]
        self.increment_avl(ost_left, coal_mass_index, hull, 1)

    def add(self, individual, label=0):
        """
        Inserts the specified individual into this population.
        """
        assert individual.label == label
        self._ancestors[label].append(individual)

    def __iter__(self):
        # will default to label 0
        # iter_label() extends behavior
        return iter(self._ancestors[0])

    def iter_label(self, label):
        """
        Iterates ancestors in popn from a label
        """
        return iter(self._ancestors[label])

    def iter_ancestors(self):
        """
        Iterates over all ancestors in a population over all labels.
        """
        for ancestors in self._ancestors:
            yield from ancestors

    def find_indv(self, indv):
        """
        find the index of an ancestor in population
        """
        return self._ancestors[indv.label].index(indv)

    def get_emerged_from_lineage(self, index, label):
        """
        Returns the indices of lineages that emerged from a given one.
        """
        return [i for i, s in enumerate(self._ancestors[label]) if index.issubset(s.origin)]


class Pedigree:
    """
    Class representing a pedigree for use with the DTWF model, as implemented
    in C library
    """

    def __init__(self, tables):
        self.ploidy = 2
        self.individuals = []

        ts = tables.tree_sequence()
        for tsk_ind in ts.individuals():
            assert len(tsk_ind.nodes) == self.ploidy
            assert len(tsk_ind.parents) == self.ploidy
            time = ts.node(tsk_ind.nodes[0]).time
            # All nodes must be equivalent
            assert len({ts.node(node).flags for node in tsk_ind.nodes}) == 1
            assert len({ts.node(node).time for node in tsk_ind.nodes}) == 1
            assert len({ts.node(node).population for node in tsk_ind.nodes}) == 1
            ind = Individual(
                tsk_ind.id,
                ploidy=self.ploidy,
                nodes=list(tsk_ind.nodes),
                parents=list(tsk_ind.parents),
                time=time,
            )
            assert ind.id == len(self.individuals)
            self.individuals.append(ind)

    def print_state(self):
        print("Pedigree")
        print("-------")
        print("Individuals = ")
        for ind in self.individuals:
            print("\t", ind)
        print("-------")


class Individual:
    """
    Class representing a diploid individual in the DTWF pedigree model.
    """

    def __init__(self, id_, *, ploidy, nodes, parents, time):
        self.id = id_
        self.ploidy = ploidy
        self.nodes = nodes
        self.parents = parents
        self.time = time
        self.common_ancestors = [[] for i in range(ploidy)]

    def __str__(self):
        return (
            f"(ID: {self.id}, time: {self.time}, "
            + f"parents: {self.parents}, nodes: {self.nodes}, "
            + f"common_ancestors: {self.common_ancestors})"
        )

    def add_common_ancestor(self, head, ploid):
        """
        Adds the specified ancestor (represented by the head of a segment
        chain) to the list of ancestors that find a common ancestor in
        the specified ploid of this individual.
        """
        heapq.heappush(self.common_ancestors[ploid], (head.left, head))


class TrajectorySimulator:
    """
    Class to simulate an allele frequency trajectory on which to condition
    the coalescent simulation.
    """

    def __init__(self, initial_freq, end_freq, alpha, time_slice):
        self._initial_freq = initial_freq
        self._end_freq = end_freq
        self._alpha = alpha
        self._time_slice = time_slice
        self._reset()

    def _reset(self):
        self._allele_freqs = []
        self._times = []

    def _genic_selection_stochastic_forwards(self, dt, freq, alpha):
        ux = (alpha * freq * (1 - freq)) / np.tanh(alpha * freq)
        sign = 1 if random.random() < 0.5 else -1
        freq += (ux * dt) + sign * np.sqrt(freq * (1.0 - freq) * dt)
        return freq

    def _simulate(self):
        """
        Proposes a sweep trajectory and returns the acceptance probability.
        """
        x = self._end_freq  # backward time
        current_size = 1
        t_inc = self._time_slice
        t = 0
        while x > self._initial_freq:
            self._allele_freqs.append(max(x, self._initial_freq))
            self._times.append(t)
            # just a note below
            # current_size = self._size_calculator(t)
            #
            x = 1.0 - self._genic_selection_stochastic_forwards(
                t_inc, 1.0 - x, self._alpha * current_size
            )
            t += self._time_slice
        # will want to return current_size / N_max
        # for prototype this always equals 1
        return 1

    def run(self):
        while random.random() > self._simulate():
            self.reset()
        return self._allele_freqs, self._times


class RateMap:
    def __init__(self, positions, rates):
        """
        # For gene conversion:
        positions = [0, sequence_length]
        rates = [gene_conversion_rate, 0]
        cumulative = [0, sequence_length * gene_conversion_rate]
        """
        self.positions = positions
        self.rates = rates
        self.cumulative = RateMap.recomb_mass(positions, rates)

    @staticmethod
    def recomb_mass(positions, rates):
        recomb_mass = 0
        cumulative = [recomb_mass]
        for i in range(1, len(positions)):
            recomb_mass += (positions[i] - positions[i - 1]) * rates[i - 1]
            cumulative.append(recomb_mass)
        return cumulative

    @property
    def sequence_length(self):
        return self.positions[-1]

    @property
    def total_mass(self):
        return self.cumulative[-1]  # sequence_length * gene_conversion_rate

    @property
    def mean_rate(self):
        return self.total_mass / self.sequence_length  # gene_conversion_rate

    def mass_between(self, left, right):
        left_mass = self.position_to_mass(left)
        right_mass = self.position_to_mass(right)
        return right_mass - left_mass

    def position_to_mass(self, pos):
        if pos == self.positions[0]:
            return 0
        if pos >= self.positions[-1]:
            return self.cumulative[-1]

        index = self._search(self.positions, pos)
        assert index > 0
        index -= 1
        offset = pos - self.positions[index]
        return self.cumulative[index] + offset * self.rates[index]

    def mass_to_position(self, recomb_mass):
        if recomb_mass == 0:
            return 0
        index = self._search(self.cumulative, recomb_mass)
        assert index > 0
        index -= 1
        mass_in_interval = recomb_mass - self.cumulative[index]
        pos = self.positions[index] + (mass_in_interval / self.rates[index])
        return pos

    def shift_by_mass(self, pos, mass):
        result_mass = self.position_to_mass(pos) + mass
        return self.mass_to_position(result_mass)

    def _search(self, values, query):
        left = 0
        right = len(values) - 1
        while left < right:
            m = (left + right) // 2
            if values[m] < query:
                left = m + 1
            else:
                right = m
        return left


class OverlapCounter:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.overlaps = self._make_segment(0, seq_length, 0)

    def overlaps_at(self, pos):
        assert 0 <= pos < self.seq_length
        curr_interval = self.overlaps
        while curr_interval is not None:
            if curr_interval.left <= pos < curr_interval.right:
                return curr_interval.node
            curr_interval = curr_interval.next
        raise ValueError("Bad overlap count chain")

    def increment_interval(self, left, right):
        """
        Increment the count that spans the interval
        [left, right), creating additional intervals in overlaps
        if necessary.
        """
        curr_interval = self.overlaps
        while left < right:
            if curr_interval.left == left:
                if curr_interval.right <= right:
                    curr_interval.node += 1
                    left = curr_interval.right
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, right)
                    curr_interval.node += 1
                    break
            else:
                if curr_interval.right < left:
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, left)
                    curr_interval = curr_interval.next

    def _split(self, seg, bp):  # noqa: A002
        """
        Split the segment at breakpoint and add in another segment
        from breakpoint to seg.right. Set the original segment's
        right endpoint to breakpoint
        """
        right = self._make_segment(bp, seg.right, seg.node)
        if seg.next is not None:
            seg.next.prev = right
            right.next = seg.next
        right.prev = seg
        seg.next = right
        seg.right = bp

    def _make_segment(self, left, right, count):
        seg = Segment(0)
        seg.left = left
        seg.right = right
        seg.node = count
        return seg


class Hull:
    """
    A hull keeps track of the outermost boundaries (left, right) of
    a segment chain (lineage_head). Hulls allow us to efficiently
    keep track of overlapping lineages when simulating under the SMC_K.
    """

    def __init__(self, index):
        self.left = None
        self.right = None
        self.lineage_head = None
        self.index = index
        self.insertion_order = math.inf

    def __lt__(self, other):
        return (self.left, self.insertion_order) < (other.left, other.insertion_order)

    def __repr__(self):
        return f"l:{self.left}, r:{self.right}, io:{self.insertion_order}"

    def intersects_with(self, other):
        return self.left < other.right and other.left < self.right


class HullEnd:
    """
    Each HullEnd is associated with a single Hull and keeps track of
    Hull.right. This object is used to keep track of the order of Hulls
    based on Hull.right in a separate AVLTree when simulating the SMC_K.
    """

    def __init__(self, x):
        self.x = x
        self.insertion_order = math.inf

    def __lt__(self, other):
        return (self.x, self.insertion_order) < (other.x, other.insertion_order)

    def __repr__(self):
        return f"x:{self.x}, io:{self.insertion_order}"


class OrderStatisticsTree:
    """
    Bintrees AVL tree with added functionality to keep track of the rank
    of all nodes in the AVL tree. This is needed for the SMC_K implementation.
    The C AVL library has this functionality already baked in.
    """

    def __init__(self):
        self.avl = bintrees.AVLTree()
        self.rank = {}
        self.size = 0
        self.min = None

    def __len__(self):
        return self.size

    def __setitem__(self, key, value):
        first = True
        rank = 0
        if self.min is not None:
            if self.min < key:
                prev_key = self.avl.floor_key(key)
                rank = self.rank[prev_key]
                rank += 1
                first = False
        if first:
            self.min = key
        self.avl[key] = value
        self.rank[key] = rank
        self.size += 1
        self.update_ranks(key, rank)

    def __getitem__(self, key):
        return self.avl[key], self.rank[key]

    def get_rank(self, key):
        return self.rank[key]

    def update_ranks(self, key, rank, increment=1):
        while rank < self.size - 1:
            key = self.avl.succ_key(key)
            self.rank[key] += increment
            rank += 1

    def pop(self, key):
        if self.min == key:
            if len(self) == 1:
                self.min = None
            else:
                self.min = self.avl.succ_key(key)
        rank = self.rank.pop(key)
        self.update_ranks(key, rank, -1)
        value = self.avl.pop(key)
        self.size -= 1
        return value, rank

    def succ_key(self, key):
        rank = self.rank[key]
        if rank < self.size - 1:
            key = self.avl.succ_key(key)
            rank += 1
            return key, rank
        else:
            return None, None

    def prev_key(self, key):
        if key == self.min:
            return None, None
        else:
            key = self.avl.prev_key(key)
            rank = self.rank[key]
            return key, rank

    def floor_key(self, key):
        if len(self) == 0:
            return None
        if key < self.min:
            return None
        return self.avl.floor_key(key)

    def ceil_key(self, key):
        if len(self) == 0:
            return None
        return self.avl.ceiling_key(key)


class Simulator:
    """
    A reference implementation of the multi locus simulation algorithm.
    """

    def __init__(
        self,
        *,
        tables,
        recombination_map,
        migration_matrix,
        population_growth_rates,
        population_sizes,
        population_growth_rate_changes,
        population_size_changes,
        migration_matrix_element_changes,
        bottlenecks,
        census_times,
        model="hudson",
        max_segments=100,
        num_labels=1,
        sweep_trajectory=None,
        coalescing_segments_only=True,
        additional_nodes=None,
        time_slice=None,
        hgt_rate=0.0,
        gene_conversion_rate=0.0,
        gene_conversion_length=1,
        discrete_genome=True,
        coalescent_events_nwk=None,
        coalescent_events_ts=None,
    ):
        # Must be a square matrix.
        N = len(migration_matrix)
        assert len(tables.populations) == N
        assert len(population_growth_rates) == N
        assert len(population_sizes) == N
        for j in range(N):
            assert N == len(migration_matrix[j])
            assert migration_matrix[j][j] == 0
        assert gene_conversion_length >= 1

        self.tables = tables
        self.model = model
        self.L = tables.sequence_length
        self.recomb_map = recombination_map
        self.gc_map = RateMap([0, self.L], [gene_conversion_rate, 0])
        self.hgt_map = RateMap([0, self.L], [hgt_rate, 0])
        self.hgt_edges = []
        self.hgt_tract_length = 1
        self.bin_hgt_mask = 0b10

        self.tract_length = gene_conversion_length
        self.discrete_genome = discrete_genome
        self.migration_matrix = migration_matrix
        self.num_labels = num_labels
        self.num_populations = N
        self.max_segments = max_segments
        self.coalescing_segments_only = coalescing_segments_only
        self.additional_nodes = msprime.NodeType(additional_nodes)
        if self.additional_nodes.value > 0:
            assert not self.coalescing_segments_only
        self.pedigree = None
        self.segment_stack = []
        self.segments = [None for j in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            s = Segment(j + 1)
            self.segments[j + 1] = s
            self.segment_stack.append(s)
        self.hull_stack = []
        self.hulls = [None for _ in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            h = Hull(j + 1)
            self.hulls[j + 1] = h
            self.hull_stack.append(h)
        self.P = [Population(id_, num_labels, max_segments, model) for id_ in range(N)]

        if self.recomb_map.total_mass == 0:
            self.recomb_mass_index = None
        else:
            self.recomb_mass_index = [FenwickTree(self.max_segments) for _ in range(num_labels)]

        if self.gc_map.total_mass == 0:
            self.gc_mass_index = None
        else:
            self.gc_mass_index = [FenwickTree(self.max_segments) for _ in range(num_labels)]

        if self.hgt_map.total_mass == 0:
            self.hgt_mass_index = None
        else:
            self.hgt_mass_index = [FenwickTree(self.max_segments) for _ in range(num_labels)]

        self.S = bintrees.AVLTree()
        for pop in self.P:
            pop.set_start_size(population_sizes[pop.id])
            pop.set_growth_rate(population_growth_rates[pop.id], 0)
        self.edge_buffer = []

        self.fixed_coalescent_events = False
        self.coalescent_events = []

        if coalescent_events_nwk:
            self.fixed_coalescent_events = True
            self.parse_nwk(coalescent_events_nwk)
            self.coalescent_events.sort()
            logger.debug(self.coalescent_events)
        elif coalescent_events_ts:
            self.fixed_coalescent_events = True
            self.parse_ts(coalescent_events_ts)
            self.coalescent_events.sort()
            logger.debug(self.coalescent_events)

        # set hull_offset for smc_k, deviates from actual pattern
        # implemented using `ParametricAncestryModel()`
        self.hull_offset = None
        if self.model == "smc_k":
            self.hull_offset = 0.0

        self.initialise(tables.tree_sequence())

        self.num_ca_events = 0
        self.num_re_events = 0
        self.num_gc_events = 0
        self.num_hgt_events = 0

        # Sweep variables
        self.sweep_site = (self.L // 2) - 1  # need to add options here
        self.sweep_trajectory = sweep_trajectory
        self.time_slice = time_slice

        self.modifier_events = [(sys.float_info.max, None, None)]
        for time, pop_id, new_size in population_size_changes:
            self.modifier_events.append(
                (time, self.change_population_size, (int(pop_id), new_size))
            )
        for time, pop_id, new_rate in population_growth_rate_changes:
            self.modifier_events.append(
                (
                    time,
                    self.change_population_growth_rate,
                    (int(pop_id), new_rate, time),
                )
            )
        for time, pop_i, pop_j, new_rate in migration_matrix_element_changes:
            self.modifier_events.append(
                (
                    time,
                    self.change_migration_matrix_element,
                    (int(pop_i), int(pop_j), new_rate),
                )
            )
        for time, pop_id, intensity in bottlenecks:
            self.modifier_events.append((time, self.bottleneck_event, (int(pop_id), 0, intensity)))
        for time in census_times:
            self.modifier_events.append((time[0], self.census_event, time))
        self.modifier_events.sort()

    def initialise(self, ts):
        root_time = np.max(self.tables.nodes.time)
        self.t = root_time

        root_segments_head = [None for _ in range(ts.num_nodes)]
        root_segments_tail = [None for _ in range(ts.num_nodes)]
        last_S = -1
        for tree in ts.trees():
            left, right = tree.interval
            S = 0 if tree.num_roots == 1 else tree.num_roots
            if S != last_S:
                self.S[left] = S
                last_S = S
            # If we have 1 root this is a special case and we don't add in
            # any ancestral segments to the state.
            if tree.num_roots > 1:
                for root in tree.roots:
                    population = ts.node(root).population
                    if root_segments_head[root] is None:
                        seg = self.alloc_segment(left, right, root, population)
                        root_segments_head[root] = seg
                        root_segments_tail[root] = seg
                    else:
                        tail = root_segments_tail[root]
                        if tail.right == left:
                            tail.right = right
                        else:
                            seg = self.alloc_segment(left, right, root, population, tail)
                            tail.next = seg
                            root_segments_tail[root] = seg
        self.S[self.L] = -1

        # Insert the segment chains into the algorithm state.
        for node in range(ts.num_nodes):
            seg = root_segments_head[node]
            if seg is not None:
                seg.origin = {seg.node}
                left_end = seg.left
                pop = seg.population
                label = seg.label
                self.P[seg.population].add(seg)
                while seg is not None:
                    self.set_segment_mass(seg)
                    seg = seg.next

        if self.model == "smc_k":
            for node in range(ts.num_nodes):
                seg = root_segments_head[node]
                if seg is not None:
                    left_end = seg.left
                    pop = seg.population
                    label = seg.label
                    lineage_head = seg
                    right_end = root_segments_tail[node].right
                    new_hull = self.alloc_hull(left_end, right_end, lineage_head)
                    # insert Hull
                    floor = self.P[pop].hulls_left[label].floor_key(new_hull)
                    insertion_order = 0
                    if floor is not None:
                        if floor.left == new_hull.left:
                            insertion_order = floor.insertion_order + 1
                    new_hull.insertion_order = insertion_order
                    self.P[pop].hulls_left[label][new_hull] = -1

            # initialise the correct coalesceable pairs count
            for pop in self.P:
                for label, ost_left in enumerate(pop.hulls_left):
                    avl = ost_left.avl
                    ost_right = pop.hulls_right[label]
                    count = 0
                    for hull in avl.keys():
                        floor = ost_right.floor_key(HullEnd(hull.left))
                        num_ending_before_hull = 0
                        if floor is not None:
                            num_ending_before_hull = ost_right.rank[floor] + 1
                        num_pairs = count - num_ending_before_hull
                        avl[hull] = num_pairs
                        pop.coal_mass_index[label].set_value(hull.index, num_pairs)
                        # insert HullEnd
                        hull_end = HullEnd(hull.right)
                        floor = ost_right.floor_key(hull_end)
                        insertion_order = 0
                        if floor is not None:
                            if floor.x == hull.right:
                                insertion_order = floor.insertion_order + 1
                        hull_end.insertion_order = insertion_order
                        ost_right[hull_end] = -1
                        count += 1

    def ancestors_remain(self):
        """
        Returns True if the simulation is not finished, i.e., there is some ancestral
        material that has not fully coalesced.
        """
        return sum(pop.get_num_ancestors() for pop in self.P) != 0

    def change_population_size(self, pop_id, size):
        self.P[pop_id].set_start_size(size)

    def change_population_growth_rate(self, pop_id, rate, time):
        self.P[pop_id].set_growth_rate(rate, time)

    def change_migration_matrix_element(self, pop_i, pop_j, rate):
        self.migration_matrix[pop_i][pop_j] = rate

    def alloc_hull(self, left, right, lineage_head):
        alpha = lineage_head
        hull = self.hull_stack.pop()
        hull.left = left
        hull.right = min(right + self.hull_offset, self.L)
        while alpha.prev is not None:
            alpha = alpha.prev
        assert alpha is not None
        hull.lineage_head = alpha
        alpha.hull = hull
        return hull

    def alloc_segment(
        self,
        left,
        right,
        node,
        population,
        prev=None,
        next=None,  # noqa: A002
        label=0,
        hull=None,
        origin=None,
    ):
        """
        Pops a new segment off the stack and sets its properties.
        """
        s = self.segment_stack.pop()
        s.left = left
        s.right = right
        s.node = node
        s.population = population
        s.next = next
        s.prev = prev
        s.label = label
        s.hull = hull
        s.origin = origin if origin is not None else set()
        return s

    def copy_segment(self, segment):
        return self.alloc_segment(
            left=segment.left,
            right=segment.right,
            node=segment.node,
            population=segment.population,
            next=segment.next,
            prev=segment.prev,
            label=segment.label,
            origin=segment.origin,
        )

    def free_segment(self, u):
        """
        Frees the specified segment making it ready for reuse and
        setting its weight to zero.
        """
        if self.recomb_mass_index is not None:
            self.recomb_mass_index[u.label].set_value(u.index, 0)
        if self.gc_mass_index is not None:
            self.gc_mass_index[u.label].set_value(u.index, 0)
        if self.hgt_mass_index is not None:
            self.hgt_mass_index[u.label].set_value(u.index, 0)
        self.segment_stack.append(u)

    def free_hull(self, u):
        """
        Frees the specified hull making it ready for reuse.
        """
        u.left = None
        u.right = None
        u.lineage_head = None
        u.insertion_order = math.inf
        self.hull_stack.append(u)

    def store_node(self, population, flags=0, metadata=None):
        self.flush_edges()
        return self.tables.nodes.add_row(
            time=self.t, flags=flags, population=population, metadata=metadata
        )

    def flush_edges(self):
        """
        Flushes the edges in the edge buffer to the table, squashing any adjacent edges.
        """
        if len(self.edge_buffer) > 0:
            self.edge_buffer.sort(key=lambda e: (e.child, e.left))
            left = self.edge_buffer[0].left
            right = self.edge_buffer[0].right
            child = self.edge_buffer[0].child
            parent = self.edge_buffer[0].parent
            for e in self.edge_buffer[1:]:
                assert e.parent == parent
                if e.left != right or e.child != child:
                    self.tables.edges.add_row(left, right, parent, child)
                    left = e.left
                    child = e.child
                right = e.right
            self.tables.edges.add_row(left, right, parent, child)
            self.edge_buffer = []

    def update_node_flag(self, node_id, flag):
        node_obj = self.tables.nodes[node_id]
        node_obj = node_obj.replace(flags=node_obj.flags | flag)
        self.tables.nodes[node_id] = node_obj

    def store_edge(self, left, right, parent, child):
        """
        Stores the specified edge to the output tree sequence.
        """
        if len(self.edge_buffer) > 0:
            last_edge = self.edge_buffer[-1]
            if last_edge.parent != parent:
                self.flush_edges()

        self.edge_buffer.append(tskit.Edge(left=left, right=right, parent=parent, child=child))

    def finalise(self):
        """
        Finalises the simulation returns an msprime tree sequence object.
        """
        self.flush_edges()

        # Insert unary edges for any remainining lineages.
        current_time = self.t
        for population in self.P:
            for ancestor in population.iter_ancestors():
                node = tskit.NULL
                # See if there is already a node in this ancestor at the
                # current time
                seg = ancestor
                while seg is not None:
                    if self.tables.nodes[seg.node].time == current_time:
                        node = seg.node
                        break
                    seg = seg.next
                if node == tskit.NULL:
                    # Add a new node for the current ancestor
                    node = self.tables.nodes.add_row(
                        flags=0, time=current_time, population=population.id
                    )
                # Add in edges pointing to this ancestor
                seg = ancestor
                while seg is not None:
                    if seg.node != node:
                        self.tables.edges.add_row(seg.left, seg.right, node, seg.node)
                    seg = seg.next

        # Need to work around limitations in tskit Python API to prevent
        # individuals from getting unsorted:
        # https://github.com/tskit-dev/tskit/issues/1726
        ind_col = self.tables.nodes.individual
        ind_table = self.tables.individuals.copy()
        self.tables.sort()
        self.tables.individuals.clear()
        for ind in ind_table:
            self.tables.individuals.append(ind)
        self.tables.nodes.individual = ind_col

        return self.tables

    def simulate(self, end_time):
        self.verify()
        if self.model == "hudson":
            self.hudson_simulate(end_time)
        elif self.model == "dtwf":
            self.dtwf_simulate()
        elif self.model == "fixed_pedigree":
            self.pedigree_simulate()
        elif self.model == "single_sweep":
            self.single_sweep_simulate()
        elif self.model == "smc_k":
            self.hudson_simulate(end_time)
        else:
            print("Error: bad model specification -", self.model)
            raise ValueError
        try:
            return self.finalise()
        except Exception as e:
            self.print_state()
            print("invalid edges:")
            nodes = list(self.tables.nodes)
            for edge in self.tables.edges:
                if nodes[edge.parent].time <= nodes[edge.child].time:
                    print(edge)
            raise e

    def get_potential_destinations(self):
        """
        For each population return the set of populations for which it has a
        non-zero migration into.
        """
        N = len(self.P)
        potential_destinations = [set() for _ in range(N)]
        for j in range(N):
            for k in range(N):
                if self.migration_matrix[j][k] > 0:
                    potential_destinations[j].add(k)
        return potential_destinations

    def get_total_recombination_rate(self, label):
        total_rate = 0
        if self.recomb_mass_index is not None:
            total_rate = self.recomb_mass_index[label].get_total()
        return total_rate

    def get_total_gc_rate(self, label):
        total_rate = 0
        if self.gc_mass_index is not None:
            total_rate = self.gc_mass_index[label].get_total()
        return total_rate

    def get_total_gc_left_rate(self, label):
        gc_left_total = self.get_total_gc_left(label)
        return gc_left_total

    def get_total_gc_left(self, label):
        gc_left_total = 0
        num_ancestors = sum(pop.get_num_ancestors() for pop in self.P)
        mean_gc_rate = self.gc_map.mean_rate
        gc_left_total = num_ancestors * mean_gc_rate * self.tract_length
        return gc_left_total

    def find_cleft_individual(self, label, cleft_value):
        mean_gc_rate = self.gc_map.mean_rate
        individual_index = math.floor(cleft_value / (mean_gc_rate * self.tract_length))
        for pop in self.P:
            num_ancestors = pop.get_num_ancestors()
            if individual_index < num_ancestors:
                return pop._ancestors[label][individual_index]
            individual_index -= num_ancestors
        raise AssertionError()

    def get_total_hgt_rate(self, label):
        total_rate = 0
        if self.hgt_mass_index is not None:
            total_rate = self.hgt_mass_index[label].get_total()
        return total_rate

    def get_total_hgt_left_rate(self, label):
        hgt_left_total = self.get_total_hgt_left(label)
        return hgt_left_total

    def get_total_hgt_left(self, label):
        hgt_left_total = 0
        num_ancestors = sum(pop.get_num_ancestors() for pop in self.P)
        mean_hgt_rate = self.hgt_map.mean_rate
        hgt_left_total = num_ancestors * mean_hgt_rate * self.hgt_tract_length
        return hgt_left_total

    def find_hgt_left_individual(self, label, hgt_left_value):
        mean_hgt_rate = self.hgt_map.mean_rate
        individual_index = math.floor(hgt_left_value / (mean_hgt_rate * self.hgt_tract_length))
        for pop in self.P:
            num_ancestors = pop.get_num_ancestors()
            if individual_index < num_ancestors:
                return pop._ancestors[label][individual_index]
            individual_index -= num_ancestors
        raise AssertionError()

    #@profile
    def hudson_simulate(self, end_time):
        """
        Simulates the algorithm until all loci have coalesced.
        """
        infinity = sys.float_info.max
        non_empty_pops = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
        potential_destinations = self.get_potential_destinations()

        hgt_end_time = infinity
        if self.hgt_mass_index is not None:
            n = self.P[0].get_num_ancestors()
            hgt_end_time = 5 * (n - 1) / n

        # only worried about label 0 below
        while len(non_empty_pops) > 0:
            #self.verify()
            if self.t >= end_time:
                break
            # self.print_state()
            re_rate = self.get_total_recombination_rate(label=0)
            t_re = infinity
            if re_rate > 0:
                t_re = random.expovariate(re_rate)

            # Gene conversion can occur within segments ..
            gc_rate = self.get_total_gc_rate(label=0)
            t_gcin = infinity
            if gc_rate > 0:
                t_gcin = random.expovariate(gc_rate)
            # ... or to the left of the first segment.
            gc_left_rate = self.get_total_gc_left_rate(label=0)
            t_gc_left = infinity
            if gc_left_rate > 0:
                t_gc_left = random.expovariate(gc_left_rate)

            # Common ancestor events occur within demes.
            t_ca = infinity
            for index in non_empty_pops:
                pop = self.P[index]
                assert pop.get_num_ancestors() > 0
                t = pop.get_common_ancestor_waiting_time(self.t)
                if t < t_ca:
                    t_ca = t
                    ca_population = index

            # Horizontal Gene Transfer
            hgt_rate = self.get_total_hgt_rate(label=0)
            t_hgt = infinity
            if hgt_rate > 0 and self.t < hgt_end_time:
                t_hgt = random.expovariate(hgt_rate)

            hgt_left_rate = self.get_total_hgt_left_rate(label=0)
            t_hgt_left = infinity
            if hgt_left_rate > 0 and self.t < hgt_end_time:
                t_hgt_left = random.expovariate(hgt_left_rate)

            t_mig = infinity
            # Migration events happen at the rates in the matrix.
            for j in non_empty_pops:
                source_size = self.P[j].get_num_ancestors()
                assert source_size > 0
                # for k in range(len(self.P)):
                for k in potential_destinations[j]:
                    rate = source_size * self.migration_matrix[j][k]
                    assert rate > 0
                    t = random.expovariate(rate)
                    if t < t_mig:
                        t_mig = t
                        mig_source = j
                        mig_dest = k

            min_time = min(t_re, t_ca, t_gcin, t_gc_left, t_hgt, t_hgt_left, t_mig)
            # pop.print_state()

            # TODO ACHTUNG, fixiert auf hgt
            # min_time = t_hgt

            logger.debug(f"{min_time=}")
            logger.debug(
                "Ancestors: ",
                sorted([j.node for j in pop._ancestors[0]]),
                pop.get_num_ancestors(0),
            )

            assert min_time != infinity
            if self.t + min_time > self.modifier_events[0][0]:
                t, func, args = self.modifier_events.pop(0)
                self.t = t
                func(*args)
                # Don't bother trying to maintain the non-zero lists
                # through demographic events, just recompute them.
                non_empty_pops = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
                potential_destinations = self.get_potential_destinations()
                event = "MOD"
            else:
                self.t += min_time

                if (
                    self.fixed_coalescent_events
                    and self.coalescent_events
                    and self.coalescent_events[0][0] < self.t
                ):
                    # self.print_state()
                    # Fixed Coalescent event.
                    logger.debug("Fixed CA @", self.t, self.coalescent_events[0])
                    # print("Fixed CA @", self.t, self.coalescent_events[0])
                    event = "CA"
                    ce = self.coalescent_events.pop(0)

                    # Reset to fixed time
                    prev_time = self.t - min_time
                    # Check if the current event should have happen earlier
                    # or at the same time as the last event.
                    if ce[0] <= prev_time:
                        # Add epsilon as two events can't happen simultaneously.
                        self.t = prev_time + 0.00000001
                    else:
                        # Reset to time of ce event and add epsilon to avoid
                        # collision with leaf nodes.
                        self.t = ce[0] + 0.00000001

                    self.common_ancestor_event(
                        ca_population,
                        0,
                        lineage_a=ce[1],
                        lineage_b=ce[2],
                    )

                    if self.P[ca_population].get_num_ancestors() == 0:
                        non_empty_pops.remove(ca_population)

                elif min_time == t_re:
                    event = "RE"
                    self.hudson_recombination_event(0)
                elif min_time == t_gcin:
                    event = "GCI"
                    self.wiuf_gene_conversion_within_event(0)
                elif min_time == t_gc_left:
                    event = "GCL"
                    self.wiuf_gene_conversion_left_event(0)
                elif min_time == t_ca:
                    event = "CA"
                    self.common_ancestor_event(ca_population, 0)
                    if self.P[ca_population].get_num_ancestors() == 0:
                        non_empty_pops.remove(ca_population)
                elif min_time == t_hgt:
                    event = "HGT"
                    # print("HGT EVENT @", self.t)
                    self.hgt_branch_event(0, left_event=False)
                elif min_time == t_hgt_left:
                    event = "HGTL"
                    self.hgt_branch_event(0, left_event=True)

                else:
                    event = "MIG"
                    self.migration_event(mig_source, mig_dest)
                    if self.P[mig_source].get_num_ancestors() == 0:
                        non_empty_pops.remove(mig_source)
                    assert self.P[mig_dest].get_num_ancestors() > 0
                    non_empty_pops.add(mig_dest)
            logger.info(
                "%s time=%f n=%d",
                event,
                self.t,
                sum(pop.get_num_ancestors() for pop in self.P),
            )
            X = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
            assert non_empty_pops == X

    def single_sweep_simulate(self):
        """
        Does a structed coalescent until end_freq is reached, using
        information in self.weep_trajectory.

        """
        allele_freqs, times = self.sweep_trajectory
        sweep_traj_step = 0
        x = allele_freqs[sweep_traj_step]

        assert self.num_populations == 1

        # go through segments and assign labels
        # a bit ugly with the two loops because
        # of dealing with the pops
        indices = []
        for idx, u in enumerate(self.P[0].iter_label(0)):
            if random.random() < x:
                self.set_labels(u, 1)
                indices.append(idx)
            else:
                assert u.label == 0
        popped = 0
        for i in indices:
            tmp = self.P[0].remove(i - popped, 0)
            popped += 1
            self.P[0].add(tmp, 1)

        # main loop time
        t_inc_orig = self.time_slice
        e_time = 0.0
        while self.ancestors_remain() and sweep_traj_step < len(times) - 1:
            self.verify()
            event_prob = 1.0
            while event_prob > random.random() and sweep_traj_step < len(times) - 1:
                sweep_traj_step += 1
                x = allele_freqs[sweep_traj_step]
                e_time += times[sweep_traj_step]
                # self.t = self.t + times[sweep_traj_step]
                sweep_pop_sizes = [
                    self.P[0].get_num_ancestors(label=0),
                    self.P[0].get_num_ancestors(label=1),
                ]
                p_rec_b = self.get_total_recombination_rate(0) * t_inc_orig
                p_rec_B = self.get_total_recombination_rate(1) * t_inc_orig

                # JK NOTE: We should probably factor these pop size calculations
                # into a method in Population like get_common_ancestor_waiting_time().
                # That way we can handle exponentially growing populations as well?
                p_coal_b = (
                    (sweep_pop_sizes[0] * (sweep_pop_sizes[0] - 1))
                    / (1.0 - x)
                    * t_inc_orig
                    / self.P[0].start_size
                )
                p_coal_B = (
                    (sweep_pop_sizes[1] * (sweep_pop_sizes[1] - 1))
                    / x
                    * t_inc_orig
                    / self.P[0].start_size
                )
                sweep_pop_tot_rate = p_rec_b + p_rec_B + p_coal_b + p_coal_B

                total_rate = sweep_pop_tot_rate
                if total_rate == 0:
                    break
                event_prob *= 1.0 - total_rate

            if total_rate == 0:
                break
            if self.t + e_time > self.modifier_events[0][0]:
                t, func, args = self.modifier_events.pop(0)
                self.t = t
                func(*args)
            else:
                self.t += e_time
                # choose which event happened
                if random.random() < sweep_pop_tot_rate / total_rate:
                    # even in sweeping pop, choose which kind
                    r = random.random()
                    e_sum = p_coal_B
                    if r < e_sum / sweep_pop_tot_rate:
                        # coalescent in B
                        self.common_ancestor_event(0, 1)
                    else:
                        e_sum += p_coal_b
                        if r < e_sum / sweep_pop_tot_rate:
                            # coalescent in b
                            self.common_ancestor_event(0, 0)
                        else:
                            e_sum += p_rec_B
                            if r < e_sum / sweep_pop_tot_rate:
                                # recomb in B
                                self.hudson_recombination_event_sweep_phase(1, self.sweep_site, x)
                            else:
                                # recomb in b
                                self.hudson_recombination_event_sweep_phase(
                                    0, self.sweep_site, 1.0 - x
                                )
        # clean up the labels at end
        for idx, u in enumerate(self.P[0].iter_label(1)):
            tmp = self.P[0].remove(idx, u.label)
            self.set_labels(u, 0)
            self.P[0].add(tmp)

    def pedigree_simulate(self):
        """
        Simulates through the provided pedigree, stopping at the top.
        """
        self.pedigree = Pedigree(self.tables)
        self.dtwf_climb_pedigree()

    def dtwf_simulate(self):
        """
        Simulates the algorithm until all loci have coalesced.
        """
        while self.ancestors_remain():
            self.t += 1
            self.verify()
            self.dtwf_generation()

    def dtwf_generation(self):
        """
        Evolves one generation of a Wright Fisher population
        """
        # Migration events happen at the rates in the matrix.
        for j in range(len(self.P)):
            source_size = self.P[j].get_num_ancestors()
            for k in range(len(self.P)):
                if j == k:
                    continue
                mig_rate = source_size * self.migration_matrix[j][k]
                num_migs = min(source_size, np.random.poisson(mig_rate))
                for _ in range(num_migs):
                    mig_source = j
                    mig_dest = k
                    self.migration_event(mig_source, mig_dest)

        for pop_idx, pop in enumerate(self.P):
            # Cluster haploid inds by parent
            parent_inds = pop.get_ind_range(self.t)
            offspring = bintrees.AVLTree()
            for anc in pop.iter_label(0):
                parent_index = np.random.choice(parent_inds.stop - parent_inds.start)
                parent = parent_inds.start + parent_index
                if parent not in offspring:
                    offspring[parent] = []
                offspring[parent].append(anc)

            # Draw recombinations in children and sort segments by
            # inheritance direction
            for children in offspring.values():
                parent_nodes = [-1, -1]
                H = [[], []]
                for child in children:
                    segs_pair = self.dtwf_recombine(child, parent_nodes)
                    for seg in segs_pair:
                        if seg is not None and seg.index != child.index:
                            pop.add(seg)

                    self.verify()
                    # Collect segments inherited from the same individual
                    for i, seg in enumerate(segs_pair):
                        if seg is None:
                            continue
                        assert seg.prev is None
                        heapq.heappush(H[i], (seg.left, seg))

                # Merge segments
                for ploid, h in enumerate(H):
                    segments_to_merge = len(h)
                    if segments_to_merge == 1:
                        if self.additional_nodes.value & msprime.NODE_IS_PASS_THROUGH > 0:
                            parent_nodes[ploid] = self.store_additional_nodes_edges(
                                msprime.NODE_IS_PASS_THROUGH,
                                parent_nodes[ploid],
                                h[0][1],
                            )
                        h = []
                    elif segments_to_merge >= 2:
                        for _, individual in h:
                            pop.remove_individual(individual)
                        # parent_nodes[ploid] does not need to be updated here
                        if segments_to_merge == 2:
                            self.merge_two_ancestors(
                                pop_idx, 0, h[0][1], h[1][1], parent_nodes[ploid]
                            )
                        else:
                            self.merge_ancestors(h, pop_idx, 0, parent_nodes[ploid])  # label 0 only
            self.verify()

    def process_pedigree_common_ancestors(self, ind, ploid):
        """
        Merge the ancestral material that has been inherited on this "ploid"
        (i.e., single genome, chromosome strand, tskit node) of this
        individual, then recombine and distribute the remaining ancestral
        material among its parent ploids.
        """
        common_ancestors = ind.common_ancestors[ploid]
        if len(common_ancestors) == 0:
            # No ancestral material inherited on this ploid of this individual
            return

        # All the segment chains in common_ancestors reach a common
        # ancestor in this ploid of this individual. First we remove
        # them from the populations they are stored in:
        for _, anc in common_ancestors:
            pop = self.P[anc.population]
            pop.remove_individual(anc)

        # Merge together these lists of ancestral segments to create the
        # monoploid genome for this ploid of this individual.
        # If any coalescences occur, they use the corresponding node ID.
        # FIXME update the population/label here
        node = ind.nodes[ploid]
        genome = self.merge_ancestors(common_ancestors, 0, 0, node)
        if ind.parents[ploid] == tskit.NULL:
            # If this individual is a founder we need to make sure that all
            # lineages that are present are marked with unary nodes to show
            # where they emerged from the pedigree. These can then be
            # picked up as ancient samples by later simulations. Note that
            # pedigree simulations are a special case here in that we
            # don't want to extend unary edges to the last time point in the
            # simulation because we are *not* simulating the entire
            # population process, only the subset that we have information
            # about within the pedigree.
            seg = genome
            while seg is not None:
                if seg.node != node:
                    self.store_edge(seg.left, seg.right, parent=node, child=seg.node)
                seg = seg.next
            pop.remove_individual(genome)
        else:
            # If this individual is not a founder, it inherited the current
            # monoploid genome as a gamete from one parent. This gamete was
            # created by recombining between the parent's monoploid genomes
            # to create two independent lines of ancestry.
            parent = self.pedigree.individuals[ind.parents[ploid]]
            parent_ancestry = self.dtwf_recombine(genome, parent.nodes)
            for parent_ploid in range(ind.ploidy):
                seg = parent_ancestry[parent_ploid]
                if seg is not None:
                    # Add this segment chain of ancestry to the accumulating
                    # set in the parent on the corresponding ploid.
                    parent.add_common_ancestor(seg, ploid=parent_ploid)
                    if seg != genome:
                        # Add the recombined ancestor to the population
                        pop.add(seg)

        self.flush_edges()
        self.verify()

    def dtwf_climb_pedigree(self):
        """
        Simulates transmission of ancestral material through a pre-specified
        pedigree
        """
        assert self.num_populations == 1  # Single pop/pedigree for now
        pop = self.P[0]

        # Go through the extant lineages and gather the ancestral material
        # into the corresponding pedigree individuals.
        for anc in pop.iter_ancestors():
            node = self.tables.nodes[anc.node]
            assert node.individual != tskit.NULL
            ind = self.pedigree.individuals[node.individual]
            ind.add_common_ancestor(anc, ploid=ind.nodes.index(anc.node))

        # Visit pedigree individuals in time order.
        visit_order = sorted(self.pedigree.individuals, key=lambda x: (x.time, x.id))
        for ind in visit_order:
            self.t = ind.time
            for ploid in range(ind.ploidy):
                self.process_pedigree_common_ancestors(ind, ploid)

    def store_arg_edges(self, segment, u=-1):
        if u == -1:
            u = len(self.tables.nodes) - 1
        # Store edges pointing to current node to the left
        x = segment
        while x is not None:
            if x.node != u:
                self.store_edge(x.left, x.right, u, x.node)
            x.node = u
            x = x.prev
        # Store edges pointing to current node to the right
        x = segment
        while x is not None:
            if x.node != u:
                self.store_edge(x.left, x.right, u, x.node)
            x.node = u
            x = x.next

    def migration_event(self, j, k):
        """
        Migrates an individual from population j to population k.
        Only does label 0
        """
        label = 0
        source = self.P[j]
        dest = self.P[k]
        index = random.randint(0, source.get_num_ancestors(label) - 1)
        x = source.remove(index, label)
        hull = x.get_hull()
        assert (self.model == "smc_k") == (hull is not None)
        dest.add(x, label)
        if self.model == "smc_k":
            source.remove_hull(label, hull)
            dest.add_hull(label, hull)
        if self.additional_nodes.value & msprime.NODE_IS_MIG_EVENT > 0:
            self.store_node(k, flags=msprime.NODE_IS_MIG_EVENT)
            self.store_arg_edges(x)
        # Set the population id for each segment also.
        u = x
        while u is not None:
            u.population = k
            u = u.next

    def get_recomb_left_bound(self, seg):
        """
        Returns the left bound for genomic region over which the specified
        segment represents recombination events.
        """
        if seg.prev is None:
            left_bound = seg.left + 1 if self.discrete_genome else seg.left
        else:
            left_bound = seg.prev.right
        return left_bound

    def get_gc_left_bound(self, seg):
        # TODO remove me
        return self.get_recomb_left_bound(seg)

    def get_hgt_left_bound(self, seg):
        # TODO remove me
        return self.get_recomb_left_bound(seg)

    def set_segment_mass(self, seg):
        """
        Sets the mass for the specified segment. All links *must* be
        appropriately set before calling this function.
        """
        if self.recomb_mass_index is not None:
            mass_index = self.recomb_mass_index[seg.label]
            recomb_left_bound = self.get_recomb_left_bound(seg)
            recomb_mass = self.recomb_map.mass_between(recomb_left_bound, seg.right)
            mass_index.set_value(seg.index, recomb_mass)
        if self.gc_mass_index is not None:
            mass_index = self.gc_mass_index[seg.label]
            gc_left_bound = self.get_gc_left_bound(seg)
            gc_mass = self.gc_map.mass_between(gc_left_bound, seg.right)
            mass_index.set_value(seg.index, gc_mass)
        if self.hgt_mass_index is not None:
            mass_index = self.hgt_mass_index[seg.label]
            hgt_left_bound = self.get_hgt_left_bound(seg)
            hgt_mass = self.hgt_map.mass_between(hgt_left_bound, seg.right)
            mass_index.set_value(seg.index, hgt_mass)

    def set_labels(self, segment, new_label):
        """
        Move the specified segment to the specified label.
        """
        mass_indexes = [self.recomb_mass_index, self.gc_mass_index, self.hgt_mass_index]
        while segment is not None:
            masses = []
            for mass_index in mass_indexes:
                if mass_index is not None:
                    masses.append(mass_index[segment.label].get_value(segment.index))
                    mass_index[segment.label].set_value(segment.index, 0)
            segment.label = new_label
            for mass, mass_index in zip(masses, mass_indexes):
                if mass_index is not None:
                    mass_index[segment.label].set_value(segment.index, mass)
            segment = segment.next

    def choose_breakpoint(self, mass_index, rate_map):
        assert mass_index.get_total() > 0
        random_mass = random.uniform(0, mass_index.get_total())
        y = self.segments[mass_index.find(random_mass)]
        y_cumulative_mass = mass_index.get_cumulative_sum(y.index)
        y_right_mass = rate_map.position_to_mass(y.right)
        bp_mass = y_right_mass - (y_cumulative_mass - random_mass)
        bp = rate_map.mass_to_position(bp_mass)
        if self.discrete_genome:
            bp = math.floor(bp)
        return y, bp

    def hudson_recombination_event(self, label, return_heads=False):
        """
        Implements a recombination event.
        """
        self.num_re_events += 1
        y, bp = self.choose_breakpoint(self.recomb_mass_index[label], self.recomb_map)
        logger.debug(f"Breakpoint {y}, {bp}")
        x = y.prev
        if y.left < bp:
            #   x         y
            # =====  ===|====  ...
            #          bp
            # becomes
            #   x     y
            # =====  ===  α        (LHS)
            #           =====  ... (RHS)
            alpha = self.copy_segment(y)
            alpha.left = bp
            alpha.prev = None
            if y.next is not None:
                y.next.prev = alpha
            y.next = None
            y.right = bp
            self.set_segment_mass(y)
            lhs_tail = y
        else:
            #   x            y
            # =====  |   =========  ...
            #
            # becomes
            #   x
            # =====          α          (LHS)
            #            =========  ... (RHS)
            x.next = None
            y.prev = None
            alpha = y
            lhs_tail = x

        if self.model == "smc_k":
            # modify original hull
            pop = alpha.population
            lhs_hull = lhs_tail.get_hull()
            rhs_right = lhs_hull.right
            lhs_hull.right = min(lhs_tail.right + self.hull_offset, self.L)
            self.P[pop].reset_hull_right(label, lhs_hull, rhs_right, lhs_hull.right)

            # create hull for alpha
            alpha_hull = self.alloc_hull(alpha.left, rhs_right - self.hull_offset, alpha)
            self.P[alpha.population].add_hull(label, alpha_hull)

        self.set_segment_mass(alpha)
        self.P[alpha.population].add(alpha, label)
        if self.additional_nodes.value & msprime.NODE_IS_RE_EVENT > 0:
            self.store_node(lhs_tail.population, flags=msprime.NODE_IS_RE_EVENT)
            self.store_arg_edges(lhs_tail)
            self.store_node(alpha.population, flags=msprime.NODE_IS_RE_EVENT)
            self.store_arg_edges(alpha)
        ret = None
        if return_heads:
            x = lhs_tail
            # Seek back to the head of the x chain
            while x.prev is not None:
                x = x.prev
            ret = x, alpha
        return ret

    def generate_gc_tract_length(self):
        # generate tract length
        if self.discrete_genome:
            tl = np.random.geometric(1 / self.tract_length)
        else:
            tl = np.random.exponential(self.tract_length)
        return tl

    def wiuf_gene_conversion_within_event(self, label):
        """
        Implements a gene conversion event that starts within a segment
        """
        # we're not trying to simulate the full GC process with this
        # TODO This is more complicated than it needs to be now because
        # one event anymore. Look into what bits can be dropped now
        # that we're simulating gc_left separately again.
        y, left_breakpoint = self.choose_breakpoint(self.gc_mass_index[label], self.gc_map)
        x = y.prev
        # generate tract_length
        tl = self.generate_gc_tract_length()
        assert tl > 0
        right_breakpoint = left_breakpoint + tl
        if y.left >= right_breakpoint:
            #                  y
            # ...  |   |   ========== ...
            #     lbp rbp
            return None
        self.num_gc_events += 1
        hull = y.get_hull()
        assert (self.model == "smc_k") == (hull is not None)
        pop = y.population
        reset_right = -1
        # Process left break

        insert_alpha = True
        if left_breakpoint <= y.left:
            #  x             y
            # =====  |  ==========
            #       lbp
            #
            # becomes
            #  x
            # =====         α
            #           ==========
            if x is None:
                # In this case we *don't* insert alpha because it is already
                # the head of a segment chain
                insert_alpha = False
            else:
                x.next = None
                reset_right = x.right
            y.prev = None
            alpha = y
            tail = x
        else:
            #  x             y
            # =====     ====|=====
            #              lbp
            #
            # becomes
            #  x         y
            # =====     ====   α
            #               ======
            alpha = self.copy_segment(y)
            alpha.left = left_breakpoint
            alpha.prev = None
            if y.next is not None:
                y.next.prev = alpha
            y.next = None
            y.right = left_breakpoint
            self.set_segment_mass(y)
            tail = y
            reset_right = left_breakpoint
        self.set_segment_mass(alpha)

        # Find the segment z that the right breakpoint falls in
        z = alpha
        hull_left = z.left
        hull_right = -1
        while z is not None and right_breakpoint >= z.right:
            hull_right = z.right
            z = z.next

        # Process the right break
        head = None
        if z is not None:
            if z.left < right_breakpoint:
                #   tail             z
                # ======
                #       ...  ===|==========
                #              rbp
                #
                # becomes
                #  tail              head
                # =====         ===========
                #      ...   ===
                #             z
                head = self.copy_segment(z)
                head.left = right_breakpoint
                if z.next is not None:
                    z.next.prev = head
                z.right = right_breakpoint
                z.next = None
                self.set_segment_mass(z)
                hull_right = right_breakpoint
            else:
                #   tail             z
                # ======
                #   ...   |   =============
                #        rbp
                #
                # becomes
                #  tail             z
                # ======      =============
                #  ...
                if z.prev is not None:
                    z.prev.next = None
                head = z
            if tail is not None:
                tail.next = head
            head.prev = tail
            self.set_segment_mass(head)
        else:
            # rbp lies beyond segment chain, regular recombination logic applies
            if insert_alpha and self.model == "smc_k":
                assert reset_right >= 0
                self.P[pop].reset_hull_right(
                    label, hull, hull_right + self.hull_offset, reset_right
                )

        #        y            z
        #  |  ========== ... ===== |
        # lbp                     rbp
        # When y and z are the head and tail of the segment chains, then
        # this GC event does nothing. This logic takes are of this situation.
        new_individual_head = None
        if insert_alpha:
            new_individual_head = alpha
        elif head is not None:
            new_individual_head = head
        if new_individual_head is not None:
            if self.model == "smc_k":
                assert hull_left < hull_right
                hull = self.alloc_hull(hull_left, hull_right, new_individual_head)
                self.P[new_individual_head.population].add_hull(new_individual_head.label, hull)
            self.P[new_individual_head.population].add(
                new_individual_head, new_individual_head.label
            )

    def wiuf_gene_conversion_left_event(self, label):
        """
        Implements a gene conversion event that started left of a first segment.
        """
        random_gc_left = random.uniform(0, self.get_total_gc_left(label))
        # Get segment where gene conversion starts from left
        y = self.find_cleft_individual(label, random_gc_left)
        assert y is not None

        # generate tract_length
        tl = self.generate_gc_tract_length()

        bp = y.left + tl
        while y is not None and y.right <= bp:
            y = y.next

        # if the gene conversion spans the whole individual nothing happens
        if y is None:
            #    last segment
            # ... ==========   |
            #                  bp
            # stays in current state
            return None

        self.num_gc_events += 1
        x = y.prev
        pop = y.population
        lhs_hull = y.get_hull()
        assert (self.model == "smc_k") == (lhs_hull is not None)
        if y.left < bp:
            #  x          y
            # =====   =====|====
            #              bp
            # becomes
            #  x         y
            # =====   =====
            #              =====
            #                α
            alpha = self.copy_segment(y)
            alpha.left = bp
            alpha.prev = None
            if alpha.next is not None:
                alpha.next.prev = alpha
            y.next = None
            y.right = bp
            self.set_segment_mass(y)
            right = y.right
        else:
            #  x          y
            # ===== |  =========
            #       bp
            # becomes
            #  x
            # =====
            #          =========
            #              α
            # split the link between x and y.
            x.next = None
            y.prev = None
            alpha = y
            right = x.right

        if self.model == "smc_k":
            # lhs logic is identical to the lhs recombination event
            rhs_right = lhs_hull.right
            lhs_hull.right = right + self.hull_offset
            self.P[pop].reset_hull_right(label, lhs_hull, rhs_right, lhs_hull.right)

            # rhs
            hull = self.alloc_hull(alpha.left, rhs_right, alpha)
            self.P[pop].add_hull(label, hull)

        self.set_segment_mass(alpha)
        assert alpha.prev is None
        self.P[alpha.population].add(alpha, label)

    #@profile
    def create_hgt_segment(self, y, lbp, rbp, population_index=0):
        """
        Creates an alpha segment of length tl of segment with index i.
        Returns alpha, alpha'
        """
        pop = self.P[population_index]

        if lbp < y.left or y.right < rbp:
            raise RuntimeError(f"Breakpoints {lbp} - {rbp} out of bounds {y.left} - {y.right}")

        x = y.prev
        z = y.next
        logger.debug("Creating HGT segemnt from: ", y, "Prev: ", x, "Next: ", z)

        # Segment
        #   x                  y                z
        #  ====   ======|=============|=====   ====
        #           left_bp      right_bp

        # becomes
        #  x        y       alpha        y'      z
        #  ====   ======  ============  =====   ====
        #                 ============
        #                   alpha'

        alpha = self.copy_segment(y)
        alpha.left = lbp
        alpha.right = rbp
        alpha_qual_y = False
        if y.left == lbp and y.right == rbp:
            # Alpha is identical to y
            alpha_qual_y = True

        elif y.left == lbp:
            # Alpha at left most end of segment
            y.left = rbp
            y.prev = alpha
            alpha.prev = x
            alpha.next = y

            if x is not None:
                x.next = alpha
            else:
                # y is head of lineage
                pop.remove_individual(y)
                pop.add(alpha)

        elif y.right == rbp:
            # Alpha at right most end of segment
            y.right = lbp
            y.next = alpha

            alpha.prev = y
            alpha.next = z
            if z is not None:
                z.prev = alpha
        else:
            # Alpha in middle of segment
            y_prime = self.copy_segment(y)

            y.right = lbp
            y.next = alpha

            y_prime.left = rbp
            y_prime.prev = alpha
            y_prime.next = z
            if z is not None:
                z.prev = y_prime

            alpha.prev = y
            alpha.next = y_prime

            self.set_segment_mass(y_prime)

        # update masses
        self.set_segment_mass(y)
        self.set_segment_mass(alpha)

        if alpha_qual_y:
            alpha_prime = alpha
            alpha = y
        else:
            alpha_prime = self.copy_segment(alpha)

        alpha_prime.prev = None
        alpha_prime.next = None
        alpha_prime.origin = set()

        self.set_segment_mass(alpha_prime)
        pop.add(alpha_prime)
        return alpha, alpha_prime

    #@profile
    def hgt_branch_event(self, label, left_event):
        population_index = 0

        # Choose lineage id
        # lineage_id = random.randint(0, pop.get_num_ancestors(label) - 1)
        if left_event:
            random_hgt_left = random.uniform(0, self.get_total_hgt_left(0))
            recipient = self.find_hgt_left_individual(label, random_hgt_left)
            left_breakpoint = recipient.left
        else:
            recipient, left_breakpoint = self.choose_breakpoint(
                self.hgt_mass_index[label], self.hgt_map
            )

        # Follow the segment chain to the randomly chosen breakpoint.
        while recipient.right < left_breakpoint:
            recipient = recipient.next
            if recipient is None:
                return
        if left_event:
            left_breakpoint = recipient.left
        right_breakpoint = left_breakpoint + self.hgt_tract_length

        if not (recipient.left <= left_breakpoint and right_breakpoint <= recipient.right):
            # print("Short segment, skipping.")
            return

        self.num_hgt_events += 1

        # TODO implement real flags
        HORIZONTAL_MERGE = 1 << 23
        HORIZONTAL_PARENT = 1 << 22
        HORIZONTAL_GENE = 1 << 21

        # Add Parent Node (Responsible for recipient node)
        hm_node_id = self.store_node(population_index, flags=HORIZONTAL_MERGE)
        hg_node_id = self.store_node(population_index, flags=HORIZONTAL_GENE)
        self.t += 0.0000001

        # Add Parent Node (Responsible for HG node)
        hp_node_id = self.store_node(
            population_index,
            flags=HORIZONTAL_PARENT,
        )

        # Create alpha & alpha' segments out of the chosen segment
        alpha, alpha_prime = self.create_hgt_segment(
            recipient,
            left_breakpoint,
            right_breakpoint,
            population_index=population_index,
        )

        assert 0 < self.hgt_tract_length
        assert left_breakpoint + self.hgt_tract_length == right_breakpoint
        assert alpha.left == alpha_prime.left == left_breakpoint
        assert alpha.right == alpha_prime.right == right_breakpoint
        assert alpha_prime.origin == set()
        assert alpha_prime.prev is None
        assert alpha_prime.next is None

        # Insert alpha to AVLTree
        if left_breakpoint not in self.S:
            j = self.S.floor_key(left_breakpoint)
            self.S[left_breakpoint] = self.S[j]

        if right_breakpoint not in self.S:
            j = self.S.floor_key(right_breakpoint)
            self.S[right_breakpoint] = self.S[j]

        self.S[left_breakpoint] += 1

        """
        # Defrag AVL Tree
        j = 0
        k = 0
        while k < self.L:
            k = self.S.succ_key(j)
            if self.S[j] == self.S[k]:
                del self.S[k]
            else:
                j = k
        """

        #     HP
        #      |
        #     HG                 |                    HM                    <---- t + 0.0000001
        #      |                 |                    /\
        #     _|_                |     ______________/  \______________
        #    /   \               |    /                                \    <----- Edge from HM to Donor Node over complete span and HP to RN over alpha
        #     ===                |     RN      RN     RN     RN     RN      <----- HG or RN as segment.node
        #    alpha'
        #                        |   ======= =======  ===  ======= =======
        #                        |      x       y      a      y'      z

        # Place all edges
        # Edges for ..., x, y, alpha, y', ....
        alpha_prime.node = hg_node_id
        self.store_arg_edges(alpha_prime, hp_node_id)
        self.store_arg_edges(alpha, hm_node_id)

        # Alpha prime
        alpha_prime.node = hp_node_id
        self.hgt_edges.append(
            tskit.Edge(
                left=left_breakpoint,
                right=right_breakpoint,
                parent=hp_node_id,
                child=hm_node_id,
                metadata=self.bin_hgt_mask.to_bytes(1),  # HGT Mask
            )
        )

    def hudson_recombination_event_sweep_phase(self, label, sweep_site, pop_freq):
        """
        Implements a recombination event in during a selective sweep.
        """
        lhs, rhs = self.hudson_recombination_event(label, return_heads=True)

        r = random.random()
        if sweep_site < rhs.left:
            if r < 1.0 - pop_freq:
                # move rhs to other population
                t_idx = self.P[rhs.population].find_indv(rhs)
                self.P[rhs.population].remove(t_idx, rhs.label)
                self.set_labels(rhs, 1 - label)
                self.P[rhs.population].add(rhs, rhs.label)
        else:
            if r < 1.0 - pop_freq:
                # move lhs to other population
                t_idx = self.P[lhs.population].find_indv(lhs)
                self.P[lhs.population].remove(t_idx, lhs.label)
                self.set_labels(lhs, 1 - label)
                self.P[lhs.population].add(lhs, lhs.label)

    def dtwf_generate_breakpoint(self, start):
        left_bound = start + 1 if self.discrete_genome else start
        mass_to_next_recomb = np.random.exponential(1.0)
        bp = self.recomb_map.shift_by_mass(left_bound, mass_to_next_recomb)
        if self.discrete_genome:
            bp = math.floor(bp)
        return bp

    def dtwf_recombine(self, x, ind_nodes):
        """
        Chooses breakpoints and returns segments sorted by inheritance
        direction, by iterating through segment chain starting with x
        """
        u = self.alloc_segment(-1, -1, -1, -1, None, None)
        v = self.alloc_segment(-1, -1, -1, -1, None, None)
        seg_tails = [u, v]

        # TODO Should this be the recombination rate going foward from x.left?
        if self.recomb_map.total_mass > 0:
            k = self.dtwf_generate_breakpoint(x.left)
        else:
            k = np.inf

        ix = np.random.randint(2)
        seg_tails[ix].next = x

        while x is not None:
            seg_tails[ix] = x
            y = x.next

            if x.right > k:
                assert x.left < k
                self.num_re_events += 1
                ix = (ix + 1) % 2
                # Make new segment
                if seg_tails[ix] is u or seg_tails[ix] is v:
                    tail = None
                else:
                    tail = seg_tails[ix]
                z = self.copy_segment(x)
                z.left = k
                z.prev = tail
                self.set_segment_mass(z)
                if x.next is not None:
                    x.next.prev = z
                seg_tails[ix].next = z
                seg_tails[ix] = z
                x.next = None
                x.right = k
                self.set_segment_mass(x)
                x = z
                k = self.dtwf_generate_breakpoint(k)
            elif x.right <= k and y is not None and y.left >= k:
                # Recombine between segment and the next
                assert seg_tails[ix] == x
                x.next = None
                y.prev = None
                while y.left >= k:
                    self.num_re_events += 1
                    ix = (ix + 1) % 2
                    k = self.dtwf_generate_breakpoint(k)
                seg_tails[ix].next = y
                if seg_tails[ix] is u or seg_tails[ix] is v:
                    tail = None
                else:
                    tail = seg_tails[ix]
                y.prev = tail
                self.set_segment_mass(y)
                seg_tails[ix] = y
                x = y
            else:
                # No recombination between x.right and y.left
                x = y

        # Remove sentinel segments - this can be handled more simply
        # with pointers in C implemetation
        s = u
        u = s.next
        self.free_segment(s)

        s = v
        v = s.next
        self.free_segment(s)

        if self.additional_nodes.value & msprime.NODE_IS_RE_EVENT > 0:
            re_event = all(segment is not None for segment in [u, v])
            if re_event:
                for ploid, segment in enumerate([u, v]):
                    ind_nodes[ploid] = self.store_additional_nodes_edges(
                        msprime.NODE_IS_RE_EVENT,
                        ind_nodes[ploid],
                        segment,
                    )

        return u, v

    def census_event(self, time):
        for pop in self.P:
            for ancestor in pop.iter_ancestors():
                seg = ancestor
                u = self.tables.nodes.add_row(
                    time=time, flags=msprime.NODE_IS_CEN_EVENT, population=pop.id
                )
                while seg is not None:
                    # Add an edge joining the segment to the new node.
                    self.store_edge(seg.left, seg.right, u, seg.node)
                    seg.node = u
                    seg = seg.next

    def bottleneck_event(self, pop_id, label, intensity):
        # self.print_state()
        # Merge some of the ancestors.
        pop = self.P[pop_id]
        H = []
        for _ in range(pop.get_num_ancestors()):
            if random.random() < intensity:
                x = pop.remove(0)
                heapq.heappush(H, (x.left, x))
        self.merge_ancestors(H, pop_id, label)

    def store_additional_nodes_edges(self, flag, new_node_id, z):
        if self.additional_nodes.value & flag > 0:
            if new_node_id == -1:
                new_node_id = self.store_node(z.population, flags=flag)
            else:
                self.update_node_flag(new_node_id, flag)
            self.store_arg_edges(z, new_node_id)
        return new_node_id

    def merge_ancestors(self, H, pop_id, label, new_node_id=-1):
        pop = self.P[pop_id]
        defrag_required = False
        coalescence = False
        pass_through = len(H) == 1
        alpha = None
        z = None
        merged_head = None
        while len(H) > 0:
            alpha = None
            left = H[0][0]
            X = []
            r_max = self.L
            while len(H) > 0 and H[0][0] == left:
                x = heapq.heappop(H)[1]
                X.append(x)
                r_max = min(r_max, x.right)
            if len(H) > 0:
                r_max = min(r_max, H[0][0])
            if len(X) == 1:
                x = X[0]
                if len(H) > 0 and H[0][0] < x.right:
                    alpha = self.alloc_segment(x.left, H[0][0], x.node, x.population)
                    alpha.label = label
                    x.left = H[0][0]
                    heapq.heappush(H, (x.left, x))
                else:
                    if x.next is not None:
                        y = x.next
                        heapq.heappush(H, (y.left, y))
                    alpha = x
                    alpha.next = None
            else:
                coalescence = True
                if new_node_id == -1:
                    new_node_id = self.store_node(pop_id)
                # We must also break if the next left value is less than
                # any of the right values in the current overlap set.
                if left not in self.S:
                    j = self.S.floor_key(left)
                    self.S[left] = self.S[j]
                if r_max not in self.S:
                    j = self.S.floor_key(r_max)
                    self.S[r_max] = self.S[j]
                # Update the number of extant segments.
                if self.S[left] == len(X):
                    self.S[left] = 0
                    right = self.S.succ_key(left)
                else:
                    right = left
                    while right < r_max and self.S[right] != len(X):
                        self.S[right] -= len(X) - 1
                        right = self.S.succ_key(right)
                    alpha = self.alloc_segment(left, right, new_node_id, pop_id)
                # Update the heaps and make the record.
                for x in X:
                    if x.node != new_node_id:  # required for dtwf and fixed_pedigree
                        self.store_edge(left, right, new_node_id, x.node)
                    if x.right == right:
                        self.free_segment(x)
                        if x.next is not None:
                            y = x.next
                            heapq.heappush(H, (y.left, y))
                    elif x.right > right:
                        x.left = right
                        heapq.heappush(H, (x.left, x))

            # loop tail; update alpha and integrate it into the state.
            if alpha is not None:
                if z is None:
                    pop.add(alpha, label)
                    merged_head = alpha
                else:
                    if (coalescence and not self.coalescing_segments_only) or (
                        self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0
                    ):
                        defrag_required |= z.right == alpha.left
                    else:
                        defrag_required |= z.right == alpha.left and z.node == alpha.node
                    z.next = alpha
                alpha.prev = z
                self.set_segment_mass(alpha)
                z = alpha
        if coalescence:
            if not self.coalescing_segments_only:
                self.store_arg_edges(z, new_node_id)
        else:
            if not pass_through:
                if self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0:
                    new_node_id = self.store_additional_nodes_edges(
                        msprime.NODE_IS_CA_EVENT, new_node_id, z
                    )
            else:
                if self.additional_nodes.value & msprime.NODE_IS_PASS_THROUGH > 0:
                    assert new_node_id != -1
                    assert self.model == "fixed_pedigree"
                    new_node_id = self.store_additional_nodes_edges(
                        msprime.NODE_IS_PASS_THROUGH, new_node_id, z
                    )

        if defrag_required:
            self.defrag_segment_chain(z)
        if coalescence:
            self.defrag_breakpoints()
        return merged_head

    def defrag_segment_chain(self, z):
        y = z
        while y.prev is not None:
            x = y.prev
            if x.right == y.left and x.node == y.node:
                x.right = y.right
                x.next = y.next
                if y.next is not None:
                    y.next.prev = x
                self.set_segment_mass(x)
                self.free_segment(y)
            y = x

    def defrag_breakpoints(self):
        # Defrag the breakpoints set
        j = 0
        k = 0
        while k < self.L:
            k = self.S.succ_key(j)
            if self.S[j] == self.S[k]:
                del self.S[k]
            else:
                j = k

    def get_random_pair(self, pop, label):
        num_pairs = self.P[pop].get_num_pairs(label)
        random_mass = random.randint(1, num_pairs)
        mass_index = self.P[pop].coal_mass_index[label]

        # get first element of pair
        hull1_index = mass_index.find(random_mass)
        hull1_cumulative_mass = mass_index.get_cumulative_sum(hull1_index)
        remaining_mass = hull1_cumulative_mass - random_mass

        # get second element of pair
        avl = self.P[pop].hulls_left[label].avl
        hull1 = self.hulls[hull1_index]
        left = hull1.left
        hull2 = hull1
        while remaining_mass >= 0:
            hull2 = avl.prev_key(hull2)
            if hull2.left == left or hull2.right > left:
                remaining_mass -= 1

        return (hull1_index, hull2.index)

    def is_blocked_ancestor(self, i: int, pop, label) -> int:
        """
        Checks if the ancestor at index i is required for a future common ancestor event.

        Returns
        -------
            0: If the ancestor is not required or multiple similar ancestor exists.
            1: If blocked.
            2: If exactly two ancestor with the same origin exists.
        """
        # If no fixed events happen in the future (the list is empty):
        if not self.coalescent_events:
            return 0

        # Check if ancestor i is required as ancestor in a future event.
        _, ceA, ceB = zip(*self.coalescent_events)

        ancestor = pop._ancestors[label][i]
        # Stricter version
        # return any(ancestor.origin.issubset(j) for j in ceA + ceB)

        # Relevant Common Ancestor events
        ca_events = [j for j in ceA + ceB if ancestor.origin.issubset(j)]

        if not ca_events:
            return 0

        # Check if multiple ancestors with this origin exists
        matching_origins = 0
        for j in pop._ancestors[label]:
            # Has j same origin as selected ancestor.
            if ancestor.origin.issubset(j.origin) and any(j.origin.issubset(k) for k in ca_events):
                matching_origins += 1
                # If more than two is there -> ancestor can be chosen.
                if matching_origins == 3:
                    return 0

        return matching_origins  # 1 or 2

    def common_ancestor_event(
        self,
        population_index,
        label,
        lineage_a=None,
        lineage_b=None,
    ):
        """
        Implements a coancestry event.
        If lineage_a and lineage_b are set, only lineages emerged from those
        will be selected. Raises an error if only one lineage is provided.
        """
        pop = self.P[population_index]

        if self.model == "smc_k":
            # Choose two ancestors uniformly according to hulls_left weights
            random_pair = self.get_random_pair(population_index, label)
            hull_i_ptr, hull_j_ptr = random_pair
            hull_i = self.hulls[hull_i_ptr]
            hull_j = self.hulls[hull_j_ptr]
            x = hull_i.lineage_head
            y = hull_j.lineage_head
            pop.remove_individual(x, label)
            pop.remove_hull(label, hull_i)
            pop.remove_individual(y, label)
            pop.remove_hull(label, hull_j)
            self.free_hull(hull_i)
            self.free_hull(hull_j)

        else:
            if (lineage_a is None) ^ (lineage_b is None):  # a xor b
                raise RuntimeError(
                    "For a fixed Common Ancestor event, both lineages must be named."
                )

            if lineage_a is None and lineage_b is None:
                # Choose uniformly from all lineages
                all_lineage_ids = range(0, pop.get_num_ancestors(label) - 1)
                i, j = random.choices(all_lineage_ids, k=2)
                # random sampling without replacement

                blocked_i = self.is_blocked_ancestor(i, pop, label)
                blocked_j = self.is_blocked_ancestor(j, pop, label)
                if blocked_i or blocked_j:
                    # Check edge case where two identical lineages were chosen:
                    if blocked_i == blocked_j == 2:
                        if pop._ancestors[label][i].origin == pop._ancestors[label][j].origin:
                            # Skip as they are required.
                            return
                    else:
                        return

                x = pop.remove(i, label)
                y = pop.remove(j, label)

            else:
                # Get all lineages that emerged from lineage_a and lineage_b
                lineage_ids_a = pop.get_emerged_from_lineage(lineage_a, label)
                lineage_ids_b = pop.get_emerged_from_lineage(lineage_b, label)
                if len(lineage_ids_a) == 1 and lineage_ids_a == lineage_ids_b:
                    return

                i = None
                j = None
                while i == j:
                    i = random.choice(lineage_ids_a)
                    j = random.choice(lineage_ids_b)

                x = pop.remove(i, label)
                if i < j:
                    j -= 1
                y = pop.remove(j, label)

        self.merge_two_ancestors(population_index, label, x, y)

    def merge_two_ancestors(self, population_index, label, x, y, u=-1):
        pop = self.P[population_index]
        self.num_ca_events += 1
        z = None
        merged_head = None
        coalescence = False
        defrag_required = False
        new_origin = set()
        i = x
        j = y
        while i is not None:
            new_origin = new_origin | i.origin
            i = i.next
        while j is not None:
            new_origin = new_origin | j.origin
            j = j.next

        i = x
        j = y
        while i is not None:
            i.origin = new_origin
            i = i.next
        while j is not None:
            j.origin = new_origin
            j = j.next

        while x is not None or y is not None:
            alpha = None
            if x is None or y is None:
                if x is not None:
                    alpha = x
                    x = None
                if y is not None:
                    alpha = y
                    y = None
            else:
                if y.left < x.left:
                    beta = x
                    x = y
                    y = beta
                if x.right <= y.left:
                    alpha = x
                    x = x.next
                    alpha.next = None
                elif x.left != y.left:
                    alpha = self.copy_segment(x)
                    alpha.prev = None
                    alpha.next = None
                    alpha.right = y.left
                    x.left = y.left
                else:
                    if not coalescence:
                        coalescence = True
                        if u == -1:
                            self.store_node(population_index)
                            u = len(self.tables.nodes) - 1
                    # Put in breakpoints for the outer edges of the coalesced
                    # segment
                    left = x.left
                    r_max = min(x.right, y.right)
                    if left not in self.S:
                        j = self.S.floor_key(left)
                        self.S[left] = self.S[j]
                    if r_max not in self.S:
                        j = self.S.floor_key(r_max)
                        self.S[r_max] = self.S[j]
                    # Update the number of extant segments.
                    if self.S[left] == 2:
                        self.S[left] = 0
                        right = self.S.succ_key(left)
                    else:
                        right = left
                        while right < r_max and self.S[right] != 2:
                            self.S[right] -= 1
                            right = self.S.succ_key(right)
                        alpha = self.alloc_segment(
                            left=left,
                            right=right,
                            node=u,
                            population=population_index,
                            label=label,
                            origin=new_origin,
                        )
                    if x.node != u:  # required for dtwf and fixed_pedigree
                        self.store_edge(left, right, u, x.node)
                    if y.node != u:  # required for dtwf and fixed_pedigree
                        self.store_edge(left, right, u, y.node)
                    # Now trim the ends of x and y to the right sizes.
                    if x.right == right:
                        self.free_segment(x)
                        x = x.next
                    else:
                        x.left = right
                    if y.right == right:
                        self.free_segment(y)
                        y = y.next
                    else:
                        y.left = right

            # loop tail; update alpha and integrate it into the state.
            if alpha is not None:
                if z is None:
                    pop.add(alpha, label)
                    merged_head = alpha
                else:
                    if (coalescence and not self.coalescing_segments_only) or (
                        self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0
                    ):
                        defrag_required |= z.right == alpha.left
                    else:
                        defrag_required |= z.right == alpha.left and z.node == alpha.node
                    z.next = alpha
                alpha.prev = z
                self.set_segment_mass(alpha)
                z = alpha

        if coalescence:
            if not self.coalescing_segments_only:
                self.store_arg_edges(z, u)
        else:
            if self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0:
                self.store_additional_nodes_edges(msprime.NODE_IS_CA_EVENT, u, z)

        if defrag_required:
            self.defrag_segment_chain(z)
        if coalescence:
            self.defrag_breakpoints()

        if merged_head is not None and self.model == "smc_k":
            assert merged_head.prev is None
            hull = self.alloc_hull(merged_head.left, merged_head.right, merged_head)
            while merged_head is not None:
                right = merged_head.right
                merged_head = merged_head.next
            hull.right = min(right + self.hull_offset, self.L)
            pop.add_hull(label, hull)

    def parse_nwk(self, nwk: str) -> None:
        """
        Parses newick string.
        """
        # Remove Tail and whitespace
        nwk = nwk.strip().strip(";")
        nwk = nwk.replace(" ", "")

        # Remove Root Time
        head, tail = nwk.rsplit(":", 1)
        if ")" not in tail:
            nwk = head

        # Remove Root Label
        if not nwk.endswith(")"):
            nwk = nwk.rsplit(")", 1)[0] + ")"

        # Remove redundant brackets
        if nwk.startswith("(") and nwk.endswith(")"):
            nwk = nwk[1:-1]

        # Labels of leaf nodes
        leaf_nwk = [
            label[0] for n in nwk.replace("(", ",").split(",") if (label := n.split(":"))[0]
        ]

        # Create initial mapping for labels (str) to id (int).
        self.leaf_mapping = {label: i for i, label in enumerate(leaf_nwk)}
        self.label_mapping = {i: label for i, label in enumerate(leaf_nwk)}

        if len(self.tables.nodes) < len(leaf_nwk):
            raise ValueError(
                "Population in newick string is larger than the provided sample_size. "
                f"Please increase to {len(leaf_nwk)} or greater."
            )

        if len(leaf_nwk) < len(self.tables.nodes):
            # TODO check if warning library can be imported
            print("Population in newick string is smaller than the provided sample_size.")

        self.parse_nwk_str(nwk, 0)

    def bracket_split(self, nkw: str):
        """
        Splits a string at "," if and only if it is not enclosed by brackets.
        Input:
            nwk: str, string to split.
        Returns:
            splits: List[str], list of substrings.
        """
        splits = []
        level = 0
        next_split = []
        for c in nkw:
            if c == "," and level == 0:
                splits.append("".join(next_split))
                next_split = []
            else:
                if c == "(":
                    level += 1
                elif c == ")":
                    level -= 1
                next_split.append(c)
        splits.append("".join(next_split))
        return splits

    def parse_nwk_str(self, newick_str: str, time: float):
        """
        Parses a newick string recursivly.
        """
        # Split nwk intro left and right part.
        left_node, right_node = self.bracket_split(newick_str)

        left_time = float(left_node.split(":")[-1].split(")", 1)[0])
        right_time = float(right_node.split(":")[-1].split(")", 1)[0])

        # Clean left node and remove label
        left_node = left_node.rsplit(":", 1)[0]
        left_node = left_node[1:] if left_node.startswith("(") else left_node
        left_node = left_node[:-1] if left_node.endswith(")") else left_node

        # Clean right node and remove label
        right_node = right_node.rsplit(":", 1)[0]
        right_node = right_node[1:] if right_node.startswith("(") else right_node
        right_node = right_node[:-1] if right_node.endswith(")") else right_node

        # Check if left node is a leaf
        if "," not in left_node:
            if left_node in self.leaf_mapping:
                # Replace leaf name with mapped id
                node_id = self.leaf_mapping[left_node]
            else:
                raise RuntimeError(f"Invalid newick structure: {left_node}")
            left_node = [node_id]
        else:
            left_node, sub_left_time = self.parse_nwk_str(left_node, time + left_time)
            left_time = left_time + sub_left_time

        # Check if left node is a leaf
        if "," not in right_node:
            if right_node in self.leaf_mapping:
                # Replace leaf name with mapped id
                node_id = self.leaf_mapping[right_node]
            else:
                raise RuntimeError(f"Invalid newick structure {right_node}")

            right_node = [node_id]

        else:
            right_node, sub_right_time = self.parse_nwk_str(right_node, time + right_time)
            right_time = right_time + sub_right_time

        # Left and right time should be euqal.
        # To handle potential inconsistentcies the maximum time is used.
        ce_time = max(left_time, right_time)
        self.coalescent_events.append((ce_time, set(left_node), set(right_node)))
        node_name = left_node + right_node

        return node_name, ce_time

    def parse_ts(self, ts: tskit.TreeSequence) -> None:
        """
        Parses coalescent events from tree sequence file.
        Does only work for one welldefined tree without gene conversion or recombination.
        """
        tables = ts.dump_tables()

        edges = list(tables.edges)
        events = []
        while edges:
            e1 = edges.pop(0)
            # Sweep until sibling edge is found. Usually the next in list.
            # Can be simplified if that assumption is made.
            for i, e2 in enumerate(edges):
                if e1.parent == e2.parent:
                    e2 = edges.pop(i)
                    break
            # Get time of parent node (coalescent time)
            t = [n.time for i, n in enumerate(tables.nodes) if i == e1.parent][0]
            events.append((t, e1.child, e2.child, e1.parent))
        events.sort()

        # Add all leafs
        node_mapping = {i: {i} for i, n in enumerate(tables.nodes) if n.time == 0}
        if len(self.tables.nodes) < len(node_mapping):
            raise ValueError(
                "Population in tree sequence is larger than the provided sample_size. "
                f"Please increase to {len(node_mapping)} or greater."
            )
        if len(node_mapping) < len(self.tables.nodes):
            # TODO check if warning library can be imported
            print("Population in tree sequence is smaller than the provided sample_size.")

        # While loop is only required if events happen at
        # the same time and are not ordered correctly.
        # Can be simplified if that assumption is made.
        while events:
            t, e1, e2, p = events.pop(0)
            if e1 not in node_mapping or e2 not in node_mapping:
                # Add event to end of the queue.
                events.append(t, e1, e2, p)
                continue

            e1 = node_mapping[e1]
            e2 = node_mapping[e2]
            self.coalescent_events.append((t, e1, e2))

            if p in node_mapping:
                p = node_mapping[p]
            else:
                # Add parent node to mapping
                node_mapping[p] = e1 | e2

    def print_state(self, verify=False):
        print("State @ time ", self.t)
        for label in range(self.num_labels):
            print(
                "Recomb mass = ",
                0 if self.recomb_mass_index is None else self.recomb_mass_index[label].get_total(),
            )
            print(
                "GC mass = ",
                0 if self.gc_mass_index is None else self.gc_mass_index[label].get_total(),
            )
            print(
                "HGT mass = ",
                0 if self.hgt_mass_index is None else self.hgt_mass_index[label].get_total(),
            )
        print("Modifier events = ")
        for t, f, args in self.modifier_events:
            print("\t", t, f, args)
        print("Population sizes:", [pop.get_num_ancestors() for pop in self.P])
        print("Migration Matrix:")
        for row in self.migration_matrix:
            print("\t", row)
        for population in self.P:
            population.print_state()
        if self.pedigree is not None:
            self.pedigree.print_state()
        print("Overlap counts", len(self.S))
        for k, x in self.S.items():
            print("\t", k, "\t:\t", x)
        for label in range(self.num_labels):
            if self.recomb_mass_index is not None:
                print(
                    "recomb_mass_index [%d]: %d"
                    % (label, self.recomb_mass_index[label].get_total())
                )
                for j in range(1, self.max_segments + 1):
                    s = self.recomb_mass_index[label].get_value(j)
                    if s != 0:
                        seg = self.segments[j]
                        left_bound = self.get_recomb_left_bound(seg)
                        sp = self.recomb_map.mass_between(left_bound, seg.right)
                        print("\t", j, "->", s, sp)
            if self.gc_mass_index is not None:
                print("gc_mass_index [%d]: %d" % (label, self.gc_mass_index[label].get_total()))
                for j in range(1, self.max_segments + 1):
                    s = self.gc_mass_index[label].get_value(j)
                    if s != 0:
                        seg = self.segments[j]
                        left_bound = self.get_gc_left_bound(seg)
                        sp = self.gc_map.mass_between(left_bound, seg.right)
                        print("\t", j, "->", s, sp)
            if self.hgt_mass_index is not None:
                print("hgt_mass_index [%d]: %d" % (label, self.hgt_mass_index[label].get_total()))
                for j in range(1, self.max_segments + 1):
                    s = self.hgt_mass_index[label].get_value(j)
                    if s != 0:
                        seg = self.segments[j]
                        left_bound = self.get_hgt_left_bound(seg)
                        sp = self.hgt_map.mass_between(left_bound, seg.right)
                        print("\t", j, "->", s, sp)
        print("nodes")
        print(self.tables.nodes)
        print("edges")
        print(self.tables.edges)
        print("hgt edges")
        for e in self.hgt_edges:
            print(e)
        if verify:
            self.verify()

    def verify_segments(self):
        for pop in self.P:
            for label in range(self.num_labels):
                for head in pop.iter_label(label):
                    assert head.prev is None
                    prev = head
                    u = head.next
                    while u is not None:
                        assert prev.next is u
                        assert u.prev is prev
                        assert u.left >= prev.right
                        assert u.label == head.label
                        assert u.population == head.population
                        prev = u
                        u = u.next

    def verify_overlaps(self):
        overlap_counter = OverlapCounter(self.L)
        for pop in self.P:
            for label in range(self.num_labels):
                for u in pop.iter_label(label):
                    while u is not None:
                        overlap_counter.increment_interval(u.left, u.right)
                        u = u.next

        for pos, count in self.S.items():
            if pos != self.L:
                assert count == overlap_counter.overlaps_at(pos)

        assert self.S[self.L] == -1
        # Check the ancestry tracking.
        A = bintrees.AVLTree()
        A[0] = 0
        A[self.L] = -1
        for pop in self.P:
            for label in range(self.num_labels):
                for u in pop.iter_label(label):
                    while u is not None:
                        if u.left not in A:
                            k = A.floor_key(u.left)
                            A[u.left] = A[k]
                        if u.right not in A:
                            k = A.floor_key(u.right)
                            A[u.right] = A[k]
                        k = u.left
                        while k < u.right:
                            A[k] += 1
                            k = A.succ_key(k)
                        u = u.next
        # Now, defrag A
        j = 0
        k = 0
        while k < self.L:
            k = A.succ_key(j)
            if A[j] == A[k]:
                del A[k]
            else:
                j = k
        assert list(A.items()) == list(self.S.items())

    def verify_mass_index(self, label, mass_index, rate_map, compute_left_bound):
        assert mass_index is not None
        total_mass = 0
        alt_total_mass = 0
        for pop_index, pop in enumerate(self.P):
            for u in pop.iter_label(label):
                assert u.prev is None
                left = compute_left_bound(u)
                while u is not None:
                    assert u.population == pop_index
                    assert u.left < u.right
                    left_bound = compute_left_bound(u)
                    s = rate_map.mass_between(left_bound, u.right)
                    right = u.right
                    index_value = mass_index.get_value(u.index)
                    total_mass += index_value
                    assert math.isclose(s, index_value, abs_tol=1e-6)
                    v = u.next
                    if v is not None:
                        assert v.prev == u
                        assert u.right <= v.left
                    u = v

                s = rate_map.mass_between(left, right)
                alt_total_mass += s
        assert math.isclose(total_mass, mass_index.get_total(), abs_tol=1e-6)
        assert math.isclose(total_mass, alt_total_mass, abs_tol=1e-6)

    def verify_hulls(self):
        for pop in self.P:
            for label in range(self.num_labels):
                # num ancestors and num hulls should be identical
                num_lineages = len(pop._ancestors[label])
                assert num_lineages == len(pop.hulls_left[label])
                if num_lineages > 0:
                    assert max(pop.hulls_left[label].rank.values()) == num_lineages - 1
                    assert max(pop.hulls_right[label].rank.values()) == num_lineages - 1
                # verify counts in avl tree
                count = 0
                for a, b in itertools.combinations(pop._ancestors[label], 2):
                    # make_hulls:
                    a_hull = make_hull(a, self.L, self.hull_offset)
                    b_hull = make_hull(b, self.L, self.hull_offset)
                    count += a_hull.intersects_with(b_hull)
                avl_pairs = avl_count_pairs(pop.hulls_left[label])
                assert count == avl_pairs
                fenwick_pairs = pop.coal_mass_index[label].get_total()
                assert count == fenwick_pairs

                avl = pop.hulls_left[label].avl
                io = 0
                left = None
                for key in avl.keys():
                    if left is None:
                        left = key.left
                    else:
                        if left == key.left:
                            io += 1
                        else:
                            io = 0
                    assert io == key.insertion_order
                    left = key.left

    def verify(self):
        """
        Checks that the state of the simulator is consistent.
        """
        self.verify_segments()
        if self.model != "fixed_pedigree":
            # The fixed_pedigree model doesn't maintain a bunch of stuff.
            # It would probably be simpler if it did.
            self.verify_overlaps()
            for label in range(self.num_labels):
                if self.recomb_mass_index is None:
                    assert self.recomb_map.total_mass == 0
                else:
                    self.verify_mass_index(
                        label,
                        self.recomb_mass_index[label],
                        self.recomb_map,
                        self.get_recomb_left_bound,
                    )

                if self.gc_mass_index is None:
                    assert self.gc_map.total_mass == 0
                else:
                    self.verify_mass_index(
                        label,
                        self.gc_mass_index[label],
                        self.gc_map,
                        self.get_gc_left_bound,
                    )
                if self.hgt_mass_index is None:
                    assert self.hgt_map.total_mass == 0
                else:
                    self.verify_mass_index(
                        label,
                        self.hgt_mass_index[label],
                        self.hgt_map,
                        self.get_hgt_left_bound,
                    )
        if self.model == "smc_k":
            self.verify_hulls()


def make_hull(a, L, offset=0):
    hull = Hull(-1)
    assert a.prev is None
    b = a
    tracked_hull = a.get_hull()
    while b is not None:
        right = b.right
        b = b.next
    hull.left = a.left
    hull.right = min(right + offset, L)
    assert tracked_hull.left == hull.left
    assert tracked_hull.right == hull.right
    assert tracked_hull.lineage_head == a
    return hull


def avl_count_pairs(ost):
    return sum(value for value in ost.avl.values())

#@profile
def run_simulate(args):
    """
    Runs the simulation and outputs the results in text.
    """
    n = args.sample_size
    m = args.sequence_length
    rho = args.recombination_rate
    hgt_rate = args.hgt_rate
    gc_rate = args.gene_conversion_rate[0]
    mean_tract_length = args.gene_conversion_rate[1]
    num_populations = args.num_populations
    migration_matrix = [
        [args.migration_rate * int(j != k) for j in range(num_populations)]
        for k in range(num_populations)
    ]
    sample_configuration = [0 for j in range(num_populations)]
    population_growth_rates = [0 for j in range(num_populations)]
    population_sizes = [1 for j in range(num_populations)]
    sample_configuration[0] = n
    if args.sample_configuration is not None:
        sample_configuration = args.sample_configuration
    if args.population_growth_rates is not None:
        population_growth_rates = args.population_growth_rates
    if args.population_sizes is not None:
        population_sizes = args.population_sizes
    if args.recomb_positions is None or args.recomb_rates is None:
        recombination_map = RateMap([0, m], [rho, 0])
    else:
        positions = args.recomb_positions
        rates = args.recomb_rates
        recombination_map = RateMap(positions, rates)
    num_labels = 1

    if args.ce_from_nwk and args.ce_from_ts:
        raise RuntimeError(
            "Can't load coalescent events from newick and tree sequence simultaneously."
        )

    coalescent_events_nwk = args.ce_from_nwk
    coalescent_events_ts = args.ce_from_ts

    if coalescent_events_ts:
        coalescent_events_ts = tskit.load(coalescent_events_ts)

    sweep_trajectory = None
    if args.model == "single_sweep":
        if num_populations > 1:
            raise ValueError("Multiple populations not currently supported")
        # Compute the trajectory
        if args.trajectory is None:
            raise ValueError("Must provide trajectory (init_freq, end_freq, alpha)")
        init_freq, end_freq, alpha = args.trajectory
        traj_sim = TrajectorySimulator(init_freq, end_freq, alpha, args.time_slice)
        sweep_trajectory = traj_sim.run()
        num_labels = 2
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 1)

    if args.from_ts is None:
        tables = tskit.TableCollection(m)
        for pop_id, sample_count in enumerate(sample_configuration):
            tables.populations.add_row()
            for _ in range(sample_count):
                tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=pop_id)
    else:
        from_ts = tskit.load(args.from_ts)
        tables = from_ts.dump_tables()

    s = Simulator(
        tables=tables,
        recombination_map=recombination_map,
        migration_matrix=migration_matrix,
        model=args.model,
        population_growth_rates=population_growth_rates,
        population_sizes=population_sizes,
        population_growth_rate_changes=args.population_growth_rate_change,
        population_size_changes=args.population_size_change,
        migration_matrix_element_changes=args.migration_matrix_element_change,
        bottlenecks=args.bottleneck,
        census_times=args.census_time,
        max_segments=100000,
        num_labels=num_labels,
        coalescing_segments_only=not args.all_segments,
        additional_nodes=args.additional_nodes,
        sweep_trajectory=sweep_trajectory,
        time_slice=args.time_slice,
        hgt_rate=hgt_rate,
        gene_conversion_rate=gc_rate,
        gene_conversion_length=mean_tract_length,
        discrete_genome=args.discrete,
        coalescent_events_nwk=coalescent_events_nwk,
        coalescent_events_ts=coalescent_events_ts,
    )
    tables = s.simulate(args.end_time)

    ts = tables.tree_sequence()
    if args.output_file is not None:
        ts.dump(args.output_file)

    if args.svg:
        from IPython.display import SVG

        with open(f"{args.output_file}.svg", "w") as f:
            f.write(SVG(ts.draw_svg()).data)

    if args.d3:
        import tskit_arg_visualizer

        d3arg = tskit_arg_visualizer.D3ARG.from_ts(ts=ts)
        d3arg.draw(
            width=750,
            height=750,
            y_axis_scale="time",
            edge_type="line",
            variable_edge_width=True,
        )
    if args.verbose:
        s.print_state()

    return ts, s.hgt_edges


def add_simulator_arguments(parser):
    parser.add_argument("sample_size", type=int)
    parser.add_argument("output_file", default=None)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument(
        "-l",
        "--log-level",
        type=int,
        help="Set log-level to the specified value",
        default=logging.WARNING,
    )
    parser.add_argument("--random-seed", "-s", type=int, default=1)
    parser.add_argument("--sequence-length", "-L", type=int, default=100)
    parser.add_argument("--discrete", "-d", action="store_true")
    parser.add_argument("--num-replicates", "-R", type=int, default=1000)
    parser.add_argument("--recombination-rate", "-r", type=float, default=0.01)
    parser.add_argument("--hgt-rate", "-t", type=float, default=0)
    parser.add_argument("--recomb-positions", type=float, nargs="+", default=None)
    parser.add_argument("--recomb-rates", type=float, nargs="+", default=None)
    parser.add_argument("--gene-conversion-rate", "-c", type=float, nargs=2, default=[0, 3])
    parser.add_argument("--num-populations", "-p", type=int, default=1)
    parser.add_argument("--migration-rate", "-g", type=float, default=1)
    parser.add_argument("--sample-configuration", type=int, nargs="+", default=None)
    parser.add_argument("--population-growth-rates", type=float, nargs="+", default=None)
    parser.add_argument("--population-sizes", type=float, nargs="+", default=None)
    parser.add_argument(
        "--population-size-change", type=float, nargs=3, action="append", default=[]
    )
    parser.add_argument(
        "--population-growth-rate-change",
        type=float,
        nargs=3,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--migration-matrix-element-change",
        type=float,
        nargs=4,
        action="append",
        default=[],
    )
    parser.add_argument("--bottleneck", type=float, nargs=3, action="append", default=[])
    parser.add_argument("--census-time", type=float, nargs=1, action="append", default=[])
    parser.add_argument(
        "--ce-from-nwk",
        default="",
        help="""Specify the coalescent events as newick string.
        Example: Merge A and B at time 0.5. Merge (A,B) with C at time 0.8.
        --> "((A:0.5, B:0.5):0.3, C:0.8);"
        """,
    )
    parser.add_argument(
        "--ce-from-ts",
        default="",
        help="""Load coalescent events from a tree sequence file.""",
    )
    parser.add_argument(
        "--trajectory",
        type=float,
        nargs=3,
        default=None,
        help="Parameters for the allele frequency trajectory simulation",
    )
    parser.add_argument(
        "--all-segments",
        action="store_true",
        default=False,
        help="Only record edges along coalescing segments.",
    )
    parser.add_argument(
        "--additional-nodes",
        type=int,
        default=0,
        help="Record edges along all segments for coalescing nodes.",
    )
    parser.add_argument(
        "--time-slice",
        type=float,
        default=1e-6,
        help="The delta_t value for selective sweeps",
    )
    parser.add_argument("--model", default="hudson")
    parser.add_argument(
        "--from-ts",
        "-F",
        default=None,
        help=(
            "Specify the tree sequence to complete. The sample_size argument "
            "is ignored if this is provided"
        ),
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Create visualisation.",
    )

    parser.add_argument(
        "--d3",
        action="store_true",
        help="Create visualisation using tskit-arg-visualizer.",
    )
    parser.add_argument("--end-time", type=float, default=np.inf, help="The end for simulations.")


def main(args=None):
    parser = argparse.ArgumentParser()
    add_simulator_arguments(parser)
    args = parser.parse_args(args)
    print(args)
    log_output = daiquiri.output.Stream(
        sys.stdout,
        formatter=daiquiri.formatter.ColorFormatter(fmt="[%(levelname)s] %(message)s"),
    )
    daiquiri.setup(level=args.log_level, outputs=[log_output])
    run_simulate(args)


if __name__ == "__main__":
    main()
