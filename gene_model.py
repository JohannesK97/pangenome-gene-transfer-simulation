"""
Simulation of gene gain and loss in random or fixed phylogenetic trees,
based on the infinitely many genes model and capable of simulating horizontal gene transfer (HGT).
"""

import msprime
import tskit
import numpy as np
from typing import List, Union
from multiprocessing import Pool
import warnings
from random import randint

import hgt_simulation
import hgt_mutations
import hgt_sim_args
from gfs import gfs_from_matrix

alleles = ["absent", "present"]

@profile
def gene_model(
    theta: int,
    rho: float,
    ts: Union[tskit.TreeSequence, None] = None,
    hgt_edges: Union[List[tskit.Edge], None] = None,
    num_samples: Union[int, None] = None,
    num_sites: Union[int, None] = None,
    gene_conversion_rate: float = 0,
    recombination_rate: int = 0,
    hgt_rate: float = 0,
    ce_from_nwk: Union[str, None] = None,
    ce_from_ts: Union[tskit.TreeSequence, None] = None,
    check_double_gene_gain=True,
    double_site_relocation=False,
    relocate_double_gene_gain=False,
    ancestral_genes=False,
) -> tskit.TreeSequence:
    """
    Simulate a gene model with gain and loss mutations using msprime.

    Parameters
    ----------

    theta: int :
        Gene gain rate per time-step.
    rho: float :
        Gene loss rate per gene.

    Either provide a `TreeSequence`:
        ts: tskit.TreeSequence:
            Provide a TreeSequence to simulate gene gain and loss directly on this tree.
        hgt_edges: Union[List[tskit.Edge], None]:
            (Default value = None)
            List of HGT edges that can't be represented in the TreeSequence.
    Or simulate directly:
        num_samples: int :
            Number of samples.
        num_sites: int :
            Number of sites in the genome.
        gene_conversion_rate: float :
            (Default value = 0)
            Rate at which gene conversion events are initiated.
        recombination_rate: int :
            (Default value = 0)
            Rate at which recombination events are initiated.
        hgt_rate: float :
            (Default value = 0)
            Rate at which horizontal gene transfer events are initiated.
        ce_from_nwk: Union[str, None] :
            (Default value = None)
            Tree structure in Newick format to follow during simulation.
        ce_from_ts: Union[tskit.TreeSequence, None] :
            (Default value = None)
            Tree structure in TreeSequence format to follow during simulation.

    check_double_gene_gain: bool :
         (Default value = True).
         Checks the number of double gene gain events.
         Raises a warning if more than 1% of all events are double gene gain events.
    double_site_relocation: bool :
        (Default value = False).
        Simulates with double `num_sites` and fixes double gain mutations. Not possible when a `TreeSequence` is provided.
        Raises an error if used in combination with `ts`.
    relocate_double_gene_gain: bool :
        (Default value = False).
        Repositions double gene gain mutations. Only possible if `recombination_rate` and `gene_conversion_rate` = 0.
        Raises an error if used in combination with `recombination_rate` or `gene_conversion_rate`.
        Raises an error if the number of double gene gain events is to high.

    Returns
    -------
    mts (tskit.TreeSequence):
        The tskit.TreeSequence object resulting from the gene gain and loss events.

    Raises
    ------
    ValueError: If a `TreeSequence` and parameters for simulation were provided.
    ValueError: If rho is too small / large compared to theta, causing an invalid root probability.
    ValueError: If relocate_double_gene_gain is used with recombination_rate or gene_conversion_rate.
    RuntimeError: If the number of double gene gain events is to high to be repositioned.
    RuntimeWarning: If num_sites is too small, causing many double (present -> present) mutations.

    """
    if ts is not None and (num_samples is not None or num_sites is not None):
        raise ValueError(
            "A TreeSequence (ts) and parameters for simulation were provided. Choose either."
        )

    if relocate_double_gene_gain and double_site_relocation:
        raise ValueError(
            "Repositioning of double gene gain mutations (relocate_double_gene_gain) can't be used in combination with double_site_relocation."
        )

    if relocate_double_gene_gain and (recombination_rate or gene_conversion_rate):
        raise ValueError(
            "Repositioning of double gene gain mutations (relocate_double_gene_gain) can't be used in combination with recombination_rate or gene_conversion_rate."
        )

    if ts is not None:
        # get num_sites from ts
        num_sites = int(ts.sequence_length)

    elif num_sites is not None and num_samples is not None:
        sim_num_sites = num_sites
        if double_site_relocation:
            sim_num_sites *= 2

        if hgt_rate == 0 and ce_from_nwk is None and ce_from_ts is None:
            # Regular simulation.
            ts: tskit.TreeSequence = msprime.sim_ancestry(
                samples=num_samples,
                sequence_length=sim_num_sites,
                ploidy=1,
                recombination_rate=recombination_rate,
                gene_conversion_rate=gene_conversion_rate,
                gene_conversion_tract_length=1,  # One gene
            )
        else:
            # Simulation using custom model that supports hgt and tree fixation.
            args = hgt_sim_args.Args(
                sample_size=num_samples,
                num_sites=sim_num_sites,
                gene_conversion_rate=gene_conversion_rate,
                recombination_rate=recombination_rate,
                hgt_rate=hgt_rate,
                ce_from_ts=ce_from_ts,
                ce_from_nwk=ce_from_nwk,
                random_seed=randint(1, int(2**32 - 2)),
            )
            #theta = theta * 2
            #rho = rho * 2

            ts, hgt_edges = hgt_simulation.run_simulate(args)
            #print("HGT simulation done")
    else:
        raise ValueError("Neither a TreeSequence, nor simulation parameters were provided.")

    if theta == rho == 0:
        return ts

    theta_total_events = theta
    if rho != 0:                 # CHANGE!
        theta_total_events = 0   # CHANGE!
    rho_total_events = rho * num_sites

    root_proba = theta_total_events / (rho_total_events if rho_total_events != 0 else theta)
    if not (0 <= root_proba <= 1):
        raise ValueError(f"Invalid theta / rho resulting in a root probability of {root_proba}")

    event_rate = rho_total_events + theta_total_events
    theta_proba = theta_total_events / event_rate
    rho_proba = rho_total_events / event_rate
    event_rate /= num_sites

    gain_loss_model = msprime.MatrixMutationModel(
        alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [1 - theta_proba, theta_proba],
            [rho_proba, 1 - rho_proba],
        ],
    )

    tables = ts.dump_tables()

    # Set the ancestral state for each site.
    poisson = np.random.poisson(theta / (rho if rho != 0 else 1))
    poisson = min(poisson, num_sites)
    poisson = num_sites

    position = np.arange(0, num_sites, dtype="uint32")
    position = np.random.choice(position, poisson, replace=False)
    position.sort()

    if not ancestral_genes:
        ancestral_state = [alleles[0]] * poisson
    else:
        ancestral_state = [alleles[1]] * poisson
    ancestral_state, ancestral_state_offset = tskit.pack_strings(ancestral_state)

    tables.sites.set_columns(
        position=position,
        ancestral_state=ancestral_state,
        ancestral_state_offset=ancestral_state_offset,
    )
    ts = tables.tree_sequence()
    """
    if not hgt_edges:
        # Regular mutation simulation.
        mts = msprime.sim_mutations(
            ts,
            rate=event_rate,
            model=gain_loss_model,
        )
        print("0 hgt events observed.")
    else:
        # Custom mutation simulation that supports hgt.
    """    
    # Simulate gains first (one per site):
    if theta != 0:
        mts = hgt_mutations.sim_mutations(
            ts,
            hgt_edges=hgt_edges,
            rho=rho,
            event_rate=event_rate,
            model=gain_loss_model,
            one_mutation=True
        )

    # Add losses afterwards (random number per site):
    if rho != 0 and theta != 0:
        mts = hgt_mutations.sim_mutations(
            mts,
            hgt_edges=hgt_edges,
            rho=rho,
            event_rate=event_rate,
            model=gain_loss_model,
            one_mutation=False
        )
    elif rho != 0 and theta == 0:
        mts = hgt_mutations.sim_mutations(
            ts,
            hgt_edges=hgt_edges,
            rho=rho,
            event_rate=event_rate,
            model=gain_loss_model,
            one_mutation=False
        )

    return mts




def tables_double_site_relocation(tables: tskit.TableCollection) -> tskit.TreeSequence:

    """
    Splits the genome in half and removes double gene gain mutations on the left and single gene gain on the right half.

        Simulate a gene model with gain and loss mutations using msprime.

    Parameters
    ----------
    tables: tskit.TableCollection :
        Table representation of the tree sequence.

    Returns
    -------
    mts: tskit.TreeSequence :
        Cleaned version of the tree.
    """

    site_split = len(tables.sites) // 2

    # Get all double mutations
    derived_state, parent_state, metadata_state = _unpack_tables(tables)

    # Create mask of left and right half
    mask_left = tables.mutations.site <= site_split
    mask_right = np.logical_not(mask_left)

    # Create mask of single and double gene gain mutations that are not sentinel mutations
    mask_double = _get_double_mask(derived_state, parent_state, metadata_state)

    mask_double_left = np.logical_and(mask_left, mask_double)
    mask_single_right = np.logical_and(mask_right, np.logical_not(mask_double))

    mask_keep = np.logical_not(np.logical_or(mask_double_left, mask_single_right))

    alleles_str_len = max(len(a) for a in alleles)

    # Custom dtype to improve performance.
    mutation_dtype = np.dtype(
        [
            ("id", np.int32),
            ("site", np.int32),
            ("node", np.int32),
            ("derived_state", f"S{alleles_str_len}"),
            ("parent", np.int32),
            ("time", np.double),
            ("metadata", bytes),
        ]
    )

    # Create array of all mutations directly to avoid slow MutationTableRow Object creation
    filtered_mutations = np.zeros(sum(mask_keep), dtype=mutation_dtype)

    filtered_mutations["id"] = np.arange(0, filtered_mutations.shape[0])
    filtered_mutations["node"] = tables.mutations.node[mask_keep]
    filtered_mutations["site"] = tables.mutations.site[mask_keep]
    filtered_mutations["derived_state"] = derived_state[mask_keep]
    filtered_mutations["parent"] = tables.mutations.parent[mask_keep]
    filtered_mutations["time"] = tables.mutations.time[mask_keep]
    filtered_mutations["metadata"] = np.array(
        tskit.unpack_bytes(tables.mutations.metadata, tables.mutations.metadata_offset)
    )[mask_keep]

    # Save filtered mutations to MutationTable
    tables.mutations.clear()
    for m in filtered_mutations:
        tables.mutations.add_row(
            site=m["site"],
            node=m["node"],
            derived_state=m["derived_state"],
            parent=m["parent"],
            metadata=m["metadata"],
            time=m["time"],
        )

    site_dtype = np.dtype(
        [
            ("position", np.int32),
            ("ancestral_state", f"S{alleles_str_len}"),
            ("metadata", bytes),
        ]
    )

    # Create array of all sites directly to avoid slow SiteTableRow Object creation
    all_sites = np.zeros(len(tables.sites), dtype=site_dtype)
    all_sites["position"] = tables.sites.position
    all_sites["ancestral_state"] = tskit.unpack_strings(
        tables.sites.ancestral_state, tables.sites.ancestral_state_offset
    )
    all_sites["metadata"] = tskit.unpack_bytes(tables.sites.metadata, tables.sites.metadata_offset)

    # Set acestral state of right side to "absent"
    all_sites["ancestral_state"][site_split:] = alleles[0]

    tables.sites.clear()
    for s in all_sites:
        # Add all site to the empty tables
        tables.sites.add_row(
            position=s["position"],
            ancestral_state=s["ancestral_state"],
            metadata=s["metadata"],
        )

    tables.compute_mutation_parents()
    filtered_mts = tables.tree_sequence()
    return filtered_mts


def tables_relocate_double_gene_gain(
    tables: tskit.TableCollection,
    num_sites: int,
) -> tskit.TreeSequence:
    """
    Relocates double gene gain mutations to unused sites.
    Can't be used if gene conversion or recombination was active during the simulation.

    Parameters
    ----------
    tables: tskit.TableCollection :
        Table representation of the tree sequence.
    num_sites: int:
        Number of sites in the genome.

    Returns
    -------
    mts: tskit.TreeSequence :
        Cleaned version of the tree.
    """

    derived_state, parent_state, metadata_state = _unpack_tables(tables)

    # Create mask of single and double gene gain mutations that are not sentinel mutations
    mask_double = _get_double_mask(derived_state, parent_state, metadata_state)

    num_new_mutations = sum(mask_double)

    # Set of position that are not yet used
    unused_positions = np.setdiff1d(np.arange(0, num_sites), tables.sites.position)

    if len(unused_positions) < num_new_mutations:
        raise RuntimeError(
            "No unused sites left. Please increase num_sites in the initial simulation."
        )

    alleles_str_len = max(len(a) for a in alleles)
    mutation_dtype = np.dtype(
        [
            ("id", np.int32),
            ("site", np.int32),
            ("node", np.int32),
            ("derived_state", f"S{alleles_str_len}"),
            ("parent", np.int32),
            ("time", np.double),
            ("metadata", bytes),
        ]
    )

    # Create array of all mutations directly to avoid slow MutationTableRow Object creation
    all_mutations = np.zeros(len(tables.mutations), dtype=mutation_dtype)

    # Convert and copy data with care
    all_mutations["id"] = np.arange(0, all_mutations.shape[0])
    all_mutations["node"] = tables.mutations.node
    all_mutations["site"] = tables.mutations.site
    all_mutations["derived_state"] = tskit.unpack_strings(
        tables.mutations.derived_state, tables.mutations.derived_state_offset
    )
    all_mutations["parent"] = tables.mutations.parent
    all_mutations["time"] = tables.mutations.time
    all_mutations["metadata"] = tskit.unpack_bytes(
        tables.mutations.metadata, tables.mutations.metadata_offset
    )

    site_dtype = np.dtype(
        [
            ("id", np.int32),
            ("position", np.int32),
            ("ancestral_state", f"S{alleles_str_len}"),
            ("metadata", bytes),
        ]
    )

    # Create array of all sites directly to avoid slow SiteTableRow Object creation
    all_sites = np.zeros(len(tables.sites), dtype=site_dtype)
    all_sites["id"] = np.arange(0, all_sites.shape[0])
    all_sites["position"] = tables.sites.position
    all_sites["ancestral_state"] = tskit.unpack_strings(
        tables.sites.ancestral_state, tables.sites.ancestral_state_offset
    )
    all_sites["metadata"] = tskit.unpack_bytes(tables.sites.metadata, tables.sites.metadata_offset)

    new_positions = unused_positions[:num_new_mutations]
    new_mutation_ids = np.arange(
        all_mutations.shape[0], all_mutations.shape[0] + num_new_mutations, dtype=np.int32
    )

    new_site_ids = np.arange(
        all_sites.shape[0],
        all_sites.shape[0] + num_new_mutations,
        dtype=np.int32,
    )

    # Add sites
    new_sites = np.empty(num_new_mutations, dtype=site_dtype)
    new_sites["id"] = new_site_ids
    new_sites["position"] = new_positions
    new_sites["ancestral_state"] = (alleles[0],)

    # Copy only double mutations
    new_mutations = np.copy(all_mutations[mask_double])

    # Add mutations / site ids
    new_mutations["id"] = new_mutation_ids
    new_mutations["site"] = new_site_ids

    # Has no parent as its a new mutation
    new_mutations["parent"] = tskit.NULL

    # Potentially add metadata information that it is repositioned

    all_sites = np.concatenate((all_sites, new_sites))
    all_mutations = np.concatenate((all_mutations, new_mutations))

    # Sort sites so that they can be added to the tables
    all_sites.sort(order="position")

    tables.sites.clear()
    for s in all_sites:
        # Add all site to the empty tables
        tables.sites.add_row(
            position=s["position"],
            ancestral_state=s["ancestral_state"],
            metadata=s["metadata"],
        )

    # Store new site id after sorting
    # (as the id is just the position in the table)
    site_id_mapping = np.zeros(all_sites.shape[0])
    site_id_mapping[all_sites["id"]] = np.arange(all_sites.shape[0])  # Fill in the mappings

    # Fix/Map the site of mutation to new site as they have changed
    all_mutations["site"] = site_id_mapping[all_mutations["site"]]

    # Mutation must be sorted by site and -time
    all_mutations["time"] = all_mutations["time"] * -1
    all_mutations = np.sort(all_mutations, order=["site", "time"], kind="stable")
    all_mutations["time"] = all_mutations["time"] * -1

    # Store new mutation id after sorting
    mutation_id_mapping = np.zeros(all_mutations.shape[0] + 1)
    mutation_id_mapping[all_mutations["id"]] = np.arange(
        all_mutations.shape[0]
    )  # Fill in the mappings
    mutation_id_mapping[-1] = tskit.NULL  # -1

    # Fix/Map the parent id to new id as mutations are now sorted
    all_mutations["parent"] = mutation_id_mapping[all_mutations["parent"]]

    tables.mutations.clear()
    for m in all_mutations:
        tables.mutations.add_row(
            site=m["site"],
            node=m["node"],
            derived_state=m["derived_state"],
            parent=m["parent"],
            metadata=m["metadata"],
            time=m["time"],
        )

    # Generate tree out of tables
    new_mts = tables.tree_sequence()
    return new_mts


def _get_genotype_matrix(pool_args):
    args, kwargs = pool_args
    mts = gene_model(*args, **kwargs)
    gm = mts.genotype_matrix(alleles=tuple(alleles))
    return gm


def multi_genotype_matrices(
    n: int = 10,
    processes: int = 10,
    *args,
    **kwargs,
) -> List[np.ndarray]:
    """Run many simulations of a gene gain and loss model and return the genotype matrices.

    Parameters:
    -----------
    n: int :
        (Default value = 10)
        Number of simulations.
    processes: int :
         (Default value = 1)
         Maximum number of parallel processes.
    *args:
        Arguments passed to the `gene_model` function.
    **kwargs:
        Keyword arguments passed to the `gene_model` function.

    Returns:
    --------
    lmts: List[tskit.TreeSequence] :
        List of mutation tree sequences.

    Raises:
    -------
    See "Raises" section of `gene_model` function.
    """

    pool_args = [(args, kwargs) for _ in range(n)]

    with Pool(processes=processes) as pool:
        result = pool.imap_unordered(_get_genotype_matrix, pool_args)
        lgm = list(result)

    return lgm


def _unpack_tables(tables):
    """
    Unpacks a `TableCollection` and returns the derived and parental state of each mutation with the respective metadata.

    Parameters:
    -----------
    tables: tskit.TableCollection :
        Tables to unpack.

    Returns:
    --------
    derived_state: np.ndarray[str] :
        Derived state of each mutation.
    parent_state: np.ndarray[str] :
        Parental state of each mutation.
    metadata_state: np.ndarray[int] :
        Metadata of each mutation.

    """
    derived_state = np.array(
        tskit.unpack_strings(tables.mutations.derived_state, tables.mutations.derived_state_offset)
    )
    ancestral_state = np.array(
        tskit.unpack_strings(tables.sites.ancestral_state, tables.sites.ancestral_state_offset)
    )
    parent_state = np.where(
        tables.mutations.parent == tskit.NULL,
        ancestral_state[tables.mutations.site],
        derived_state[tables.mutations.parent],
    )
    metadata_state = np.array(
        tskit.unpack_bytes(tables.mutations.metadata, tables.mutations.metadata_offset),
    ).view("uint8")

    return derived_state, parent_state, metadata_state


def _get_double_mask(derived_state, parent_state, metadata_state):
    """
    Creates a mask of double gene gain mutations that are not sentinel mutations.

    Parameters:
    -----------
    derived_state: np.ndarray[str] :
        Derived state of each mutation.
    parent_state: np.ndarray[str] :
        Parental state of each mutation.
    metadata_state: np.ndarray[int] :
        Metadata of each mutation.

    Returns:
    --------
    mask_double: np.ndarray[bool] :
        Mask of non-sentinel double gene gain mutations.

    """
    bin_sentinel_mask = 0b01
    mask_not_sentinel = np.logical_not(np.bitwise_and(metadata_state, bin_sentinel_mask))
    mask_present = derived_state == alleles[1]
    mask_parent_present = parent_state == alleles[1]
    mask_double = np.logical_and(mask_not_sentinel, mask_present, mask_parent_present)
    return mask_double


def _get_mts(pool_args):
    args, kwargs = pool_args
    mts = gene_model(*args, **kwargs)
    return mts


def multi_mts(
    n: int = 10,
    processes: int = 10,
    *args,
    **kwargs,
) -> List[tskit.TreeSequence]:
    """Run many simulations of a gene gain and loss model and return mutation trees.

    Parameters:
    -----------
    n: int :
        (Default value = 10)
        Number of simulations.
    processes: int :
         (Default value = 1)
         Maximum number of parallel processes.
    *args:
        Arguments passed to the `gene_model` function.
    **kwargs:
        Keyword arguments passed to the `gene_model` function.

    Returns:
    --------
    lmts: List[tskit.TreeSequence] :
        List of mutation tree sequences.

    Raises:
    -------
    See "Raises" section of `gene_model` function.
    """

    pool_args = [(args, kwargs) for _ in range(n)]

    with Pool(processes=processes) as pool:
        result = pool.imap_unordered(_get_mts, pool_args)
        lmts = list(result)

    return lmts


def double_gain_probability(theta: int, rho: float, num_sites: int) -> float:
    """Calculates the probaility that at least one of the sites is hit by a double gene gain event.

    Parameters:
    -----------
    theta: int :
        Gene gain rate per time-step.
    rho: float :
        Gene loss rate per gene.
    num_sites: int :
        Number of sites in the genome.

    Returns:
    --------
    p: float :
        Probability that at least one of the sites is hit by a double gene gain event.
    """

    return 1 - (1 - (theta / (theta + (rho * num_sites))) ** 2) ** num_sites


def _get_gfs(pool_args):
    args, kwargs = pool_args
    mts = gene_model(*args, **kwargs)
    gm = mts.genotype_matrix(alleles=tuple(alleles))
    single_gfs = np.array(gfs_from_matrix(gm, kwargs["num_samples"]))
    return single_gfs


def multi_gfs(
    n: int = 10,
    processes: int = 10,
    *args,
    **kwargs,
) -> List[np.ndarray]:
    """Run many simulations of a gene gain and loss model and return only their GFS.

    Parameters:
    -----------
    n: int :
        (Default value = 10)
        Number of simulations.
    processes: int :
         (Default value = 1)
         Maximum number of parallel processes.
    *args:
        Arguments passed to the `gene_model` function.
    **kwargs:
        Keyword arguments passed to the `gene_model` function.

    Returns:
    --------
    lgfs: List[tskit.TreeSequence] :
        List of gene frequency spectra.

    Raises:
    -------
    See "Raises" section of `gene_model` function.
    """

    pool_args = [(args, kwargs) for _ in range(n)]

    with Pool(processes=processes) as pool:
        result = pool.imap_unordered(_get_gfs, pool_args)
        lgfs = list(result)

    return lgfs
