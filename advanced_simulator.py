import msprime
import tskit
import hgt_simulation
import hgt_sim_args
import numpy as np
import random
import torch
import os
import time

from collections import namedtuple
from collections import defaultdict
from collections import deque
from typing import List, Union
from sbi.utils import BoxUniform

from concurrent.futures import ProcessPoolExecutor

@profile
def simulator(
    theta: int = 1,
    rho: float = 0.3,
    hgt_rate: float = 0,
    num_sites: int = 100,
    ce_from_nwk: Union[str, None] = None,
    num_samples: Union[int, None] = None,
    infinite_sites_factor: int = 3,
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
            )
    
        ce_from_nwk = core_tree.first().newick()


    ### Calculate hgt events:

    args = hgt_sim_args.Args(
        sample_size=num_samples,
        num_sites=num_sites,
        gene_conversion_rate=0,
        recombination_rate=0,
        hgt_rate=hgt_rate,
        ce_from_ts=None,
        ce_from_nwk=ce_from_nwk,
        random_seed=1
    )

    ts, hgt_edges = hgt_simulation.run_simulate(args)


    ### Place mutations

    alleles = ["absent", "present"]
    
    # Place one mutation per site, e.g. genome position
    
    tables = ts.dump_tables()
    
    Mutation = namedtuple("Mutation", ["node", "time"])
    
    present_mutations = {}
    
    site = 0
    for tree in ts.trees():
        
        nodes = [n for n in tree.nodes()]
        probabilities = np.array([tree.branch_length(n) for n in nodes]) / tree.total_branch_length
        
        for i in range(int(tree.span)):
    
            mutation_node = np.random.choice(nodes, size=1, p=probabilities)[0]
    
            mutation_time = random.uniform(tree.time(mutation_node), tree.time(tree.parent(mutation_node)))
            
            tables.mutations.add_row(
                site = site,
                node = mutation_node,
                derived_state = "present",
                #parent=parent_id,
                #metadata=mutation.metadata,
                time=mutation_time,
            )
            
            tables.sites.add_row(position=site, ancestral_state="absent")
    
            present_mutations[site] = Mutation(node=mutation_node, time=mutation_time)
    
            # Add sentinel mutations at the leafs:
    
            for leaf_position in range(num_samples):
                tables.mutations.add_row(
                    site = site,
                    node = leaf_position,
                    derived_state = "absent",
                    time = 0.00000000001,
                )
        
            site += 1
    
    ts_gains = tables.tree_sequence()
    
    # Place losses:
    rho = 0.3
    
    losses_model = msprime.MatrixMutationModel(
        alleles = alleles,
        root_distribution=[1, 0],
        transition_matrix=[
            [1, 0],
            [1, 0],
        ],
    )
    
    ts_gains_losses = msprime.sim_mutations(ts_gains, rate=rho, model = losses_model, keep = True)


    ### Calculate the gene absence presence matrix:

    MutationRecord = namedtuple('MutationRecord', ['site_id', 'mutation_id', 'node', 'is_hgt'])
    
    tables = ts_gains_losses.dump_tables()
    
    hgt_parent_nodes = [edge.parent-1 for edge in hgt_edges]
    hgt_parent_children = defaultdict(list)
        
    for parent in hgt_parent_nodes:
        hgt_parent_children[parent].append(parent-1)
            
    for tree in ts_gains_losses.trees():
        
        for site in tree.sites():
            mutations = site.mutations
            present_mutation = [m for m in mutations if m.derived_state == "present"][0]
            absent_mutations = [m for m in mutations if m.derived_state == "absent"]
            absent_mutation_nodes = {m.node for m in mutations if m.derived_state == "absent"}
    
            nodes_to_process = deque([(present_mutation.node, False, False)])
            # The second variable describes if a hgt edge has passed in the whole way down to the actual node. 
            # The third describes if a hgt edge was passed in the last step.
    
            child_mutations = []
            
            #visited = set()  # Prevents duplicate processing of nodes.
    
            if present_mutation.node < num_samples: # Gain directly above leaf:
                if present_mutation.id == sorted([mut for mut in mutations if mut.node == present_mutation.node], key=lambda m: m.time)[1].id:
                    sentinel_mutation = min([mut for mut in absent_mutations if mut.node == present_mutation.node], key=lambda m: m.time)
                    child_mutations.append(MutationRecord(
                        site_id=site.id,
                        mutation_id=sentinel_mutation.id,
                        node=sentinel_mutation.node,
                        is_hgt=False
                    ))
    
            else:
                while nodes_to_process:
                    #print(nodes_to_process)
                    child_node = nodes_to_process.popleft()

                    """
                    # Process the node only if it has not been visited yet.
                    if child_node in visited:
                        continue
                    visited.add(child_node)
                    """
        
                    # If there is a mutation on the edge, find the earliest one.
                    if not child_node[2] and child_node[0] in absent_mutation_nodes:
                        
                        absent_mutation_after_gain_at_node = [mut for mut in absent_mutations if mut.node == child_node[0] and mut.time < present_mutation.time]
                        
                        if absent_mutation_after_gain_at_node: # not empty
                            earliest_mutation = max(
                                [mut for mut in absent_mutations if mut.node == child_node[0] and mut.time < present_mutation.time], 
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
                        else:
                            for child in tree.children(child_node[0]):
                                nodes_to_process.extend([(child, child_node[1], False)])
                            if hgt_parent_children[child_node[0]]:
                                nodes_to_process.extend([(hgt_parent_children[child_node[0]][0], True, True)])  
                            
            
                    # If there is no mutation, add child nodes.
                    else:
                        for child in tree.children(child_node[0]):
                            nodes_to_process.extend([(child, child_node[1], False)])
                        if hgt_parent_children[child_node[0]]:
                            nodes_to_process.extend([(hgt_parent_children[child_node[0]][0], True, True)])
    
            child_mutations.sort(key=lambda mut: not mut.is_hgt) # Will set is_hgt to False later if there are paths without hgt events to the leaf.
    
            for mutation in child_mutations:
    
                tables.mutations[mutation.mutation_id] = tables.mutations[mutation.mutation_id].replace(derived_state = "present")
                tables.mutations[mutation.mutation_id] = tables.mutations[mutation.mutation_id].replace(metadata = bytes([mutation.is_hgt]))
    
        
    mts = tables.tree_sequence()


    ### Print the computation time.
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Success: hgt_rate = {hgt_rate}, Total computation time = {elapsed_time:.6f} seconds.")

    return mts



def read_simulation_results(file_path):
    hgt_rates = []
    results = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("hgt_rate"):
            try:
                # Extrahiere hgt_rate
                hgt_rate, result_start = line.split(":")
                hgt_rate = float(hgt_rate.split()[1])

                # Ergebnisse sammeln (mehrzeilige Arrays)
                array_lines = [result_start.strip()]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("hgt_rate"):
                    array_lines.append(lines[i].strip())
                    i += 1
                
                # Kombiniere und formatiere die Array-Zeilen
                array_text = " ".join(array_lines)
                #array_text = array_text.replace(' ', ',').replace(',,', ',')  # Leerzeichen durch Kommas ersetzen
                array_text = re.sub(r'[ ,]+', ',', array_text.strip())
                array_text = array_text.replace("[,", "[").replace(",]", "]")  # Korrekt formatieren
                
                # Konvertiere in ein Numpy-Array
                result = np.array(eval(array_text))
                
                # Ergebnisse speichern
                hgt_rates.append(hgt_rate)
                results.append(result)
            except Exception as e:
                print(f"Fehler beim Verarbeiten von hgt_rate {line}:\n{e}")
        else:
            i += 1

    return hgt_rates, results


@profile
def simulate_and_store(theta, rho, num_samples, num_sites, hgt_rate, ce_from_nwk, output_file):
    
    mts = simulator(theta = theta, rho = rho, num_samples = num_samples, num_sites = num_sites, hgt_rate = hgt_rate, ce_from_nwk = ce_from_nwk)

    gene_absence_presence_matrix = []

    for var in mts.variants():
        gene_absence_presence_matrix.append(var.genotypes)
    gene_absence_presence_matrix = np.array(gene_absence_presence_matrix)
    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
    with open(output_file, 'a') as f:
        f.write(f"hgt_rate {hgt_rate}: {gene_absence_presence_matrix}\n")

@profile
def run_simulation(num_simulations, output_file, theta, rho, hgt_rates, num_samples, num_sites):

    core_tree = msprime.sim_ancestry(
            samples=num_samples,
            sequence_length=1,
            ploidy=1,
            recombination_rate=0,
            gene_conversion_rate=0,
            gene_conversion_tract_length=1,  # One gene
        )

    ce_from_nwk = core_tree.first().newick()

    """
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx in range(num_simulations):
            hgt_rate = hgt_rates[idx].item()
            futures.append(executor.submit(simulate_and_store, theta, rho, num_samples, hgt_rate, ce_from_nwk, output_file))

        # Warten auf alle Futures
        for future in futures:
            future.result()  # blockiert bis die Aufgabe abgeschlossen ist
    """
    hgt_rate = 0.5
    simulate_and_store(theta, rho, num_samples, num_sites, hgt_rate, ce_from_nwk, output_file)

if __name__ == '__main__':
    
    num_simulations = 1
    
    hgt_rate_max = 1 # Maximum hgt rate
    hgt_rate_min = 0 # Minimum hgt rate
    
    theta = 1000
    rho = 0.3
    num_samples = 20
    num_sites = 2000

    prior = BoxUniform(low=hgt_rate_min * torch.ones(1), high=hgt_rate_max * torch.ones(1))
    
    hgt_rates = prior.sample((num_simulations,))
    #hgt_rates = torch.linspace(hgt_rate_min, hgt_rate_max, num_simulations)
    
    output_file = 'simulation_results.txt'

    if os.path.exists(output_file):
        os.remove(output_file)

    run_simulation(num_simulations, output_file, theta, rho, hgt_rates, num_samples, num_sites)

