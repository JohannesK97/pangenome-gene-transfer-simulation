
import gene_model

import math
from typing import List, Union
import msprime
import tskit
import numpy as np

import h5py
from concurrent.futures import ProcessPoolExecutor
import queue
import os

import torch
from sbi.utils import BoxUniform

# Given: rho, theta, hgt_rate, core_tree

def simulator(
    theta: int = 1,
    rho: float = 0.1,
    hgt_rate: float = 0,
    core_tree: Union[str, None] = None,
    num_samples: Union[int, None] = None,
    infinite_sites_factor: int = 3,
) -> tskit.TreeSequence:

    if core_tree is not None and num_samples is not None:
        raise ValueError(
            "A core tree and parameters for simulation were provided. Choose either."
        )
        
    if core_tree is None and num_samples is None:
        raise ValueError(
            "Neither a core tree or parameters for simulation were provided. Choose either."
        )
    
    if core_tree is None:
        core_tree = msprime.sim_ancestry(
                samples=num_samples,
                sequence_length=1,
                ploidy=1,
                recombination_rate=0,
                gene_conversion_rate=0,
                gene_conversion_tract_length=1,  # One gene
            )
    
    ce_from_nwk = core_tree.first().newick()
    
    # Calculate the number of genes in the root of the core tree:
    
    expected_number_of_genes_in_core_root = 1
    for k in range(1, 100):
        term = (hgt_rate ** k) / math.prod(1 + rho + i for i in range(k))
        expected_number_of_genes_in_core_root += term
    expected_number_of_genes_in_core_root = expected_number_of_genes_in_core_root * theta / rho
    
    number_of_genes_in_core_root = np.random.poisson(lam=expected_number_of_genes_in_core_root)
    
    if number_of_genes_in_core_root > 0:
        mts_core_root = gene_model.gene_model(
            theta=0,
            rho=rho,
            gene_conversion_rate=0,
            recombination_rate=0,
            hgt_rate=hgt_rate,
            num_samples=num_samples,
            num_sites=number_of_genes_in_core_root,
            double_site_relocation=False,
            core_genes=True,
        )
    
    #  Calculate the tree and the number of genes which were observed:
    
    num_sites = round(infinite_sites_factor * expected_number_of_genes_in_core_root)
    
    mts = gene_model.gene_model(
        theta=theta,
        rho=rho,
        gene_conversion_rate=0,
        recombination_rate=0,
        hgt_rate=hgt_rate,
        num_samples=num_samples,
        num_sites=num_sites,
        double_site_relocation=False,
        core_genes=False,
    )
    
    # Calculate the lenghts of the tree at each site:
    
    tree_lengths = np.array([mts.at(i).total_branch_length for i in range(0,num_sites)])
    tree_lengths_average = sum(tree_lengths)/len(tree_lengths)
    
    # Calculate the number of gene gains:
    number_of_gains = np.random.poisson(lam=tree_lengths_average*theta)
    
    while (number_of_gains > num_sites):
        number_of_gains = np.random.poisson(lam=tree_lengths_average*theta)
    
    # Make sure all lengths are positive
    if np.any(tree_lengths <= 0):
        raise ValueError("Tree with length 0 found.")
    
    # Choose number_of_gains sites:
    
    selected_trees_indexes = []
    k = 0 # This index describes how often a site has gained multiple mutations
    
    while (len(selected_trees_indexes) < number_of_gains):
    
        if k == 0:
            probabilities = tree_lengths / sum(tree_lengths)
        else: 
            probabilities = np.ones(num_sites)
            for i in range(0,k+1):
                probabilities = probabilities - math.comb(number_of_gains, i) * (1-tree_lengths / sum(tree_lengths)) ** (number_of_gains-i) * (tree_lengths / sum(tree_lengths)) ** i
            probabilities[selected_trees_indexes] = 0
            probabilities = probabilities / sum(probabilities)
    
        selected_trees = np.random.choice(np.array(range(0, num_sites)), size=number_of_gains - len(selected_trees_indexes), replace=True, p=probabilities)
        selected_trees_indexes.extend(np.unique(selected_trees))
        k = k + 1
    
    gene_absence_presence_matrix = []
    
    # Add the genes present in the root of the core tree:
    if number_of_genes_in_core_root > 0:
        for var in mts_core_root.variants():
            if not var.alleles == ('present',): # At least one loss in the leaves
                gene_absence_presence_matrix.append([1 if x == 0 else 0 for x in var.genotypes]) # 0 for absence and 1 for presence have to be swapped.
            else: # No losses in the leaves
                gene_absence_presence_matrix.append(np.ones(num_samples))
    
    for var in mts.variants():
        if (var.site.position in selected_trees_indexes):
            gene_absence_presence_matrix.append(var.genotypes)
    gene_absence_presence_matrix = np.array(gene_absence_presence_matrix)

    print(gene_absence_presence_matrix)

    return gene_absence_presence_matrix


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
                array_text = array_text.replace(' ', ',').replace(',,', ',')  # Leerzeichen durch Kommas ersetzen
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

def simulator_SBI(theta, rho, num_samples, hgt_rate):    
    return simulator(theta=theta, rho=rho, num_samples=num_samples, hgt_rate=hgt_rate)


def simulate_and_store(theta, rho, num_samples, hgt_rate, output_file):
    
    result = simulator_SBI(theta, rho, num_samples, hgt_rate)
    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
    with open(output_file, 'a') as f:
        f.write(f"hgt_rate {hgt_rate}: {result}\n")

def run_simulation(num_simulations, output_file, theta, rho, hgt_rates, num_samples):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx in range(num_simulations):
            hgt_rate = hgt_rates[idx].item()
            # futures.append(executor.submit(simulate_and_store, theta, rho, num_samples, hgt_rate, output_file, idx))
            futures.append(executor.submit(simulate_and_store, theta, rho, num_samples, hgt_rate, output_file))

        # Warten auf alle Futures
        for future in futures:
            future.result()  # blockiert bis die Aufgabe abgeschlossen ist

if __name__ == '__main__':
    
    num_simulations = 500
    
    hgt_rate_max = 0.1 # Maximum hgt rate
    hgt_rate_min = 0 # Minimum hgt rate
    
    theta = 100
    rho = 0.1
    num_samples = 20
    
    prior = BoxUniform(low=hgt_rate_min * torch.ones(1), high=hgt_rate_max * torch.ones(1))

    hgt_rates = prior.sample((num_simulations,))
    
    output_file = 'simulation_results.txt'

    if os.path.exists(output_file):
        os.remove(output_file)
    
    run_simulation(num_simulations, output_file, theta, rho, hgt_rates, num_samples)



