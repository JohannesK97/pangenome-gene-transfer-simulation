
# Packages

import torch
import numpy as np
import multiprocessing

import gene_model
import gfs
import color_scheme

from concurrent.futures import ProcessPoolExecutor, as_completed

# Simulation Parameters
theta = 5 # Gene Gain rate
rho = 0.2 # Gene Loss rate

gene_conv = 0 # Gene Conversion rate
recomb = 0 # Recombination rate
hgt_rate_max = 0.2 # Maximum hgt rate
hgt_rate_min = 0 # Minimum hgt rate

num_sites = 500 # Number of sites to simulate
num_samples = 5 # Number of samples / individuals


def simulation(rate): # rate is one single hgt rate.

    matrix = np.zeros((num_sites*2, num_samples))
    gfs_matrix = np.zeros((num_samples+1))

    mts = gene_model.gene_model(
        theta=theta,
        rho=rho,
        gene_conversion_rate=gene_conv,
        recombination_rate=recomb,
        hgt_rate=rate.item(),
        num_samples=num_samples,
        num_sites=num_sites,
        double_site_relocation=True, # Fix double gene gain events, won't hide the warning.
    )

    for var in mts.variants():
        if 'present' in var.alleles and 'absent' in var.alleles: # Gene present and absent in different samples
            matrix[int(var.site.position),:] = var.genotypes
        elif 'present' in var.alleles: # Gene present in all samples
            matrix[int(var.site.position),:] = np.ones(num_samples)

    for var in mts.variants():
        gfs_matrix = mts.allele_frequency_spectrum(span_normalise = False, polarised = True)

    print(gfs_matrix)
    
    return torch.from_numpy(gfs_matrix)
    # return torch.from_numpy(matrix)


def simulator(hgt_rate): # hgt_rate is a vector of hgt rates.

    if not isinstance(hgt_rate, torch.Tensor):
        raise ValueError("hgt_rate must be a tensor.")

    num_samples = 10 # Number of samples / individuals
    gfs_matrix = np.zeros((len(hgt_rate), num_samples+1))

    #with ProcessPoolExecutor(max_workers=12) as executor:
    #    futures = [executor.submit(simulation, rate) for rate in hgt_rate.numpy()]
    #    results = [future.result() for future in as_completed(futures)]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulation, rate) for rate in hgt_rate.numpy()]
        results = [future.result() for future in as_completed(futures)]
        #for future in futures:
        #    rate = futures[future]
        #    try:
        #        result = future.result(timeout=120)  # Timeout
        #        print(f"Rate {rate} abgeschlossen mit Ergebnis {result}")
        #    except Exception as e:
        #        print(f"Rate {rate} wurde mit Fehler {exc} beendet.")
        #        future.cancel()
                
    # with ProcessPoolExecutor() as executor:
        # Parallel processing
    #    results = list(executor.map(simulation, hgt_rate.numpy()))

    for i, result in enumerate(results):
        gfs_matrix[i, :] = result

    return torch.from_numpy(gfs_matrix)

# Globale Definition
def simulate_site(rate):

    mts = gene_model.gene_model(
        theta=0,
        rho=0,
        gene_conversion_rate=0,
        recombination_rate=0,
        hgt_rate=rate,
        num_samples=num_samples,
        num_sites=1,
        double_site_relocation=False,
    )
    return mts