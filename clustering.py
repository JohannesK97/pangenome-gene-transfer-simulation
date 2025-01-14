import simulator
import torch
import re
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ProcessPoolExecutor



def reorder_matrix(matrix):
    """
    Reorders the rows and columns of the matrix based on Hamming distance,
    with each row normalized by dividing by its maximum value before comparison.
    """
    # Compute Hamming distance between columns
    column_hamming_distances = pdist(matrix.T, metric='hamming')  # Transpose to compare columns
    
    # Perform hierarchical clustering for columns
    column_linkage_matrix = linkage(column_hamming_distances, method='average', optimal_ordering = True)  # Average linkage clustering
    
    # Get the optimal order of columns
    column_order = leaves_list(column_linkage_matrix)
    
    # Reorder the columns based on the new order
    matrix = matrix[:, column_order]

    # Normalize rows after column reordering
    #row_maxes = matrix.max(axis=1, keepdims=True)
    #normalized_matrix = matrix / np.maximum(row_maxes, 1e-8)  # Avoid division by zero
    
    # Compute Hamming distance between rows
    row_hamming_distances = pdist(matrix, metric='cosine')  # Compare normalized rows
    
    # Perform hierarchical clustering for rows
    row_linkage_matrix = linkage(row_hamming_distances, method='average', optimal_ordering = True)  # Average linkage clustering
    
    # Get the optimal order of rows
    row_order = leaves_list(row_linkage_matrix)
    
    # Step 10: Reorder the rows based on the new order
    reordered_matrix = matrix[row_order, :]
    
    return reordered_matrix

def custom_distance(matrix):
    """
    Computes a custom distance matrix for rows in the matrix.
    Distance = distance / (Sum of 1s in both rows).
    
    Args:
    - matrix (numpy.ndarray): Input matrix
    
    Returns:
    - numpy.ndarray: Custom distance matrix (condensed form)
    """
    num_rows = matrix.shape[0]
    distance_matrix = []
    
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            # Calculate distance
            dist = np.sum(abs(matrix[i] - matrix[j]))
            # Calculate the sum of 1s in both rows
            sum_of_ones = np.sum(matrix[i]) + np.sum(matrix[j])
            # Compute custom distance
            distance = dist / sum_of_ones if sum_of_ones != 0 else np.inf
            distance_matrix.append(distance)
    
    return np.array(distance_matrix)

def cluster_and_sum_rows(matrix, num_clusters):
    """
    Clusters the rows of a matrix based on a custom distance and sums the rows within each cluster.
    
    Args:
    - matrix (numpy.ndarray): Input matrix
    - num_clusters (int): Number of clusters to create
    
    Returns:
    - summed_clusters (numpy.ndarray): Summed rows for each cluster
    - labels (list): Cluster labels for each row
    """
    # Step 1: Compute custom distance matrix
    custom_distances = custom_distance(matrix)
    
    # Step 2: Perform hierarchical clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=num_clusters, metric='precomputed', linkage='complete'
    )
    clustering_model.fit(squareform(custom_distances))
    labels = clustering_model.labels_
    
    # Step 3: Sum rows within each cluster
    summed_clusters = []
    for cluster_id in range(num_clusters):
        # Select rows belonging to the current cluster
        cluster_rows = matrix[labels == cluster_id]
        # Compute the sum of the rows in this cluster
        summed_clusters.append(cluster_rows.sum(axis=0))
    
    return np.array(summed_clusters)


def cluster_and_store(gene_presence_absence_matrice, hgt_rate, output_file, num_clusters):
    with open(output_file, 'a') as file_handle:  # Use append mode to avoid overwriting
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        summed_clusters = cluster_and_sum_rows(gene_presence_absence_matrice, num_clusters)
        file_handle.write(f"hgt_rate {hgt_rate}: {summed_clusters}\n")
    print("Hgt_rate:", hgt_rate, "successfull.")

def run_clustering(gene_presence_absence_matrices, hgt_rates):
    
    num_clusters = min(matrix.shape[0] for matrix in gene_presence_absence_matrices)
    num_clusters = 908   #!!!!!!!!!!!!!!!!!
    print("Number of clusters:", num_clusters)

    output_file = 'clustered_simulated_data.txt'

    if os.path.exists(output_file):
        os.remove(output_file)
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx in range(len(hgt_rates)):
            hgt_rate = hgt_rates[idx]
            matrix = gene_presence_absence_matrices[idx]
            futures.append(
                executor.submit(cluster_and_store, matrix, hgt_rate, output_file, num_clusters)
            )
        for future in futures:
            future.result()

if __name__ == '__main__':
    
    file = 'simulation_results.txt'
    
    print("Load data.")
    hgt_rates, gene_presence_absence_matrices_unfiltered = simulator.read_simulation_results(file)

    gene_presence_absence_matrices = [
        np.array([
            row for row in gene_presence_absence_matrices_unfiltered[i]
            if not (np.all(row == 0) or np.all(row == 1))
        ])
        for i in range(len(gene_presence_absence_matrices_unfiltered))
    ]
    print("Data is loaded.")
    run_clustering(gene_presence_absence_matrices, hgt_rates)

