import pandas as pd
import numpy as np

# Load the CSV
df_pairs = pd.read_csv('max_dist_v_sim.csv', usecols=['station1', 'station2', 'dist'])

# Get all unique station IDs from both columns
stations = sorted(set(df_pairs['station1']).union(df_pairs['station2']))
n = len(stations)
print(f"Number of unique stations: {n}")  # Debug: should be 104

# Initialize a square matrix with NaN
dist_matrix = np.full((n, n), np.nan)

# Fill the matrix with distances
for _, row in df_pairs.iterrows():
    i = stations.index(row['station1'])
    j = stations.index(row['station2'])
    dist_matrix[i, j] = row['dist']
    dist_matrix[j, i] = row['dist']  # Symmetry

# Set diagonal to 0
np.fill_diagonal(dist_matrix, 0)

# Convert to DataFrame
df_dist = pd.DataFrame(dist_matrix, index=stations, columns=stations)

# Save to CSV
df_dist.to_csv('distance_matrix.csv')

print(df_dist.head())

binary_matrix = np.zeros_like(df_dist)  # Initialize with zeros
for col in df_dist.columns:
    # Get indices of the 8 smallest values (excluding NaN)
    smallest_indices = df_dist[col].nsmallest(8 + 1).index # 8 closest stations including itself, so we can remove itself later
    # Set those positions to 1 in the binary matrix
    binary_matrix[df_dist.index.isin(smallest_indices), df_dist.columns.get_loc(col)] = 1

# Convert back to DataFrame with station IDs
binary_matrix = np.minimum(1, binary_matrix + binary_matrix.T)
np.fill_diagonal(binary_matrix, 0)
df_binary = pd.DataFrame(binary_matrix, index=stations, columns=stations)

# Save to CSV
df_binary.to_csv('adj_matrix.csv')

print("First few rows of binary matrix:")
print(df_binary.head())

weight_matrix = np.zeros_like(df_dist)
weight_vector = np.zeros(n)
for i in range(n): 
    weight_vector[i] = np.sum(np.exp(-np.pow(dist_matrix[i][binary_matrix[i].astype(bool)]/1000,2)))

for i in range(n):
    for j in range(n):
        if (binary_matrix[i,j] == 0):
            weight_matrix[i,j] = 0
        else:
            weight_matrix[i,j] = np.exp(-np.pow(dist_matrix[i,j]/1000,2))/np.sqrt(weight_vector[i]*weight_vector[j])

df_weights = pd.DataFrame(weight_matrix, index=stations, columns=stations)
df_weights.to_csv('weight_matrix.csv')

print("First few rows of weight matrix:")
print(df_weights.head())

eigvals, eigvecs = np.linalg.eigh(weight_matrix)
D = np.diag(eigvals)
P = eigvecs
P_inv = P.T

df_eigs = pd.DataFrame(eigvecs, index=stations, columns=eigvals)
df_eigs.to_csv('eigvecs.csv')

print("First few rows of eigs matrix:")
print(df_eigs.head()) 
