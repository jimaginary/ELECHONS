import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
from pathlib import Path

def load_and_plot_stations(adj_matrix_file, eigenvector_file):
    """
    Plot stations with edges, colored by eigenvector values from a selected eigenvalue.
    
    Parameters:
        adj_matrix_file (str): Path to the binary adjacency matrix CSV.
        eigenvector_file (str): Path to the eigenvectors CSV (eigenvalues as columns).
    """
    # Load station coordinates
    stations_df = pd.read_csv('../datasets/acorn_stations', 
                              usecols=['station number', 'station name', 'lat', 'long'])
    stations_df.columns = stations_df.columns.str.strip()
    stations_df = stations_df.dropna(subset=['lat', 'long'])
    stations_df['lat'] = stations_df['lat'].astype(float)
    stations_df['long'] = stations_df['long'].astype(float)

    # Load adjacency matrix
    adj_df = pd.read_csv(adj_matrix_file, index_col=0)
    adj_matrix = adj_df.to_numpy()
    station_ids = adj_df.index.tolist()

    # Load eigenvector matrix
    eigen_df = pd.read_csv(eigenvector_file, index_col=0)
    eigenvalues = eigen_df.columns.astype(float).tolist()  # Column headers as floats
    eigenvectors = eigen_df.to_numpy()

    # Filter stations_df to match adjacency matrix
    stations_df = stations_df[stations_df['station number'].isin(station_ids)]
    stations_df = stations_df.set_index('station number').reindex(station_ids)

    # Extract coordinates and metadata
    latitudes = stations_df['lat'].to_numpy()
    longitudes = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    # Prompt user to select eigenvalue
    print("Available eigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"{i}: {val:.3f}, ", end="")
    print()
    while True:
        try:
            choice = int(input("Enter the index of the eigenvalue to use (e.g., 0 for first): "))
            if 0 <= choice < len(eigenvalues):
                break
            print(f"Please enter a number between 0 and {len(eigenvalues) - 1}")
        except ValueError:
            print("Please enter a valid integer")
    
    selected_eigenvalue = eigenvalues[choice]
    eigenvector = eigenvectors[:, choice]

    # Create scatter plot with eigenvector coloring
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(longitudes, latitudes, c=eigenvector, s=50, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Eigenvector Value')

    # Add edges
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            if adj_matrix[i, j] == 1:
                plt.plot([longitudes[i], longitudes[j]], 
                         [latitudes[i], latitudes[j]], 
                         'k-', alpha=0.2, linewidth=0.5)

    # Customize plot
    plt.title(f'Weather Stations (Colored by Eigenvector for Eigenvalue {selected_eigenvalue:.3f})', fontsize=14)
    plt.xlabel('Longitude (°E)', fontsize=12)
    plt.ylabel('Latitude (°S)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add hover functionality
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = int(sel.index)
        sel.annotation.set_text(f"Station: {names[index]}\nID: {ids[index]}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    # Show plot
    plt.tight_layout()
    plt.show()

def main():
    """Command-line interface to plot stations with edges and eigenvector coloring."""
    parser = argparse.ArgumentParser(description='Plot weather stations with edges and eigenvector coloring.')
    parser.add_argument('adj_matrix_file', type=str, help='Path to the binary adjacency matrix CSV (e.g., binary_distance_matrix.csv)')
    parser.add_argument('eigenvector_file', type=str, help='Path to the eigenvectors CSV (e.g., eigenvectors.csv)')
    args = parser.parse_args()

    try:
        load_and_plot_stations(args.adj_matrix_file, args.eigenvector_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
