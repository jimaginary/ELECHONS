import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
from pathlib import Path
import station_handler

def plot_stations_with_eigenvector(adj_matrix_file, eigenvector_file):
    stations_df = station_handler.stations
    stations_df['lat'] = stations_df['lat'].astype(float)
    stations_df['long'] = stations_df['long'].astype(float)

    adj_df = pd.read_csv(adj_matrix_file, index_col=0)
    adj_matrix = adj_df.to_numpy()

    eigen_df = pd.read_csv(eigenvector_file, index_col=0)
    eigenvalues = eigen_df.columns.astype(float).tolist()
    eigenvectors = eigen_df.to_numpy()

    lat = stations_df['lat'].to_numpy()
    long = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    print("Eigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"{i}: {val:.3f}, ", end="")
    print()
    choice = int(input("Enter the index of the eigenvalue to use: "))
    
    selected_eigenvalue = eigenvalues[choice]
    eigenvector = eigenvectors[:, choice]

    scatter = plt.scatter(long, lat, c=eigenvector, cmap='bwr', zorder=5)
    plt.colorbar(scatter, label='Eigenvector Value')

    for i in range(len(stations_df)):
        for j in range(i + 1, len(stations_df)):
            if adj_matrix[i, j] == 1:
                plt.plot([long[i], long[j]], 
                         [lat[i], lat[j]], 
                         'k-', alpha=0.2, zorder=0)

    plt.title(f'Weather Stations (Colored by Eigenvector for Eigenvalue {selected_eigenvalue:.3f})')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°S)')

    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = int(sel.index)
        sel.annotation.set_text(f"Station: {names[index]}\nID: {ids[index]}")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot weather stations with edges and eigenvector coloring.')
    parser.add_argument('adj_matrix_file', type=str, help='Path to the binary adjacency matrix CSV (e.g., binary_distance_matrix.csv)')
    parser.add_argument('eigenvector_file', type=str, help='Path to the eigenvectors CSV (e.g., eigenvectors.csv)')
    args = parser.parse_args()

    try:
        plot_stations_with_eigenvector(args.adj_matrix_file, args.eigenvector_file)
    except FileNotFoundError as e:
        print(f"Error: File not found")

if __name__ == '__main__':
    main()
