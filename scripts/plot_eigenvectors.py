import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
from elechons.data import station_handler
from elechons.processing import edges
from elechons import config

def plot_stations_with_eigenvector(neighbours):
    stations_df = station_handler.STATIONS
    stations_df['lat'] = stations_df['lat'].astype(float)
    stations_df['long'] = stations_df['long'].astype(float)

    adj_matrix = edges.K_nearest(edges.distance_matrix(stations_df), neighbours)
    eigvals, eigvecs = np.linalg.eigh(edges.closeness_matrix(stations_df, config.SCALE_KM, neighbours))

    lat = stations_df['lat'].to_numpy()
    long = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    print("Eigenvalues:")
    for i, val in enumerate(eigvals):
        print(f"{i}: {val:.3f}, \t", end="")
    print()
    choice = int(input("Enter the index of the eigenvalue to use: "))
    
    selected_eigenvalue = eigvals[choice]
    eigenvector = eigvecs[:, choice]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot weather stations with edges and eigenvector coloring.')
    parser.add_argument('--neighbours', type=int, help='number of nearest neighbour edge connections')
    args = parser.parse_args()

    plot_stations_with_eigenvector(args.neighbours)
