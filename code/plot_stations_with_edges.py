import pandas as pd
import numpy as np
import station_handler
import matplotlib.pyplot as plt
import mplcursors
import argparse
from pathlib import Path

def plot_stations_with_edges(adj_matrix_file):
    stations_df = station_handler.stations
    stations_df['lat'] = stations_df['lat'].astype(float)
    stations_df['long'] = stations_df['long'].astype(float)

    adj_df = pd.read_csv(adj_matrix_file, index_col=0)
    adj_matrix = adj_df.to_numpy()

    lat = stations_df['lat'].to_numpy()
    long = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    scatter = plt.scatter(long, lat, zorder=5)

    for i in range(len(stations_df)):
        for j in range(i + 1, len(stations_df)):  # Upper triangle to avoid duplicates
            if adj_matrix[i, j] == 1:
                plt.plot([long[i], long[j]], 
                         [lat[i], lat[j]], 
                         'k-', alpha=0.2, zorder=0)

    plt.title('Weather Stations in Australia with Nearest Neighbor Edges')
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
    """Command-line interface to plot stations with edges from an adjacency matrix."""
    parser = argparse.ArgumentParser(description='Plot weather stations with edges from a binary adjacency matrix.')
    parser.add_argument('adj_matrix_file', type=str, help='Path to the binary adjacency matrix CSV (e.g., binary_distance_matrix.csv)')
    args = parser.parse_args()

    try:
        plot_stations_with_edges(args.adj_matrix_file)
    except FileNotFoundError:
        print(f"Error: File not found")

if __name__ == '__main__':
    main()
