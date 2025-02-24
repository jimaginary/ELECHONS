import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
from pathlib import Path

def load_and_plot_stations(adj_matrix_file):
    """
    Load station data and adjacency matrix, then plot stations with edges.
    
    Parameters:
        adj_matrix_file (str): Path to the binary adjacency matrix CSV.
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
    station_ids = adj_df.index.tolist()  # e.g., ['1019', '3003', ...]

    # Filter stations_df to match adjacency matrix stations
    stations_df = stations_df[stations_df['station number'].isin(station_ids)]
    stations_df = stations_df.set_index('station number').reindex(station_ids)

    # Extract data
    latitudes = stations_df['lat'].to_numpy()
    longitudes = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(longitudes, latitudes, c='blue', s=50, alpha=0.6, label='Stations')

    # Add edges based on adjacency matrix
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):  # Upper triangle to avoid duplicates
            if adj_matrix[i, j] == 1:
                plt.plot([longitudes[i], longitudes[j]], 
                         [latitudes[i], latitudes[j]], 
                         'k-', alpha=0.2, linewidth=0.5)

    # Customize plot
    plt.title('Weather Stations in Australia with Nearest Neighbor Edges', fontsize=14)
    plt.xlabel('Longitude (°E)', fontsize=12)
    plt.ylabel('Latitude (°S)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

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
    """Command-line interface to plot stations with edges from an adjacency matrix."""
    parser = argparse.ArgumentParser(description='Plot weather stations with edges from a binary adjacency matrix.')
    parser.add_argument('adj_matrix_file', type=str, help='Path to the binary adjacency matrix CSV (e.g., binary_distance_matrix.csv)')
    args = parser.parse_args()

    try:
        load_and_plot_stations(args.adj_matrix_file)
    except FileNotFoundError:
        print(f"Error: File '{args.adj_matrix_file}' not found")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
