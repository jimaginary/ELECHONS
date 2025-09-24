import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
from elechons.data import station_handler
from elechons.processing import edges

def plot_stations(neighbours):
    stations_df = station_handler.STATIONS
    stations_df['lat'] = stations_df['lat'].astype(float)
    stations_df['long'] = stations_df['long'].astype(float)

    lat = stations_df['lat'].to_numpy()
    long = stations_df['long'].to_numpy()
    names = stations_df['station name'].to_numpy()
    ids = stations_df.index.to_numpy()

    # adj_matrix = edges.K_nearest(edges.distance_matrix(stations_df), neighbours)

    # for i in range(len(stations_df)):
    #     for j in range(i + 1, len(stations_df)):  # Upper triangle to avoid duplicates
    #         if adj_matrix[i, j] == 1:
    #             plt.plot([long[i], long[j]], 
    #                     [lat[i], lat[j]], 
    #                     'k-', alpha=0.2, zorder=0)

    scatter = plt.scatter(long, lat, zorder=5)

    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = int(sel.index)
        sel.annotation.set_text(f"Station: {names[index]}\nID: {ids[index]}")
    
    plt.title('Weather Stations in Australia with Nearest Neighbor Edges')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°S)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot weather stations with edges to nearest neighbours.')
    parser.add_argument('--neighbours', default=0, type=int, help='Number of neighbours')
    args = parser.parse_args()

    plot_stations(args.neighbours)
