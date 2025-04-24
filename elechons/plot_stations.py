import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import station_handler
import argparse

def plot_stations():
    df = station_handler.stations
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)

    lat = df['lat'].to_numpy()
    long = df['long'].to_numpy()
    names = df['station name'].to_numpy()
    ids = df['station number'].to_numpy()

    plt.scatter(long, lat, s=10)

    for i, id in enumerate(ids):
       plt.annotate(id, (long[i], lat[i]), size=10)

    plt.title('Weather Stations in Australia')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°S)')

    plt.show()

def main():
    plot_stations()

if __name__ == '__main__':
    main()
