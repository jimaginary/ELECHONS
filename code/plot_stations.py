import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import station_handler
import mplcursors
import argparse

def plot_stations():
    df = station_handler.stations
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)

    lat = df['lat'].to_numpy()
    long = df['long'].to_numpy()
    names = df['station name'].to_numpy()
    ids = df['station number'].to_numpy()

    scatter = plt.scatter(long, lat, c='blue')

    plt.title('Weather Stations in Australia')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°S)')

    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = int(sel.index)
        sel.annotation.set_text(f"Station: {names[index]}\nID: {ids[index]}")
    plt.show()

def main():
    """Command-line interface to plot stations from a CSV file."""
    parser = argparse.ArgumentParser(description='Plot weather stations interactively from a CSV file.')

    plot_stations()

if __name__ == '__main__':
    main()
