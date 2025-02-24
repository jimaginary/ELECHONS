import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse

def load_and_plot_stations():
    """
    Load station data from CSV and create an interactive scatter plot.
    """
    # Load CSV, specifying relevant columns
    df = pd.read_csv('../datasets/acorn_stations', usecols=[
        'station number', 'station name', 'lat', 'long'
    ])

    # Clean up column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Remove rows with missing lat/long
    df = df.dropna(subset=['lat', 'long'])
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)

    # Extract data as NumPy arrays
    latitudes = df['lat'].to_numpy()
    longitudes = df['long'].to_numpy()
    names = df['station name'].to_numpy()
    ids = df['station number'].to_numpy()

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(longitudes, latitudes, c='blue', s=50, alpha=0.6)

    # Customize plot
    plt.title('Weather Stations in Australia', fontsize=14)
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
    plt.show()

def main():
    """Command-line interface to plot stations from a CSV file."""
    parser = argparse.ArgumentParser(description='Plot weather stations interactively from a CSV file.')

    try:
        load_and_plot_stations()
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
