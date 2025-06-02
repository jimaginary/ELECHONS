import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from elechons import config
from elechons.data import station_handler
from elechons.processing import series

def plot_autocorrelation(station, stat):
    df = station_handler.get_timeseries(station, stat)
    df['date'] = pd.to_datetime(df['date'])

    full_stat = config.STAT_TYPES[stat]
    temps = df[f'{full_stat} temperature (degC)'].to_numpy()
    
    max_lag = min(3 * 365, len(temps) - 1)  # Max 3 years or series length
    lags = np.arange(max_lag + 1)
    autocorr = series.autocorr(temps, max_lag)
    
    plt.plot(lags, autocorr)
    
    for lag, label in [(1, '1 Day'), (30, '1 Month'), (365, '1 Year')]:
        plt.axvline(x=lag, color='red', linestyle='--', alpha=0.5)
        plt.text(lag, 0.95, label, verticalalignment='top', color='red')
    
    plt.xscale('log')
    plt.xlabel('Time Lag (days, log scale)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation of {stat.capitalize()} Temperature (Station {station})')
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot autocorrelation of temperature data.')
    parser.add_argument('station_id', type=str, help='Station ID')
    parser.add_argument('stat', choices=['max', 'min', 'mean'], help='Statistic (max, min, mean)')
    args = parser.parse_args()

    try:
        plot_autocorrelation(args.station_id, args.stat)
    except FileNotFoundError:
        print(f"Error: File not found")

if __name__ == '__main__':
    main()
