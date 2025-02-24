import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def compute_autocorrelation(series, max_lag):
    """
    Compute autocorrelation for a series up to max_lag.
    
    Parameters:
        series (np.array): Time series data.
        max_lag (int): Maximum lag to compute.
    
    Returns:
        np.array: Autocorrelation values for lags 0 to max_lag.
    """
    n = len(series)
    mean = np.mean(series)
    variance = np.var(series)
    series_centered = series - mean
    
    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0  # R(0) = 1 by definition
        else:
            # Compute correlation for lag
            autocorr[lag] = np.sum(series_centered[:-lag] * series_centered[lag:]) / ((n - lag) * variance)
    
    return autocorr

def plot_autocorrelation(file_path, stat, station_id):
    """
    Plot autocorrelation with log-time delay, marking 1 day, 1 month, 1 year.
    
    Parameters:
        file_path (str): Path to the CSV file.
        stat (str): Statistic (max, min, mean).
        station_id (str): Station ID (e.g., 008315).
    """
    # Load CSV
    df = pd.read_csv(file_path, usecols=['date', f'{stat}imum temperature (degC)'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temperature series, dropping NaN
    temp_col = f'{stat}imum temperature (degC)'
    temps = df[temp_col].dropna().to_numpy()
    
    # Compute autocorrelation (up to 3 years for visibility)
    max_lag = min(3 * 365, len(temps) - 1)  # Max 3 years or series length
    lags = np.arange(max_lag + 1)
    autocorr = compute_autocorrelation(temps, max_lag)
    
    # Plot with log-scale x-axis
    plt.figure(figsize=(12, 6))
    plt.plot(lags, autocorr, color='blue', label='Autocorrelation')
    
    # Mark specific lags
    for lag, label in [(1, '1 Day'), (30, '1 Month'), (365, '1 Year')]:
        if lag <= max_lag:
            plt.axvline(x=lag, color='red', linestyle='--', alpha=0.5)
            plt.text(lag, 0.95, label, rotation=90, verticalalignment='top', color='red')
    
    # Customize plot
    plt.xscale('log')
    plt.xlabel('Time Lag (days, log scale)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.title(f'Autocorrelation of {stat.capitalize()} Temperature (Station {station_id})', fontsize=14)
    plt.grid(True, which="both", ls='--', alpha=0.7)
    plt.ylim(-1, 1)
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

def main():
    """Command-line interface to plot autocorrelation for a station and statistic."""
    parser = argparse.ArgumentParser(description='Plot autocorrelation of temperature data.')
    parser.add_argument('stat', choices=['max', 'min', 'mean'], help='Statistic (max, min, mean)')
    parser.add_argument('station_id', type=str, help='Station ID (e.g., 008315)')
    args = parser.parse_args()

    # Construct file path
    folder = Path(f"../datasets/acorn_sat_v2.5.0_daily_t{args.stat}/")
    file_name = f"t{args.stat}.{args.station_id}.daily.csv"
    file_path = folder / file_name

    try:
        plot_autocorrelation(file_path, args.stat, args.station_id)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Check station ID and folder.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
