import pandas as pd
import numpy as np
import argparse

def compute_similarity(station1, station2, stat):
    """
    Compute the similarity between temperature datasets from two CSV files,
    ignoring empty or NaN entries and adjusting sample size accordingly.
    
    Parameters:
        station1 (str): Weather station id.
        station2 (str): Weather station id.
        stat     (str): Temp state (max, min, mean).
    
    Returns:
        tuple: (similarity, number of valid samples)
        
    Raises:
        ValueError: If no valid overlapping data points remain after cleaning.
    """

    full_stat = {
            'max': 'maximum',
            'min': 'minimum',
            'mean': 'mean'
    }.get(stat)

    file1 = f'../datasets/acorn_sat_v2.5.0_daily_t{stat}/t{stat}.{station1}.daily.csv'  
    file2 = f'../datasets/acorn_sat_v2.5.0_daily_t{stat}/t{stat}.{station2}.daily.csv'

    # Load the CSV files
    df1 = pd.read_csv(file1, usecols=['date', f'{full_stat} temperature (degC)'])
    df2 = pd.read_csv(file2, usecols=['date', f'{full_stat} temperature (degC)'])

    # Convert 'date' to datetime
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])

    # Merge on 'date' to align datasets (inner join for overlapping dates)
    merged_df = pd.merge(df1, df2, on='date', suffixes=('_1', '_2'))

    # Drop rows where either temperature is NaN or empty
    merged_df = merged_df.dropna(subset=[f'{full_stat} temperature (degC)_1', f'{full_stat} temperature (degC)_2'])

    # Check if there's enough valid data
    n_samples = len(merged_df)
    if n_samples == 0:
        raise ValueError("No valid overlapping data points found after removing empty/NaN entries")
    if n_samples < 2:
        raise ValueError(f"Only {n_samples} valid sample(s) found; need at least two for similarity")

    # Extract temperature columns as NumPy arrays
    temp1 = merged_df[f'{full_stat} temperature (degC)_1'].to_numpy()
    temp2 = merged_df[f'{full_stat} temperature (degC)_2'].to_numpy()

    # Compute similarity (sample similarity with ddof=1)
    similarity = np.sqrt(np.sum(np.pow(temp1-temp2,2))/n_samples)

    return similarity, n_samples

def main():
    """Command-line interface to compute and print similarity between two CSV files."""
    parser = argparse.ArgumentParser(description='Compute similarity between two temperature datasets.')
    parser.add_argument('station1', type=str, help='Weather station 1 e.g. 066062')
    parser.add_argument('station2', type=str, help='Weather station 2 e.g. 086071')
    parser.add_argument('stat', choices=['max', 'min', 'mean'], help='Temperature statistic (max, min, mean)')
    args = parser.parse_args()

    try:
        # Compute similarity and get sample count
        sim, n_samples = compute_similarity(args.station1, args.station2, args.stat)
        print(f"Covariance between {args.station1} and {args.station2} {args.stat} temperature: {sim:.4f}")
        print(f"Number of valid samples: {n_samples}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
