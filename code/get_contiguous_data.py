import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def find_longest_contiguous_days(folder_path, station_ids, stat):
    """
    Find the longest contiguous set of days where all stations have temperature data.
    
    Parameters:
        folder_path (str): Path to the folder with temperature CSV files.
        station_ids (list): List of station IDs (e.g., '009518', '014015', ...).
        stat (str): Statistic type ('max', 'min', 'mean') for column naming.
    
    Returns:
        pd.DataFrame: Subset with the longest contiguous period.
    """
    folder = Path(folder_path)
    temp_col = f'{stat}imum temperature (degC)'
    
    # Load first file to initialize merged DataFrame
    first_file = folder / f't{stat}.{station_ids[0]}.daily.csv'
    df_merged = pd.read_csv(first_file, usecols=['date', temp_col])
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_merged = df_merged.rename(columns={temp_col: station_ids[0]})
    
    # Merge with remaining files
    for station_id in station_ids[1:]:
        file_path = folder / f't{stat}.{station_id}.daily.csv'
        df_temp = pd.read_csv(file_path, usecols=['date', temp_col])
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp = df_temp.rename(columns={temp_col: station_id})
        df_merged = pd.merge(df_merged, df_temp, on='date', how='inner')
    
    # Drop rows with any NaN (missing temps)
    df_complete = df_merged.dropna()

    # Find contiguous blocks
    df_complete['date_diff'] = df_complete['date'].diff().dt.days
    df_complete['block'] = (df_complete['date_diff'] != 1).cumsum()
    
    # Group by blocks and find the longest
    block_sizes = df_complete.groupby('block').size()
    longest_block_id = block_sizes.idxmax()
    longest_block_size = block_sizes.max()
    
    # Extract the longest contiguous period
    longest_period = df_complete[df_complete['block'] == longest_block_id]
    
    print(f"Longest contiguous period: {longest_block_size} days")
    print(f"Start date: {longest_period['date'].min()}")
    print(f"End date: {longest_period['date'].max()}")
    
    return longest_period.drop(columns=['date_diff', 'block'])

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find longest contiguous period across temperature time series.')
    parser.add_argument('stat', choices=['max', 'min', 'mean'], help='Statistic type (max, min, mean)')
    args = parser.parse_args()

    # Construct folder path
    folder_path = f'../datasets/acorn_sat_v2.5.0_daily_t{args.stat}'
    folder = Path(folder_path)
    
    # Find all station IDs from filenames
    station_ids = [f.stem.split('.')[1] for f in folder.glob(f't{args.stat}.*.daily.csv')]
    
    if not station_ids:
        print(f"Error: No files found in {folder_path} matching t{args.stat}.*.daily.csv")
        return
    
    print(f"Found {len(station_ids)} stations")
    if len(station_ids) != 104:
        print(f"Warning: Expected 104 stations, found {len(station_ids)}")

    # Find longest contiguous period
    longest_df = find_longest_contiguous_days(folder_path, station_ids, args.stat)
    
    # Save to CSV
    output_file = f'longest_contiguous_t{args.stat}.csv'
    longest_df.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'")

if __name__ == '__main__':
    main()
