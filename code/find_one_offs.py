import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def find_one_off_missing(folder_path, station_ids, stat):
    full_stat = {
        'max': "maximum",
        'min': 'minimum',
        'mean': 'mean'
    }[stat]

    folder = Path(folder_path)
    temp_col = f'{full_stat} temperature (degC)'
    missing_dates = {}

    for station_id in station_ids:
        file_path = folder / f't{stat}.{station_id}.daily.csv'
        df = pd.read_csv(file_path, usecols=['date', temp_col])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert invalid dates to NaT

        # Identify NaN in temperature column
        is_nan = df[temp_col].isna()

        # Shift to check previous and next days
        prev_day = is_nan.shift(1, fill_value=False)  # False for first row
        next_day = is_nan.shift(-1, fill_value=False)  # False for last row

        # One-off missing: NaN where prev and next are not NaN
        one_off_mask = is_nan & ~prev_day & ~next_day
        # Filter out NaT dates and get valid Timestamps
        one_off_dates = df.loc[one_off_mask & df['date'].notna(), 'date'].tolist()

        if one_off_dates:
            missing_dates[station_id] = one_off_dates

    return missing_dates

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find one-off missing temperature values across stations.')
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

    # Find one-off missing values
    missing_dates = find_one_off_missing(folder_path, station_ids, args.stat)
    
    # Print results
    if not missing_dates:
        print("No one-off missing values found across all stations.")
    else:
        print("One-off missing values found:")
        for station_id, dates in missing_dates.items():
            print(f"Station {station_id}: {len(dates)} missing dates")
            for date in dates:
                print(f"  {date.strftime('%Y-%m-%d')}")  # Safe now, NaT filtered out

    # Save to CSV (optional)
    if missing_dates:
        output_data = []
        for station_id, dates in missing_dates.items():
            for date in dates:
                output_data.append({'station_id': station_id, 'date': date})
        df_output = pd.DataFrame(output_data)
        output_file = f'one_off_missing_t{args.stat}.csv'
        df_output.to_csv(output_file, index=False)
        print(f"Saved to '{output_file}'")

if __name__ == '__main__':
    main()
