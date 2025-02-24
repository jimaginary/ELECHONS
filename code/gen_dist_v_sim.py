import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import get_similarity
import earth_distance

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot temperature similarity v distance plot for some stat in the acorn dataset.')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--output', type=str, help='output file to store the data to')
args = parser.parse_args()

station_list = pd.read_csv(f'../datasets/acorn_stations_list', dtype={'stations': str})
lat_long_table = pd.read_csv(f'../datasets/acorn_stations', usecols=['station number', 'lat', 'long'], dtype={'station number': str})
filepaths = {station: f'../datasets/acorn_sat_v2.5.0_daily_t{args.stat}/t{args.stat}.{station}.daily.csv' for station in station_list['stations']}
full_stat = {
    'max': 'maximum',
    'min': 'minimum',
    'mean': 'mean'
}.get(args.stat)

output = pd.DataFrame(['station1', 'station2', 'dist', 'sim'])

def filt(df):
    df['date'] = pd.to_datetime(df['date'])

#    if args.year:
#        df = df[df['date'].dt.year == args.year]
#        if df.empty:
#            raise ValueError(f"No data found for year {args.year} in the dataset")
#
#    if args.average == 'month':
#        # Group by year and month, calculate mean
#        df = df.groupby(df['date'].dt.to_period('M')).mean(numeric_only=True).reset_index()
#        df['date'] = df['date'].dt.to_timestamp()  # Convert period back to datetime for plotting
#    elif args.average == 'year':
#        # Group by year, calculate mean
#        df = df.groupby(df['date'].dt.year).mean(numeric_only=True).reset_index()
#        df['date'] = pd.to_datetime(df['date'], format='%Y')  # Convert year to datetime

loc = 0
rows, _ = station_list.shape
row = 0

for station1 in station_list['stations']:
    print(f'{100*row/rows:2f} complete')
    row += 1
    for station2 in station_list['stations']:
        df1 = pd.read_csv(filepaths[station1], usecols=['date', f'{full_stat} temperature (degC)'])
        df2 = pd.read_csv(filepaths[station2], usecols=['date', f'{full_stat} temperature (degC)'])

        filt(df1)
        filt(df2)

        lat_long1 = lat_long_table.loc[lat_long_table['station number'] == station1]
        lat_long2 = lat_long_table.loc[lat_long_table['station number'] == station2]

        dist = earth_distance.haversine(lat_long1['lat'].iloc[0], lat_long1['long'].iloc[0], lat_long2['lat'].iloc[0], lat_long2['long'].iloc[0])
        sim, _ = get_similarity.compute_similarity(station1, station2, args.stat)
        data = {'station1': station1, 'station2': station2, 'dist': dist, 'sim': sim}
        output = pd.concat([output, pd.DataFrame(data, index=[0])], ignore_index=True)
        loc += 1

name = args.output
if (args.output == None):
    name = 'out' 
output.to_csv(f'./{name}', index=False)       
