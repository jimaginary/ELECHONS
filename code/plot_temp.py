import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import station_handler
import argparse

parser = argparse.ArgumentParser(description='Plot temp timeseries')
parser.add_argument('station', type=str, help='Station id')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file')
parser.add_argument('--average', choices=['month', 'year'], help='Average data by month or year instead of daily')
parser.add_argument('--year', type=int, help='Plot only data for the specified year')
args = parser.parse_args()

df = station_handler.get_timeseries(args.station, args.stat)
df['date'] = pd.to_datetime(df['date'])

full_stat = station_handler.get_full_stat_name(args.stat)

# Filter
if args.year:
    df = df[df['date'].dt.year == args.year]

# Average
if args.average == 'month':
    df = df.groupby(df['date'].dt.to_period('M')).mean(numeric_only=True).reset_index()
    df['date'] = df['date'].dt.to_timestamp()
    title_suffix = ' (Monthly Average)'
elif args.average == 'year':
    df = df.groupby(df['date'].dt.year).mean(numeric_only=True).reset_index()
    df['date'] = pd.to_datetime(df['date'], format='%Y')
    title_suffix = ' (Yearly Average)'
else:
    title_suffix = ' (Daily)'

dates = df['date'].to_numpy()
values = df[f'{full_stat} temperature (degC)'].to_numpy()
plt.plot(dates, values)
base_title = f'Daily {full_stat.capitalize()} Temperature'
if args.year:
    base_title += f' for {args.year}'
plt.title(f'{base_title}{title_suffix}')
plt.xlabel('date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()

# Output
if args.save_png:
    output_file = args.station + '.png'
    plt.savefig(output_file)
else:
    plt.show()

plt.close() 
