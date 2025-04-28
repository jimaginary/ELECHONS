import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elechons.data import station_handler as sh
from elechons import config
import argparse

parser = argparse.ArgumentParser(description='Plot temp timeseries')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file')
parser.add_argument('--average', choices=['month', 'year'], help='Average data by month or year instead of daily')
parser.add_argument('--year', type=int, help='Plot only data for the specified year')
args = parser.parse_args()

full_stat = config.STAT_TYPES[args.stat]

timeseries = []
for i, station in enumerate(sh.STATIONS['station number']):
    df = sh.get_timeseries(station, args.stat).iloc[-config.OVERLAP_LENGTH:]
    df['date'] = pd.to_datetime(df['date'])

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
    if i == 0:
        dates = df['date']
    
    timeseries.append(df[f'{full_stat} temperature (degC)'])

timeseries = np.array(timeseries)

mins = np.min(timeseries, axis=0)
q1s = np.percentile(timeseries, 25, axis=0)
q2s = np.percentile(timeseries, 50, axis=0)
q3s = np.percentile(timeseries, 75, axis=0)
maxs = np.max(timeseries, axis=0)

plt.plot(dates, mins, color='blue', linestyle='--', label='min')
plt.plot(dates, q1s, color='blue', linestyle='-', label='q1')
plt.plot(dates, q2s, color='black', linestyle='-', label='q2')
plt.plot(dates, q3s, color='red', linestyle='-', label='q3')
plt.plot(dates, maxs, color='red', linestyle='--', label='max')

plt.fill_between(dates, q1s, q2s, color='blue', alpha=0.3)
plt.fill_between(dates, q2s, q3s, color='red', alpha=0.3)

base_title = f'Daily {full_stat.capitalize()} Temperature'
if args.year:
    base_title += f' for {args.year}'
plt.title(f'{base_title}{title_suffix}')
plt.xlabel('date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.legend()

# Output
if args.save_png:
    output_file = f'{config.PLOTS_DIR}/{args.stat}.png'
    plt.savefig(output_file)
else:
    plt.show()

plt.close() 
