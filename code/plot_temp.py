import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot a 1D CSV dataset (variable vs. date) as a line plot.')
parser.add_argument('station', type=str, help='Station id (e.g. 066062)')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file instead of displaying it')
parser.add_argument('--average', choices=['month', 'year'], help='Average data by month or year instead of daily')
parser.add_argument('--year', type=int, help='Plot only data for the specified year (e.g., 2024)')
args = parser.parse_args()

filepath = f'../datasets/acorn_sat_v2.5.0_daily_t{args.stat}/t{args.stat}.{args.station}.daily.csv'
full_stat = {
    'max': 'maximum',
    'min': 'minimum',
    'mean': 'mean'
}.get(args.stat)

# Load the CSV into a pandas DataFrame
# Assuming BoM format; adjust column names if needed
df = pd.read_csv(filepath, usecols=['date', f'{full_stat} temperature (degC)'])

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter by year if specified
if args.year:
    df = df[df['date'].dt.year == args.year]
    if df.empty:
        raise ValueError(f"No data found for year {args.year} in the dataset")

# Average data if specified
if args.average == 'month':
    # Group by year and month, calculate mean
    df = df.groupby(df['date'].dt.to_period('M')).mean(numeric_only=True).reset_index()
    df['date'] = df['date'].dt.to_timestamp()  # Convert period back to datetime for plotting
    title_suffix = ' (Monthly Average)'
elif args.average == 'year':
    # Group by year, calculate mean
    df = df.groupby(df['date'].dt.year).mean(numeric_only=True).reset_index()
    df['date'] = pd.to_datetime(df['date'], format='%Y')  # Convert year to datetime
    title_suffix = ' (Yearly Average)'
else:
    title_suffix = ' (Daily)'

# Extract data as NumPy arrays
dates = df['date'].to_numpy()
values = df[f'{full_stat} temperature (degC)'].to_numpy()

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(dates, values, label=f'{args.stat.capitalize()} Temperature', color='red', linewidth=1.5)

# Customize the plot
base_title = f'Daily {full_stat.capitalize()} Temperature'
if args.year:
    base_title += f' for {args.year}'
plt.title(f'{base_title}{title_suffix}', fontsize=14)
plt.xlabel('date', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Handle output
if args.save_png:
    output_file = args.station.rsplit('.', 1)[0] + '.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")
else:
    plt.show()

# Close the plot
plt.close() 
