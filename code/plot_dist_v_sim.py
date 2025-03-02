import pandas as pd
import matplotlib.pyplot as plt
import station_handler
import argparse

def plot_dist_vs_sim(file_path):
    df = pd.read_csv(file_path, usecols=['station1', 'station2', 'dist', 'sim'])

    distances = df['dist']
    similarities = df['sim']

    plt.scatter(distances, similarities, s=2)
    plt.title(f'Distance vs. RMS Temperature Difference ({file_path.split("_")[0].capitalize()})')
    plt.xlabel('Distance (km)')
    plt.ylabel('RMS Distance (°C)')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot distance vs. RMS temperature difference.')
    parser.add_argument('stat', choices=['max', 'mean', 'min'], help='Statistic to plot (max, mean, min)')
    args = parser.parse_args()

    # Construct file name based on stat
    file_name = f"{args.stat}_dist_v_sim.csv"

    try:
        plot_dist_vs_sim(file_name)
    except FileNotFoundError:
        print(f"Error: File not found.")

if __name__ == '__main__':
    main()
