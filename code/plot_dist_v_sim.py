import pandas as pd
import matplotlib.pyplot as plt
import station_handler
import argparse

def plot_dist_vs_sim(file_path):
    df = pd.read_csv(file_path, usecols=['station1', 'station2', 'dist', 'sim'])

    distances = df['dist']
    similarities = df['sim']

    plt.scatter(distances, similarities, s=2)
    plt.title(f'Distance vs. RMS Temperature Difference ({file_path.split("_")[0].capitalize()})', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('RMS Distance (°C)', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    """Command-line interface to plot distance vs. similarity based on a statistic."""
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
