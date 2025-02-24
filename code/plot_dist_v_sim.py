import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_dist_vs_sim(file_path):
    """
    Plot Distance (km) vs. RMS Distance (°C) from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file (e.g., max_dist_v_sim.csv).
    """
    # Load CSV
    df = pd.read_csv(file_path, usecols=['station1', 'station2', 'dist', 'sim'])

    # Extract data
    distances = df['dist']
    similarities = df['sim']
    print(df[df['sim'] > 30])

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, similarities, c='blue', alpha=0.6, s=50)

    # Customize plot
    plt.title(f'Distance vs. RMS Temperature Difference ({file_path.split("_")[0].capitalize()})', fontsize=14)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('RMS Distance (°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
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
        print(f"Error: File '{file_name}' not found. Ensure it exists in the current directory.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
