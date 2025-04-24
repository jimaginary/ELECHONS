import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import station_handler as sh
import argparse

parser = argparse.ArgumentParser(description='Plot temp spectrum')
parser.add_argument('station', type=str, help='Station id')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file')
args = parser.parse_args()

df = sh.get_fft(args.station, args.stat)

full_stat = sh.get_full_stat_name(args.stat)

freqs = df['f (/years)'].to_numpy()
mags = [np.abs(complex(c)) for c in df['component'].to_numpy()]

plt.scatter(freqs, 20*np.log10(mags), s=10)
plt.xlabel('f (/years)')
plt.ylabel('Magnitude (dB)')
plt.title(f'Frequency spectrum of {sh.get_full_stat_name(args.stat)} temperature data for station {args.station}')

# Output
if args.save_png:
    output_file = args.station + '.png'
    plt.savefig(output_file)
else:
    plt.show()

plt.close() 
