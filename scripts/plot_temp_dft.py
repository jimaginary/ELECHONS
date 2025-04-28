import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elechons.data import station_handler as sh
from elechons import config
import argparse

parser = argparse.ArgumentParser(description='Plot temp spectrum')
parser.add_argument('station', type=str, help='Station id')
parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file')
args = parser.parse_args()

full_stat = config.STAT_TYPES[args.stat]

data = sh.get_timeseries(args.station, args.stat)[full_stat + ' temperature (degC)']
spectra = np.fft.fftshift(np.fft.fft(data))

freqs = np.fft.fftshift(np.fft.fftfreq(len(spectra))) * 365.25

plt.scatter(freqs, 20*np.log10(np.abs(spectra)), s=10)
plt.xlabel('f (/years)')
plt.ylabel('Magnitude (dB)')
plt.title(f'Frequency spectrum of {full_stat} temperature data for station {args.station}')

# Output
if args.save_png:
    output_file = args.station + '.png'
    plt.savefig(output_file)
else:
    plt.show()

plt.close() 
