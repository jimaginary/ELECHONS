import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elechons.data import station_handler
from elechons import config
import argparse

def plot_temp_dft(station, stat, save_png):
    full_stat = config.STAT_TYPES[stat]

    data = station_handler.get_timeseries(station, stat)[full_stat + ' temperature (degC)']
    spectra = np.fft.fftshift(np.fft.fft(data))

    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectra))) * 365.25

    plt.scatter(freqs, 20*np.log10(np.abs(spectra)), s=10)
    plt.xlabel('f (/years)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Frequency spectrum of {full_stat} temperature data for station {station}')

    # Output
    if save_png:
        output_file = station + '.png'
        plt.savefig(output_file)
    else:
        plt.show()

    plt.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot temp spectrum')
    parser.add_argument('station', type=str, help='Station id')
    parser.add_argument('stat', choices=['max','mean','min'], help='Stat desired to plot')
    parser.add_argument('--save-png', action='store_true', help='Save the plot as a PNG file')
    args = parser.parse_args()

    plot_temp_dft(args.station, args.stat, args.save_png)
