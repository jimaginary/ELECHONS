import pandas as pd
import numpy as np
import station_handler as sh
import matplotlib.pyplot as plt
import sys

time_eigs = np.exp(-complex(0,1)*2*np.pi*np.fft.fftfreq(sh.overlap_length))

graph_eigen_df = pd.read_csv('eigvecs.csv', index_col=0)
graph_eigenvalues = graph_eigen_df.columns.astype(float).tolist()
graph_eigenvectors = graph_eigen_df.to_numpy()

compression_factors = [50, 20, 15, 10, 7, 5, 3, 2, 1]

for stat in ['max', 'min', 'mean']:
    print(f'{stat} data compression RMSE')
    stat_stations_spectra = []
    stat_stations_timeseries = []
    for i, station in enumerate(sh.stations['station number']):
        stat_stations_spectra.append([complex(c) for c in sh.get_fft(station, stat)['component']])
        stat_stations_timeseries.append(sh.get_timeseries(station, stat)[f'{sh.get_full_stat_name(stat)} temperature (degC)'][-sh.overlap_length:])
    stat_stations_spectra = np.array(stat_stations_spectra)
    stat_stations_timeseries = np.array(stat_stations_timeseries)
    spectra = graph_eigenvectors.T @ stat_stations_spectra
    rows, cols = np.unravel_index(np.argsort(np.abs(spectra).flatten())[::-1], spectra.shape)
    ordered_args = list(zip(rows, cols))

    N = np.prod(spectra.shape)

    total_power = np.sum(np.pow(np.abs(stat_stations_timeseries),2))

    [print(f'1/{C},\t', end='') for C in compression_factors]
    print()
    for C in compression_factors:
        compressed_spectra = np.zeros_like(spectra)
        for i in range(N // C):
            compressed_spectra[*ordered_args[i]] = spectra[*ordered_args[i]]
        compressed_time_spectra = graph_eigenvectors @ compressed_spectra
        compressed_data = []
        for time_spectra in compressed_time_spectra:
            compressed_data.append(np.fft.ifft(time_spectra))
        compressed_data = np.array(compressed_data)
        rmse = np.sqrt(np.sum(np.pow(np.abs((stat_stations_timeseries - compressed_data)),2))/total_power)
        print(f'{rmse*100:.2f},\t', end='')
    print()

