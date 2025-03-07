import pandas as pd
import numpy as np
import station_handler as sh
import edge_computations as ec
import matplotlib.pyplot as plt
import sys

time_eigs = np.exp(-complex(0,1)*2*np.pi*np.fft.fftfreq(sh.overlap_length))

# graph_eigen_df = pd.read_csv('eigvecs.csv', index_col=0)
graph_eigenvalues, graph_eigenvectors = np.linalg.eigh(ec.laplacian_matrix(sh.stations, 8))#graph_eigen_df.columns.astype(float).tolist()
# graph_eigenvectors = graph_eigen_df.to_numpy()

compression_factors = [50, 20, 15, 10, 7, 5, 3, 2, 1]

for stat in ['max', 'min', 'mean']:
    print(f'\t{stat} data compression RMSE')
    stat_stations_timeseries = sh.get_series_matrix(stat)
    stat_stations_spectra = np.fft.fft(stat_stations_timeseries, axis=1)

    N = np.prod(stat_stations_timeseries.shape)

    total_power = np.sum(np.pow(np.abs(stat_stations_timeseries),2))

    [print(f'1/{C},\t', end='') for C in compression_factors]
    print()
    print('Time-only Compression:')
    rows, cols = np.unravel_index(np.argsort(np.abs(stat_stations_spectra).flatten())[::-1], stat_stations_spectra.shape)
    ordered_args = list(zip(rows, cols))
    for C in compression_factors:
        compressed_spectra = np.zeros_like(stat_stations_spectra)
        for i in range(N // C):
            compressed_spectra[*ordered_args[i]] = stat_stations_spectra[*ordered_args[i]]
        compressed_data = []
        for time_spectra in compressed_spectra:
            compressed_data.append(np.fft.ifft(time_spectra))
        compressed_data = np.array(compressed_data)
        rmse = np.sqrt(np.sum(np.pow(np.abs((stat_stations_timeseries - compressed_data)),2))/total_power)
        print(f'{rmse*100:.2f},\t', end='')
    print()

    print('Time and Space Compression:')
    spectra = graph_eigenvectors.T @ stat_stations_spectra
    rows, cols = np.unravel_index(np.argsort(np.abs(spectra).flatten())[::-1], spectra.shape)
    ordered_args = list(zip(rows, cols))
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

