import pandas as pd
import numpy as np
import station_handler as sh
import edge_computations as ec
import matplotlib.pyplot as plt
import sys

def rmse(x):
    return np.sqrt(np.mean(np.pow(np.real(x), 2)))

time_eigs = np.exp(-complex(0,1)*2*np.pi*np.fft.fftfreq(sh.overlap_length))

# graph_eigen_df = pd.read_csv('eigvecs.csv', index_col=0)
graph_eigenvalues, graph_eigenvectors = np.linalg.eigh(ec.laplacian_matrix(sh.stations, 8))#graph_eigen_df.columns.astype(float).tolist()
# graph_eigenvectors = graph_eigen_df.to_numpy()

compression_factors = [50, 20, 15, 10, 7, 5, 3, 2, 1]

for stat in ['max', 'min', 'mean']:
    print(f'\t{stat} data compression RMSE')
    stations_timeseries = sh.get_series_matrix(stat)
    stations_timespectra = np.fft.fft(stations_timeseries, axis=1)

    N = np.prod(stations_timeseries.shape)

    [print(f'1/{C},\t', end='') for C in compression_factors]
    print()
    print('Time-only Compression:')
    rows, cols = np.unravel_index(np.argsort(np.abs(stations_timespectra).flatten())[::-1], stations_timespectra.shape)
    ordered_args = list(zip(rows, cols))
    for C in compression_factors:
        compressed_spectra = np.zeros_like(stations_timespectra)
        for i in range(N // C):
            compressed_spectra[*ordered_args[i]] = stations_timespectra[*ordered_args[i]]
        compressed_data = np.fft.ifft(compressed_spectra, axis=1)
        print(f'{100 * rmse(stations_timeseries - compressed_data) / rmse(stations_timeseries):.2f},\t', end='')
    print()

    print('Time and Space Compression:')
    stations_spacespectra = graph_eigenvectors.T @ stations_timespectra
    rows, cols = np.unravel_index(np.argsort(np.abs(stations_spacespectra).flatten())[::-1], stations_spacespectra.shape)
    ordered_args = list(zip(rows, cols))
    for C in compression_factors:
        compressed_spectra = np.zeros_like(stations_spacespectra)
        for i in range(N // C):
            compressed_spectra[*ordered_args[i]] = stations_spacespectra[*ordered_args[i]]
        compressed_time_spectra = graph_eigenvectors @ compressed_spectra
        compressed_data = np.fft.ifft(compressed_time_spectra, axis=1)
        print(f'{100 * rmse(stations_timeseries - compressed_data) / rmse(stations_timeseries):.2f},\t', end='')
    print()
