
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from elechons import config
# from models import spectral_compression
from elechons.processing import edges
from elechons.data import station_handler as sh

def main():
    parser = argparse.ArgumentParser(description="Compare GFT and DFT compression performance using percentile-based thresholding.")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile (0-100) of size reduction")
    parser.add_argument("--neighbours", type=float, default=8, help="Number of nearest neighbour edges")
    parser.add_argument("--stat", choices=config.STAT_TYPES.keys(), default="mean", help="Statistic to use (max, min, mean)")
    args = parser.parse_args()

    _, eigvecs = np.linalg.eigh(edges.closeness_matrix(sh.STATIONS, config.SCALE_KM, args.neighbours))
    data = sh.get_series_matrix(args.stat)

    # GFT transform
    gft_spectrum = eigvecs.T @ data
    gft_energies = np.abs(gft_spectrum)
    gft_threshold = np.percentile(gft_energies, args.percentile)
    gft_compressed = np.where(gft_energies >= gft_threshold, gft_spectrum, 0)
    gft_decompressed = eigvecs @ gft_compressed

    # DFT transform
    dft_spectrum = np.fft.fft(data, axis=1)
    dft_energies = np.abs(dft_spectrum)
    dft_threshold = np.percentile(dft_energies, args.percentile)
    dft_compressed = np.where(dft_energies >= dft_threshold, dft_spectrum, 0)
    dft_decompressed = np.fft.ifft(dft_compressed, axis=1)

    # Both
    dgft_spectrum = eigvecs.T @ dft_spectrum
    dgft_energies = np.abs(dgft_spectrum)
    dgft_threshold = np.percentile(dgft_energies, args.percentile)
    dgft_compressed = np.where(dgft_energies >= dgft_threshold, dgft_spectrum, 0)
    dgft_decompressed = eigvecs @ np.fft.ifft(dgft_compressed, axis=1)
    

    print(f"GFT ({args.percentile:.1f}% compressed) RMSE (%): {spectral_compression.compute_rmse(data, gft_decompressed):.4f}")
    print(f"DFT ({args.percentile:.1f}% components) RMSE (%): {spectral_compression.compute_rmse(data, dft_decompressed):.4f}")
    print(f"Both ({args.percentile:.1f}% components) RMSE (%): {spectral_compression.compute_rmse(data, dgft_decompressed):.4f}")

if __name__ == "__main__":
    main()
