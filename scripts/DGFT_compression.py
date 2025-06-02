
import numpy as np
import argparse
from elechons import config
from elechons.models import spectral_compression
from elechons.processing import edges
from elechons.data import station_handler
from elechons.utils import metrics

def DGFT_compression(percentile, neighbours, stat):
    _, eigvecs = np.linalg.eigh(edges.closeness_matrix(station_handler.STATIONS, config.SCALE_KM, neighbours))
    data = station_handler.get_series_matrix(stat)

    # GFT transform
    gft_compressed = spectral_compression.gft_compress(data, eigvecs, percentile)
    gft_decompressed = spectral_compression.gft_decompress(gft_compressed, eigvecs)

    # DFT transform
    dft_compressed = spectral_compression.dft_compress(data, percentile)
    dft_decompressed = spectral_compression.dft_decompress(dft_compressed)

    # Both
    dgft_compressed = spectral_compression.dgft_compress(data, eigvecs, percentile)
    dgft_decompressed = spectral_compression.dgft_decompress(dgft_compressed, eigvecs)

    print(f"GFT ({percentile:.1f}% compressed) RMSE (%): {metrics.rmse_percent(data, gft_decompressed):.4f}")
    print(f"DFT ({percentile:.1f}% compressed) RMSE (%): {metrics.rmse_percent(data, dft_decompressed):.4f}")
    print(f"Both ({percentile:.1f}% compressed) RMSE (%): {metrics.rmse_percent(data, dgft_decompressed):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GFT and DFT compression performance using percentile-based threstation_handlerolding.")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile (0-100) of size reduction")
    parser.add_argument("--neighbours", type=int, default=8, help="Number of nearest neighbour edges")
    parser.add_argument("--stat", choices=config.STAT_TYPES.keys(), default="mean", help="Statistic to use (max, min, mean)")
    args = parser.parse_args()

    DGFT_compression(args.percentile, args.neighbours, args.stat)
