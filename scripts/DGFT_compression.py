
import numpy as np
import argparse
from elechons import config
from elechons.models import spectral_compression
from elechons.processing import edges
from elechons.data import station_handler as sh
from elechons.utils import metrics

def main():
    parser = argparse.ArgumentParser(description="Compare GFT and DFT compression performance using percentile-based thresholding.")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile (0-100) of size reduction")
    parser.add_argument("--neighbours", type=float, default=8, help="Number of nearest neighbour edges")
    parser.add_argument("--stat", choices=config.STAT_TYPES.keys(), default="mean", help="Statistic to use (max, min, mean)")
    args = parser.parse_args()

    _, eigvecs = np.linalg.eigh(edges.closeness_matrix(sh.STATIONS, config.SCALE_KM, args.neighbours))
    data = sh.get_series_matrix(args.stat)

    # GFT transform
    gft_compressed = spectral_compression.gft_compress(data, eigvecs, args.percentile)
    gft_decompressed = spectral_compression.gft_decompress(gft_compressed, eigvecs)

    # DFT transform
    dft_compressed = spectral_compression.dft_compress(data, args.percentile)
    dft_decompressed = spectral_compression.dft_decompress(dft_compressed)

    # Both
    dgft_compressed = spectral_compression.dgft_compress(data, eigvecs, args.percentile)
    dgft_decompressed = spectral_compression.dgft_decompress(dgft_compressed, eigvecs)
    

    print(f"GFT ({args.percentile:.1f}% compressed) RMSE (%): {metrics.rmse_percent(data, gft_decompressed):.4f}")
    print(f"DFT ({args.percentile:.1f}% components) RMSE (%): {metrics.rmse_percent(data, dft_decompressed):.4f}")
    print(f"Both ({args.percentile:.1f}% components) RMSE (%): {metrics.rmse_percent(data, dgft_decompressed):.4f}")

if __name__ == "__main__":
    main()
