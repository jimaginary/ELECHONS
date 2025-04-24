
import numpy as np
import pandas as pd

def dft_matrix(N):
    omega = np.exp(-2j * np.pi / N)
    return np.array([[omega**(i * j) for j in range(N)] for i in range(N)]) / np.sqrt(N)

def compute_rmse(original, compressed):
    return np.sqrt(np.mean(np.square(original - compressed)))

def compress_with_gft(data, eigvecs, k):
    # Transform to GFT domain
    spectrum = eigvecs.T @ data
    # Zero all but k largest components (by energy)
    energies = np.sum(np.abs(spectrum)**2, axis=1)
    top_k_indices = np.argsort(energies)[-k:]
    mask = np.zeros_like(spectrum)
    mask[top_k_indices, :] = spectrum[top_k_indices, :]
    # Inverse GFT
    return eigvecs @ mask

def compress_with_dft(data, dft_basis, k):
    # Transform to DFT domain
    spectrum = dft_basis.T @ data
    # Zero all but k largest components (by energy)
    energies = np.sum(np.abs(spectrum)**2, axis=1)
    top_k_indices = np.argsort(energies)[-k:]
    mask = np.zeros_like(spectrum)
    mask[top_k_indices, :] = spectrum[top_k_indices, :]
    # Inverse DFT
    return dft_basis @ mask
