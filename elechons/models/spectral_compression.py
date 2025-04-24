import numpy as np

def compute_rmse(original, compressed):
    return np.sqrt(np.mean(np.square(original - compressed)))

# compresses and decompresses data using k graph eigenvectors
def compress_with_gft(data, eigvecs, k):
    spectrum = eigvecs.T @ data
    energies = np.sum(np.abs(spectrum)**2, axis=1)
    top_k_indices = np.argsort(energies)[-k:]
    mask = np.zeros_like(spectrum)
    mask[top_k_indices, :] = spectrum[top_k_indices, :]
    return eigvecs @ mask

# compresses and decompresses data using k DFT eigenvectors
def compress_with_dft(data, dft_basis, k):
    spectrum = dft_basis.T @ data
    energies = np.sum(np.abs(spectrum)**2, axis=1)
    top_k_indices = np.argsort(energies)[-k:]
    mask = np.zeros_like(spectrum)
    mask[top_k_indices, :] = spectrum[top_k_indices, :]
    return dft_basis @ mask
