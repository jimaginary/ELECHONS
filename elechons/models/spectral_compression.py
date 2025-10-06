import numpy as np

# compress data with gft retaining p% of graph eigenvectors
def gft_compress(data, eigvecs, p):
    gft_spectrum = eigvecs.T @ data
    gft_energies = np.abs(gft_spectrum)
    gft_threshold = np.percentile(gft_energies, p)
    return np.where(gft_energies >= gft_threshold, gft_spectrum, 0)

# decompress gft data given eigenvectors
def gft_decompress(gft_compressed, eigvecs):
    return eigvecs @ gft_compressed

# compress data with dft retaining p% of eigenvectors
def dft_compress(data, p):
    # DFT transform
    dft_spectrum = np.fft.fft(data, axis=1)
    dft_energies = np.abs(dft_spectrum)
    dft_threshold = np.percentile(dft_energies, p)
    return np.where(dft_energies >= dft_threshold, dft_spectrum, 0)

# decompress data with dft
def dft_decompress(dft_compressed):
    return np.fft.ifft(dft_compressed, axis=1)

# compress data with dft and gft retaining p% of eigenvectors
def dgft_compress(data, eigvecs, p):
    dgft_spectrum = eigvecs.T @ np.fft.fft(data, axis=1)
    dgft_energies = np.abs(dgft_spectrum)
    dgft_threshold = np.percentile(dgft_energies, p)
    return np.where(dgft_energies >= dgft_threshold, dgft_spectrum, 0)

# decompress data with dft and gft
def dgft_decompress(dgft_compressed, eigvecs):
    return np.fft.ifft(eigvecs @ dgft_compressed, axis=1)