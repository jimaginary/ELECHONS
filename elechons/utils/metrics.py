import numpy as np

def rmse_percent(original, compressed):
    return 100*np.sqrt(np.mean(np.square(np.abs(original - compressed))))/np.sqrt(np.mean(np.square(np.abs(original))))