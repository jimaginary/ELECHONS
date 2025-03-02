import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def autocorr(v, max_lag):
    var = np.var(v)
    n = len(v)
    v_centered = v - np.mean(v)
    C = np.zeros(max_lag + 1)
    C[0] = 1
    C[1:] = [np.sum(v_centered[:-lag] * v_centered[lag:]) / ((n - lag) * var) for lag in range(1, max_lag+1)] 
    return C

def series_dist(u, v):
    return np.sqrt(np.sum(np.pow(u - v, 2))/len(u))
