import numpy as np
import scipy.special as special
from scipy.linalg import toeplitz, solve_toeplitz

# A the covariance matrix, or with inv=True, A the precision matrix (inverse covariance matrix)
def partial_corr(A, inv=False):
    if not inv:
        A = np.linalg.inv(A)
    B = np.reciprocal(np.sqrt(np.outer(np.diagonal(A), np.diagonal(A))))
    return - A * B

# root mean squared error
def rmse(x):
    return np.sqrt(np.mean(np.pow(np.real(x), 2)))

# least squares minimizer
def least_squares(params, func, d, p):
    return np.sqrt(np.mean(np.pow(p - func(params, d), 2)))

# takes in validity vector and reports largest contiguous invalidity
def get_largest_backfill(v):
        l = 0
        s = 0
        for i in v:
            if (not i):
                s += 1
                l = max(s, l)
            else:
                s = 0
        return l

# lambda, distance
def cov_2d_model(l, d):
    # normalised for regression coeff
    return np.where(d<1e-12, 1, d * l * special.kv(1, d * l))

def cov_3d_model(l, d):
    return np.exp(-l * d)

# X the matrix of data to regress against, y the data to regress
# returns regression coefficients & fit
def regress(X, y):
    proj = np.linalg.pinv(X.T @ X) @ X.T
    params = proj @ y
    return params, X @ params

def auto_regress(y, delay):
    X = np.array([y[i:-delay + i] for i in range(delay)])
    return regress(X.T, y[delay:])

def auto_regress_long(y, delays):
    X = np.array([y[i : -np.max(delays) - 1 + i] for i in delays])
    return regress(X.T, y[np.max(delays) + 1:])

# temp autocorr model
w = 2 * np.pi / 365.25
def seasonal_autocorr_model(params, t):
    return params[3] + (1 - params[0] - params[1] - params[3]) * np.exp(-params[2] * t) + params[0] * np.cos(w * t) + params[1] * np.cos(2 * w * t)

# get unbiased autocorrelation estimates up to delay_max
def autocorr(y, delay_max):
    var = np.var(y)
    n = len(y)
    y_centered = y - np.mean(y)
    C = np.zeros(delay_max + 1)
    C[0] = 1
    C[1:] = [np.sum(y_centered[:-lag] * y_centered[lag:]) / ((n - lag) * var) for lag in range(1, delay_max+1)] 
    return C

# O(n^2) Toeplitz inversion to get autoregression coefficients in O(n^2) from autocorr vector
def gohberg_inverse(autocorr):
    e1 = np.zeros(len(autocorr))
    e1[0] = 1
    en = np.zeros(len(autocorr))
    en[0] = -1

    u = solve_toeplitz(autocorr, e1)
    v = solve_toeplitz(autocorr, en)

    inv = np.outer(u, v) - np.outer(v[::-1], u[::-1])
    return inv[0]

# A the covariance matrix, or with inv=True, A the precision matrix (inverse covariance matrix)
def regression_coefficients(A, inv=False):
    if not inv:
        A = np.linalg.inv(A)
    return - A / np.diagonal(A)[:, np.newaxis]

# A the covariance matrix, or with inv=True, A the precision matrix (inverse covariance matrix)
def stochastic_adjacency_matrix(A, inv=False):
    B = partial_corr(A, inv)
    np.fill_diagonal(B, 0)
    np.fill_diagonal(B, 1 - np.sum(B, axis=0))

# D the distance matrix
# L the sensor locations
# var the sensor variances vector
# corr the correlation vs distance function
# dist the distance function between two locations in the space
def kriging_interpolator(D, L, var, corr, dist):
    krig_matrix = np.zeros([temps.shape[0] + 1, temps.shape[0] + 1])
    corr_matrix = corr(D)
    cov_matrix = corr_matrix * np.outer(np.sqrt(var), np.sqrt(var))
    # assumed variance of the location
    model_var = np.mean(var)
    
    krig_matrix = np.block([[corr_matrix, np.ones([D.shape[0], 1])], [np.ones([1, D.shape[0]]), np.zeros([1, 1])]])
    krig_inv = np.linalg.inv(krig_matrix)

    coeffs = lambda loc: np.array([corr(dist(loc, l)) for l in L])
    weights = lambda loc: krig_inv @ np.block([coeffs(loc), np.ones(1)])

    # takes in a location and returns interpolation, and its variance
    def interpolate(loc):
        interpolation[-1-j, i] = np.dot(w[:-1], day_temp)

        c = np.array([coeffs(loc) * np.sqrt(var) * np.sqrt(model_var)])
        var_block = np.block([[cov_matrix, c.T], [c, np.array([[model_var]])]])
        var_vec = np.block([weights(loc)[:-1], np.array([1])])
        
        return np.dot(weights(loc)[:-1], day_temp), np.sqrt(var_vec.T @ var_block @ var_vec)
    
    return interpolate
