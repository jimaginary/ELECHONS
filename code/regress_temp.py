import station_handler as sh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats

temps = sh.get_series_matrix('max')
valid = ~sh.get_was_nan_matrix('max')
days = np.arange(temps.shape[1])

# for each series, we want to fit on non-nans a function of form a+bsin(2pi*t/365+phi)
# params are b, phi in order
w = 2*np.pi/365
def sin_model(params, t):
    return params[0]*np.sin(w*t + params[1])

def cost_function(params, t, y, v):
    y_pred = sin_model(params, t)
    return np.sum(np.pow(y - y_pred, 2)*v)

# def jac(params, t, y, v):
#     de_dphi = np.sum(2*v*params[0]*np.cos(w*t+params[1])*(params[0]*np.sin(w*t+params[1])-y))
#     de_db = np.sum(2*v*(params[0]*np.pow(np.sin(w*t+params[1]),2)-2*y*np.sin(w*t+params[1])))
#     return np.array([de_db, de_dphi])

# incorporate average component
means = np.sum(temps*valid, axis=1) / np.sum(valid, axis=1)
temps_mean_adj = temps - np.tile(means, (temps.shape[1], 1)).T

params = np.zeros((temps.shape[0], 2))
for i in range(temps.shape[0]):
    min_obj = minimize(cost_function, np.array([0, 0]), args=(days, temps_mean_adj[i], valid[i]), method='Nelder-Mead')
    if not min_obj.success:
        print('Got a minimisation failure!')
    params[i] = min_obj.x

temps_mean_sin_adj = temps_mean_adj - np.array([sin_model(params[i], days) for i in range(temps.shape[0])])
fig, ax = plt.subplots()
# ax.plot(temps_mean_sin_adj[1])
stats.probplot(temps_mean_sin_adj[2], plot=ax)


plt.show()


