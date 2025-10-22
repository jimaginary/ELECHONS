import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from elechons.processing import edges as ec
from elechons.data import station_handler as sh

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')

r.init('mean')
dist = ec.distance_matrix(sh.STATIONS)
lat = sh.STATIONS['lat'].to_numpy()
long = sh.STATIONS['long'].to_numpy()

N = r.temps_mean_sin_adj.shape[0]
p = 2
init = np.hstack([np.eye(N) for _ in range(p)])
pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, init, alpha = 0.005)
print(f'--- var {p} l0 model')
print(f'alpha={0.005:.4f}')
print_info(pred)

E = pred.residuals()
Ecov = np.cov(E)
Epr = np.linalg.inv(Ecov)

plt.scatter(dist.flatten(), Ecov.flatten())
plt.show()

r = Ecov / np.diag(Ecov).reshape(-1, 1)

plt.scatter(dist.flatten(), r.flatten())
plt.show()

Ecov_scaled = Ecov / np.sum(Ecov, axis=1, keepdims=True)

plt.scatter(dist.flatten(), Ecov_scaled.flatten())
plt.show()

plt.scatter(Ecov.flatten(), Epr.flatten())
plt.show()

plt.scatter(r.flatten(), Epr.flatten())
plt.show()
