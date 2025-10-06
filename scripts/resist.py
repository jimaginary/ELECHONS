import elechons.regress_temp as r
import matplotlib.pyplot as plt
import elechons.data.station_handler as sh
import elechons.processing.edges as ec
import numpy as np

r.init('mean')

dist = ec.distance_matrix(sh.STATIONS)
v = np.diag(np.var(r.temps_mean_sin_adj, axis=1))

G = np.reciprocal(dist)
# G = np.exp(-0.0019 * dist)
np.fill_diagonal(G, 0)
L = np.diag(np.sum(G,axis=1)) - G

l = 0.01
Linv = np.linalg.inv(L + l * np.eye(L.shape[0]))
C = Linv @ v @ Linv
d = np.diag(C)

Req = C / np.sqrt(np.outer(d,d))

Corrs = np.corrcoef(r.temps_mean_sin_adj)

plt.scatter(Req.flatten(), Corrs.flatten())
plt.title(f'l={l}')
plt.show()