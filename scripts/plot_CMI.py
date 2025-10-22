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
CMI = pred.CMI()

Ecov = np.cov(pred.residuals())
Ecov_scaled = Ecov / np.sum(Ecov, axis=1, keepdims=True)
plt.scatter(ec.closeness_matrix(sh.STATIONS,1000,8).flatten(), CMI.flatten())
plt.colorbar()
plt.show()

plt.imshow(CMI)
plt.colorbar()
plt.show()

# drank = dist.argsort(axis=1).argsort(axis=1)
# ranksym = (drank + drank.T)/2
plt.scatter(dist.flatten(), CMI.flatten())
plt.colorbar()
# plt.show()
plt.savefig('plts/CMI_v_dist.png', dpi=300, bbox_inches='tight')

segments = [((long[i],lat[i]),(long[j],lat[j])) for i in range(N) for j in range(N)]

base_cmap = cm.get_cmap('viridis')
colors_with_alpha = base_cmap(np.linspace(0, 1, 256))
colors_with_alpha[:, -1] = np.linspace(0, 1, 256)  # alpha goes from 0 → 1
transparent_cmap = colors.ListedColormap(colors_with_alpha)

lc = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2,
    norm=plt.Normalize(0, 0.4)
)

lc.set_array(CMI.flatten())
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.scatter(long, lat, color='black', zorder=3, s=10)
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('Pairwise value')

ax.set_xlim(long.min() - 1, long.max() + 1)
ax.set_ylim(lat.min() - 1, lat.max() + 1)
ax.set_aspect('equal')
# plt.show()
plt.savefig('plts/CMI_edges.png', dpi=300, bbox_inches='tight')


# r.init('mean')
# CMI = lr.CMI(r.temps_mean_sin_adj, 2)
# CMI_list = CMI.flatten()

# from matplotlib.colors import LogNorm
# plt.imshow(CMI, norm=LogNorm(vmin=np.percentile(CMI_list,75), vmax=CMI.max()))
# plt.colorbar()
# plt.show()

# p = 25
# L = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, CMI < np.percentile(CMI_list, p))
# init = L.param_history
# print(f'finished var 2 model CMI threshhold of {p}% of params with CMI, rmse {L.rmse()}')

# for i in range(1,20):
#     pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, 2, init, alpha = 0.001*i)
#     print(f'--- var 2 l0 model')
#     print(f'alpha={0.001*i:.4f}')
#     print_info(pred)