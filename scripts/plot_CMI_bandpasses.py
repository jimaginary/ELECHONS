import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from elechons.processing import edges as ec
from elechons.data import station_handler as sh
from matplotlib.colors import Normalize
import elechons.config as config

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')

r.init('mean')
dist = ec.distance_matrix(sh.STATIONS)
lat = sh.STATIONS['lat'].to_numpy()
long = sh.STATIONS['long'].to_numpy()

N = r.temps_mean_sin_adj.shape[0]
p = 2
alpha = 0.005
if config.DATASET == 'noaa':
    alpha = 0.0125
pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, lr.VAR(r.temps_mean_sin_adj, p).param_history, alpha = alpha)
print(f'--- var {p} l0 model')
print(f'alpha={0.005:.4f}')
print_info(pred)

CMI = []
bands = [(0, np.pi/4), (np.pi/4, 2*np.pi/4), (2*np.pi/4, 3*np.pi/4), (3*np.pi/4, np.pi)]
for omega0, omega1 in bands:
    CMI_vl0_el0 = pred.CMI(l0=True, omega0=omega0, omega1=omega1)
    np.fill_diagonal(CMI_vl0_el0, np.nan)
    CMI.append(CMI_vl0_el0)

# --- compare ---
x = CMI[0].flatten()
y = CMI[3].flatten()

high = min(np.nanmax(x), np.nanmax(y))

# Plot y = x line
plt.plot([0, high], [0, high], 'r-', zorder=1)

plt.scatter(x, y, zorder=0)

plt.title('(0,π/4) v (3π/4,π) CMI')
plt.xlabel('CMI (0,π/4)')
plt.ylabel('CMI (3π/4,π)')

plt.savefig(f'plts/CMI_comparison.png', dpi=300, bbox_inches='tight')

# --- mapped ---

# k nearest neighbours visible
k = 12
p = np.nanpercentile(
    np.hstack([CMI[i] for i in range(0,4)]), 100 * (1 - k / (4 * (N - 1)))
)
# cut off at top 1% of those...
m = np.nanpercentile(
    np.hstack([CMI[i] for i in range(0,4)]), 100 * (1 - 0.01*k / (4 * (N - 1)))
)
print(p, m, p / m)

segments = [((long[i],lat[i]),(long[j],lat[j])) for i in range(N) for j in range(N)]

base_cmap = cm.get_cmap('viridis')
colors_with_alpha = base_cmap(np.linspace(0, 1, 256))
colors_with_alpha[:, -1] = np.hstack((np.linspace(0, 0, int(256 * p / m)), np.linspace(0, 1, 256 - int(256 * p / m))))  # alpha goes from 0 → 1
transparent_cmap = colors.ListedColormap(colors_with_alpha)

lc00 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2,
    norm = Normalize(vmin=0, vmax=m)
)

lc01 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2,
    norm = Normalize(vmin=0, vmax=m)
)

lc10 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2,
    norm = Normalize(vmin=0, vmax=m)
)

lc11 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2,
    norm = Normalize(vmin=0, vmax=m)
)

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

lc00.set_array(CMI[0].flatten())
axes[0,0].add_collection(lc00)
axes[0,0].scatter(long, lat, color='black', zorder=3, s=10)
axes[0,0].set_title('l0 CMI (0,π/4)')
axes[0,0].set_xlabel('long (deg)')
axes[0,0].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc00, ax=axes[0,0])
cbar.set_label('CMI')

axes[0,0].set_xlim(long.min() - 1, long.max() + 1)
axes[0,0].set_ylim(lat.min() - 1, lat.max() + 1)
axes[0,0].set_aspect('equal')

lc01.set_array(CMI[1].flatten())
axes[0,1].add_collection(lc01)
axes[0,1].scatter(long, lat, color='black', zorder=3, s=10)
axes[0,1].set_title('l0 CMI (π/4,π/2)')
axes[0,1].set_xlabel('long (deg)')
axes[0,1].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc01, ax=axes[0,1])
cbar.set_label('CMI')

axes[0,1].set_xlim(long.min() - 1, long.max() + 1)
axes[0,1].set_ylim(lat.min() - 1, lat.max() + 1)
axes[0,1].set_aspect('equal')

lc10.set_array(CMI[2].flatten())
axes[1,0].add_collection(lc10)
axes[1,0].scatter(long, lat, color='black', zorder=3, s=10)
axes[1,0].set_title('l0 CMI (π/2,3π/4)')
axes[1,0].set_xlabel('long (deg)')
axes[1,0].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc10, ax=axes[1,0])
cbar.set_label('CMI')

axes[1,0].set_xlim(long.min() - 1, long.max() + 1)
axes[1,0].set_ylim(lat.min() - 1, lat.max() + 1)
axes[1,0].set_aspect('equal')

lc11.set_array(CMI[3].flatten())
axes[1,1].add_collection(lc11)
axes[1,1].scatter(long, lat, color='black', zorder=3, s=10)
axes[1,1].set_title('l0 CMI (3π/4,π)')
axes[1,1].set_xlabel('long (deg)')
axes[1,1].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc11, ax=axes[1,1])
cbar.set_label('CMI')

axes[1,1].set_xlim(long.min() - 1, long.max() + 1)
axes[1,1].set_ylim(lat.min() - 1, lat.max() + 1)
axes[1,1].set_aspect('equal')

plt.tight_layout()
plt.savefig(f'plts/CMI_bandpasses.png', dpi=300, bbox_inches='tight')
