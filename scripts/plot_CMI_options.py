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
pred1 = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, init, alpha = 0.005)
print(f'--- var {p} l0 model')
print(f'alpha={0.005:.4f}')
print_info(pred1)
pred2 = lr.VAR(r.temps_mean_sin_adj, p)

CMI_vl0_eNotl0 = pred1.CMI(l0=False)
np.fill_diagonal(CMI_vl0_eNotl0, np.nan)
print(f'CMI_vl0_eNotl0', np.sum(CMI_vl0_eNotl0 != 0)/np.prod(CMI_vl0_eNotl0.shape))
CMI_vl0_el0 = pred1.CMI(l0=True)
np.fill_diagonal(CMI_vl0_el0, np.nan)
print(f'CMI_vl0_el0', np.sum(CMI_vl0_el0 != 0)/np.prod(CMI_vl0_el0.shape))

CMI_vNotl0_eNotl0 = pred2.CMI(l0=False)
np.fill_diagonal(CMI_vNotl0_eNotl0, np.nan)
print(f'CMI_vNotl0_eNotl0', np.sum(CMI_vNotl0_eNotl0 != 0)/np.prod(CMI_vNotl0_eNotl0.shape))
CMI_vNotl0_el0 = pred2.CMI(l0=True)
np.fill_diagonal(CMI_vNotl0_el0, np.nan)
print(f'CMI_vNotl0_el0', np.sum(CMI_vNotl0_el0 != 0)/np.prod(CMI_vNotl0_el0.shape))

fig, axes = plt.subplots(2, 2, figsize=(8, 4))

im0 = axes[0,0].imshow(CMI_vNotl0_eNotl0, cmap='viridis', vmin=0, vmax=0.2)
axes[0,0].set_title('CMI with VAR and Sample Precision')
plt.colorbar(im0, ax=axes[0,0])

im1 = axes[0,1].imshow(CMI_vNotl0_el0, cmap='viridis', vmin=0, vmax=0.2)
axes[0,1].set_title('CMI with VAR and l0 Precision')
plt.colorbar(im1, ax=axes[0,1])

im0 = axes[1,0].imshow(CMI_vl0_eNotl0, cmap='viridis', vmin=0, vmax=0.2)
axes[1,0].set_title('CMI with l0 VAR and Sample Precision')
plt.colorbar(im0, ax=axes[1,0])

im1 = axes[1,1].imshow(CMI_vl0_el0, cmap='viridis', vmin=0, vmax=0.2)
axes[1,1].set_title('CMI with l0 VAR and l0 Precision')
plt.colorbar(im1, ax=axes[1,1])

plt.tight_layout()
plt.savefig(f'plts/CMI_options.png', dpi=300, bbox_inches='tight')

# --- mapped ---

segments = [((long[i],lat[i]),(long[j],lat[j])) for i in range(N) for j in range(N)]

base_cmap = cm.get_cmap('viridis')
colors_with_alpha = base_cmap(np.linspace(0, 1, 256))
colors_with_alpha[:, -1] = np.linspace(0, 1, 256)  # alpha goes from 0 → 1
transparent_cmap = colors.ListedColormap(colors_with_alpha)

lc00 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2
)

lc01 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2
)

lc10 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2
)

lc11 = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2
)

fig, axes = plt.subplots(2, 2, figsize=(8, 4))

lc00.set_array(CMI_vNotl0_eNotl0.flatten())
axes[0,0].add_collection(lc00)
axes[0,0].scatter(long, lat, color='black', zorder=3, s=10)
axes[0,0].set_title('CMI with VAR and Sample Precision')
axes[0,0].set_xlabel('long (deg)')
axes[0,0].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc00, ax=axes[0,0])
cbar.set_label('CMI')

axes[0,0].set_xlim(long.min() - 1, long.max() + 1)
axes[0,0].set_ylim(lat.min() - 1, lat.max() + 1)
axes[0,0].set_aspect('equal')

lc01.set_array(CMI_vNotl0_el0.flatten())
axes[0,1].add_collection(lc01)
axes[0,1].scatter(long, lat, color='black', zorder=3, s=10)
axes[0,1].set_title('CMI with VAR and l0 Precision')
axes[0,1].set_xlabel('long (deg)')
axes[0,1].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc01, ax=axes[0,1])
cbar.set_label('CMI')

axes[0,1].set_xlim(long.min() - 1, long.max() + 1)
axes[0,1].set_ylim(lat.min() - 1, lat.max() + 1)
axes[0,1].set_aspect('equal')

lc10.set_array(CMI_vl0_eNotl0.flatten())
axes[1,0].add_collection(lc10)
axes[1,0].scatter(long, lat, color='black', zorder=3, s=10)
axes[1,0].set_title('CMI with l0 VAR and Sample Precision')
axes[1,0].set_xlabel('long (deg)')
axes[1,0].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc10, ax=axes[1,0])
cbar.set_label('CMI')

axes[1,0].set_xlim(long.min() - 1, long.max() + 1)
axes[1,0].set_ylim(lat.min() - 1, lat.max() + 1)
axes[1,0].set_aspect('equal')

lc11.set_array(CMI_vl0_el0.flatten())
axes[1,1].add_collection(lc11)
axes[1,1].scatter(long, lat, color='black', zorder=3, s=10)
axes[1,1].set_title('CMI with l0 VAR and l0 Precision')
axes[1,1].set_xlabel('long (deg)')
axes[1,1].set_ylabel('lat (deg)')
cbar = plt.colorbar(lc11, ax=axes[1,1])
cbar.set_label('CMI')

axes[1,1].set_xlim(long.min() - 1, long.max() + 1)
axes[1,1].set_ylim(lat.min() - 1, lat.max() + 1)
axes[1,1].set_aspect('equal')

plt.tight_layout()
plt.savefig(f'plts/CMI_edges_options.png', dpi=300, bbox_inches='tight')
