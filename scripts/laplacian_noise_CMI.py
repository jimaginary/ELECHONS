import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from elechons.processing import edges as ec
from elechons.data import station_handler as sh
import elechons.config as config
from elechons.models.laplacian_noise import two_D_cov

result, dist, single_to_tuple = two_D_cov(n=10, samples=10000)
N = result.shape[0]
filt1d = np.random.rand(N) < 0.3
argfilt1d = np.where(filt1d)[0]
result = result[filt1d]
prEl0 = lr.FastGL0(np.cov(result), result.shape[1])

R = - prEl0 / np.sqrt(np.outer(prEl0.diagonal(), prEl0.diagonal()))

CMI = - np.log(1 - np.pow(np.abs(R),2))

segments = [(single_to_tuple(i),single_to_tuple(j)) for i in argfilt1d for j in argfilt1d]
coords = np.array([single_to_tuple(i) for i in argfilt1d])

base_cmap = cm.get_cmap('viridis')
colors_with_alpha = base_cmap(np.linspace(0, 1, 256))
colors_with_alpha[:, -1] = np.linspace(0, 1, 256)  # alpha goes from 0 → 1
transparent_cmap = colors.ListedColormap(colors_with_alpha)

lc = LineCollection(
    segments,
    cmap=transparent_cmap,
    linewidths=2
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

lc.set_array(CMI.flatten())
axes[0].add_collection(lc)
axes[0].scatter(coords[:,0], coords[:,1], color='black', zorder=3, s=10)
axes[0].set_title('CMI with Sample Precision')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
cbar = plt.colorbar(lc, ax=axes[0])
cbar.set_label('CMI')

axes[1].scatter(dist.flatten()[np.outer(filt1d, filt1d).flatten()], np.corrcoef(result).flatten())
axes[1].set_title('Correlation vs distance')
axes[1].set_xlabel('dist')
axes[1].set_ylabel('correlation')

plt.tight_layout()
plt.savefig(f'plts/laplacian_noise_CMI.png', dpi=300, bbox_inches='tight')