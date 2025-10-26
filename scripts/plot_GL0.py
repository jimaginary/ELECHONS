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
pred = lr.VAR(r.temps_mean_sin_adj, 2)

n = r.temps_mean_sin_adj.shape[1]
S = np.cov(pred.residuals())

Pr = np.linalg.inv(S)
L0_Pr = lr.FastGL0(S, n)
print(np.sum(L0_Pr != 0)/np.prod(L0_Pr.shape))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

im0 = axes[0].imshow(Pr, cmap='viridis', vmin=-0.2, vmax=0.2)
axes[0].set_title('S^{-1}')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(L0_Pr, cmap='viridis', vmin=-0.2, vmax=0.2)
axes[1].set_title('Omega')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('plts/VAR_Gl0_err_pr.png', dpi=300, bbox_inches='tight')