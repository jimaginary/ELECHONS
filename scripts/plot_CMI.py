import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import matplotlib.pyplot as plt

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')

r.init('mean')
CMI = lr.CMI(r.temps_mean_sin_adj, 2)
CMI_list = CMI.flatten()

# from matplotlib.colors import LogNorm
# plt.imshow(CMI, norm=LogNorm(vmin=np.percentile(CMI_list,75), vmax=CMI.max()))
# plt.colorbar()
# plt.show()

p = 25
L = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, CMI < np.percentile(CMI_list, p))
init = L.param_history
print(f'finished var 2 model CMI threshhold of {p}% of params with CMI, rmse {L.rmse()}')

for i in range(1,20):
    pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, 2, init, alpha = 0.001*i)
    print(f'--- var 2 l0 model')
    print(f'alpha={0.001*i:.4f}')
    print_info(pred)