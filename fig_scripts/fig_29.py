import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import elechons.processing.edges as edges
import elechons.data.station_handler as sh

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')

r.init('mean')

partial_corr = np.linalg.inv(np.corrcoef(r.regression_error))

dist = edges.distance_matrix(sh.STATIONS)

N = r.temps_mean_sin_adj.shape[0]

print()
print('results of VAR-initialised l0 coordinate descent')

p = 2
L = lr.VAR(r.temps_mean_sin_adj, p)
init = L.param_history
print(f'finished initialising, rmse {L.rmse()}')
pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, init, alpha = 0.005)
print(f'--- var {p} l0 model')
print(f'alpha={0.005:.4f}')
print_info(pred)