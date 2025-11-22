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
print('results of distance-masked regression')

for i in range(6,9):
    mask = dist < 200*i
    mask = np.hstack([mask, mask])
    pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
    print(f'--- var 2 dist < {200*i}km mask model')
    print_info(pred)

print()
print('results of kNN masked model')

for i in [38,40,42]:
    mask = (edges.K_nearest(dist, i, undirected=False).T == 1)
    mask = np.hstack([mask, mask])
    pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
    print(f'--- var 2 {i}NN mask model')
    print_info(pred)

print()
print('partial correlation percentile mask model')

for i in range(5,8):
    mask = (partial_corr > np.percentile(partial_corr, 100-i*5)) + (np.eye(N) == 1)
    mask = np.hstack([mask, mask])
    pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
    print(f'--- var 2 {i*5}% partial corr mask model')
    print_info(pred)

print()
print('lasso refitter model')

for i in range(3, 6):
    pred = lr.VAR_lasso(r.temps_mean_sin_adj, 2, 0.02*i)
    mask = (pred.param_history != 0).T
    pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
    print(f'--- var 2 l={0.02*i:.2f} lasso refitter model ---')
    print_info(pred)