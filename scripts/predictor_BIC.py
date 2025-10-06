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

for i in range(1,10):
    p = 2
    L = lr.VAR(r.temps_mean_sin_adj, p)
    # L = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, i, np.hstack([dist < 1200 for _ in range(i)]))
    init = L.param_history.T
    print(f'finished initialising with dist, rmse {L.rmse()}')
    pred = lr.VAR_l0_coord_descent(r.temps_mean_sin_adj, p, init, alpha = 0.001*i)
    print(f'--- var {p} l0 model')
    print(f'alpha={0.001*i:.4f}')
    print_info(pred)

print()

# for i in range(1, 6):
#     pred = lr.VAR_lasso(r.temps_mean_sin_adj, 2, 0.02*i)
# #     pred = lr.VAR(r.temps_mean_sin_adj, 2, percentile=100-i*5)
#     mask = (pred.param_history != 0).T
#     pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
#     print(f'--- var 2 l={0.02*i:.2f} lasso refitter model ---')
#     print_info(pred)

# print()

# for i in range(1,10):
#     mask = (partial_corr > np.percentile(partial_corr, 100-i*5)) + (np.eye(N) == 1)
#     mask = np.hstack([mask, mask])
#     pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
#     print(f'--- var 2 {i*5}% partial corr mask model')
#     print_info(pred)

# print()

# for i in range(1,10):
#     mask = (edges.K_nearest(dist, i, undirected=False).T == 1)
#     mask = np.hstack([mask, mask])
#     pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
#     print(f'--- var 2 {i}NN mask model')
#     print_info(pred)

# print()

# for i in range(1,10):
#     mask = dist < 200*i
#     mask = np.hstack([mask, mask])
#     pred = lr.solve_VAR_with_mask(r.temps_mean_sin_adj, 2, mask)
#     print(f'--- var 2 dist < {200*i}km mask model')
#     print_info(pred)

# print()

# for i in range(1,6):
#     pred = lr.VAR_lasso(r.temps_mean_sin_adj, 2, 0.02*i)
#     print(f'--- var 2 lasso ({0.02*i:.2f}) model ---')
#     print_info(pred)

# print()

# for i in range(1,6):
#     pred = lr.VAR(r.temps_mean_sin_adj, i)
#     print(f'--- var {i} model ---')
#     print_info(pred)

# print()

# for i in range(0, 100, 10):
#     pred = lr.VAR(r.temps_mean_sin_adj, 2, percentile=i)
#     print(f'--- var 2 {i}% entries model ---')
#     print_info(pred)

# print()

# for i in range(1,6):
#     pred = lr.fast_var_sgd(r.temps_mean_sin_adj, i)
#     print(f'--- var {i} sgd model ---')
#     print_info(pred)

