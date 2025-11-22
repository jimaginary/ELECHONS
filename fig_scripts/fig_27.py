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
print('results of Lasso regression')

for i in range(1,6):
    pred = lr.VAR_lasso(r.temps_mean_sin_adj, 2, 0.02*i)
    print(f'--- var 2 lasso ({0.02*i:.2f}) model ---')
    print_info(pred)