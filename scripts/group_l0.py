import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import elechons.processing.edges as edges
import elechons.data.station_handler as sh

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')

r.init('mean')

# group l0
for i in range(1,5):
    init = lr.VAR(r.temps_mean_sin_adj, 2).param_history.T
    B = np.empty_like(init)
    B[:,0::2] = init[:,:init.shape[0]]
    B[:,1::2] = init[:,init.shape[0]:]
    pred = lr.VAR_group_l0(r.temps_mean_sin_adj, 2, B, alpha=(0.05*i), threshhold=100)
    print(f'--- var 2 l={(0.05*i)} group lasso model\n')
    print_info(pred)
