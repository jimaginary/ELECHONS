import elechons.models.group_lasso as gl
import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import elechons.processing.edges as edges
import elechons.data.station_handler as sh

def print_info(pred):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}\n')
        f.write(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}\n')

r.init('mean')

# for i in range(1,10):
#     pred = gl.gLasso(r.temps_mean_sin_adj, i, l1=0.12, l2=0.04/np.sqrt(i))
#     print(f'--- var {i} group lasso model')
#     # print(f'alpha={0.001*i:.4f}')
#     print_info(pred)
#     print(pred.param_history)

# for i in range(1,10):
#     pred = gl.gLasso(r.temps_mean_sin_adj, 2, l1=0, l2=1.0*i)
#     print(f'--- var l2={0.05*i} lasso model')
#     # print(f'alpha={0.001*i:.4f}')
#     print_info(pred)
#     print(pred.param_history)

for i in range(0,5):
    for j in range(2, 10):
        pred = gl.gLasso(r.temps_mean_sin_adj, j, l1=0, l2=(1.0*i))
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(f'--- var {j} l={(0.25+0.05*i)} group lasso model\n')
        print_info(pred)
