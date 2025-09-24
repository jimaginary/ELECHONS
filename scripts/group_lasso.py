import elechons.models.group_lasso as gl
import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import elechons.processing.edges as edges
import elechons.data.station_handler as sh

def print_info(pred, fname):
    print(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}')
    print(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}')
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f'rmse: {pred.rmse():.4f}, rmse (%): {pred.rmse_percent():.2f}\n')
        f.write(f'BICa: {pred.BIC():.2f}, num params: {pred.num_params}\n')

r.init('mean')

# pred = gl.gLasso_space(r.temps_mean_sin_adj, 2, l1=0, l2=(0.08))
# with open("output_space.txt", "a", encoding="utf-8") as f:
#     f.write(f'--- var {2} l={(0.08)} group lasso model\n')
# print_info(pred, 'output_space.txt')

# gLasso space
# for i in range(2,5):
#     for j in range(2, 3):
#         pred = gl.gLasso_space(r.temps_mean_sin_adj, j, l1=0, l2=(0.05*i))
#         with open("output_space.txt", "a", encoding="utf-8") as f:
#             f.write(f'--- var {j} l={(0.05*i)} group lasso model\n')
#         print_info(pred, 'output_space.txt')

# gLasso time
for i in range(0,7):
    for j in range(4, 5):
        pred = gl.gLasso_time(r.temps_mean_sin_adj, j, l1=0, l2=(2.7+0.3*i))
        with open("output_time.txt", "a", encoding="utf-8") as f:
            f.write(f'--- var {j} l={(2.7+0.3*i)} group lasso model\n')
        print_info(pred, 'output_time.txt')
