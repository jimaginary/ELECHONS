import numpy as np

# data the sample
# var the variances at each sample
# pos the position vectors of the samples
# corr the correlation function between position vectors
def unbiased_krige(data, var, pos, corr):

    krig_matrix = np.zeros([pos.shape[0] + 1, pos.shape[0] + 1])
    corr_matrix = np.array([[corr(pos[i], pos[j]) for i in pos] for j in pos])

    cov_matrix = corr_matrix * np.outer(np.sqrt(var), np.sqrt(var))
    mean_var = np.mean(var)
    
    krig_matrix = np.block([[corr_matrix, np.ones([pos.shape[0], 1])], [np.ones([1, pos.shape[0]]), np.zeros([1, 1])]])
    krig_inv = np.linalg.inv(krig_matrix)

    def corrs(interpolate_pos):
        np.array([corr(pos[i], interpolate_pos) for i in range(pos.shape[0])])

    def interpolate(interpolate_pos):
        w = krig_inv @ np.block([corrs(interpolate_pos), np.ones(1)])
        return np.dot(w[:-1], data)
    
    def krig_std_err(interpolate_pos):
        corrs_vec = corrs(interpolate_pos)
        w = krig_inv @ np.block([corrs_vec, np.ones(1)])
        c = corrs_vec * np.sqrt(var) * np.sqrt(mean_var)
        var_block = np.block([[cov_matrix, c.T], [c, np.array([[mean_var]])]])
        var_vec = np.block([w[:-1], np.array([-1])])
        return np.sqrt(var_vec.T @ var_block @ var_vec)

    return interpolate, krig_std_err

    # corrs = lambda lat1, long1: np.array([s.cov_3d_model(l, ec.earth_distance(lat1, long1, lat[i], long[i])) for i in range(temps.shape[0])])
    # weights = lambda lat1, long1: krig_inv @ np.block([corrs(lat1, long1), np.ones(1)])

    # day_temp = temps[:, t]

    # la_left = -45
    # la_right = -10
    # lo_left = 110
    # lo_right = 155
    # lats = np.linspace(la_left, la_right, (la_right - la_left)*4)
    # longs = np.linspace(lo_left, lo_right, (lo_right - lo_left)*4)
    
    # interpolation = np.zeros([len(lats), len(longs)])
    # interpolation_variance = np.zeros([len(lats), len(longs)])

    # for i, lo in enumerate(longs):
    #     if i % 10 == 0:
    #         print(f'i={i}')
    #     for j, la in enumerate(lats):
    #         w = weights(la, lo)
            
    #         interpolation[-1-j, i] = np.dot(w[:-1], day_temp)

    #         c = np.array([corrs(la, lo) * np.sqrt(var) * np.sqrt(mean_var)])
    #         var_block = np.block([[cov_matrix, c.T], [c, np.array([[mean_var]])]])
    #         var_vec = np.block([w[:-1], np.array([-1])])
    #         interpolation_variance[-1-j, i] = np.sqrt(var_vec.T @ var_block @ var_vec)
    