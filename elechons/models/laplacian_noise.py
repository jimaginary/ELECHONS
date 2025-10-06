import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

def one_D_cov(length = 1000, samples = 5000, l=0.5, g=1, mean_of_means=10, var_of_means=2):
    dist = lambda x,y: min(np.abs(x-y), length - np.abs(x-y))
    model = lambda r: (1 + l*r)*np.exp(-l*r)#(g**2 / (4 * l ** 3)) * (1 + l*r)*np.exp(-l*r)
    mean_model = g * mean_of_means / l ** 2

    dists = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            dists[i, j] = dist(i, j)

    main_diag = -2 * np.ones(length)
    off_diag = 1 * np.ones(length - 1)

    # loop laplacian
    L = (np.diag(main_diag) +
        np.diag(off_diag, k=1) +
        np.diag(off_diag, k=-1))
    L[0,-1] = 1
    L[-1, 0] = 1

    screened_poisson = L - l**2 * np.eye(length)

    solution = np.linalg.inv(screened_poisson)

    stim_means = np.random.normal(loc=mean_of_means, scale=var_of_means, size=(length))
    stim = -g * (np.random.normal(size=(length, samples)) + stim_means[:, np.newaxis])

    result = solution @ stim

    cov = np.corrcoef(result)

    dist_range = np.arange(length // 2)

    covs_by_dist = [cov.flatten()[dists.flatten() == k] for k in dist_range]
    cov_means = np.array([np.average(covs_by_dist[i]) for i in dist_range])
    std = np.array([np.std(covs_by_dist[i]) for i in dist_range])
    
    plt.plot(dist_range, cov_means, 'b', zorder=0, label="empirical mean")
    plt.plot(dist_range, cov_means + 1.96*std, 'k--', zorder=0, label="empirical 95% CI")
    plt.plot(dist_range, cov_means - 1.96*std, 'k--', zorder=0)
    plt.plot(dist_range, model(dist_range), 'r', zorder=5, label="model prediction")

    plt.title(f"correlation v dist, λ={l}, γ={g}")
    plt.xlabel("dist")
    plt.ylabel("correlation")
    plt.legend()
    
    plt.savefig(f'../plts/laplacian_noise_modelling/one_D_cov_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()

    res_means = np.mean(result, axis=1)
    plt.scatter(stim_means, res_means, s=1, c='b', zorder=0, label='empirical result')
    plt.axhline(mean_model, c='r', linestyle='--', zorder=5, label='model prediction')
    plt.xlabel('stimulus mean')
    plt.ylabel('output mean')
    plt.title('mean response vs mean input (γ adjusted)')
    plt.ylim([mean_model - (np.max(res_means) - np.min(res_means)), mean_model + (np.max(res_means) - np.min(res_means))])
    plt.legend()
    plt.savefig(f'../plts/laplacian_noise_modelling/one_D_mean_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()

def two_D_cov(n = 50, samples = 5000, l=0.5, g=1):
    tuple_to_single = lambda t: t[0]*n + t[1]
    single_to_tuple = lambda i: (i // n, i % n)
    # model = lambda r: (g**2 / (4 * np.pi * l ** 2)) * (l * r * special.kv(1, l * r) - special.kv(0, l * r))
    # model = lambda r: (g**2 / (4 * np.pi * l ** 2)) * (1 + l * r) * np.exp(-l * r)
    model = lambda r: l * r * special.kv(1, l * r) #(r * g ** 2 / (4 * np.pi * l)) * special.kv(1, l * r)

    def singles_to_dist(i, j):
        t1 = single_to_tuple(i)
        t2 = single_to_tuple(j)
        dx = min(abs(t1[0] - t2[0]), n - abs(t1[0] - t2[0]))
        dy = min(abs(t1[1] - t2[1]), n - abs(t1[1] - t2[1]))
        return np.sqrt(np.pow(dx, 2) + np.pow(dy, 2))

    def edge_locations(v):
        i, j = single_to_tuple(v)
        tuples = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
        return np.array([tuple_to_single(t) for t in tuples])

    dist = np.zeros((n**2, n**2))
    L = np.zeros((n**2, n**2))
    for i in range(n ** 2):
        L[i][edge_locations(i)] = 1
        L[i, i] = - np.sum(L[i])
        for j in range(n ** 2):
            dist[i,j] = singles_to_dist(i, j)
    
    screened_poisson = L - l ** 2 * np.eye(n ** 2)
    solution = np.linalg.inv(screened_poisson)
    stim = -g * np.random.normal(size=(n ** 2, samples))
    result = solution @ stim
    cov = np.corrcoef(result)

    dist_set = np.sort(np.unique(dist))

    covs_by_dist = [cov.flatten()[dist.flatten() == k] for k in dist_set]
    means = np.array([np.average(covs_by_dist[i]) for i in range(len(dist_set))])
    std = np.array([np.std(covs_by_dist[i]) for i in range(len(dist_set))])

    # dist_range = np.linspace(0, np.max(dist), 500)
    # plt.scatter(dist.flatten(), cov.flatten(), s = 1 , zorder=0, label="empirical result")
    plt.plot(dist_set, means, 'b', zorder=0, label="empirical mean")
    plt.plot(dist_set, means + 1.96*std, 'k--', zorder=0, label="empirical 95% CI")
    plt.plot(dist_set, means - 1.96*std, 'k--', zorder=0)
    plt.plot(dist_set, model(dist_set), 'r', zorder=5, label="model prediction")

    plt.title(f"correlation v dist 2D, λ={l}, γ={g}")
    plt.xlabel("dist")
    plt.ylabel("correlation")
    plt.legend()
    
    plt.savefig(f'../plts/laplacian_noise_modelling/two_D_cov_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()

def two_D_variograms(n = 20, samples = 1000, l=0.5, g=1, mean=0.5):
    tuple_to_single = lambda t: t[0]*n + t[1]
    single_to_tuple = lambda i: (i // n, i % n)
    variogram = lambda r: g ** 2 / (4 * np.pi * l ** 2) - r * g ** 2 / (4 * np.pi * l) * special.kv(1, l * r)
    # covariogram = lambda r, n: r * g ** 2 / (4 * np.pi * l) * special.kv(1, l * r) - n * (g ** 2 + mean ** 2)
    # this is without the -E[m(h)^2] component, so it is a bit off 
    covariogram = lambda r, n: r * g ** 2 / (4 * np.pi * l) * special.kv(1, l * r)

    def singles_to_dist(i, j):
        t1 = single_to_tuple(i)
        t2 = single_to_tuple(j)
        dx = min(abs(t1[0] - t2[0]), n - abs(t1[0] - t2[0]))
        dy = min(abs(t1[1] - t2[1]), n - abs(t1[1] - t2[1]))
        return np.sqrt(np.pow(dx, 2) + np.pow(dy, 2))

    def edge_locations(v):
        i, j = single_to_tuple(v)
        tuples = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
        return np.array([tuple_to_single(t) for t in tuples])

    dist = np.zeros((n**2, n**2))
    L = np.zeros((n**2, n**2))
    for i in range(n ** 2):
        L[i][edge_locations(i)] = 1
        L[i, i] = - np.sum(L[i])
        for j in range(n ** 2):
            dist[i,j] = singles_to_dist(i, j)
    
    screened_poisson = L - l ** 2 * np.eye(n ** 2)
    solution = np.linalg.inv(screened_poisson)

    point_mean = l ** 2 / g * mean
    stim = -g * np.random.normal(loc=point_mean, size=(n ** 2, samples))
    result = solution @ stim

    cross_avg = np.array([0.5*(result[:, i, np.newaxis].T + result[:, i, np.newaxis]) for i in range(samples)])
    cross_var = np.array([0.5*np.pow(result[:, i, np.newaxis].T - result[:, i, np.newaxis],2) for i in range(samples)])

    dist_set = np.sort(np.unique(dist))
    dist_num = np.array([np.sum(dist == d) for d in dist_set])

    dist_avgs = np.array([[np.mean(cross_avg[t, dist == k]) for k in dist_set] for t in range(samples)])

    variograms = np.array([[np.mean(cross_var[i, dist == k]) for k in dist_set] for i in range(samples)])
    mean_variogram = np.mean(variograms, axis=0)
    std_variogram = np.std(variograms, axis=0)

    covariograms = np.array([[np.mean(((result[:, i] - dist_avgs[i, j])[np.newaxis].T * (result[:, i] - dist_avgs[i, j])[np.newaxis])[dist == k]) for j, k in enumerate(dist_set)] for i in range(samples)])
    mean_covariogram = np.mean(covariograms, axis=0)
    std_covariogram = np.std(covariograms, axis=0)

    plt.plot(dist_set, mean_covariogram, 'b', zorder=0, label="empirical mean")
    plt.plot(dist_set, mean_covariogram + 1.96*std_covariogram, 'k--', zorder=0, label="empirical 95% CI")
    plt.plot(dist_set, mean_covariogram - 1.96*std_covariogram, 'k--', zorder=0)
    plt.plot(dist_set, covariogram(dist_set, dist_num), 'r', zorder=5, label="model prediction")
    
    plt.savefig(f'../plts/laplacian_noise_modelling/two_D_covariogram_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()

    plt.plot(dist_set, mean_variogram, 'b', zorder=0, label="empirical mean")
    plt.plot(dist_set, mean_variogram + 1.96*std_variogram, 'k--', zorder=0, label="empirical 95% CI")
    plt.plot(dist_set, mean_variogram - 1.96*std_variogram, 'k--', zorder=0)
    plt.plot(dist_set, variogram(dist_set), 'r', zorder=5, label="model prediction")
    
    plt.savefig(f'../plts/laplacian_noise_modelling/two_D_variogram_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()
    # plt.plot(dist_set, model(dist_set), 'r', zorder=5, label="model prediction")

    # means = np.array([np.average(covs_by_dist[i]) for i in range(len(dist_set))])
    # std = np.array([np.std(covs_by_dist[i]) for i in range(len(dist_set))])

    # # dist_range = np.linspace(0, np.max(dist), 500)
    # # plt.scatter(dist.flatten(), cov.flatten(), s = 1 , zorder=0, label="empirical result")
    # plt.plot(dist_set, means, 'b', zorder=0, label="empirical mean")
    # plt.plot(dist_set, means + 1.96*std, 'k--', zorder=0, label="empirical 95% CI")
    # plt.plot(dist_set, means - 1.96*std, 'k--', zorder=0)
    # plt.plot(dist_set, model(dist_set), 'r', zorder=5, label="model prediction")

    # plt.title(f"cov v dist 2D, λ={l}, γ={g}")
    # plt.xlabel("dist")
    # plt.ylabel("cov")
    # plt.legend()
    
    # plt.savefig(f'../plts/laplacian_noise_modelling/two_D_cov_modelling_l{l}_g{g}.png', dpi=300)
    # plt.close()