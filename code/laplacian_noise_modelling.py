import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

def one_D_cov(length = 1000, samples = 5000, l=0.5, g=1):
    dist = lambda x,y: min(np.abs(x-y), length - np.abs(x-y))
    model = lambda r: (g**2 / (4 * l ** 3)) * (1 + l*r)*np.exp(-l*r)

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

    stim = -g * np.random.normal(size=(length, samples))

    result = solution @ stim

    cov = np.cov(result)

    dist_range = np.arange(length // 2)

    covs_by_dist = [cov.flatten()[dists.flatten() == k] for k in dist_range]
    means = np.array([np.average(covs_by_dist[i]) for i in dist_range])
    std = np.array([np.std(covs_by_dist[i]) for i in dist_range])
    
    plt.plot(dist_range, means, 'b', zorder=0, label="empirical mean")
    plt.plot(dist_range, means + 1.96*std, 'k--', zorder=0, label="empirical 95% CI")
    plt.plot(dist_range, means - 1.96*std, 'k--', zorder=0)
    plt.plot(dist_range, model(dist_range), 'r', zorder=5, label="model prediction")

    plt.title(f"cov v dist, λ={l}, γ={g}")
    plt.xlabel("dist")
    plt.ylabel("cov")
    plt.legend()
    
    plt.savefig(f'../plts/laplacian_noise_modelling/one_D_cov_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()

def two_D_cov(n = 50, samples = 5000, l=0.5, g=1):
    tuple_to_single = lambda t: t[0]*n + t[1]
    single_to_tuple = lambda i: (i // n, i % n)
    # model = lambda r: (g**2 / (4 * np.pi * l ** 2)) * (l * r * special.kv(1, l * r) - special.kv(0, l * r))
    # model = lambda r: (g**2 / (4 * np.pi * l ** 2)) * (1 + l * r) * np.exp(-l * r)
    model = lambda r: (r / (4 * np.pi * l)) * special.kv(1, l * r)

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
    cov = np.cov(result)

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

    plt.title(f"cov v dist 2D, λ={l}, γ={g}")
    plt.xlabel("dist")
    plt.ylabel("cov")
    plt.legend()
    
    plt.savefig(f'../plts/laplacian_noise_modelling/two_D_cov_modelling_l{l}_g{g}.png', dpi=300)
    plt.close()