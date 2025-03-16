import numpy as np
import matplotlib.pyplot as plt

def one_D_cov(length = 1000, samples = 5000, l=1, g=1):
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

    covs_by_dist = [cov.flatten()[dists.flatten() == k] for k in range(length // 2)]
    means = np.array([np.average(covs_by_dist[i]) for i in range(length // 2)])
    std = np.array([np.std(covs_by_dist[i]) for i in range(length // 2)])

    dist_range = np.arange(length // 2)
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

def two_D_cov(n = 30, samples = 5000, l=1, g=1):
    tuple_to_single = lambda i, j: i*n + j
    single_to_tuple = lambda i: (i // n, i % n)

    def edge_locations(v):
        i, j = single_to_tuple(v)
        tuples = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
        return np.array([tuple_to_single(t) for t in tuples])

    L = np.zeros((n**2, n**2))
    for i in range(n ** 2):
        L[i, edge_locations(i)] = 1