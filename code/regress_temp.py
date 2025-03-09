import station_handler as sh
import edge_computations as ec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats

stat = None
lat = sh.stations['lat'].to_numpy()
long = sh.stations['long'].to_numpy()
temps = None
valid = None
days = None
means = None
temps_mean_adj = None
params = None
temps_mean_sin_adj = None
regression_coefficients = None
regression_error = None

# for each series, we want to fit on non-nans a function of form a+bsin(2pi*t/365+phi)
# params are b, phi in order
w = 2*np.pi/365.25
def sin_model(params, t):
    return params[0]*np.sin(w*t + params[1])

def cost_function(params, t, y, v):
    y_pred = sin_model(params, t)
    return np.sum(np.pow(y - y_pred, 2)*v)

_INIT = False
def init(_stat):
    global _INIT, stat, temps, valid, days, means, temps_mean_adj, params, temps_mean_sin_adj, regression_coefficients, regression_error
    _INIT = True
    stat = _stat
    temps = sh.get_series_matrix(stat)
    valid = ~sh.get_was_nan_matrix(stat)
    days = np.arange(temps.shape[1])

    # incorporate average component
    means = np.sum(temps*valid, axis=1) / np.sum(valid, axis=1)
    temps_mean_adj = temps - np.tile(means, (temps.shape[1], 1)).T

    params = np.zeros((temps.shape[0], 2))
    for i in range(temps.shape[0]):
        min_obj = minimize(cost_function, np.array([0, 0]), args=(days, temps_mean_adj[i], valid[i]), method='Nelder-Mead')
        if not min_obj.success:
            print(f'Got a minimisation failure at station {sh.stations.iloc[i]['station number']}!')
            print(f'params: {min_obj.x}')
        params[i] = min_obj.x

    temps_mean_sin_adj = temps_mean_adj - np.array([sin_model(params[i], days) for i in range(temps.shape[0])])

    def auto_regress(y, v):
        V = v[1:]*v[:-1]
        return np.dot(V*y[1:],V*y[:-1])/np.dot(V*y[:-1],V*y[:-1])

    regression_coefficients = np.array([auto_regress(temps_mean_sin_adj[i], valid[i]) for i in range(temps.shape[0])])

    regression_error = np.array([temps_mean_sin_adj[i][1:] - regression_coefficients[i]*temps_mean_sin_adj[i][:-1] for i in range(temps.shape[0])])

def save_hist_qq_subplots():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    for st in range(temps.shape[0]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
        # Rice's rule: bins = 2*cbrt(obsv)
        no_bins = int(2*np.cbrt(temps.shape[1]))

        axes[0,0].hist(temps_mean_adj[st], bins=no_bins)
        axes[0,0].set_title(f'mean-adjusted {stat} temperature distribution for station {sh.stations.iloc[st]['station number']}')
        axes[0,0].set_xlabel('degC')
        axes[0,0].set_ylabel('no. samples')

        axes[0,1].hist(temps_mean_sin_adj[st], bins=no_bins)
        axes[0,1].set_title(f'season-adjusted {stat} temperature distribution for station {sh.stations.iloc[st]['station number']}')
        axes[0,1].set_xlabel('degC')
        axes[0,1].set_ylabel('no. samples')

        stats.probplot(temps_mean_adj[st], plot=axes[1,0])
        axes[1,0].set_title(f'mean-adj {stat} temp q-q plot')

        stats.probplot(temps_mean_sin_adj[st], plot=axes[1,1])
        axes[1,1].set_title(f'sin-adj {stat} temp q-q plot')

        plt.tight_layout()
        fig.savefig(f'../plts/distribution_imgs/{stat}_dist_{sh.stations.iloc[st]['station number']}.png', bbox_inches='tight')
        plt.close()

def test_normality(st):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    print(f'Anderson-Darling test results for temp data at station: {sh.stations.iloc[st]['station number']}')
    and_r = stats.anderson(temps_mean_adj[st], dist='norm')
    print(and_r.statistic)
    print(and_r.critical_values)
    print(and_r.significance_level)

    print(f'Anderson-Darling test results for temp data sinusoid regressed at station: {sh.stations.iloc[st]['station number']}')
    and_r = stats.anderson(temps_mean_sin_adj[st], dist='norm')
    print(and_r.statistic)
    print(and_r.critical_values)
    print(and_r.significance_level)

def plot_dist_by_loc():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    # temps_var = np.var(temps_mean_adj, axis=1)
    # temps_skew = stats.skew(temps_mean_adj, axis=1)
    # temps_kurt = stats.kurtosis(temps_mean_adj, axis=1)
    temps_var = np.var(temps_mean_sin_adj, axis=1)
    temps_skew = stats.skew(temps_mean_sin_adj, axis=1)
    temps_kurt = stats.kurtosis(temps_mean_sin_adj, axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6,10))

    scatter = axes[0].scatter(long, lat, c=temps_var, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Var')
    axes[0].set_title(f'variance in sin-adj {stat} temp data by location')

    scatter = axes[1].scatter(long, lat, c=temps_skew, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Skew')
    axes[1].set_title(f'skew in sin-adj {stat} temp data by location')

    scatter = axes[2].scatter(long, lat, c=temps_kurt, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Kurtosis')
    axes[2].set_title(f'kurtosis in sin-adj {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'../plts/distribution_imgs/{stat}_dists_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_seasonality_by_loc():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # normalise phase/amp: add half a year when amplitude is negative
    # subtract 2/12 from phase because our data starts in march
    phase = ((params[:,1]+0.5*(params[:,0] < 0) + 2/12) % 1)/(2*np.pi)
    amp = np.abs(params[:,0])
    
    scatter = axes[0].scatter(long, lat, c=phase, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Phase (years)')
    axes[0].set_title(f'seasonal phase of {stat} temp data by location')

    scatter = axes[1].scatter(long, lat, c=amp, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Amplitude (degC)')
    axes[1].set_title(f'seasonal amplitude of {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'../plts/sin_fit_imgs/{stat}_params_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autoregression_by_loc():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

    scatter = axes.scatter(long, lat, c=regression_coefficients, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='autoregression coefficient')
    axes.set_title(f'autoregression coefficient for {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'../plts/autoregression_fit_imgs/{stat}_coeff_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autoregression_partial_corrs(max_delay = 20):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    T_partials = np.zeros((104, max_delay))
    for j, temp in enumerate(temps_mean_sin_adj):
        T = np.array([temp[max_delay+1-i:temp.shape[0]-i] for i in range(max_delay+1)])
        T_cov = np.cov(T)
        T_precision = np.linalg.inv(T_cov)
        T_partials[j] = np.array([-T_precision[0, i] / np.sqrt(T_precision[0, 0] * T_precision[i, i]) for i in range(1, max_delay+1)])
    for row, partials in enumerate(T_partials.T):
        plt.scatter((row+1)*np.ones_like(partials), partials, s=1, c='b')
    plt.xlabel('Delay')
    plt.ylabel('Partial Correlation')
    plt.title(f'{stat} temp autoregression partial correlations for max delay = {max_delay}')
    
    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_partial_corr_by_delay_{max_delay}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_correlation_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    space_error_correlation = np.corrcoef(regression_error).flatten()
    dist = ec.distance_matrix(sh.stations).flatten()

    plt.scatter(dist, space_error_correlation, s=2)
    plt.xlabel('distance (km)')
    plt.ylabel('correlation coefficient')
    plt.title(f'{stat} temp dist v Pearson correlation for regression error')
    
    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_error_corr_by_dist.png', bbox_inches='tight')
    plt.close()

def plot_regression_coeff_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    dist = ec.distance_matrix(sh.stations)
    cov = np.cov(regression_error)

    betas = cov @ (np.eye(cov.shape[0])*np.reciprocal(cov))

    plt.scatter(dist.flatten(), betas.flatten(), s=2)
    plt.xlabel('distance (km)')
    plt.ylabel('single regression coefficient')
    plt.title(f'{stat} temp dist v single regression coefficient for regression error')
    
    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_error_regression_coeff_by_dist.png', bbox_inches='tight')
    plt.close()

def plot_partial_corr_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    dist = ec.distance_matrix(sh.stations)
    order = np.argsort(dist, axis=1)

    space_error_cov = np.cov(regression_error)
    space_error_precision = np.linalg.inv(space_error_cov)
    partial_corr = np.zeros_like(space_error_precision)
    for i in range(temps.shape[0]):
        for j in range(temps.shape[0]):
            partial_corr[i, j] = -space_error_precision[i, j] / np.sqrt(space_error_precision[i, i] * space_error_precision[j, j])

    # r_squared = 1 - np.reciprocal(np.diagonal(space_error_cov)*np.diagonal(space_error_precision))
    # print(r_squared)
    
    order = order[partial_corr > -1+1e-12]
    dist = dist[partial_corr > -1+1e-12]
    partial_corr = partial_corr[partial_corr > -1+1e-12]

    scatter = plt.scatter(dist, partial_corr, s=1, c=order, cmap='rainbow', alpha=0.4)
    plt.xlabel('distance (km)')
    plt.ylabel('partial correlation coefficient')
    plt.title(f'{stat} temp dist v partial correlation for regression error')
    plt.colorbar(scatter, label='distance order')

    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_nearest_error_partial_corr_by_dist.png', bbox_inches='tight', dpi=300)
    plt.close()