import station_handler as sh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats

stat = 'min'
lat = sh.stations['lat'].to_numpy()
long = sh.stations['long'].to_numpy()
temps = sh.get_series_matrix(stat)
valid = ~sh.get_was_nan_matrix(stat)
days = np.arange(temps.shape[1])

# for each series, we want to fit on non-nans a function of form a+bsin(2pi*t/365+phi)
# params are b, phi in order
w = 2*np.pi/365
def sin_model(params, t):
    return params[0]*np.sin(w*t + params[1])

def cost_function(params, t, y, v):
    y_pred = sin_model(params, t)
    return np.sum(np.pow(y - y_pred, 2)*v)

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

def save_hist_qq_subplots():
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
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

    scatter = axes.scatter(long, lat, c=regression_coefficients, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='autoregression coefficient')
    axes.set_title(f'autoregression coefficient for {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'../plts/autoregression_fit_imgs/{stat}_coeff_by_loc.png', bbox_inches='tight')
    plt.close()