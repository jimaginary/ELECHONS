import station_handler as sh
import edge_computations as ec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from matplotlib.colors import TwoSlopeNorm
import scipy.special as special
import yeo_johnson_transform as yj

stat = None
lat = sh.stations['lat'].to_numpy()
long = sh.stations['long'].to_numpy()
temps = None
valid = None
days = None
means = None
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
    global _INIT, stat, temps, valid, days, means, params, temps_mean_sin_adj, regression_coefficients, regression_error
    _INIT = True
    stat = _stat
    temps = sh.get_series_matrix(stat)
    valid = ~sh.get_was_nan_matrix(stat)
    days = np.arange(temps.shape[1])

    X = np.array([[1 for _ in range(temps.shape[1])], np.sin(w*days), np.cos(w*days)]).T
    proj = np.linalg.inv(X.T @ X) @ X.T
    params = proj @ temps.T
    temps_mean_sin_adj = temps - params.T @ X.T

    def auto_regress(y, v):
        V = v[1:]*v[:-1]
        return np.dot(V*y[1:],V*y[:-1])/np.dot(V*y[:-1],V*y[:-1])

    regression_coefficients = np.array([auto_regress(temps_mean_sin_adj[i], valid[i]) for i in range(temps.shape[0])])

    regression_error = np.array([temps_mean_sin_adj[i][1:] - regression_coefficients[i]*temps_mean_sin_adj[i][:-1] for i in range(temps.shape[0])])

def plot_all():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    print('plotting save_hist_qq_subplots')
    save_hist_qq_subplots()
    print('plotting plot_dist_by_loc')
    plot_dist_by_loc()
    print('plotting plot_seasonality_by_loc')
    plot_seasonality_by_loc()
    print('plotting plot_autoregression_by_loc')
    plot_autoregression_by_loc()
    print('plotting plot_autoregression_partial_corrs delay 20')
    plot_autoregression_partial_corrs()
    print('plotting plot_autoregression_partial_corrs delay 5')
    plot_autoregression_partial_corrs(5)
    print('plotting plot_correlation_v_dist')
    plot_correlation_v_dist()
    print('plotting plot_regression_coeff_v_dist')
    plot_regression_coeff_v_dist()
    print('plotting plot_partial_corr_v_dist')
    plot_partial_corr_v_dist()
    print('plotting plot_autoreg_residues')
    plot_autoreg_residues()
    print('plotting explicit_spatial_fit')
    explicit_spatial_fit()

def save_hist_qq_subplots():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    for st in range(temps.shape[0]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
        # Rice's rule: bins = 2*cbrt(obsv)
        no_bins = int(2*np.cbrt(temps.shape[1]))

        axes[0,0].hist(temps[st], bins=no_bins)
        axes[0,0].set_title(f'{stat} temperature distribution for station {sh.stations.iloc[st]['station number']}')
        axes[0,0].set_xlabel('degC')
        axes[0,0].set_ylabel('no. samples')

        axes[0,1].hist(temps_mean_sin_adj[st], bins=no_bins)
        axes[0,1].set_title(f'season-adjusted {stat} temperature distribution for station {sh.stations.iloc[st]['station number']}')
        axes[0,1].set_xlabel('degC')
        axes[0,1].set_ylabel('no. samples')

        stats.probplot(temps[st], plot=axes[1,0])
        axes[1,0].set_title(f'{stat} temp q-q plot')

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
    and_r = stats.anderson(temps[st], dist='norm')
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

    # subtract 2/12 from phase because our data starts in march
    phase = np.arctan(- params[1,:] / params[2,:]) / (2 * np.pi) - (2 / 12)
    amp = np.sqrt(np.pow(params[1,:],2) + np.pow(params[2,:],2))
    
    scatter = axes[0].scatter(long, lat, c=phase, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Phase (years)')
    axes[0].set_title(f'seasonal phase of {stat} temp data by location')

    scatter = axes[1].scatter(long, lat, c=amp, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Amplitude (degC)')
    axes[1].set_title(f'seasonal amplitude of {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'../plts/sin_fit_imgs/{stat}_params_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autoregression_by_loc_post_yj_transform():
    station_skews = stats.skew(temps_mean_sin_adj, axis=1)
    # skew ~ 1.9(l-1):
    station_lambdas = 1 + station_skews/1.9

    yj_data = np.array([yj.yj(temps_mean_sin_adj[i], station_lambdas[i]) for i in range(temps.shape[0])])
    yj_autoreg_coeffs = np.array([np.dot(yj_data[i, :-1], yj_data[i, 1:]) / np.dot(yj_data[i, :-1], yj_data[i, :-1]) for i in range(temps.shape[0])])
    yj_autoreg_fit = np.array([yj_autoreg_coeffs[i] * yj_data[i,:-1] for i in range(temps.shape[0])])
    autoreg_error = np.array([temps_mean_sin_adj[i,1:] - yj.yj_inv(yj_autoreg_fit[i], station_lambdas[i]) for i in range(temps.shape[0])])
    
    RMSE = np.sqrt(np.sum(np.pow(autoreg_error, 2), axis=1) / temps.shape[1])

    RMSE_tot = np.sqrt(np.sum(np.pow(RMSE,2))/temps.shape[0])

    scatter = plt.scatter(long, lat, c=RMSE, cmap='rainbow')
    plt.title(f'RMSE from autoreg via YJ skew adjustment. Total RMSE={RMSE_tot:.4f}')
    plt.xlabel('long')
    plt.ylabel('lat')
    plt.colorbar(scatter, label='RMSE (degC)')

    plt.tight_layout()
    plt.savefig(f'../plts/YJ/{stat}_RMSE_by_loc.png', bbox_inches='tight')
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
    
    def exp_model(params, d):
        return params[0]*np.exp(-params[1]*d)
    
    def cov_2d_model(params, d):
        # return np.where(d<1e-12, (params[0] ** 2 / (4 * np.pi * params[1] ** 2)), (d * params[0] ** 2 / (4 * np.pi * params[1])) * special.kv(1, params[1] * d))
        # normlised for regression coeff
        return np.where(d<1e-12, 1, d * params[0] * special.kv(1, d * params[0]))
    
    def least_squares(params, func, d, p):
        return np.sqrt(np.sum(np.pow(p - func(params, d), 2))/len(d))

    space_error_correlation = np.corrcoef(regression_error).flatten()
    dist = ec.distance_matrix(sh.stations).flatten()

    exp_model_min_obj = minimize(least_squares, np.array([0,0]), args=(exp_model, dist, space_error_correlation), method='Nelder-Mead')
    if not exp_model_min_obj.success:
        print(f'failed to minimise with exp model')
    exp_model_params = exp_model_min_obj.x

    cov_2d_model_min_obj = minimize(least_squares, np.array([0.01]), args=(cov_2d_model, dist, space_error_correlation), method='Nelder-Mead')
    if not cov_2d_model_min_obj.success:
        print(f'failed to minimise with cov 1d model')
    cov_2d_model_params = cov_2d_model_min_obj.x

    order = dist.argsort()
    plt.plot(dist[order], exp_model(exp_model_params, dist)[order], label=f'exp model cost {least_squares(exp_model_params, exp_model, dist, space_error_correlation):.4f}', c='r')
    plt.plot(dist[order], cov_2d_model(cov_2d_model_params, dist)[order], label=f'cov 2d model cost {least_squares(cov_2d_model_params, cov_2d_model, dist, space_error_correlation):.4f}', c='g')

    plt.scatter(dist, space_error_correlation, s=2, c='b', alpha=0.1)
    plt.xlabel('distance (km)')
    plt.ylabel('correlation coefficient')
    plt.title(f'{stat} temp dist v Pearson correlation for regression error, fitted λ={cov_2d_model_params[0]:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_error_corr_by_dist.png', bbox_inches='tight', dpi=300)
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

def plot_autoreg_residues():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    merged_valid = valid[:,1:]*valid[:,:-1]
    station_error = np.array([np.sqrt(np.sum(np.pow(error,2)*merged_valid[i]) / np.sum(merged_valid[i])) for i, error in enumerate(regression_error)])

    scatter = plt.scatter(long, lat, c=station_error, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='RMSE (degC)')
    plt.title(f'error after autoregression for {stat} temp data by location')

    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_RMSE_by_loc.png', bbox_inches='tight')
    plt.close()

def explicit_spatial_fit():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    def exp_model(params, d):
        return params[0]*np.exp(-params[1]*d)
    
    def least_squares(params, func, d, p):
        return np.sum(np.pow(p - func(params, d), 2))/len(d)

    space_error_correlation = np.corrcoef(regression_error).flatten()
    dist = ec.distance_matrix(sh.stations).flatten()

    exp_model_min_obj = minimize(least_squares, np.array([0,0]), args=(exp_model, dist, space_error_correlation), method='Nelder-Mead')
    if not exp_model_min_obj.success:
        print(f'failed to minimise with exp model')
    exp_model_params = exp_model_min_obj.x

    P = exp_model(exp_model_params, ec.distance_matrix(sh.stations))
    np.fill_diagonal(P, 0)

    prev_errors = regression_error[:,:-1]
    betas = []
    for station, post_errors in enumerate(regression_error[:,1:]):
        c = prev_errors.T @ P[station]
        if (np.dot(c,c) == 0):
            betas.append(0)
        else:
            betas.append((P[station].T @ prev_errors @ post_errors) / (np.dot(c,c)))
    
    betas = np.array(betas)
    print(f'spatial fit betas: max {np.max(betas):.4f}, min {np.min(betas):.4f}, mean {np.mean(betas):.4f}, std {np.std(betas):.4f}')

    error_pred = []
    for temp in prev_errors.T:
        error_pred.append(betas * (P @ temp))
    error_pred = np.array(error_pred)

    merged_valid = valid.T[1:,:]*valid.T[:-1,:]
    valids = np.sum(merged_valid)

    spatial_cost = np.sqrt(np.sum(np.pow(error_pred - regression_error.T[1:,:],2)*merged_valid[1:,:]) / valids)
    print(f'cost: {spatial_cost:.4f} degC')
    
    prior_cost = np.sqrt(np.sum(np.pow(regression_error.T[1:,:], 2)*merged_valid[1:,:]) / valids)
    print(f'prior cost: {prior_cost:.4f} degC')

    scatter = plt.scatter(long, lat, c=betas, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='spatial autoregression coefficient')
    plt.title(f'spatial autoregression coefficient for {stat} temp data by location, fit cost={spatial_cost:.4f} degC')

    plt.tight_layout()
    plt.savefig(f'../plts/autoregression_fit_imgs/{stat}_space_coeff_by_loc.png', bbox_inches='tight')
    plt.close()

def direct_error_regressed_on_spatial_data():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    data = temps_mean_sin_adj.T[1:,:]
    X = temps_mean_sin_adj.T[:-1,:]
    proj = np.linalg.inv(X.T @ X) @ X.T
    spatial_params = proj @ data
    pred = X @ spatial_params
    spatial_error = data - pred

    RMSE = np.sqrt(np.average(np.pow(spatial_error,2)))

    vmax = np.max(np.abs(spatial_params))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))

    im1 = ax1.imshow(spatial_params, cmap='RdBu', norm=norm)
    plt.colorbar(im1, ax=ax1, label='Regression Coefficient')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    ax1.set_title(f'Spatial Regression Matrix, fit RMSE = {RMSE:.4f}')

    np.fill_diagonal(spatial_params, 0)
    vmax = np.max(np.abs(spatial_params))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im2 = ax2.imshow(spatial_params, cmap='RdBu', norm=norm)
    plt.colorbar(im2, ax=ax2, label='Regression Coefficient')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    ax2.set_title(f'Spatial Regression Matrix w/out autoregression (main diagonal)')

    im3 = ax3.imshow(ec.distance_matrix(sh.stations), cmap='RdBu')
    plt.colorbar(im3, ax=ax3, label='Distances')
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Row Index')
    ax3.set_title('Distances Matrix')

    plt.tight_layout()
    fig.savefig(f'../plts/spatial_fit_imgs/{stat}_error_regressed_on_spatial_data.png', bbox_inches='tight', dpi=300)
    plt.close()