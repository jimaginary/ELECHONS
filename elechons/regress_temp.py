from elechons.processing import edges as ec
from elechons.data import station_handler as sh
from elechons.models import yeo_johnson_transform as yj
from elechons import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from matplotlib.colors import TwoSlopeNorm
import scipy.special as special
import datetime
from elechons.models import spatial_methods as s
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

stat = None
lat = sh.STATIONS['lat'].to_numpy()
long = sh.STATIONS['long'].to_numpy()
ids = np.array([i for i in sh.STATIONS['station number'].to_numpy()])
temps = None
valid = None
days = None
params = None
temps_mean_sin_adj = None
regression_coefficients = None
regression_error = None
arima_error = None

w = 2*np.pi/365.25

_INIT = False
def init(_stat, remove_backfills_above=10000, delay=2):
    global _INIT, stat, temps, valid, ids, days, params, temps_mean_sin_adj, regression_coefficients, regression_error, lat, long
    _INIT = True
    stat = _stat
    temps = sh.get_series_matrix(stat)
    valid = ~sh.get_was_nan_matrix(stat)
    days = np.arange(temps.shape[1])

    # small_backfills = np.array([s.get_largest_backfill(v) <= remove_backfills_above for v in valid])
    # temps = temps[small_backfills]
    # valid = valid[small_backfills]
    # lat = lat[small_backfills]
    # long = long[small_backfills]
    # ids = ids[small_backfills]

    X = np.array([[1 for _ in range(temps.shape[1])], np.sin(w*days), np.cos(w*days), np.sin(2*w*days), np.cos(2*w*days)]).T
    proj = np.linalg.inv(X.T @ X) @ X.T
    params = proj @ temps.T
    temps_mean_sin_adj = temps - params.T @ X.T

    print(f'seasonality RMSE {s.rmse(temps_mean_sin_adj):.4f}')

    regression_coefficients = []
    regression_fits = []
    for temp in temps_mean_sin_adj:
        regression = s.auto_regress(temp, delay)
        regression_coefficients.append(regression[0])
        regression_fits.append(regression[1])
    regression_coefficients = np.array(regression_coefficients)
    regression_error = temps_mean_sin_adj[:, delay:] - np.array(regression_fits)

    print(f'season-adjusted regression RMSE {s.rmse(regression_error):.4f}')

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
    print('plotting plot_precision_v_dist')
    plot_precision_v_dist()
    print('plotting plot_autoreg_residues')
    plot_autoreg_residues()

def save_hist_qq_subplots():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    for st in range(temps.shape[0]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
        # Rice's rule: bins = 2*cbrt(obsv)
        no_bins = int(2*np.cbrt(temps.shape[1]))

        axes[0,0].hist(temps[st], bins=no_bins)
        axes[0,0].set_title(f'{stat} temperature distribution for station {sh.STATIONS.iloc[st]['station number']}')
        axes[0,0].set_xlabel('degC')
        axes[0,0].set_ylabel('no. samples')

        axes[0,1].hist(temps_mean_sin_adj[st], bins=no_bins)
        axes[0,1].set_title(f'season-adjusted {stat} temperature distribution for station {sh.STATIONS.iloc[st]['station number']}')
        axes[0,1].set_xlabel('degC')
        axes[0,1].set_ylabel('no. samples')

        stats.probplot(temps[st], plot=axes[1,0])
        axes[1,0].set_title(f'{stat} temp q-q plot')

        stats.probplot(temps_mean_sin_adj[st], plot=axes[1,1])
        axes[1,1].set_title(f'sin-adj {stat} temp q-q plot')

        plt.tight_layout()
        fig.savefig(f'{config.PLOTS_DIR}/distribution_imgs/{stat}_dist_{sh.STATIONS.iloc[st]['station number']}.png', bbox_inches='tight')
        plt.close()

def test_normality(st):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    print(f'Anderson-Darling test results for temp data at station: {sh.STATIONS.iloc[st]['station number']}')
    and_r = stats.anderson(temps[st], dist='norm')
    print(and_r.statistic)
    print(and_r.critical_values)
    print(and_r.significance_level)

    print(f'Anderson-Darling test results for temp data sinusoid regressed at station: {sh.STATIONS.iloc[st]['station number']}')
    and_r = stats.anderson(temps_mean_sin_adj[st], dist='norm')
    print(and_r.statistic)
    print(and_r.critical_values)
    print(and_r.significance_level)

def plot_ar_lb_success(lags=10, arima=False):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    if arima and arima_error is None:
        print(f'Requested ARIMA error but fit_arima() has not been called')
        return
    
    testset = arima_error if arima_error is not None else regression_error

    successes_per_lag = []
    for lag in range(1, lags + 1):
        successes = 0
        for err in testset:
            successes += (acorr_ljungbox(err, lags=[lag], return_df=True)['lb_pvalue'].to_numpy() > 0.05)
        successes_per_lag.append(successes)
    
    plt.scatter(np.arange(1, lags + 1), successes_per_lag)
    plt.title('No. timeseries passing Ljung-Box test')
    plt.xlabel('Delay (days)')
    plt.ylabel('No. timeseries with p>0.05')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_ljung_box.png', bbox_inches='tight')
    plt.close()


def plot_dist_by_loc():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    temps_var = np.var(temps_mean_sin_adj, axis=1)
    temps_skew = stats.skew(temps_mean_sin_adj, axis=1)
    temps_kurt = stats.kurtosis(temps_mean_sin_adj, axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6,10))

    scatter = axes[0].scatter(long, lat, c=temps_var, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Var (degC^2)')
    axes[0].set_title(f'variance in sin-adj {stat} temp data by location')

    scatter = axes[1].scatter(long, lat, c=temps_skew, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Skew')
    axes[1].set_title(f'skew in sin-adj {stat} temp data by location')

    scatter = axes[2].scatter(long, lat, c=temps_kurt, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Kurtosis')
    axes[2].set_title(f'kurtosis in sin-adj {stat} temp data by location')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/distribution_imgs/{stat}_dists_by_loc.png', bbox_inches='tight')
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
    axes[0].set_xlabel('long (deg)')
    axes[0].set_ylabel('lat (deg)')

    scatter = axes[1].scatter(long, lat, c=amp, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Amplitude (degC)')
    axes[1].set_title(f'seasonal amplitude of {stat} temp data by location')
    axes[1].set_xlabel('long (deg)')
    axes[1].set_ylabel('lat (deg)')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/sin_fit_imgs/{stat}_params_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autocorrs(max_delay = 1000):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    for st in range(temps.shape[0]):
        days = np.arange(0, max_delay + 1)
        autocorr = s.autocorr(temps[st], max_delay)
        autocorr_model_min_obj = minimize(s.least_squares, np.array([0.7, 0.01, 0.2, 0.0]), args=(s.seasonal_autocorr_model, days, autocorr), method='Nelder-Mead')
        if not autocorr_model_min_obj.success:
            print(f'failed to minimise with cov 1d model')
        autocorr_model_params = autocorr_model_min_obj.x

        plt.plot(days, autocorr, 'b', label='true')
        plt.plot(days, s.seasonal_autocorr_model(autocorr_model_params, days), 'r', label='fit')
        plt.xlabel('delay (days)')
        plt.ylabel('autocorrelation')
        plt.title(f'autocorrelation v time for station {sh.STATIONS.iloc[st]['station number']}, with fit RMSE {s.least_squares(autocorr_model_params, s.seasonal_autocorr_model, days, autocorr):.4f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{config.PLOTS_DIR}/autocorr_levinson/{stat}_autocorr_{sh.STATIONS.iloc[st]['station number']}.png', bbox_inches='tight')
        plt.close()

        autoreg = s.gohberg_inverse(s.seasonal_autocorr_model(autocorr_model_params, days))
        autoreg = -autoreg / autoreg[0]
        plt.plot(days[2:], autoreg[2:])
        plt.xlabel('delay (days)')
        plt.ylabel('autoreg')
        plt.title(f'autoreg v time for station {sh.STATIONS.iloc[st]['station number']}, with fitted autocorr')

        plt.tight_layout()
        plt.savefig(f'{config.PLOTS_DIR}/autocorr_levinson/{stat}_autoreg_{sh.STATIONS.iloc[st]['station number']}.png', bbox_inches='tight')
        plt.close()

def fit_arima():
    global arima_error
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    err = []
    for i, temp in enumerate(temps_mean_sin_adj):
        model = ARIMA(temp, order=(2, 0, 2))
        fit = model.fit()

        predictions = fit.fittedvalues

        err.append(predictions - temp)

        print(f'ARIMA 2,0,2 RMSE {s.rmse(predictions - temp)} for station {ids[i]}')
        # print(f'summary {fit.summary()}')
    arima_error = np.array(err)

def plot_mean_v_variance():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    temps_mean = params[0, :]
    temps_var = np.var(temps_mean_sin_adj, axis=1)

    plt.scatter(temps_mean, temps_var, s=1)
    plt.xlabel('mean (degC)')
    plt.ylabel('var (degC^2)')
    plt.title(f'{stat} Daily Temperature Mean v Variance')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/distribution_imgs/{stat}_mean_v_var', bbox_inches='tight')
    plt.close()

def plot_windowed_regression_v_time(station, W=182, overlap=164):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    idx = np.where(ids == station)
    if len(idx) == 0:
        print(f'No such station!')
        return
    idx = idx[0]

    windowed_regression_coefficients = np.array([auto_regress(temps_mean_sin_adj[idx, j*(W-overlap):j*(W-overlap)+W][0]) for j in range((temps.shape[1] - W) // (W-overlap))])
    years = (W - overlap) * np.arange(len(windowed_regression_coefficients[:, 0])) / 365

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,7))
    
    axes[0].plot(years, windowed_regression_coefficients[:, 0])
    axes[0].set_title(f'delay = 2 days regression coefficients vs time at station {station} with window size {W}')
    axes[0].set_xlabel('years')
    axes[0].set_ylabel('autoregression coefficient')

    axes[1].plot(years, windowed_regression_coefficients[:, 1])
    axes[1].set_title(f'delay = 1 days regression coefficients vs time at station {station} with window size {W}')
    axes[1].set_xlabel('years')
    axes[1].set_ylabel('autoregression coefficient')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{station}_{stat}_autoreg_v_time', bbox_inches='tight', dpi=300)
    plt.close()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,7))

    f = np.linspace(0, 365 / (W - overlap), len(windowed_regression_coefficients[:, 0]))
    
    axes[0].plot(f, np.abs(np.fft.fft(windowed_regression_coefficients[:, 0])))
    axes[0].set_title(f'delay = 2 days regression coefficients vs time at station {station} with window size {W}')
    axes[0].set_xlabel('/years')
    axes[0].set_ylabel('autoregression coefficient')

    axes[1].plot(f, np.abs(np.fft.fft(windowed_regression_coefficients[:, 1])))
    axes[1].set_title(f'delay = 1 days regression coefficients vs time at station {station} with window size {W}')
    axes[1].set_xlabel('/years')
    axes[1].set_ylabel('autoregression coefficient')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{station}_{stat}_autoreg_v_time_fft', bbox_inches='tight', dpi=300)
    plt.close()

def fit_seasonal_autoregression():
    global regression_error
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    def seasonal_model(y):
        return np.array([y[:-2], y[1:-1], y[:-2]*np.sin(w*days)[:-2], y[:-2]*np.cos(w*days)[:-2], y[1:-1]*np.sin(w*days)[1:-1], y[1:-1]*np.cos(w*days)[1:-1]]).T

    def seasonal_autoregress(y):
        X = seasonal_model(y)
        proj = np.linalg.pinv(X.T @ X) @ X.T
        params = proj @ y[2:].T
        return params
    
    regression_coefficients = np.array([seasonal_autoregress(temps_mean_sin_adj[i]) for i in range(temps.shape[0])])

    regression_error = np.array([temps_mean_sin_adj[i][2:] - seasonal_model(temps_mean_sin_adj[i]) @ regression_coefficients[i] for i in range(temps.shape[0])])

    print(regression_coefficients)
    print(s.rmse(regression_error))

def plot_windowed_regression_std(W=182):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    windowed_regression_coefficients = np.array([[auto_regress(temps_mean_sin_adj[i][j*W:(j+1)*W]) for i in range(temps.shape[0])] for j in range(temps.shape[1] // W)])
    regression_std = np.std(windowed_regression_coefficients, axis=0)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,7))

    scatter = axes[0].scatter(long, lat, c=regression_std[:, 0], cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Std')
    axes[0].set_title(f'Std of delay = 2 days regression coefficients over {W} day windows by location')
    axes[0].set_xlabel('long')
    axes[0].set_ylabel('lat')

    scatter = axes[1].scatter(long, lat, c=regression_std[:, 1], cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Std')
    axes[1].set_title(f'Std of delay = 1 day regression coefficients over {W} day windows by location')
    axes[1].set_xlabel('long')
    axes[1].set_ylabel('lat')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_regression_std_by_loc.png', bbox_inches='tight')
    plt.close()
    
    # we expect stds to be ~ 1 - \xi_2^2, so we compare.
    expected_regression_std = np.sqrt((1 - np.pow(regression_coefficients[:, 0], 2))/W)
    delay_2_ratio = np.divide(regression_std[:, 0], expected_regression_std)
    delay_1_ratio = np.divide(regression_std[:, 1], expected_regression_std)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,7))

    scatter = axes[0].scatter(long, lat, c=delay_2_ratio, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Std / Expected Std')
    axes[0].set_title(f'delay = 2 days std/(expected std) by location')
    axes[0].set_xlabel('long')
    axes[0].set_ylabel('lat')

    scatter = axes[1].scatter(long, lat, c=delay_1_ratio, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Std / Expected Std')
    axes[1].set_title(f'delay = 1 days std/(expected std) by location')
    axes[1].set_xlabel('long')
    axes[1].set_ylabel('lat')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_regression_std_ratio_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_largest_backfill():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    backfills = np.array([s.get_largest_backfill(v) for v in valid])

    scatter = plt.scatter(long, lat, c=backfills, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Backfill (days)')
    plt.title(f'largest backfill of {stat} temp data by location')
    plt.xlabel('long (deg)')
    plt.ylabel('lat (deg)')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_backfills_by_loc.png', bbox_inches='tight')
    plt.close()
            

def autoregression_with_yj_transform(skew_coeff=0.25):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    station_skews = stats.skew(temps_mean_sin_adj, axis=1)
    station_lambdas = 1 - skew_coeff*station_skews

    yj_data = np.array([yj.yj(temps_mean_sin_adj[i], station_lambdas[i]) for i in range(temps.shape[0])])
    yj_means = np.mean(yj_data, axis=1)
    yj_mean_adj_data = np.array([yj_data[i] - yj_means[i] for i in range(temps.shape[0])])

    yj_autoreg_coeffs = np.array([auto_regress(yj_mean_adj_data[i]) for i in range(temps.shape[0])])
    yj_autoreg_fit = np.array([yj_means[i] + yj_autoreg_coeffs[i, 0] * yj_mean_adj_data[i,:-2] + yj_autoreg_coeffs[i, 1] * yj_mean_adj_data[i,1:-1] for i in range(temps.shape[0])])
    autoreg_error = np.array([temps_mean_sin_adj[i,2:] - yj.yj_inv(yj_autoreg_fit[i], station_lambdas[i]) for i in range(temps.shape[0])])
    
    RMSE = np.sqrt(np.sum(np.pow(autoreg_error, 2), axis=1) / temps.shape[1])

    RMSE_tot = np.sqrt(np.sum(np.pow(RMSE,2)) / temps.shape[0])

    scatter = plt.scatter(long, lat, c=RMSE, cmap='rainbow')
    plt.title(f'RMSE from autoreg via YJ skew adjustment. Total RMSE={RMSE_tot:.4f}')
    plt.xlabel('long')
    plt.ylabel('lat')
    plt.colorbar(scatter, label='RMSE (degC)')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/YJ/{stat}_RMSE_by_loc.png', bbox_inches='tight')
    plt.close()

    temps_skew = stats.skew(yj_data, axis=1)

    scatter = plt.scatter(long, lat, c=temps_skew, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='Skew')
    plt.title(f'skew in Yeo-Johnson transformed {stat} temp data by location')
    plt.xlabel('long')
    plt.ylabel('lat')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/YJ/{stat}_dists_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autoregression_by_loc():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    scatter = axes[0].scatter(long, lat, c=regression_coefficients[:, 0], cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='autoregression coefficient')
    axes[0].set_title(f'delay = 2 day autoreg coefficient for {stat} temp')
    axes[0].set_xlabel('long (deg)')
    axes[0].set_ylabel('lat (deg)')

    scatter = axes[1].scatter(long, lat, c=regression_coefficients[:, 1], cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='autoregression coefficient')
    axes[1].set_title(f'delay = 1 day autoreg coefficient for {stat} temp')
    axes[1].set_xlabel('long (deg)')
    axes[1].set_ylabel('lat (deg)')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_coeff_by_loc.png', bbox_inches='tight')
    plt.close()

def plot_autoregression_partial_corrs(max_delay = 20):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    T_partials = np.zeros((temps.shape[0], max_delay))
    for j, temp in enumerate(temps_mean_sin_adj):
        T = np.array([temp[max_delay+1-i:temp.shape[0]-i] for i in range(max_delay+1)])
        T_cov = np.cov(T)
        T_precision = np.linalg.inv(T_cov)
        T_partials[j] = np.array([-T_precision[0, i] / np.sqrt(T_precision[0, 0] * T_precision[i, i]) for i in range(1, max_delay+1)])
    for row, partials in enumerate(T_partials.T):
        plt.scatter((row + 1)*np.ones_like(partials), partials, s=1, c='b')
    
    plt.xlabel('Delay')
    plt.ylabel('Partial Correlation')
    plt.title(f'{stat} temp autoregression partial correlations for max delay = {max_delay}')
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_partial_corr_by_delay_{max_delay}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_kriging_from_cov_model(t=0, l=0.0019375):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    dist = ec.distance_matrix(sh.STATIONS)
    krig_matrix = np.zeros([temps.shape[0] + 1, temps.shape[0] + 1])
    corr = np.array(s.cov_3d_model(l, dist))

    var = np.var(temps_mean_sin_adj, axis=1)
    model_cov = corr * np.outer(np.sqrt(var), np.sqrt(var))
    model_var = np.mean(var)
    
    krig_matrix = np.block([[corr, np.ones([temps.shape[0], 1])], [np.ones([1, temps.shape[0]]), np.zeros([1, 1])]])
    krig_inv = np.linalg.inv(krig_matrix)

    corrs = lambda lat1, long1: np.array([s.cov_3d_model(l, ec.earth_distance(lat1, long1, lat[i], long[i])) for i in range(temps.shape[0])])
    weights = lambda lat1, long1: krig_inv @ np.block([corrs(lat1, long1), np.ones(1)])

    day_temp = temps[:, t]

    la_left = -45
    la_right = -10
    lo_left = 110
    lo_right = 155
    lats = np.linspace(la_left, la_right, (la_right - la_left)*4)
    longs = np.linspace(lo_left, lo_right, (lo_right - lo_left)*4)
    
    interpolation = np.zeros([len(lats), len(longs)])
    interpolation_variance = np.zeros([len(lats), len(longs)])

    for i, lo in enumerate(longs):
        if i % 10 == 0:
            print(f'i={i}')
        for j, la in enumerate(lats):
            w = weights(la, lo)
            
            interpolation[-1-j, i] = np.dot(w[:-1], day_temp)

            c = np.array([corrs(la, lo) * np.sqrt(var) * np.sqrt(model_var)])
            var_block = np.block([[model_cov, c.T], [c, np.array([[model_var]])]])
            var_vec = np.block([w[:-1], np.array([-1])])
            interpolation_variance[-1-j, i] = np.sqrt(var_vec.T @ var_block @ var_vec)
    
    # Kriging estimate

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(interpolation, cmap='rainbow', interpolation='none', extent=[lo_left, lo_right, la_left, la_right], zorder=0)
    plt.colorbar(im, ax=ax, label="Temp (degC)")

    ax.scatter(long, lat, c=day_temp, cmap='rainbow', edgecolors="black", linewidths=0.5, s=3, zorder=5)

    date = (datetime.datetime.strptime(sh.latest_start, "%Y-%m-%d") + datetime.timedelta(days=t)).strftime('%d/%m/%Y')

    ax.set_title(f'Kriging estimate of Australian {stat} temperature on {date}')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/kriging/{stat}_temp_model_krig.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Kriging error

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(interpolation_variance, cmap='rainbow', interpolation='none', extent=[lo_left, lo_right, la_left, la_right], zorder=0)
    plt.colorbar(im, ax=ax, label="Temp (degC)")

    ax.scatter(long, lat, c='black', s=3, zorder=5)

    date = (datetime.datetime.strptime(sh.latest_start, "%Y-%m-%d") + datetime.timedelta(days=t)).strftime('%d/%m/%Y')

    ax.set_title(f'Kriging standard error of Australian {stat} temperature on {date}')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/kriging/{stat}_temp_model_krig_error.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    space_error_correlation = np.corrcoef(regression_error).flatten()
    dist = ec.distance_matrix(sh.STATIONS).flatten()
    print(len(sh.STATIONS), temps_mean_sin_adj.shape, regression_error.shape, space_error_correlation.shape, dist.shape)

    cov_3d_model_min_obj = minimize(s.least_squares, np.array([0.01]), args=(s.cov_3d_model, dist, space_error_correlation), method='Nelder-Mead')
    if not cov_3d_model_min_obj.success:
        print(f'failed to minimise with exp model')
    cov_3d_model_params = cov_3d_model_min_obj.x

    cov_2d_model_min_obj = minimize(s.least_squares, np.array([0.01]), args=(s.cov_2d_model, dist, space_error_correlation), method='Nelder-Mead')
    if not cov_2d_model_min_obj.success:
        print(f'failed to minimise with cov 1d model')
    cov_2d_model_params = cov_2d_model_min_obj.x

    print(cov_3d_model_params, cov_2d_model_params)
    
    order = dist.argsort()
    plt.plot(dist[order], s.cov_3d_model(cov_3d_model_params, dist)[order], label=f'cov 3d (exp) model RMSE {s.least_squares(cov_3d_model_params, s.cov_3d_model, dist, space_error_correlation):.4f}, λ={cov_3d_model_params[0]:.4f}', c='r')
    plt.plot(dist[order], s.cov_2d_model(cov_2d_model_params, dist)[order], label=f'cov 2d model RMSE {s.least_squares(cov_2d_model_params, s.cov_2d_model, dist, space_error_correlation):.4f}, λ={cov_2d_model_params[0]:.4f}', c='g')
    
    plt.scatter(dist, space_error_correlation, s=2, c='b', alpha=0.1)
    plt.xlabel('distance (km)')
    plt.ylabel('correlation coefficient')
    plt.title(f'{stat} temp dist v Pearson correlation for regression error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_error_corr_by_dist.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_v_dist_angle():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    space_error_correlation = np.corrcoef(temps_mean_sin_adj).flatten()
    dist = ec.distance_matrix(sh.STATIONS).flatten()
    d_long = long[:, np.newaxis] - long[np.newaxis, :]
    d_lat = lat[:, np.newaxis] - lat[np.newaxis, :]
    angle = np.where(d_long == 0, np.pi/2, np.arctan(d_lat/d_long))
    # m_lat = np.abs((lat[:, np.newaxis] + lat[np.newaxis, :]).flatten()) / 2
    # m_long = np.abs((long[:, np.newaxis] + long[np.newaxis, :]).flatten()) / 2
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # scatter = ax.scatter(m_lat, m_long, dist, s=2, c=space_error_correlation, cmap='rainbow', depthshade=False)
    # ax.set_xlabel('lat (1000km)')
    # ax.set_ylabel('long (1000km)')
    # ax.set_zlabel('correlation coefficient')
    # ax.set_title(f'{stat} temp dist v Pearson correlation for regression error')

    # cbar = fig.colorbar(scatter, ax=ax)
    # cbar.set_label('correlation')

    scatter = plt.scatter(dist, angle, s=2, c=space_error_correlation, cmap='rainbow')
    plt.xlabel('distance (km)')
    plt.ylabel('angle (rad)')
    plt.title(f'distance v correlation for seasonality-adjusted {stat} temperature')
    plt.colorbar(scatter, label='correlation')
    
    # plt.show()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_error_corr_by_dist_angle_2.png', bbox_inches='tight', dpi=300)
    plt.close()

    scatter = plt.scatter(dist, space_error_correlation, s=2, c=angle, cmap='hsv')
    plt.xlabel('distance (km)')
    plt.ylabel('correlation')
    plt.title(f'distance v correlation for seasonality-adjusted {stat} temperature')
    plt.colorbar(scatter, label='angle (rad)')
    
    # plt.show()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_error_corr_by_dist_angle.png', bbox_inches='tight', dpi=300)
    plt.close()

def fit_correlation():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    space_error_correlation = np.corrcoef(regression_error).flatten()
    quadruples = np.array([[lat[i], long[i], lat[j], long[j]] for j in range(len(lat)) for i in range(len(long))])

    def anisotropic_model(params, inputs):
        l, k, mu_dist, mu_angle, sigma_dist, sigma_angle = params
        lat1 = inputs[:, 0]
        long1 = inputs[:, 1]
        lat2 = inputs[:, 2]
        long2 = inputs[:, 3]

        d_lat = np.abs(lat2 - lat1)
        d_long = np.abs(long2 - long1)
        m_lat = (lat2 + lat1) / 2
        m_long = (long2 + long1) / 2

        dist = ec.earth_distance(lat1, long1, lat2, long2)
        angle = np.where(d_long == 0, np.pi/2, np.arctan(d_lat/d_long))
        angle_to_mu = np.min([np.pow(angle - mu_angle, 2), np.pow(angle - mu_angle - np.pi, 2)])
        phase_dist_to_mu = np.sqrt(angle_to_mu*sigma_angle + np.pow(dist - mu_dist, 2)*sigma_dist)

        return np.exp(-l*dist) - k * np.exp(-np.pow(phase_dist_to_mu,2)/2)
    
    min_obj = minimize(s.least_squares, np.array([0.01, 0.2, 2250.0, 0.0, 200.0, 1.0]), args=(anisotropic_model, quadruples, space_error_correlation), method='Nelder-Mead')
    if not min_obj.success:
        print(f'failed to minimise with anisotropic model')
    min_params = min_obj.x

    print(f'params {min_params}')
    print(f'RMSE {s.least_squares(min_params, anisotropic_model, quadruples, space_error_correlation)}')

def plot_regression_coeff_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    dist = ec.distance_matrix(sh.STATIONS)
    cov = np.cov(regression_error)

    betas = cov @ (np.eye(cov.shape[0])*np.reciprocal(cov))

    plt.scatter(dist.flatten(), betas.flatten(), s=2)
    plt.xlabel('distance (km)')
    plt.ylabel('single regression coefficient')
    plt.title(f'{stat} temp dist v single regression coefficient for regression error')
    
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_error_regression_coeff_by_dist.png', bbox_inches='tight')
    plt.close()

def plot_precision_v_dist():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return

    dist = ec.distance_matrix(sh.STATIONS)
    order = np.argsort(dist, axis=1)

    # space_error_cov = np.cov(regression_error)
    space_error_cov = np.cov(temps_mean_sin_adj)
    space_error_precision = np.linalg.inv(space_error_cov)

    cov_order = np.argsort(space_error_cov, axis=1)
    # partial_corr = np.zeros_like(space_error_precision)
    # for i in range(temps.shape[0]):
    #     for j in range(temps.shape[0]):
    #         partial_corr[i, j] = -space_error_precision[i, j] / np.sqrt(space_error_precision[i, i] * space_error_precision[j, j])

    # r_squared = 1 - np.reciprocal(np.diagonal(space_error_cov)*np.diagonal(space_error_precision))
    # print(r_squared)

    def model(params, d):
        return -params[0] / d
    
    param = [16.25]

    L = model(param, dist)
    G = -ec.closeness_matrix(sh.STATIONS, 1000, 8)
    np.fill_diagonal(G, -np.sum(G, axis=0))
    np.fill_diagonal(L, 0)
    np.fill_diagonal(L, -np.sum(L, axis=0))
    l=0.0019375
    space_model_cov = s.cov_3d_model(l, dist)
    l = L.flatten()
    g = G.flatten()
    c = space_model_cov.flatten()
    p = space_error_precision.flatten()
    cosine = lambda u, v: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    print(cosine(l, p), cosine(g, p), cosine(c, p))
    # print(s.rmse(L), s.rmse(space_error_precision))
    # print(L, space_error_precision, s.rmse(L - space_error_precision)/s.rmse(space_error_precision))
    # print(s.rmse(G), s.rmse(space_error_precision))
    # print(G, space_error_precision, s.rmse(G - space_error_precision)/s.rmse(space_error_precision))
    # print(s.rmse(space_model_cov), s.rmse(space_error_precision))
    # print(space_model_cov, space_error_precision, s.rmse(space_model_cov - space_error_precision)/s.rmse(space_error_precision))

    
    not_eye = ~np.eye(dist.shape[0], dtype=bool)
    order = order[not_eye]
    dist = dist[not_eye]
    space_error_cov = space_error_cov[not_eye]
    cov_order = cov_order[not_eye]
    space_error_precision = space_error_precision[not_eye]
    

    # min_obj = minimize(s.least_squares, np.array([10]), args=(model, dist, space_error_precision), method='Nelder-Mead')
    # if not min_obj.success:
    #     print(f'failed to minimise with model')
    # min_params = min_obj.x
    # print(min_params)
    # print(s.rmse(model(min_params, dist) - space_error_precision))

    scatter = plt.scatter(dist, space_error_precision, s=1, c=order, cmap='rainbow', alpha=0.4)
    plt.xlabel('distance (km)')
    plt.ylabel('precision coefficient')
    plt.title(f'dist v precision for seasonality-adjusted {stat} temperature')
    plt.colorbar(scatter, label='distance order')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_nearest_error_precision_by_dist.png', bbox_inches='tight', dpi=300)
    plt.close()

    scatter = plt.scatter(space_error_cov, space_error_precision, s=1, c=cov_order, cmap='rainbow', alpha=0.4)
    plt.xlabel('covariance')
    plt.ylabel('precision coefficient')
    plt.title(f'covariance v precision for seasonality-adjusted {stat} temperature')
    plt.colorbar(scatter, label='covariance order')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_nearest_error_precision_by_cov.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_autoreg_residues():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    merged_valid = valid[:,1:]*valid[:,:-1]
    station_error = np.array([s.rmse(error) for error in regression_error])

    scatter = plt.scatter(long, lat, c=station_error, cmap='rainbow', zorder=5)
    plt.colorbar(scatter, label='RMSE (degC)')
    plt.title(f'error after autoregression for {stat} temp data by location')

    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_RMSE_by_loc.png', bbox_inches='tight')
    plt.close()

def spatial_spectra(l=0.0019375):
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    dist = ec.distance_matrix(sh.STATIONS)
    space_model_cov = s.cov_3d_model(l, dist)
    # space_model_precision = np.linalg.inv(space_model_cov)

    # partial_corr = np.zeros_like(space_model_precision)
    # for i in range(temps.shape[0]):
    #     for j in range(temps.shape[0]):
    #         partial_corr[i, j] = -space_model_precision[i, j] / np.sqrt(space_model_precision[i, i] * space_model_precision[j, j])
    # stochasticity
    # np.fill_diagonal(partial_corr, 0)

    ############
    # space_error_cov = np.cov(regression_error)
    space_error_cov = np.cov(temps_mean_sin_adj)
    space_error_precision = np.linalg.inv(space_error_cov)

    def model(params, d):
        return -params[0] / d
    
    param = [16.25]

    L = model(param, dist)
    G = -ec.closeness_matrix(sh.STATIONS, 1000, 8)
    np.fill_diagonal(G, -np.sum(G, axis=0))
    np.fill_diagonal(L, 0)
    np.fill_diagonal(L, -np.sum(L, axis=0))
    C = np.linalg.inv(space_model_cov)
    ###########
    
    eigvals, eigvecs = np.linalg.eigh(space_model_cov)
    # eigvals = eigvals

    GFT = eigvecs.T
    spectra = GFT @ temps_mean_sin_adj

    mean_spectra = np.mean(np.pow(spectra, 2), axis=1)

    plt.scatter(eigvals, mean_spectra, label='mean power spectrum')

    plt.title('mean of signal\'s spatial power spectrum')
    plt.xlabel('eigenvalue')
    plt.ylabel('Power (degC^2)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/{stat}_spatial_spectra.png', bbox_inches='tight')
    plt.close()

    noise_spectra = GFT @ np.random.normal(size=temps.shape)

    mean_noise_spectra = np.mean(np.pow(noise_spectra, 2), axis=1)

    plt.scatter(eigvals, mean_noise_spectra, label='mean power spectrum')

    plt.title('mean of noise\'s spatial power spectra')
    plt.xlabel('eigenvalue')
    plt.ylabel('Power (degC^2)')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0, 1.5])
    plt.legend()

    plt.savefig(f'{config.PLOTS_DIR}/autoregression_fit_imgs/noise_spatial_spectra.png', bbox_inches='tight')
    plt.close()

    # temp_rmse = s.rmse(temps_mean_sin_adj)
    # noise_percent = np.array([1, 5, 10, 20, 50, 100])
    # for per in noise_percent:
    #     wiener_filter = np.divide(mean_spectra, mean_spectra + (temp_rmse * per / 100) ** 2)
    #     noise = np.random.normal(scale=(temp_rmse * per / 100), size=temps.shape)
    #     spectrum = GFT @ (temps_mean_sin_adj + noise)
    #     reconstructed = GFT.T @ (spectrum * wiener_filter[:, np.newaxis])
    #     print(f'noise {per:.2f}%,\t RMSE {100*s.rmse(temps_mean_sin_adj - reconstructed) / temp_rmse:.2f}%')
    
    temp_rmse = s.rmse(temps_mean_sin_adj)
    noise_percent = np.array([1, 5, 10, 20, 50, 100])
    for op in [L, G, space_error_precision, C]:
        eigvals, eigvecs = np.linalg.eigh(op)
        GFT = eigvecs.T
        spectra = GFT @ temps_mean_sin_adj
        mean_spectra = np.mean(np.pow(spectra, 2), axis=1)
        print('Measured spectra filter:')
        for per in noise_percent:
            wiener_filter = np.divide(mean_spectra, mean_spectra + (temp_rmse * per / 100) ** 2)
            noise = np.random.normal(scale=(temp_rmse * per / 100), size=temps.shape)
            spectrum = GFT @ (temps_mean_sin_adj + noise)
            reconstructed = GFT.T @ (spectrum * wiener_filter[:, np.newaxis])
            print(f'noise {per:.2f}%,\t RMSE {100*s.rmse(temps_mean_sin_adj - reconstructed) / temp_rmse:.2f}%')
        
        print('Implicit spectra filter:')
        for per in noise_percent:
            wiener_filter = np.reciprocal(1 + eigvals*(temp_rmse * per / 100) ** 2)
            noise = np.random.normal(scale=(temp_rmse * per / 100), size=temps.shape)
            spectrum = GFT @ (temps_mean_sin_adj + noise)
            reconstructed = GFT.T @ (spectrum * wiener_filter[:, np.newaxis])
            print(f'noise {per:.2f}%,\t RMSE {100*s.rmse(temps_mean_sin_adj - reconstructed) / temp_rmse:.2f}%')

def direct_error_regressed_on_spatial_data():
    if not _INIT:
        print(f'Regress temp module uninitialised, run {__name__}.init(stat) with stat in {{\'max\', \'min\', \'mean\'}}')
        return
    
    data = temps_mean_sin_adj.T[1:,:]
    X = temps_mean_sin_adj.T[:-1,:]
    spatial_params, pred = s.regress(X, data)

    RMSE = s.rmse(data - pred)

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

    im3 = ax3.imshow(ec.distance_matrix(sh.STATIONS), cmap='RdBu')
    plt.colorbar(im3, ax=ax3, label='Distances')
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Row Index')
    ax3.set_title('Distances Matrix')

    plt.tight_layout()
    fig.savefig(f'{config.PLOTS_DIR}/spatial_fit_imgs/{stat}_error_regressed_on_spatial_data.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    init('mean')
    plot_all()