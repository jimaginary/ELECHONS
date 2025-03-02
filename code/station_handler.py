import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

git_root = '..'

stations = pd.read_csv(f'{git_root}/datasets/acorn_stations', usecols=['station number', 'station name', 'lat', 'long'])

stats = ['max', 'min', 'mean']

latest_start = '1975-03-01'
overlap_length = 17838

def get_station(station):
    station = str(station)
    station = '0'*(6 - len(station)) + station
    return stations.loc[station, 'station number']

# stat in {max, min, mean}
def get_timeseries(station, stat):
    station = str(station)
    station = '0'*(6 - len(station)) + station
    return pd.read_csv(f'{git_root}/datasets/filled_acorn_sat_v2.5.0_daily_t{stat}/t{stat}.{station}.daily.csv')

def get_fft(station, stat):
    station = str(int(station))
    return pd.read_csv(f'{git_root}/datasets/fft_t{stat}/t{stat}.{station}.csv')

def get_full_stat_name(stat):
    return {
        'max': 'maximum',
        'min': 'minimum',
        'mean': 'mean'
    }[stat]