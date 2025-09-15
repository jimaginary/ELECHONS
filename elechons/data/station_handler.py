import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elechons import config

git_root = '..'

STATIONS = pd.read_csv(config.STATIONS_FILE, usecols=['station number', 'station name', 'lat', 'long'])

def get_station(station):
    station = str(station)
    station = '0'*(6 - len(station)) + station
    return STATIONS.loc[station, 'station number']

# stat in {max, min, mean}
def get_timeseries(station, stat):
    station = str(station)
    station = '0'*(6 - len(station)) + station
    return pd.read_csv(config.STAT_DIR[stat] + f'/t{stat}.{station}.daily.csv')

def get_series_matrix(stat):
    timeseries = []
    for station in STATIONS['station number']:
        timeseries.append(get_timeseries(station, stat)[config.STAT_TYPES[stat] + ' temperature (degC)'])
    return np.array(timeseries)

def get_was_nan_matrix(stat):
    was_nan = []
    for station in STATIONS['station number']:
        was_nan.append(get_timeseries(station, stat)[f'was nan'])
    return np.array(was_nan)