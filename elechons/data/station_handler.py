import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from elechons import config

git_root = '..'

if config.DATASET == 'bom':
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
    
elif config.DATASET == 'noaa':
    STATIONS = pd.read_csv(config.STATIONS_FILE, header=None)
    STATIONS = STATIONS.drop(columns=[3,4,5])
    STATIONS.columns = ['station number', 'lat', 'long']
    STATIONS['station name'] = STATIONS['station number']
    csvs = {os.path.splitext(f)[0] for f in os.listdir(config.TEMP_DIR) if f.endswith('.csv')}
    STATIONS = STATIONS[STATIONS['station number'].isin(csvs)]
    STATIONS = STATIONS[(STATIONS['lat'] > 0) & (STATIONS['long'] < -30)]

    def get_station(station):
        return STATIONS.loc[station, 'station number']

    # stat in {max, min, mean}
    def get_timeseries(station, stat):
        return pd.read_csv(config.TEMP_DIR + f'/{station}.csv')

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