from elechons import config
from elechons.data import bffill
import pandas as pd
import glob
import os

csv_files = glob.glob(os.path.join(config.RAW_DATA_DIR, '*.csv'))

for file in csv_files:
    df = pd.read_csv(file, header=None)
    df = df[df[2]=='TAVG']
    df = df[[1,3]]
    df.columns = ['date', 'mean temperature (degC)']
    df['mean temperature (degC)'] /= 10
    df.to_csv(os.path.join(config.TEMP_DIR, os.path.basename(file)), index=False)

bffill.fill('mean')