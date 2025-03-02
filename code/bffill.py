import pandas as pd
import glob
import sys

# sys.argv[1] should be full stat name {maximum, minimum, mean}

csv_files = glob.glob('*.csv')
for file in csv_files:
    df = pd.read_csv(file)

    df[f'{sys.argv[1]} temperature (degC)'] = df[f'{sys.argv[1]} temperature (degC)'].fillna(method='bfill')
    df[f'{sys.argv[1]} temperature (degC)'] = df[f'{sys.argv[1]} temperature (degC)'].fillna(method='ffill')

    df.to_csv(file)