import pandas as pd
import glob
import sys
import station_handler as sh

# sys.argv[1] should be full stat name {maximum, minimum, mean}

csv_files = glob.glob('*.csv')
for file in csv_files:
    df = pd.read_csv(file, usecols=['date', f'{sys.argv[1]} temperature (degC)'])

    df['date'] = pd.to_datetime(df['date'])

    start_date = pd.to_datetime(sh.latest_start)
    end_date = df['date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df = df.set_index('date').reindex(full_date_range).reset_index()
    df = df.rename(columns={'index': 'date'})

    nan_mask = df[f'{sys.argv[1]} temperature (degC)'].isna()

    # nan_groups = (nan_mask != nan_mask.shift(fill_value=False)).cumsum()
    # nan_lengths = nan_mask.groupby(nan_groups).sum()
    # max_stretch = max(nan_lengths)
    # if max_stretch > 100:
    #     max_group = nan_lengths.idxmax()
    #     stretch_mask = nan_mask & (nan_groups == max_group)
    #     stretch_dates = df['date'][stretch_mask]
    #     print(f'{file} has max stretch of nans {max_stretch} between {stretch_dates.iloc[0]} and {stretch_dates.iloc[-1]}')

    df[f'{sys.argv[1]} temperature (degC)'] = df[f'{sys.argv[1]} temperature (degC)'].bfill()
    df[f'{sys.argv[1]} temperature (degC)'] = df[f'{sys.argv[1]} temperature (degC)'].ffill()
    df['was nan']=nan_mask

    df.to_csv(file)