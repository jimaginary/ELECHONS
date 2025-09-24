import pandas as pd
import glob
import sys
import os
from elechons import config

# stat should be full stat name {maximum, minimum, mean}
def fill(stat):
    csv_files = glob.glob(os.path.join(config.TEMP_DIR, '*.csv'))
    for file in csv_files:
        df = pd.read_csv(file, usecols=['date', f'{stat} temperature (degC)'])

        df['date'] = pd.to_datetime(df['date'].astype(str))

        start_date = pd.to_datetime(str(config.LATEST_START))
        end_date = pd.to_datetime(str(config.EARLIEST_END))
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        df = df.set_index('date').reindex(full_date_range).reset_index()
        df = df.rename(columns={'index': 'date'})

        nan_mask = df[f'{stat} temperature (degC)'].isna()

        # nan_groups = (nan_mask != nan_mask.shift(fill_value=False)).cumsum()
        # nan_lengths = nan_mask.groupby(nan_groups).sum()
        # max_stretch = max(nan_lengths)
        # if max_stretch > 1000:
        #     max_group = nan_lengths.idxmax()
        #     stretch_mask = nan_mask & (nan_groups == max_group)
        #     stretch_dates = df['date'][stretch_mask]
        #     print(f'{file} has max stretch of nans {max_stretch} between {stretch_dates.iloc[0]} and {stretch_dates.iloc[-1]}')

        df[f'{stat} temperature (degC)'] = df[f'{stat} temperature (degC)'].bfill()
        df[f'{stat} temperature (degC)'] = df[f'{stat} temperature (degC)'].ffill()
        df['was nan']=nan_mask

        df.to_csv(file)