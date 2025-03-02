import pandas as pd
import numpy as np
import station_handler as sh
import sys

freqs = np.fft.fftfreq(sh.overlap_length)
for stat in ['max', 'min', 'mean']:
    print()
    print(f'stat {stat}')
    for i, station in enumerate(sh.stations['station number']):
        sys.stdout.write(f"\rProgress: {i}/{len(sh.stations)}")
        sys.stdout.flush()

        timeseries = sh.get_timeseries(station, stat)[f'{sh.get_full_stat_name(stat)} temperature (degC)']
        fft = np.fft.fft(timeseries[-sh.overlap_length:])

        df = pd.DataFrame({'f (/days)': freqs, 'component': fft})
        df.to_csv(f'{sh.git_root}/datasets/fft_t{stat}/t{stat}.{station}.csv')
