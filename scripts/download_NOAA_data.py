from elechons import config
import gzip
import requests
import os
import io
import pandas as pd

def download():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    with open(config.STATIONS_FILE, 'wb') as f:
        f.write(requests.get(config.DOWNLOAD_STATION_GUIDE).content)
    
    stations = pd.read_csv(config.RAW_STATIONS_FILE, sep='\s+', header=None)
    print(stations)
    stations = stations[(stations[3]=='TAVG') & (stations[5] >= 2025) & (stations[4] <= 1960)]
    stations.to_csv(config.STATIONS_FILE, index=False, header=False)

    for station in stations[0]:
        href = station + '.csv.gz'
        path = os.path.join(config.RAW_DATA_DIR, station + '.csv')
        full_url = config.DOWNLOAD + href
        print("Downloading", href)
        file_r = requests.get(full_url)
        file_r.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(file_r.content)) as f_in:
            with open(path, "wb") as f:
                f.write(f_in.read())
    
    print('all datasets downloaded and extracted into elechons/data/raw/')

if __name__ == '__main__':
    download()
