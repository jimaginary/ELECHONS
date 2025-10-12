import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

PLOTS_DIR = os.path.join(PROJECT_ROOT, "..", "plts")
LOG_DIR = os.path.join(PROJECT_ROOT, "..", "output", "logs")

EARTH_RADIUS_KM = 6371.0
W_YEARLY = 2 * 3.141592653589793 / 365.25

SCALE_KM = 1000

### change this to NOAA/BOM to use the USA/AUS data
DATASET = 'bom'
###

DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", DATASET)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

if DATASET == 'bom':
    FTP_SERVER = "ftp.bom.gov.au"
    FTP_DIR = "/anon/home/ncc/www/change/ACORN_SAT_daily/"

    ACORN_VERSION = 'v2.5.0'
    ACORN_PREFIX = f'acorn_sat_{ACORN_VERSION}_daily_t'

    # for some reason the station data is not stored in the up-to-date data folders on
    # BOM's FTP server so must be retrieved seperately
    STATION_LOCATION_FILE = f'acorn_sat_v2_daily_tmax'
    RAW_STATIONS_FILE = f'acorn_sat_v2_stations.txt'

    STATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, "acorn_stations.csv")
    MAX_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmax")
    MIN_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmin")
    MEAN_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmean")
    STAT_DIR = {"max": MAX_DIR, "min": MIN_DIR, "mean": MEAN_DIR}

    OVERLAP_LENGTH = 17838
    LATEST_START = "1975-03-01"

    STAT_TYPES = {"max": "maximum", "min": "minimum", "mean": "mean"}
elif DATASET == 'noaa':
    DOWNLOAD = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/"
    DOWNLOAD_STATIONS = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    DOWNLOAD_STATION_GUIDE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"

    RAW_STATIONS_FILE = os.path.join(RAW_DATA_DIR, "stations.txt")
    STATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, "stations.txt")
    TEMP_DIR = os.path.join(PROCESSED_DATA_DIR, "temps")
    RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "temps")

    OVERLAP_LENGTH = 23717
    LATEST_START = 19601229
    EARLIEST_END = 20250301

    STAT_DIR = {"mean" : TEMP_DIR}
    STAT_TYPES = {"mean" : "mean"}
else:
    raise ValueError('config dataset does not exist')

