import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

STATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, "acorn_stations.csv")
MAX_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmax")
MIN_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmin")
MEAN_DIR = os.path.join(PROCESSED_DATA_DIR, "filled_acorn_sat_v2.5.0_daily_tmean")
STAT_DIR = {"max": MAX_DIR, "min": MIN_DIR, "mean": MEAN_DIR}

PLOTS_DIR = os.path.join(PROJECT_ROOT, "..", "output", "figures")
LOG_DIR = os.path.join(PROJECT_ROOT, "..", "output", "logs")

EARTH_RADIUS_KM = 6371.0
W_YEARLY = 2 * 3.141592653589793 / 365.25

SCALE_KM = 1000

OVERLAP_LENGTH = 17838
LATEST_START = "1975-03-01"

STAT_TYPES = {"max": "maximum", "min": "minimum", "mean": "mean"}

