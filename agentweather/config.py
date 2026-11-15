"""Central settings. Change the city here (or with env vars) and nowhere else."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    timezone: str


CITY = City(
    name=os.getenv("AW_CITY", "islamabad"),
    lat=float(os.getenv("AW_LAT", "33.6844")),
    lon=float(os.getenv("AW_LON", "73.0479")),
    timezone=os.getenv("AW_TZ", "Asia/Karachi"),
)

# Hopsworks feature group that stores one raw row per hour.
RAW_FG_NAME = "raw_observations"
RAW_FG_VERSION = 1

# HTTP behaviour for every API call.
REQUEST_TIMEOUT_S = 20
MAX_RETRIES = 3
BACKOFF_S = 2.0
