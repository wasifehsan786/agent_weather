"""Fetch and parse raw readings from each data source.

Each source has a pure ``parse_*`` function (easy to unit test) and a ``fetch_*``
function that calls the API. Column prefixes say where a value came from:
  ow_ = OpenWeather, aq_ = AQICN, om_ = Open-Meteo.
Sources are kept as separate columns here; fusing them is the feature pipeline's job.
"""
import os

from .config import CITY
from .http import get_json


def _num(value) -> float | None:
    """Convert to float; AQICN sends '-' when a reading is missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------- OpenWeather: current weather + air pollution ----------

def parse_openweather(weather: dict, pollution: dict) -> dict:
    main = weather.get("main", {})
    wind = weather.get("wind", {})
    item = (pollution.get("list") or [{}])[0]
    comp = item.get("components", {})
    return {
        "ow_temp_c": _num(main.get("temp")),
        "ow_humidity_pct": _num(main.get("humidity")),
        "ow_pressure_hpa": _num(main.get("pressure")),
        "ow_wind_speed_ms": _num(wind.get("speed")),
        "ow_wind_deg": _num(wind.get("deg")),
        "ow_clouds_pct": _num(weather.get("clouds", {}).get("all")),
        # OpenWeather omits "rain" when it is dry, so missing means 0 mm.
        "ow_rain_1h_mm": _num(weather.get("rain", {}).get("1h", 0.0)),
        "ow_aqi_1to5": _num(item.get("main", {}).get("aqi")),
        "ow_pm2_5": _num(comp.get("pm2_5")),
        "ow_pm10": _num(comp.get("pm10")),
        "ow_no2": _num(comp.get("no2")),
        "ow_o3": _num(comp.get("o3")),
        "ow_so2": _num(comp.get("so2")),
        "ow_co": _num(comp.get("co")),
    }


def fetch_openweather() -> dict:
    key = os.environ["OPENWEATHER_API_KEY"]
    base = "https://api.openweathermap.org/data/2.5"
    common = {"lat": CITY.lat, "lon": CITY.lon, "appid": key}
    weather = get_json(f"{base}/weather", {**common, "units": "metric"})
    pollution = get_json(f"{base}/air_pollution", common)
    return parse_openweather(weather, pollution)


# ---------- AQICN: station AQI ----------

def parse_aqicn(payload: dict) -> dict:
    if payload.get("status") != "ok":
        raise ValueError(f"AQICN status: {payload.get('status')}")
    data = payload.get("data", {})
    iaqi = data.get("iaqi", {})

    def sub(name: str) -> float | None:
        return _num(iaqi.get(name, {}).get("v"))

    return {
        "aq_aqi": _num(data.get("aqi")),
        "aq_pm25": sub("pm25"),
        "aq_pm10": sub("pm10"),
        "aq_no2": sub("no2"),
        "aq_o3": sub("o3"),
        "aq_so2": sub("so2"),
        "aq_co": sub("co"),
    }


def fetch_aqicn() -> dict:
    token = os.environ["AQICN_TOKEN"]
    url = f"https://api.waqi.info/feed/geo:{CITY.lat};{CITY.lon}/"
    return parse_aqicn(get_json(url, {"token": token}))


# ---------- Open-Meteo: UV, rain probability, second opinion (no key) ----------

def parse_openmeteo(weather: dict, air: dict) -> dict:
    cur = weather.get("current", {})
    air_cur = air.get("current", {})
    return {
        "om_temp_c": _num(cur.get("temperature_2m")),
        "om_humidity_pct": _num(cur.get("relative_humidity_2m")),
        "om_precip_mm": _num(cur.get("precipitation")),
        "om_wind_speed_ms": _num(cur.get("wind_speed_10m")),
        "om_wind_deg": _num(cur.get("wind_direction_10m")),
        "om_uv_index": _num(air_cur.get("uv_index")),
        "om_us_aqi": _num(air_cur.get("us_aqi")),
        "om_pm2_5": _num(air_cur.get("pm2_5")),
        "om_pm10": _num(air_cur.get("pm10")),
    }


def fetch_openmeteo() -> dict:
    loc = {"latitude": CITY.lat, "longitude": CITY.lon, "timezone": "UTC"}
    weather = get_json(
        "https://api.open-meteo.com/v1/forecast",
        {**loc, "wind_speed_unit": "ms",
         "current": "temperature_2m,relative_humidity_2m,precipitation,"
                    "wind_speed_10m,wind_direction_10m"},
    )
    air = get_json(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        {**loc, "current": "us_aqi,pm2_5,pm10,uv_index"},
    )
    return parse_openmeteo(weather, air)


# Every numeric column the raw table can contain, in a fixed order.
# A fixed schema matters: the feature store rejects rows whose columns change.
RAW_COLUMNS = (
    list(parse_openweather({}, {}))
    + ["aq_aqi", "aq_pm25", "aq_pm10", "aq_no2", "aq_o3", "aq_so2", "aq_co"]
    + list(parse_openmeteo({}, {}))
)

SOURCES = {
    "ow": fetch_openweather,
    "aq": fetch_aqicn,
    "om": fetch_openmeteo,
}
