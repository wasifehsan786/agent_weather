"""Backfill and gap-fill hourly history from Open-Meteo (free, no key).

Why: GitHub's hourly schedule skips runs, so the live table has holes, and training
needs a long history. This fills every missing hour with Open-Meteo data and never
touches hours that already exist (live rows from OpenWeather/AQICN are kept).

Backfilled rows are recognisable by the flags: om_ok=1, ow_ok=0, aq_ok=0.

Examples:
  python -m agentweather.backfill --days 3 --dry-run        # fetch, print, no write
  python -m agentweather.backfill --start 2025-01-01        # long history, writes
  python -m agentweather.backfill --days 3                  # daily gap-heal
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from . import config
from .http import get_json
from .sources import RAW_COLUMNS

log = logging.getLogger("agentweather.backfill")

CHUNK_DAYS = 90

WEATHER_MAP = {
    "temperature_2m": "om_temp_c",
    "relative_humidity_2m": "om_humidity_pct",
    "precipitation": "om_precip_mm",
    "wind_speed_10m": "om_wind_speed_ms",
    "wind_direction_10m": "om_wind_deg",
}
AIR_MAP = {
    "uv_index": "om_uv_index",
    "us_aqi": "om_us_aqi",
    "pm2_5": "om_pm2_5",
    "pm10": "om_pm10",
}


def _hourly_frame(payload: dict, mapping: dict) -> pd.DataFrame:
    """Turn an Open-Meteo 'hourly' block into a DataFrame indexed by UTC hour."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    data = {"ts": pd.to_datetime(times, utc=True)}
    for api_name, col in mapping.items():
        values = hourly.get(api_name)
        data[col] = pd.Series(values if values is not None else [None] * len(times),
                              dtype="float64")
    return pd.DataFrame(data)


def parse_history(weather: dict, air: dict) -> pd.DataFrame:
    """Combine weather + air-quality payloads into rows with the raw-table schema."""
    w = _hourly_frame(weather, WEATHER_MAP)
    a = _hourly_frame(air, AIR_MAP)
    df = w.merge(a, on="ts", how="outer").sort_values("ts").reset_index(drop=True)

    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df[RAW_COLUMNS] = df[RAW_COLUMNS].astype("float64")
    df["ow_ok"] = 0
    df["aq_ok"] = 0
    df["om_ok"] = 1
    for col in ("ow_ok", "aq_ok", "om_ok"):
        df[col] = df[col].astype("int64")
    df.insert(0, "city", config.CITY.name)
    return df[["city", "ts"] + RAW_COLUMNS + ["ow_ok", "aq_ok", "om_ok"]]


def fetch_history(start: date, end: date) -> pd.DataFrame:
    """Fetch [start, end] in chunks. Open-Meteo returns hourly data in UTC."""
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        loc = {"latitude": config.CITY.lat, "longitude": config.CITY.lon,
               "timezone": "UTC", "start_date": cursor.isoformat(),
               "end_date": chunk_end.isoformat()}
        weather = get_json(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            {**loc, "wind_speed_unit": "ms", "hourly": ",".join(WEATHER_MAP)},
        )
        air = get_json(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {**loc, "hourly": ",".join(AIR_MAP)},
        )
        frames.append(parse_history(weather, air))
        log.info("Fetched %s to %s (%d hours)", cursor, chunk_end, len(frames[-1]))
        cursor = chunk_end + timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def keep_missing_hours(history: pd.DataFrame, existing_ts: pd.Series,
                       now: datetime | None = None) -> pd.DataFrame:
    """Drop hours that already exist, hours in the future, and rows with no data at all."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("h")
    out = history[~history["ts"].isin(pd.to_datetime(existing_ts, utc=True))]
    out = out[out["ts"] <= now]
    has_data = out[[c for c in RAW_COLUMNS if c.startswith("om_")]].notna().any(axis=1)
    return out[has_data].reset_index(drop=True)


def _feature_group():
    import hopsworks

    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT") or None,
        host=os.environ.get("HOPSWORKS_HOST") or None,
    )
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=config.RAW_FG_NAME, version=config.RAW_FG_VERSION,
        description="Raw hourly readings per source (OpenWeather, AQICN, Open-Meteo).",
        primary_key=["city", "ts"], event_time="ts", online_enabled=False,
    )


def existing_hours(fg) -> pd.Series:
    try:
        return fg.select(["ts"]).read()["ts"]
    except Exception as exc:  # noqa: BLE001 - empty table or first run
        log.warning("Could not read existing hours (%s); treating table as empty.",
                    type(exc).__name__)
        return pd.Series([], dtype="datetime64[ns, UTC]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first day, YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="or: how many past days to check")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and summarise, do not touch Hopsworks")
    args = parser.parse_args()
    if not args.start and not args.days:
        parser.error("give --start YYYY-MM-DD or --days N")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    end = datetime.now(timezone.utc).date()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)

    history = fetch_history(start, end)
    if history.empty:
        log.error("No data returned.")
        return 1

    if args.dry_run:
        log.info("Dry run: %d hours from %s to %s", len(history),
                 history["ts"].min(), history["ts"].max())
        print(history.head(3).T.to_string())
        return 0

    from .ingest import _insert_with_retry

    fg = _feature_group()
    new_rows = keep_missing_hours(history, existing_hours(fg))
    log.info("%d of %d hours are missing and will be inserted", len(new_rows), len(history))
    for i in range(0, len(new_rows), 2000):
        _insert_with_retry(fg, new_rows.iloc[i:i + 2000])
        log.info("Inserted rows %d-%d", i, min(i + 2000, len(new_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
