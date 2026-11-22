from datetime import datetime, timezone

import pandas as pd

from agentweather import backfill, sources


WEATHER = {"hourly": {
    "time": ["2026-11-20T00:00", "2026-11-20T01:00", "2026-11-20T02:00"],
    "temperature_2m": [10.0, 9.5, None],
    "relative_humidity_2m": [80, 82, 85],
    "precipitation": [0.0, 0.0, 0.2],
    "wind_speed_10m": [1.0, 1.2, 1.1],
    "wind_direction_10m": [200, 210, 220],
}}
AIR = {"hourly": {
    "time": ["2026-11-20T00:00", "2026-11-20T01:00", "2026-11-20T02:00"],
    "uv_index": [0, 0, 0], "us_aqi": [150, 160, 155],
    "pm2_5": [60.0, 65.0, 62.0], "pm10": [90.0, 95.0, 92.0],
}}


def test_parse_history_matches_raw_schema():
    df = backfill.parse_history(WEATHER, AIR)
    assert list(df.columns) == ["city", "ts"] + sources.RAW_COLUMNS + ["ow_ok", "aq_ok", "om_ok"]
    assert len(df) == 3
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["om_us_aqi"].iloc[1] == 160.0
    assert pd.isna(df["om_temp_c"].iloc[2])
    assert pd.isna(df["ow_temp_c"].iloc[0])
    assert (df["om_ok"] == 1).all() and (df["ow_ok"] == 0).all() and (df["aq_ok"] == 0).all()
    assert all(df[c].dtype == "float64" for c in sources.RAW_COLUMNS)


def test_keep_missing_hours_skips_existing_and_future():
    df = backfill.parse_history(WEATHER, AIR)
    existing = pd.Series(pd.to_datetime(["2026-11-20T01:00"], utc=True))
    now = datetime(2026, 11, 20, 1, 30, tzinfo=timezone.utc)
    out = backfill.keep_missing_hours(df, existing, now)
    # 00:00 is kept; 01:00 exists; 02:00 is in the future
    assert list(out["ts"]) == [pd.Timestamp("2026-11-20 00:00", tz="UTC")]
