from datetime import datetime, timezone

import pandas as pd

from agentweather import ingest, sources


def test_openweather_parse_dry_day_means_zero_rain():
    weather = {"main": {"temp": 18.4, "humidity": 55, "pressure": 1016},
               "wind": {"speed": 2.1, "deg": 310}, "clouds": {"all": 20}}
    pollution = {"list": [{"main": {"aqi": 4}, "components": {"pm2_5": 61.2, "pm10": 90}}]}
    out = sources.parse_openweather(weather, pollution)
    assert out["ow_temp_c"] == 18.4
    assert out["ow_rain_1h_mm"] == 0.0
    assert out["ow_aqi_1to5"] == 4.0
    assert out["ow_no2"] is None


def test_aqicn_parse_handles_dash_and_missing_pollutants():
    payload = {"status": "ok", "data": {"aqi": "-", "iaqi": {"pm25": {"v": 152}}}}
    out = sources.parse_aqicn(payload)
    assert out["aq_aqi"] is None
    assert out["aq_pm25"] == 152.0
    assert out["aq_o3"] is None


def test_aqicn_error_status_raises():
    try:
        sources.parse_aqicn({"status": "error", "data": "Invalid key"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_row_survives_a_failing_source(monkeypatch):
    def boom():
        raise RuntimeError("api down")

    monkeypatch.setitem(sources.SOURCES, "ow", lambda: {"ow_temp_c": 20.0})
    monkeypatch.setitem(sources.SOURCES, "aq", boom)
    monkeypatch.setitem(sources.SOURCES, "om", lambda: {"om_uv_index": 3.5})

    df = ingest.build_row(datetime(2026, 11, 14, 15, 42, tzinfo=timezone.utc))

    assert len(df) == 1
    assert df["ts"].iloc[0] == pd.Timestamp("2026-11-14 15:00", tz="UTC")
    assert (df["ow_ok"].iloc[0], df["aq_ok"].iloc[0], df["om_ok"].iloc[0]) == (1, 0, 1)
    assert pd.isna(df["aq_aqi"].iloc[0])
    assert df["om_uv_index"].iloc[0] == 3.5
    assert list(df.columns[:2]) == ["city", "ts"]
    assert all(df[c].dtype == "float64" for c in sources.RAW_COLUMNS)
