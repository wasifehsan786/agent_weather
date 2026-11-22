import pandas as pd

from agentweather import ow_history


def _payload():
    return {"list": [
        {"dt": 1763596800, "main": {"aqi": 4},
         "components": {"pm2_5": 80.0, "pm10": 120.0, "no2": 10.0, "o3": 5.0, "so2": 2.0, "co": 500.0}},
        {"dt": 1763600400, "main": {"aqi": 5}, "components": {"pm2_5": 90.0}},
        {"dt": 1763600400, "main": {"aqi": 5}, "components": {"pm2_5": 91.0}},  # duplicate hour
    ]}


def _raw():
    cols = ["city", "ts", "om_pm2_5"] + ow_history.POLLUTION_COLS
    ts = pd.to_datetime(["2025-11-20 00:00", "2025-11-20 01:00", "2025-11-20 02:00"], utc=True)
    df = pd.DataFrame({"city": "islamabad", "ts": ts, "om_pm2_5": [50.0, 51.0, 52.0]})
    for c in ow_history.POLLUTION_COLS:
        df[c] = float("nan")
    return df[cols]


def test_parse_history_collapses_duplicates_and_maps_columns():
    df = ow_history.parse_history(_payload())
    assert len(df) == 2
    assert df["ow_pm2_5"].iloc[0] == 80.0
    assert df["ow_pm2_5"].iloc[1] == 91.0          # last duplicate wins
    assert df["ow_aqi_1to5"].iloc[0] == 4.0
    assert pd.isna(df["ow_no2"].iloc[1])
    assert str(df["ts"].dt.tz) == "UTC"


def test_parse_history_empty_payload():
    df = ow_history.parse_history({})
    assert df.empty and "ow_pm2_5" in df.columns


def test_merge_fills_only_existing_hours_and_keeps_other_columns():
    raw = _raw()
    hist = ow_history.parse_history(_payload())
    out = ow_history.merge_missing(raw, hist)
    assert len(out) == 2                            # third raw hour has no history
    assert list(out.columns) == list(raw.columns)
    assert out["om_pm2_5"].tolist() == [50.0, 51.0]  # Open-Meteo values are kept
    assert out["ow_pm2_5"].tolist() == [80.0, 91.0]


def test_merge_never_overwrites_live_value():
    raw = _raw()
    raw.loc[0, "ow_pm2_5"] = 33.0
    hist = ow_history.parse_history(_payload())
    out = ow_history.merge_missing(raw, hist)
    first = out[out["ts"] == pd.Timestamp("2025-11-20 00:00", tz="UTC")].iloc[0]
    assert first["ow_pm2_5"] == 33.0                # live value kept
    assert first["ow_pm10"] == 120.0                # missing columns still filled


def test_merge_second_run_changes_nothing():
    raw = _raw()
    hist = ow_history.parse_history(_payload())
    first = ow_history.merge_missing(raw, hist)
    raw2 = raw.set_index("ts")
    raw2.update(first.set_index("ts"))
    assert ow_history.merge_missing(raw2.reset_index(), hist).empty
