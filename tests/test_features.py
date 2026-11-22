import numpy as np
import pandas as pd

from agentweather import features, sources


def make_raw(hours=240, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-11-01", periods=hours, freq="h", tz="UTC")
    h = np.arange(hours)
    df = pd.DataFrame({"city": "islamabad", "ts": ts})
    for c in sources.RAW_COLUMNS:
        df[c] = np.nan
    df["om_temp_c"] = 15 + 8 * np.sin(2 * np.pi * h / 24) + rng.normal(0, .2, hours)
    df["om_humidity_pct"] = 60 + 20 * np.cos(2 * np.pi * h / 24)
    df["om_precip_mm"] = np.where(h % 37 == 0, 1.5, 0.0)
    df["om_wind_speed_ms"] = 2 + np.abs(np.sin(h / 5))
    df["om_wind_deg"] = (h * 7) % 360
    df["om_uv_index"] = np.clip(6 * np.sin(2 * np.pi * (h - 6) / 24), 0, None)
    df["om_us_aqi"] = 150 + 30 * np.sin(h / 9)
    df["om_pm2_5"] = 60 + 5 * np.sin(h / 9)
    df["om_pm10"] = 90.0
    df["om_ok"] = 1
    df["ow_ok"] = 0
    df["aq_ok"] = 0
    return df


def test_missing_hours_short_gap_filled_long_gap_not():
    raw = make_raw()
    raw = raw.drop(index=[50, 51, 52]).drop(index=range(100, 110))  # 3 h and 10 h gaps
    feats = features.build_features(raw).set_index("ts")
    assert feats.index.is_monotonic_increasing and len(feats) == 240       # regular grid
    short = feats.loc[pd.Timestamp("2026-11-01", tz="UTC") + pd.Timedelta(hours=51)]
    assert short["is_complete"] == 1 and short["n_imputed"] > 0 and not np.isnan(short["temp_c"])
    long_ = feats.loc[pd.Timestamp("2026-11-01", tz="UTC") + pd.Timedelta(hours=105)]
    assert long_["is_complete"] == 0 and np.isnan(long_["temp_c"])         # not invented


def test_source_priority():
    raw = make_raw()
    raw.loc[10, "ow_temp_c"] = 99.0
    feats = features.build_features(raw).set_index("ts")
    t10 = pd.Timestamp("2026-11-01", tz="UTC") + pd.Timedelta(hours=10)
    t11 = t10 + pd.Timedelta(hours=1)
    assert feats.loc[t10, "temp_c"] == 99.0                                # OpenWeather wins
    assert feats.loc[t11, "temp_c"] == raw.loc[11, "om_temp_c"]            # fallback


def test_target_is_value_24_hours_later_and_last_day_is_forecast_only():
    feats = features.build_features(make_raw())
    assert np.allclose(feats["target_temp_c_24h"].iloc[:-24], feats["temp_c"].iloc[24:].to_numpy())
    assert (feats["has_target"].iloc[-24:] == 0).all()
    assert (feats["has_target"].iloc[:-24] == 1).all()
    assert set(feats["target_rain_24h"].dropna().unique()) <= {0.0, 1.0}
    assert feats["target_wind_deg_24h"].dropna().between(0, 360).all()


def test_no_leakage_features_ignore_the_future():
    raw = make_raw()
    base = features.build_features(raw)
    changed = raw.copy()
    changed.loc[150:, "om_temp_c"] += 50          # change everything from hour 150 on
    other = features.build_features(changed)
    cols = [c for c in base.columns if not c.startswith("target_") and c not in ("has_target",)]
    assert base.loc[:149, cols].equals(other.loc[:149, cols])


def test_wind_direction_becomes_unit_vector():
    feats = features.build_features(make_raw())
    assert np.allclose(feats["wind_sin"] ** 2 + feats["wind_cos"] ** 2, 1.0)


def test_columns_are_stable_between_runs():
    raw = make_raw()
    a = list(features.build_features(raw).columns)
    b = list(features.build_features(raw.iloc[:120]).columns)
    assert a == b
