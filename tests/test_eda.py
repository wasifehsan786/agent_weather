import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from agentweather import eda, sources  # noqa: E402


def _raw(hours=24 * 60):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    df = pd.DataFrame({"city": "islamabad", "ts": ts})
    for c in sources.RAW_COLUMNS:
        df[c] = np.nan
    h = np.arange(hours)
    df["om_temp_c"] = 15 + 8 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 1, hours)
    df["om_humidity_pct"] = np.clip(60 + rng.normal(0, 10, hours), 0, 100)
    df["om_precip_mm"] = np.where(rng.random(hours) < 0.1, rng.random(hours), 0.0)
    df["om_wind_speed_ms"] = np.abs(rng.normal(2, 1, hours))
    df["om_wind_deg"] = rng.uniform(0, 360, hours)
    df["om_uv_index"] = np.clip(5 * np.sin(2 * np.pi * h / 24), 0, None)
    df["om_us_aqi"] = np.abs(rng.normal(120, 30, hours))
    df["om_pm2_5"] = np.abs(rng.normal(50, 15, hours))
    df["om_pm10"] = df["om_pm2_5"] * 1.6
    df["ow_pm2_5"] = df["om_pm2_5"] * 1.1 + 2
    df["ow_pm10"] = df["om_pm10"]
    for c in ("ow_ok", "aq_ok", "om_ok"):
        df[c] = 1 if c == "om_ok" else 0
    return df


def test_source_comparison_bias():
    cmp_ = eda.source_comparison(_raw())
    pm = cmp_[cmp_["source_a"] == "ow_pm2_5"].iloc[0]
    assert pm["overlap_hours"] == 24 * 60
    assert pm["bias"] > 0 and pm["corr"] > 0.99
    # a pair with no overlap is listed but has no statistics
    temp = cmp_[cmp_["source_a"] == "ow_temp_c"].iloc[0]
    assert temp["overlap_hours"] == 0


def test_full_run_writes_report(tmp_path):
    text = eda.run(_raw(), tmp_path)
    assert "EDA summary" in text
    for name in ("summary.md", "data_quality.csv", "source_comparison.csv",
                 "naive_baseline.csv", "eda_aqi.png", "eda_temp_c.png", "eda_correlation.png"):
        assert (tmp_path / name).exists(), name
    quality = pd.read_csv(tmp_path / "data_quality.csv")
    assert quality["outside_limits"].sum() == 0
    naive = pd.read_csv(tmp_path / "naive_baseline.csv")
    assert (naive["naive_rmse"] >= 0).all() and len(naive) == 8
