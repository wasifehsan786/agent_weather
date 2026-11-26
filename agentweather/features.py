"""Turn the raw hourly table into model-ready features and 24-hour-ahead targets.

Steps (every step is documented because the report needs it):
 1. GRID     One row per hour. Missing hours become explicit empty rows.
 2. FUSE     One value per metric using SOURCE PRIORITY: the first available source wins.
 3. IMPUTE   Gaps of up to MAX_GAP_H hours are filled by linear interpolation.
             Longer gaps stay empty and the row is marked is_complete = 0 (not used for training).
 4. TIME     Hour, weekday, month and sine/cosine versions (local time).
 5. DERIVED  Lags, 1h/3h rate of change, rolling mean/std. They only look at the past.
 6. TARGETS  The same metric 24 hours later. The newest 24 hours have no target yet
             (has_target = 0) and are used for forecasting only.

Naive baseline for the report: the forecast for t+24 is simply the value at t
(i.e. the value 24 hours before the target time).
"""
import numpy as np
import pandas as pd

from . import config

HORIZON_H = 24
MAX_GAP_H = 6
RAIN_THRESHOLD_MM = 0.1
LAGS = (1, 3, 6, 24)
DIFFS = (1, 3)
ROLL_MEAN = (6, 24)
ROLL_STD = (24,)

# Source priority per metric (first one that has a value wins).
# Air quality (aqi, pm2_5, pm10) comes from Open-Meteo ONLY. Reason (EDA, 25 Nov 2026): the
# sources disagree a lot (PM2.5 mean 99 OpenWeather vs 38 Open-Meteo; AQI 53 AQICN vs 127
# Open-Meteo on live hours), and history exists only for Open-Meteo/OpenWeather. One source for
# history AND live keeps the target consistent. AQICN and OpenWeather pollution stay in the raw
# table for cross-checks, alerts and the report. Weather metrics keep OpenWeather first for now;
# they are to be re-checked when more live hours exist.
PRIORITY = {
    "aqi": ["om_us_aqi"],
    "temp_c": ["ow_temp_c", "om_temp_c"],
    "precip_mm": ["ow_rain_1h_mm", "om_precip_mm"],
    "humidity_pct": ["ow_humidity_pct", "om_humidity_pct"],
    "uv_index": ["om_uv_index"],
    "wind_speed_ms": ["ow_wind_speed_ms", "om_wind_speed_ms"],
    "wind_deg": ["ow_wind_deg", "om_wind_deg"],
    "pm2_5": ["om_pm2_5"],
    "pm10": ["om_pm10"],
}

# Metrics that get lags/rolling features, and metrics that we forecast.
BASE = ["aqi", "temp_c", "precip_mm", "humidity_pct", "uv_index",
        "wind_speed_ms", "wind_sin", "wind_cos", "pm2_5", "pm10"]
TARGET_METRICS = ["aqi", "temp_c", "precip_mm", "humidity_pct", "uv_index",
                  "wind_speed_ms", "wind_sin", "wind_cos"]

LIMITS = {"aqi": (0, None), "precip_mm": (0, None), "humidity_pct": (0, 100),
          "uv_index": (0, None), "wind_speed_ms": (0, None), "pm2_5": (0, None),
          "pm10": (0, None), "wind_sin": (-1, 1), "wind_cos": (-1, 1)}


def to_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """STEP 1: one row per hour from the first to the last timestamp."""
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("h")
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    return df.reindex(full).rename_axis("ts")


def fuse(grid: pd.DataFrame) -> pd.DataFrame:
    """STEP 2: source priority."""
    out = pd.DataFrame(index=grid.index)
    for metric, cols in PRIORITY.items():
        present = [c for c in cols if c in grid.columns]
        if present:
            out[metric] = grid[present].astype("float64").bfill(axis=1).iloc[:, 0]
        else:
            out[metric] = np.nan
    # Wind direction is an angle, so we model it as two numbers that behave well.
    rad = np.deg2rad(out.pop("wind_deg"))
    out["wind_sin"] = np.sin(rad)
    out["wind_cos"] = np.cos(rad)
    return out[BASE]


def _gap_length(series: pd.Series) -> pd.Series:
    """For each empty hour, how long is the run of empty hours it belongs to."""
    isna = series.isna()
    run_id = (~isna).cumsum()
    return isna.groupby(run_id).transform("sum").where(isna, 0)


def impute(fused: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """STEP 3: fill short gaps only. Returns the filled table and a per-row count."""
    filled = fused.copy()
    n_imputed = pd.Series(0, index=fused.index, dtype="int64")
    for col in fused.columns:
        s = fused[col]
        isna = s.isna()
        interp = s.interpolate(method="time", limit_area="inside")
        fill_mask = isna & (_gap_length(s) <= MAX_GAP_H) & interp.notna()
        filled.loc[fill_mask, col] = interp[fill_mask]
        n_imputed += fill_mask.astype("int64")
    for col, (lo, hi) in LIMITS.items():
        filled[col] = filled[col].clip(lower=lo, upper=hi)
    return filled, n_imputed


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """STEP 4: calendar features in local time."""
    local = index.tz_convert(config.CITY.timezone)
    hour = local.hour.to_numpy()
    doy = local.dayofyear.to_numpy()
    return pd.DataFrame({
        "hour": hour.astype("int64"),
        "day_of_week": local.dayofweek.to_numpy().astype("int64"),
        "month": local.month.to_numpy().astype("int64"),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


def derived_features(filled: pd.DataFrame) -> pd.DataFrame:
    """STEP 5: lags, rate of change, rolling stats. Row t only uses rows <= t."""
    cols = {}
    for m in BASE:
        x = filled[m]
        for k in LAGS:
            cols[f"{m}_lag{k}h"] = x.shift(k)
        for k in DIFFS:
            cols[f"{m}_diff{k}h"] = x - x.shift(k)
        for w in ROLL_MEAN:
            cols[f"{m}_mean{w}h"] = x.rolling(w, min_periods=w // 2).mean()
        for w in ROLL_STD:
            cols[f"{m}_std{w}h"] = x.rolling(w, min_periods=w // 2).std()
    return pd.DataFrame(cols, index=filled.index)


def targets(filled: pd.DataFrame) -> pd.DataFrame:
    """STEP 6: the value 24 hours later. Needs a regular hourly grid (to_grid makes one)."""
    out = {}
    for m in TARGET_METRICS:
        out[f"target_{m}_24h"] = filled[m].shift(-HORIZON_H)
    t = pd.DataFrame(out, index=filled.index)
    rain = t["target_precip_mm_24h"]
    t["target_rain_24h"] = (rain >= RAIN_THRESHOLD_MM).astype("float64").where(rain.notna())
    deg = np.degrees(np.arctan2(t["target_wind_sin_24h"], t["target_wind_cos_24h"])) % 360
    t["target_wind_deg_24h"] = deg
    return t


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw table in, feature table (one row per hour) out."""
    grid = to_grid(raw)
    filled, n_imputed = impute(fuse(grid))

    parts = [filled, time_features(filled.index), derived_features(filled), targets(filled)]
    feats = pd.concat(parts, axis=1)
    feats["n_imputed"] = n_imputed
    feats["is_complete"] = filled[BASE].notna().all(axis=1).astype("int64")
    target_cols = [c for c in feats.columns if c.startswith("target_")]
    feats["has_target"] = feats[target_cols].notna().all(axis=1).astype("int64")

    feats = feats.reset_index()
    feats.insert(0, "city", config.CITY.name)
    return feats
