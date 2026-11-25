"""Exploratory data analysis: data quality, source comparison, per-metric charts.

Reads the raw table from Hopsworks, builds the same features the pipeline builds, then writes
a folder of CSV files, PNG charts and one summary.md. Nothing is written back to Hopsworks.

  python -m agentweather.eda                     # reads Hopsworks, writes eda_output/
  python -m agentweather.eda --raw-csv raw.csv   # test on a local file instead
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed on the CI machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from . import config  # noqa: E402
from .features import BASE, LIMITS, TARGET_METRICS, build_features  # noqa: E402

log = logging.getLogger("agentweather.eda")

# Metrics shown on the dashboard, with a unit for chart labels.
METRICS = {
    "aqi": "US AQI", "pm2_5": "PM2.5 (ug/m3)", "pm10": "PM10 (ug/m3)",
    "temp_c": "Temperature (C)", "precip_mm": "Precipitation (mm/h)",
    "humidity_pct": "Humidity (%)", "uv_index": "UV index",
    "wind_speed_ms": "Wind speed (m/s)", "wind_deg": "Wind direction (deg)",
}

# (OpenWeather or AQICN column, Open-Meteo column) measured on the same hour.
SOURCE_PAIRS = [
    ("ow_pm2_5", "om_pm2_5"), ("ow_pm10", "om_pm10"),
    ("aq_aqi", "om_us_aqi"),
    ("ow_temp_c", "om_temp_c"), ("ow_humidity_pct", "om_humidity_pct"),
    ("ow_wind_speed_ms", "om_wind_speed_ms"), ("ow_rain_1h_mm", "om_precip_mm"),
]


def data_quality(feats: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: how much is missing, the range, and values outside valid limits."""
    rows = []
    for m in METRICS:
        if m not in feats:
            continue
        x = feats[m]
        lo, hi = LIMITS.get(m, (None, None))
        bad = 0
        if lo is not None:
            bad += int((x < lo).sum())
        if hi is not None:
            bad += int((x > hi).sum())
        rows.append({
            "metric": m, "rows": len(x), "missing_pct": round(100 * x.isna().mean(), 3),
            "min": x.min(), "p1": x.quantile(0.01), "median": x.median(),
            "mean": x.mean(), "p99": x.quantile(0.99), "max": x.max(), "std": x.std(),
            "zero_pct": round(100 * (x == 0).mean(), 2), "outside_limits": bad,
        })
    return pd.DataFrame(rows).round(3)


def source_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare two sources on hours where both have a value. bias = first source - Open-Meteo."""
    rows = []
    for a, b in SOURCE_PAIRS:
        if a not in raw or b not in raw:
            continue
        both = raw[[a, b]].dropna()
        row = {"source_a": a, "source_b": b, "overlap_hours": len(both)}
        if len(both) >= 3:
            diff = both[a] - both[b]
            row.update({
                "bias": diff.mean(), "mae": diff.abs().mean(),
                "rmse": float(np.sqrt((diff ** 2).mean())),
                "corr": both[a].corr(both[b]),
                "mean_a": both[a].mean(), "mean_b": both[b].mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def source_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """How many rows each source supplied (the train/serve skew question)."""
    flags = [c for c in ("ow_ok", "aq_ok", "om_ok") if c in raw]
    out = {f: int(raw[f].sum()) for f in flags}
    out["rows_total"] = len(raw)
    pm = raw["ow_pm2_5"].notna().sum() if "ow_pm2_5" in raw else 0
    out["rows_with_openweather_pm2_5"] = int(pm)
    return pd.DataFrame([out])


def naive_baseline(feats: pd.DataFrame) -> pd.DataFrame:
    """Error of 'tomorrow looks like now' (the value 24 h before the target time)."""
    rows = []
    for m in TARGET_METRICS:
        t = f"target_{m}_24h"
        if m not in feats or t not in feats:
            continue
        pair = feats[[m, t]].dropna()
        if pair.empty:
            continue
        err = pair[m] - pair[t]
        rows.append({"metric": m, "n": len(pair), "naive_rmse": float(np.sqrt((err ** 2).mean())),
                     "naive_mae": err.abs().mean(), "target_std": pair[t].std()})
    return pd.DataFrame(rows).round(3)


def autocorr(x: pd.Series, max_lag: int = 72) -> list[float]:
    return [x.autocorr(k) for k in range(1, max_lag + 1)]


def metric_chart(feats: pd.DataFrame, m: str, label: str, out_dir: Path) -> None:
    """4 panels: distribution, daily pattern, monthly trend, autocorrelation."""
    x = feats.set_index("ts")[m]
    local = x.copy()
    local.index = local.index.tz_convert(config.CITY.timezone)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{label} - {config.CITY.name.title()}", fontsize=14)

    ax[0, 0].hist(x.dropna(), bins=60, color="#2a6f97")
    ax[0, 0].set_title("Distribution")
    ax[0, 0].set_xlabel(label)

    by_hour = local.groupby(local.index.hour).agg(["mean", "std"])
    ax[0, 1].plot(by_hour.index, by_hour["mean"], color="#2a6f97")
    ax[0, 1].fill_between(by_hour.index, by_hour["mean"] - by_hour["std"],
                          by_hour["mean"] + by_hour["std"], alpha=0.2, color="#2a6f97")
    ax[0, 1].set_title("Average by hour of day (local time, +/- 1 std)")
    ax[0, 1].set_xlabel("hour")

    monthly = local.resample("MS").mean()
    ax[1, 0].plot(monthly.index, monthly.values, color="#2a6f97")
    ax[1, 0].set_title("Monthly average")

    ac = autocorr(x.dropna() if x.isna().any() else x)
    ax[1, 1].bar(range(1, len(ac) + 1), ac, color="#2a6f97")
    ax[1, 1].axvline(24, color="#c1121f", linestyle="--", linewidth=1)
    ax[1, 1].set_title("Autocorrelation (red line = 24 h, our forecast horizon)")
    ax[1, 1].set_xlabel("lag (hours)")

    fig.tight_layout()
    fig.savefig(out_dir / f"eda_{m}.png", dpi=110)
    plt.close(fig)


def extra_charts(feats: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> None:
    cols = [m for m in METRICS if m in feats]
    corr = feats[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    fig.colorbar(im)
    ax.set_title("Correlation between metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_correlation.png", dpi=110)
    plt.close(fig)

    t = feats.set_index("ts")[cols].isna().resample("MS").mean() * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in cols:
        ax.plot(t.index, t[c], label=c)
    ax.set_title("Missing values per month (%) after imputation")
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_missing.png", dpi=110)
    plt.close(fig)

    if {"ow_pm2_5", "om_pm2_5"} <= set(raw.columns):
        both = raw[["ow_pm2_5", "om_pm2_5"]].dropna()
        if len(both) > 10:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(both["om_pm2_5"], both["ow_pm2_5"], s=3, alpha=0.2)
            top = float(both.max().max())
            ax.plot([0, top], [0, top], color="#c1121f", linewidth=1)
            ax.set_xlabel("Open-Meteo PM2.5")
            ax.set_ylabel("OpenWeather PM2.5")
            ax.set_title("PM2.5: two sources, same hours")
            fig.tight_layout()
            fig.savefig(out_dir / "eda_source_pm2_5.png", dpi=110)
            plt.close(fig)


def write_summary(out_dir: Path, feats, quality, compare, coverage, naive) -> str:
    lines = [
        f"# EDA summary - {config.CITY.name.title()}", "",
        f"- Hours in feature table: {len(feats)} ({feats['ts'].min()} to {feats['ts'].max()})",
        f"- Rows with a 24 h target: {int(feats['has_target'].sum())}",
        f"- Rows that needed imputation: {int((feats['n_imputed'] > 0).sum())}",
        f"- Rows still incomplete (long gaps): {int((feats['is_complete'] == 0).sum())}", "",
        "## Source coverage", coverage.to_string(index=False), "",
        "## Data quality", quality.to_string(index=False), "",
        "## Source comparison (first source minus Open-Meteo)", compare.to_string(index=False), "",
        "## Naive baseline (value 24 h ago)", naive.to_string(index=False), "",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def run(raw: pd.DataFrame, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = raw[raw["city"] == config.CITY.name] if "city" in raw else raw
    feats = build_features(raw)

    quality = data_quality(feats, raw)
    compare = source_comparison(raw)
    coverage = source_coverage(raw)
    naive = naive_baseline(feats)
    for name, frame in [("data_quality", quality), ("source_comparison", compare),
                        ("source_coverage", coverage), ("naive_baseline", naive)]:
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    for m, label in METRICS.items():
        if m in feats and feats[m].notna().any():
            metric_chart(feats, m, label, out_dir)
    extra_charts(feats, raw, out_dir)
    return write_summary(out_dir, feats, quality, compare, coverage, naive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-csv", help="read this CSV instead of Hopsworks (for testing)")
    parser.add_argument("--out", default="eda_output", help="output folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if args.raw_csv:
        raw = pd.read_csv(args.raw_csv, parse_dates=["ts"])
    else:
        from .feature_pipeline import _login

        fs = _login()
        raw = fs.get_feature_group(config.RAW_FG_NAME, version=config.RAW_FG_VERSION).read()
    log.info("Read %d raw rows", len(raw))

    text = run(raw, Path(args.out))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
