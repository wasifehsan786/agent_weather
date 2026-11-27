"""Training pipeline: baseline + Ridge + Random Forest, time-based validation, model registry.

  python -m agentweather.train                          # read Hopsworks, train, register if better
  python -m agentweather.train --no-register            # train and report only
  python -m agentweather.train --features-csv f.csv     # test on a local file

What happens:
 1. Read the feature table and keep rows that have a 24 h target.
 2. Time split: the last VAL_DAYS days are validation; a 24 h gap before them is dropped so a
    training target never overlaps the validation period.
 3. For every metric: score the naive baseline (value right now = value 24 h before the target
    time), then fit Ridge and Random Forest on the training part and score them on validation
    with RMSE, MAE and R2.
 4. Register the better of the two per metric, but only if it beats the model already in the
    registry on THIS validation window (the old model is re-scored, not just its stored number).
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import LIMITS, RAIN_THRESHOLD_MM, TARGET_METRICS

log = logging.getLogger("agentweather.train")

VAL_DAYS = 120
GAP_H = 24
SEED = 42
# Everything we forecast: the 8 regression targets plus the rain probability.
METRICS = TARGET_METRICS + ["rain"]
NOT_FEATURES = {"city", "ts", "has_target", "is_complete", "n_imputed", "wind_deg"}


def target_col(metric: str) -> str:
    return "target_rain_24h" if metric == "rain" else f"target_{metric}_24h"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES and not c.startswith("target_")
            and pd.api.types.is_numeric_dtype(df[c])]


def usable_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows with a known target and no missing feature."""
    need = cols + [target_col(m) for m in METRICS]
    return df[df["has_target"] == 1].dropna(subset=need).sort_values("ts").reset_index(drop=True)


def split_time(df: pd.DataFrame, val_days: int = VAL_DAYS, gap_h: int = GAP_H):
    """Last val_days days = validation; gap_h hours before it are thrown away."""
    val_start = df["ts"].max() - pd.Timedelta(days=val_days)
    train = df[df["ts"] < val_start - pd.Timedelta(hours=gap_h)]
    val = df[df["ts"] >= val_start]
    return train, val


def naive_prediction(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Tomorrow looks like now."""
    if metric == "rain":
        return (df["precip_mm"] >= RAIN_THRESHOLD_MM).astype("float64").to_numpy()
    return df[metric].to_numpy()


def clip(pred: np.ndarray, metric: str) -> np.ndarray:
    lo, hi = (0.0, 1.0) if metric == "rain" else LIMITS.get(metric, (None, None))
    if lo is None and hi is None:   # e.g. temperature has no hard limits
        return pred
    return np.clip(pred, lo, hi)


def score(y_true, y_pred) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def make_model(family: str, metric: str):
    if metric == "rain":
        if family == "ridge":   # logistic regression is the linear family for a yes/no target
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features=0.33,
                                      max_samples=0.6, n_jobs=-1, random_state=SEED)
    if family == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000],
                                                       cv=TimeSeriesSplit(3)))
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.33,
                                 max_samples=0.6, n_jobs=-1, random_state=SEED)


def predict(model, X, metric: str) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1] if metric == "rain" else model.predict(X)
    return clip(np.asarray(raw, dtype="float64"), metric)


def train_all(df: pd.DataFrame, val_days: int = VAL_DAYS):
    """Return (results table, fitted models, validation frame, feature columns)."""
    cols = feature_columns(df)
    data = usable_rows(df, cols)
    train, val = split_time(data, val_days)
    log.info("Features: %d | train rows: %d (%s to %s) | validation rows: %d (%s to %s)",
             len(cols), len(train), train["ts"].min(), train["ts"].max(),
             len(val), val["ts"].min(), val["ts"].max())

    rows, models = [], {}
    for metric in METRICS:
        y_tr, y_va = train[target_col(metric)], val[target_col(metric)]
        naive = score(y_va, naive_prediction(val, metric))
        rows.append({"metric": metric, "model": "naive", **naive, "skill": 0.0})
        for family in ("ridge", "random_forest"):
            model = make_model(family, metric).fit(train[cols], y_tr)
            s = score(y_va, predict(model, val[cols], metric))
            s["skill"] = 1 - s["rmse"] / naive["rmse"] if naive["rmse"] > 0 else float("nan")
            rows.append({"metric": metric, "model": family, **s})
            models[(metric, family)] = model
            log.info("%-14s %-13s RMSE %.3f  MAE %.3f  R2 %.3f  skill vs naive %+.1f%%",
                     metric, family, s["rmse"], s["mae"], s["r2"], 100 * s["skill"])
    return pd.DataFrame(rows), models, val, cols   # not rounded: the registry compares exact scores


def best_per_metric(results: pd.DataFrame) -> dict:
    """Lowest validation RMSE among the real models (the naive baseline is not a model)."""
    real = results[results["model"] != "naive"]
    return {m: g.sort_values("rmse").iloc[0]["model"] for m, g in real.groupby("metric")}


# ---------- Model registry ----------

def model_name(metric: str) -> str:
    return f"aw_{metric}_24h"


MIN_GAIN = 0.005   # the new model must be at least 0.5% better; ignores noise, avoids churn


def should_replace(new_rmse: float, current_rmse: float | None,
                   min_gain: float = MIN_GAIN) -> bool:
    """Replace only when the new model is clearly better (or there is no current model)."""
    return current_rmse is None or new_rmse < current_rmse * (1 - min_gain)


def current_model_rmse(mr, metric: str, val: pd.DataFrame):
    """Re-score the registered model on today's validation window. None if there is none."""
    try:
        old = mr.get_best_model(model_name(metric), "val_rmse", "min")
        if old is None:
            return None
        folder = Path(old.download())
        model = joblib.load(folder / "model.joblib")
        cols = json.loads((folder / "columns.json").read_text())
        pred = predict(model, val[cols], metric)
        return score(val[target_col(metric)], pred)["rmse"]
    except Exception as exc:  # noqa: BLE001 - no model yet, or an unreadable one
        log.info("No usable registered model for %s (%s).", metric, type(exc).__name__)
        return None


def register(mr, metric: str, family: str, model, cols: list[str], row: pd.Series,
             naive_rmse: float, example: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(model, Path(tmp) / "model.joblib")
        (Path(tmp) / "columns.json").write_text(json.dumps(cols))
        entry = mr.sklearn.create_model(
            name=model_name(metric),
            metrics={"val_rmse": float(row["rmse"]), "val_mae": float(row["mae"]),
                     "val_r2": float(row["r2"]), "naive_rmse": float(naive_rmse)},
            description=f"{family} forecasting {metric} 24 h ahead for {config.CITY.name}",
            input_example=example,
        )
        entry.save(tmp)


def register_all(mr, results, models, val, cols) -> list[dict]:
    decisions = []
    for metric, family in best_per_metric(results).items():
        row = results[(results["metric"] == metric) & (results["model"] == family)].iloc[0]
        naive_rmse = results[(results["metric"] == metric)
                             & (results["model"] == "naive")].iloc[0]["rmse"]
        current = current_model_rmse(mr, metric, val)
        replace = should_replace(float(row["rmse"]), current)
        if replace:
            register(mr, metric, family, models[(metric, family)], cols, row, naive_rmse,
                     val[cols].head(1))
        decisions.append({"metric": metric, "best_family": family, "new_rmse": float(row["rmse"]),
                          "current_rmse": current, "registered": replace})
        log.info("%-14s best=%s new RMSE %.3f vs current %s -> %s", metric, family,
                 row["rmse"], "none" if current is None else f"{current:.3f}",
                 "REGISTERED" if replace else "kept the old model")
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", help="read this CSV instead of Hopsworks (testing)")
    parser.add_argument("--no-register", action="store_true", help="train and report only")
    parser.add_argument("--val-days", type=int, default=VAL_DAYS)
    parser.add_argument("--out", default="train_output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    mr = None
    if args.features_csv:
        df = pd.read_csv(args.features_csv, parse_dates=["ts"])
    else:
        from .feature_pipeline import FEATURES_FG_NAME, FEATURES_FG_VERSION
        from .hopsworks_login import login

        project = login()
        df = project.get_feature_store().get_feature_group(
            FEATURES_FG_NAME, version=FEATURES_FG_VERSION).read()
        if not args.no_register:
            mr = project.get_model_registry()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    log.info("Read %d feature rows", len(df))

    results, models, val, cols = train_all(df, args.val_days)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shown = results.round(4)
    shown.to_csv(out / "metrics.csv", index=False)
    (out / "metrics.md").write_text(shown.to_string(index=False), encoding="utf-8")
    print(shown.to_string(index=False))

    if mr is not None:
        decisions = register_all(mr, results, models, val, cols)
        pd.DataFrame(decisions).to_csv(out / "registry_decisions.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
