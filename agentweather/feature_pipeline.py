"""Read raw rows from Hopsworks, build features, write them to the feature group.

  python -m agentweather.feature_pipeline --days 3          # hourly job: refresh last 3 days
  python -m agentweather.feature_pipeline --full            # rebuild everything (after a backfill)
  python -m agentweather.feature_pipeline --dry-run         # build and print, do not write

Why refresh 3 days each hour: the target of an hour appears 24 hours later, so recent
rows must be written again once their target is known.
"""
import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from . import config
from .features import build_features

log = logging.getLogger("agentweather.feature_pipeline")

FEATURES_FG_NAME = "weather_features"
FEATURES_FG_VERSION = 1
BATCH_ROWS = 12000   # each write is slow and gets slower per call, so use few big ones


def _login():
    import hopsworks

    return hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT") or None,
        host=os.environ.get("HOPSWORKS_HOST") or None,
    ).get_feature_store()


def summarise(feats: pd.DataFrame) -> None:
    log.info("Feature table: %d rows x %d columns, %s to %s", len(feats), feats.shape[1],
             feats["ts"].min(), feats["ts"].max())
    log.info("Rows with target: %d | forecast-only (no target yet): %d | complete rows: %d | "
             "rows with imputed values: %d",
             int(feats["has_target"].sum()), int((feats["has_target"] == 0).sum()),
             int(feats["is_complete"].sum()), int((feats["n_imputed"] > 0).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=3, help="refresh the last N days (default 3)")
    parser.add_argument("--full", action="store_true", help="rebuild all history")
    parser.add_argument("--dry-run", action="store_true", help="do not write to Hopsworks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    fs = _login()
    raw_fg = fs.get_feature_group(config.RAW_FG_NAME, version=config.RAW_FG_VERSION)
    raw = raw_fg.read()
    raw = raw[raw["city"] == config.CITY.name]
    log.info("Read %d raw rows", len(raw))
    if raw.empty:
        log.error("No raw rows for %s", config.CITY.name)
        return 1

    feats = build_features(raw)
    if not args.full:
        cutoff = feats["ts"].max() - pd.Timedelta(days=args.days)
        feats = feats[feats["ts"] >= cutoff]
    summarise(feats)

    if args.dry_run:
        print(feats.tail(3).T.head(40).to_string())
        return 0

    from .ingest import _insert_with_retry

    fg = fs.get_or_create_feature_group(
        name=FEATURES_FG_NAME, version=FEATURES_FG_VERSION,
        description="Fused, imputed hourly features with 24h-ahead targets.",
        primary_key=["city", "ts"], event_time="ts", online_enabled=False,
    )
    for i in range(0, len(feats), BATCH_ROWS):
        _insert_with_retry(fg, feats.iloc[i:i + BATCH_ROWS])
        log.info("Wrote rows %d-%d", i, min(i + BATCH_ROWS, len(feats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
