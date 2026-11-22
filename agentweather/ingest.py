"""Hourly raw ingestion: fetch all sources, build one row, store it in Hopsworks.

Run locally without writing anything:   python -m agentweather.ingest --dry-run
Run for real (needs Hopsworks keys):    python -m agentweather.ingest
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from . import config
from .sources import RAW_COLUMNS, SOURCES

log = logging.getLogger("agentweather.ingest")


def build_row(now: datetime | None = None) -> pd.DataFrame:
    """Call every source and return a one-row DataFrame with a fixed schema.

    A failing source leaves its columns empty (NaN) and sets its *_ok flag to 0,
    so API downtime is recorded instead of crashing the run.
    """
    now = now or datetime.now(timezone.utc)
    row: dict = {col: None for col in RAW_COLUMNS}
    for prefix, fetch in SOURCES.items():
        try:
            row.update(fetch())
            row[f"{prefix}_ok"] = 1
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the others
            log.error("Source %s failed: %s", prefix, type(exc).__name__)
            row[f"{prefix}_ok"] = 0

    df = pd.DataFrame([row])
    df[RAW_COLUMNS] = df[RAW_COLUMNS].astype("float64")
    for prefix in SOURCES:
        df[f"{prefix}_ok"] = df[f"{prefix}_ok"].astype("int64")
    df.insert(0, "city", config.CITY.name)
    # Floor to the hour so a re-run in the same hour overwrites instead of duplicating.
    df.insert(1, "ts", pd.Timestamp(now).floor("h"))
    return df


def _insert_with_retry(fg, df: pd.DataFrame, attempts: int = 4, wait_s: float = 20.0) -> None:
    """Hopsworks storage (HopsFS) sometimes times out; retry before giving up."""
    for attempt in range(1, attempts + 1):
        try:
            fg.insert(df, write_options={"wait_for_job": False})
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("Hopsworks insert failed (attempt %d/%d): %s",
                        attempt, attempts, type(exc).__name__)
            if attempt == attempts:
                raise
            time.sleep(wait_s * attempt)


def write_to_hopsworks(df: pd.DataFrame) -> None:
    import hopsworks  # imported here so --dry-run works without the package

    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT") or None,
        host=os.environ.get("HOPSWORKS_HOST") or None,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.RAW_FG_NAME,
        version=config.RAW_FG_VERSION,
        description="Raw hourly readings per source (OpenWeather, AQICN, Open-Meteo).",
        primary_key=["city", "ts"],
        event_time="ts",
        online_enabled=False,
    )
    _insert_with_retry(fg, df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and print the row, but do not write to Hopsworks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    df = build_row()
    ok = int(df[[f"{p}_ok" for p in SOURCES]].sum(axis=1).iloc[0])
    log.info("Row for %s at %s: %d/%d sources ok", config.CITY.name,
             df["ts"].iloc[0], ok, len(SOURCES))
    if ok == 0:
        log.error("Every source failed; nothing to store.")
        return 1

    if args.dry_run:
        print(df.T.to_string(header=False))
        return 0

    write_to_hopsworks(df)
    log.info("Stored 1 row in %s v%d", config.RAW_FG_NAME, config.RAW_FG_VERSION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
