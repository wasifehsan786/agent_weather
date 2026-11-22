"""Backfill real OpenWeather air-pollution history (free endpoint, available since 2020-11-27).

Why: the project brief asks for OpenWeather data, but OpenWeather's *live* API only gives
"now", and we only have ~40 live rows. OpenWeather's air-pollution *history* endpoint is
free, so we use it to fill the ow_* pollution columns for hours that already exist.

Rules:
  * Only existing rows are updated (the weather part of a row is never touched).
  * A value that is already there (a live reading) is never overwritten.
  * The ow_ok flag is not changed: it still means "the live OpenWeather call worked".

Examples:
  python -m agentweather.ow_history --days 3 --dry-run
  python -m agentweather.ow_history --start 2023-01-01
  python -m agentweather.ow_history --days 3          # daily gap-heal
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from . import config
from .http import get_json
from .sources import _num

log = logging.getLogger("agentweather.ow_history")

URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
CHUNK_DAYS = 30
FLUSH_ROWS = 5000
FETCH_TIMEOUT_S = 60
FETCH_ATTEMPTS = 5
FETCH_BACKOFF_S = 10

# OpenWeather component name -> raw table column (same names as the live ingest uses)
COMPONENTS = {
    "pm2_5": "ow_pm2_5", "pm10": "ow_pm10", "no2": "ow_no2",
    "o3": "ow_o3", "so2": "ow_so2", "co": "ow_co",
}
POLLUTION_COLS = ["ow_aqi_1to5"] + list(COMPONENTS.values())


def parse_history(payload: dict) -> pd.DataFrame:
    """One row per hour, UTC, with the ow_* pollution columns. Duplicate hours are collapsed."""
    rows = []
    for item in payload.get("list") or []:
        if "dt" not in item:
            continue
        comp = item.get("components", {})
        row = {"ts": pd.Timestamp(int(item["dt"]), unit="s", tz="UTC").floor("h"),
               "ow_aqi_1to5": _num(item.get("main", {}).get("aqi"))}
        for api_name, col in COMPONENTS.items():
            row[col] = _num(comp.get(api_name))
        rows.append(row)
    df = pd.DataFrame(rows, columns=["ts"] + POLLUTION_COLS)
    df[POLLUTION_COLS] = df[POLLUTION_COLS].astype("float64")
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def merge_missing(raw: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return the *full* raw rows that gain at least one new OpenWeather value.

    Hopsworks upserts whole rows by primary key, so we must send the complete row
    (with its Open-Meteo values) and not only the new columns.
    """
    if history.empty or raw.empty:
        return raw.iloc[0:0].copy()
    raw = raw.copy()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    hist = history.set_index("ts")
    out = raw.set_index("ts")
    common = out.index.intersection(hist.index)
    if len(common) == 0:
        return raw.iloc[0:0].copy()
    changed = pd.Series(False, index=common)
    for col in POLLUTION_COLS:
        new = hist.loc[common, col]
        old = out.loc[common, col]
        fill = old.isna() & new.notna()          # never overwrite a live reading
        out.loc[common[fill.values], col] = new[fill].values
        changed |= fill
    return out.loc[common[changed.values]].reset_index()[list(raw.columns)]


def fetch_chunk(first: date, last: date) -> pd.DataFrame:
    start = int(datetime.combine(first, time.min, tzinfo=timezone.utc).timestamp())
    end = int(datetime.combine(last, time.max, tzinfo=timezone.utc).timestamp())
    params = {"lat": config.CITY.lat, "lon": config.CITY.lon, "start": start, "end": end,
              "appid": os.environ["OPENWEATHER_API_KEY"]}
    payload = get_json(URL, params, timeout=FETCH_TIMEOUT_S,
                       attempts=FETCH_ATTEMPTS, backoff=FETCH_BACKOFF_S)
    return parse_history(payload)


def iter_chunks(start: date, end: date):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first day, YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="or: how many past days")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch one chunk and print it, do not touch Hopsworks")
    args = parser.parse_args()
    if not args.start and not args.days:
        parser.error("give --start YYYY-MM-DD or --days N")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    end = datetime.now(timezone.utc).date()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)

    if args.dry_run:
        first, last = next(iter_chunks(start, end))
        hist = fetch_chunk(first, last)
        log.info("Dry run: %d hours from %s to %s", len(hist), hist["ts"].min(), hist["ts"].max())
        print(hist.head(3).T.to_string())
        return 0

    from .backfill import _feature_group
    from .ingest import _insert_with_retry

    fg = _feature_group()
    raw = fg.read()
    log.info("Read %d existing raw rows.", len(raw))

    buffer: list[pd.DataFrame] = []
    buffered = updated = 0
    failed: list[str] = []

    def flush() -> None:
        nonlocal buffer, buffered, updated
        if not buffer:
            return
        batch = pd.concat(buffer, ignore_index=True)
        _insert_with_retry(fg, batch)
        updated += len(batch)
        log.info("Updated %d rows (total %d)", len(batch), updated)
        buffer, buffered = [], 0

    for first, last in iter_chunks(start, end):
        try:
            hist = fetch_chunk(first, last)
        except Exception as exc:  # noqa: BLE001 - skip a bad chunk, keep the rest
            log.error("Chunk %s to %s failed: %s", first, last, type(exc).__name__)
            failed.append(f"{first}..{last}")
            continue
        rows = merge_missing(raw, hist)
        log.info("%s to %s: %d hours fetched, %d rows gain OpenWeather values",
                 first, last, len(hist), len(rows))
        if len(rows):
            buffer.append(rows)
            buffered += len(rows)
        if buffered >= FLUSH_ROWS:
            flush()
    flush()

    log.info("Done. Updated %d rows. Failed chunks: %s", updated, failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
