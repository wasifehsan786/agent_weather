# Issues and fixes (running log for the final report)

| # | Date | Issue | Evidence | Fix |
|---|---|---|---|---|
| 1 | 14-15 Nov | Tests failed in CI: `No module named 'agentweather'` | Hourly ingestion #1-#4 red at the pytest step | Added `conftest.py` so pytest finds the package (a `pytest.ini` attempt failed on a formatting error) |
| 2 | 15 Nov | Hopsworks login/host: free cluster is on `eu-west.cloud.hopsworks.ai`, and project names are unique across the cluster | Project `agentweather` already existed | Added `HOPSWORKS_HOST` secret, project named `agentweather_wasif` |
| 3 | 17-18 Nov | Hopsworks storage time-out on write (`Connection to HopsFS failed after 3 attempts`) | Hourly ingestion #13 and #14 red; all 3 sources were OK | `_insert_with_retry`: 4 tries with growing waits |
| 4 | 15-22 Nov | GitHub's hourly schedule skipped most runs: about 22 runs in 157 hours | Actions run list | Gap-filling `backfill.py` (also runs daily). Planned: external hourly trigger |
| 5 | 22 Nov | AQICN has no history; OpenWeather weather history is not free | API docs and terms | Open-Meteo used for the historical backfill; AQICN kept as live-only source |
| 6 | 22 Nov | Open-Meteo `ReadTimeout` on 90-day requests | Backfill #2 red | 30-day chunks, 90 s timeout, 5 tries, write in batches (backfill then ran: 16,479 rows, 0 failed chunks) |
| 7 | 22 Nov | Hopsworks writes get slower as the table grows (2.5 to 6.5 min per 2,160 rows) | Backfill #4 log | Feature pipeline writes 12,000 rows per batch (full rebuild of 34,116 rows took about 1.5 min) |
| 8 | 22 Nov | Open issue: about 99.9% of history is Open-Meteo, only about 30-40 live OpenWeather/AQICN rows, so train and live data may differ (train/serve skew) | Row counts | To measure in EDA; options: calibration, OpenWeather pollution history backfill, reliable hourly trigger |
