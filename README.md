# AgentWeather

24-hour forecasts of AQI, temperature, precipitation (mm and rain probability), humidity,
UV Index and wind (speed and direction) for Islamabad, built on a serverless stack:
Hopsworks (feature store and model registry), GitHub Actions, LangGraph agents and Streamlit.

## Status

- [x] Week 1: hourly raw ingestion (OpenWeather, AQICN, Open-Meteo) into Hopsworks
- [x] Week 2: historical backfill (Jan 2023 to now) and feature pipeline (122 features, 24h targets)
- [ ] Week 3: EDA, naive baseline, Ridge and Random Forest
- [ ] Week 4: SARIMAX/Prophet, LSTM, model registry, daily training, SHAP
- [ ] Weeks 5-6: agent layer (forecast, alert, weather report, drift) and logs
- [ ] Week 7: dashboard
- [ ] Week 8: final report

## How data flows

```
OpenWeather + AQICN + Open-Meteo
        |  hourly (ingest.yml)
        v
Hopsworks: raw_observations   <-- backfill.py fills history and missing hours
        |  after each ingestion (features.yml)
        v
Hopsworks: weather_features   (fused, imputed, time + lag + rolling features, 24h targets)
```

Feature steps: grid (one row per hour), fuse by source priority, impute gaps of up to 6 hours,
time features, lags / rate of change / rolling stats, targets 24 hours ahead. The newest 24 hours
have no target yet and are used for forecasting only. Details are in `agentweather/features.py`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
pytest -q
python -m agentweather.ingest --dry-run            # fetch and print, no write
python -m agentweather.backfill --days 3 --dry-run # check Open-Meteo history
```

## Layout

```
agentweather/
  config.py            city and settings
  http.py              HTTP helper with retries
  sources.py           fetch + parse for each API
  ingest.py            hourly raw ingestion
  backfill.py          history and gap filling from Open-Meteo
  features.py          raw table -> features and targets
  feature_pipeline.py  reads raw, writes features to Hopsworks
tests/                 unit tests (no network needed)
docs/issues_and_fixes.md   running log for the final report
.github/workflows/
  ingest.yml           hourly ingestion
  features.yml         feature pipeline (after ingestion; or full rebuild by hand)
  backfill.yml         daily gap-fill; long backfill by hand
  tests.yml            tests on every push
```

## Secrets (GitHub: Settings > Secrets and variables > Actions)

`OPENWEATHER_API_KEY`, `AQICN_TOKEN`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `HOPSWORKS_HOST`
