# AgentWeather

24-hour forecasts of AQI, temperature, precipitation, humidity, UV Index and wind for
Islamabad, built on a serverless stack (Hopsworks, GitHub Actions, LangGraph, Streamlit).

## Status

- [x] Week 1: hourly raw ingestion (OpenWeather, AQICN, Open-Meteo) into Hopsworks
- [ ] Week 2: feature pipeline and historical backfill
- [ ] Weeks 3-4: models, registry, explainability
- [ ] Weeks 5-6: agent layer (forecast, alert, report, drift)
- [ ] Week 7: dashboard
- [ ] Week 8: report

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
pytest -q
python -m agentweather.ingest --dry-run   # fetch and print, no write
python -m agentweather.ingest             # write one row to Hopsworks
```

## Layout

```
agentweather/
  config.py     city and settings
  http.py       HTTP helper with retries
  sources.py    fetch + parse for each API
  ingest.py     hourly raw ingestion entry point
tests/          unit tests (no network needed)
.github/workflows/ingest.yml   hourly schedule
```
