"""One small HTTP helper so every API call gets the same timeout and retries."""
import logging
import time

import requests

from . import config

log = logging.getLogger(__name__)


def get_json(url: str, params: dict) -> dict:
    """GET a URL and return parsed JSON. Retries on network errors and 5xx/429.

    Params are never logged because they contain API keys.
    """
    last_error: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT_S)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt,
                        config.MAX_RETRIES, type(exc).__name__)
            if attempt < config.MAX_RETRIES:
                time.sleep(config.BACKOFF_S * attempt)
    raise RuntimeError(f"GET {url} failed after {config.MAX_RETRIES} attempts") from last_error
