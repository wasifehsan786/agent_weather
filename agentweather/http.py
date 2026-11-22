"""One small HTTP helper so every API call gets the same timeout and retries."""
import logging
import time

import requests

from . import config

log = logging.getLogger(__name__)


def get_json(url: str, params: dict, timeout: float | None = None,
             attempts: int | None = None, backoff: float | None = None) -> dict:
    """GET a URL and return parsed JSON. Retries on network errors and 5xx/429.

    Params are never logged because they contain API keys.
    """
    timeout = timeout or config.REQUEST_TIMEOUT_S
    attempts = attempts or config.MAX_RETRIES
    backoff = backoff or config.BACKOFF_S
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt,
                        attempts, type(exc).__name__)
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"GET {url} failed after {attempts} attempts") from last_error
