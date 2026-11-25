"""One place to log in to Hopsworks, with retries.

Why: Hopsworks sometimes answers "503 Service Temporarily Unavailable" for a few minutes
(seen on 24 Nov 2026). Without a retry, that one short outage fails the whole job.
"""
import logging
import os
import time

log = logging.getLogger("agentweather.hopsworks_login")

ATTEMPTS = 5
WAIT_S = 30.0   # waits 30 s, 60 s, 90 s, 120 s between the 5 tries


def login(attempts: int = ATTEMPTS, wait_s: float = WAIT_S):
    """Return the Hopsworks project. Raises the last error if every try fails."""
    import hopsworks  # imported here so --dry-run works without the package

    for attempt in range(1, attempts + 1):
        try:
            return hopsworks.login(
                api_key_value=os.environ["HOPSWORKS_API_KEY"],
                project=os.environ.get("HOPSWORKS_PROJECT") or None,
                host=os.environ.get("HOPSWORKS_HOST") or None,
            )
        except KeyError:
            raise  # a missing secret will not fix itself, so do not retry
        except Exception as exc:  # noqa: BLE001
            log.warning("Hopsworks login failed (attempt %d/%d): %s",
                        attempt, attempts, type(exc).__name__)
            if attempt == attempts:
                raise
            time.sleep(wait_s * attempt)
