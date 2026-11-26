"""
data_collection.py
------------------
Collects daily historical weather data from the Open-Meteo Historical Weather API
(https://archive-api.open-meteo.com/v1/archive) for six cities and stores every
response, exactly as received, as a JSON file under data/raw/.

Design decisions
* One request = one city x one calendar year. This keeps each response small,
  makes the cache granular (a failed request only costs one year) and keeps the
  number of requests small (~100 for 2010-today).
* Caching: a request whose output file already exists is never sent again.
  Completed years are named  <city>_<year>.json ; the current (incomplete) year
  is named <city>_<year>_to_<end_date>.json, so a later run fetches only the new
  partial slice. Overlapping days are removed later in sensing.py
  (duplicates by location + date).
* Rate limiting: the RateLimiter counts requests in sliding windows and sleeps
  BEFORE a limit would be exceeded. Open-Meteo counts long / wide requests as
  several "API calls" (fractional weighting by #variables/10 and #days/14),
  so we count both raw requests and the estimated weighted calls, and respect
  600/min, 5,000/hour and 10,000/day on both.
* Error handling:
    - HTTP 429 / 5xx / network error -> wait (exponential back-off) and retry.
    - HTTP 400 -> the request itself is wrong. We parse the API's error message;
      if it names an unknown daily variable we drop that variable and retry,
      if the end_date is out of range we move it back one day. Otherwise we stop.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

START_DATE = date(2010, 1, 1)
# The archive is updated with a delay of a few days; stay a week behind today.
END_DATE = date.today() - timedelta(days=7)

LOCATIONS = {
    "London":    (51.5085, -0.1257),
    "New York":  (40.7143, -74.0060),
    "Sydney":    (-33.8678, 151.2073),
    "Berlin":    (52.5244, 13.4105),
    "Tokyo":     (35.6895, 139.6917),
    "Vancouver": (49.2497, -123.1193),
}

REQUIRED_VARS = [
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "temperature_2m_mean", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_direction_10m_dominant",
]
# Additional variables that are physically related to rain formation.
EXTRA_VARS = [
    "relative_humidity_2m_mean", "dew_point_2m_mean", "pressure_msl_mean",
    "cloud_cover_mean", "sunshine_duration", "wind_gusts_10m_max",
    "snowfall_sum",
]
DAILY_VARS = REQUIRED_VARS + EXTRA_VARS

MAX_RETRIES = 6
TIMEOUT_S = 60


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Sliding-window limiter. Each window is (seconds, max_count)."""

    WINDOWS = {"minute": (60, 600), "hour": (3600, 5000), "day": (86400, 10000)}

    def __init__(self) -> None:
        # deque of (timestamp, weight)
        self.history: deque[tuple[float, float]] = deque()
        self.total_requests = 0
        self.total_weight = 0.0

    def _used(self, seconds: int, now: float, weighted: bool) -> float:
        return sum(w if weighted else 1 for t, w in self.history if now - t < seconds)

    def wait_for_slot(self, weight: float) -> None:
        while True:
            now = time.time()
            # drop entries older than the longest window
            while self.history and now - self.history[0][0] >= 86400:
                self.history.popleft()
            sleep_needed = 0.0
            for name, (seconds, limit) in self.WINDOWS.items():
                for weighted, add in ((False, 1), (True, weight)):
                    if self._used(seconds, now, weighted) + add > limit:
                        # earliest moment at which enough old entries expire
                        inside = [t for t, _ in self.history if now - t < seconds]
                        wait = seconds - (now - inside[0]) + 1 if inside else 1
                        print(f"  [rate-limit] {name} limit reached, sleeping {wait:.0f}s")
                        sleep_needed = max(sleep_needed, wait)
            if sleep_needed <= 0:
                return
            time.sleep(sleep_needed)

    def register(self, weight: float) -> None:
        self.history.append((time.time(), weight))
        self.total_requests += 1
        self.total_weight += weight


def estimate_weight(n_vars: int, start: date, end: date) -> float:
    """Open-Meteo's fractional call counting (approximation)."""
    n_days = (end - start).days + 1
    return max(1.0, n_vars / 10) * max(1.0, n_days / 14)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def fetch(session: requests.Session, limiter: RateLimiter, lat: float, lon: float,
          start: date, end: date, daily_vars: list[str]) -> str:
    """Send one GET request with retries. Returns the raw response body (JSON text)."""
    daily_vars = list(daily_vars)
    attempt = 0
    while True:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": ",".join(daily_vars), "timezone": "auto",
        }
        weight = estimate_weight(len(daily_vars), start, end)
        limiter.wait_for_slot(weight)
        limiter.register(weight)
        try:
            r = session.get(API_URL, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:          # network problem
            status, reason = None, str(e)
        else:
            status = r.status_code
            if status == 200:
                return r.text
            try:
                reason = r.json().get("reason", r.text)
            except ValueError:
                reason = r.text

        if status == 400:
            # The request is malformed: fix it instead of blindly retrying.
            bad_var = re.search(r"variable '([^']+)'|Cannot initialize \w+ from invalid String value (\w+)", reason)
            if bad_var:
                name = bad_var.group(1) or bad_var.group(2)
                if name in daily_vars:
                    print(f"  [400] unknown variable '{name}' -> removing it and retrying")
                    daily_vars.remove(name)
                    continue
            if "out of allowed range" in reason and end > start:
                end = end - timedelta(days=1)
                print(f"  [400] date out of range -> end_date moved to {end}")
                continue
            raise RuntimeError(f"HTTP 400 that cannot be fixed automatically: {reason}")

        attempt += 1
        if attempt > MAX_RETRIES:
            raise RuntimeError(f"Giving up after {MAX_RETRIES} retries: {status} {reason}")
        if status == 429 or status is None or status >= 500:
            wait = min(300, 5 * 2 ** (attempt - 1))
            print(f"  [{status}] {reason[:80]!r} -> retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Unexpected HTTP {status}: {reason}")


def year_chunks(start: date, end: date):
    for year in range(start.year, end.year + 1):
        s = max(start, date(year, 1, 1))
        e = min(end, date(year, 12, 31))
        complete = e == date(year, 12, 31)
        yield year, s, e, complete


def file_name(city: str, year: int, end: date, complete: bool) -> str:
    slug = city.replace(" ", "_")
    return f"{slug}_{year}.json" if complete else f"{slug}_{year}_to_{end.isoformat()}.json"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()
    skipped = 0
    with requests.Session() as session:
        for city, (lat, lon) in LOCATIONS.items():
            for year, s, e, complete in year_chunks(START_DATE, END_DATE):
                out = RAW_DIR / file_name(city, year, e, complete)
                if out.exists():
                    skipped += 1
                    continue
                print(f"#{limiter.total_requests + 1:>4} {city:<10} {s} -> {e}")
                body = fetch(session, limiter, lat, lon, s, e, DAILY_VARS)
                json.loads(body)                       # sanity check: valid JSON
                # saved byte-for-byte as received (no processing); the location
                # is encoded in the file name and recovered by sensing.py
                out.write_text(body, encoding="utf-8")
                time.sleep(0.2)   # be polite even below the limits
    print(f"\nDone. New requests: {limiter.total_requests} "
          f"(~{math.ceil(limiter.total_weight)} weighted API calls), "
          f"cached files skipped: {skipped}")


if __name__ == "__main__":
    main()
