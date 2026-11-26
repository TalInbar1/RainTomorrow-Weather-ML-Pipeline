"""
sensing.py
----------
Turns the raw Open-Meteo JSON responses (data/raw/*.json) into tables.
NO requests are sent to the API from this file - it only reads local files.

Outputs
* data/raw/sensed_data.csv          - one row per (location, date), all daily
                                      variables exactly as returned by the API.
* data/processed/final_dataset.csv  - sensed data + the target RainTomorrow.

RainTomorrow (per location, after sorting by date):
    1  if precipitation_sum of the NEXT day >= 1.0 mm
    0  otherwise
The next-day value is used ONLY to build the label and is not kept as a feature.
The last day of every location (no "tomorrow" available) is dropped.

The functions are written so that Part B can reuse them on raw test JSON files
(json_to_frame / load_raw_folder).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SENSED_CSV = RAW_DIR / "sensed_data.csv"
FINAL_CSV = PROCESSED_DIR / "final_dataset.csv"

RAIN_THRESHOLD_MM = 1.0

# Coordinates used in data_collection.py - used to identify the location of a
# response when the file name does not carry it (e.g. test files in Part B).
KNOWN_LOCATIONS = {
    "London":    (51.5085, -0.1257),
    "New York":  (40.7143, -74.0060),
    "Sydney":    (-33.8678, 151.2073),
    "Berlin":    (52.5244, 13.4105),
    "Tokyo":     (35.6895, 139.6917),
    "Vancouver": (49.2497, -123.1193),
}


def location_from_name(path: Path) -> str | None:
    """'New_York_2015.json' -> 'New York'."""
    m = re.match(r"(.+?)_\d{4}(_to_\d{4}-\d{2}-\d{2})?$", path.stem)
    return m.group(1).replace("_", " ") if m else None


def location_from_coords(lat: float, lon: float) -> str:
    """Nearest known city (the API snaps coordinates to its grid cell)."""
    return min(KNOWN_LOCATIONS,
               key=lambda c: (KNOWN_LOCATIONS[c][0] - lat) ** 2 + (KNOWN_LOCATIONS[c][1] - lon) ** 2)


def json_to_frame(payload: dict, location: str | None = None) -> pd.DataFrame:
    """Convert one API response to a DataFrame with 'location' and 'date' columns."""
    daily = payload["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if location is None:
        location = location_from_coords(payload["latitude"], payload["longitude"])
    df.insert(0, "location", location)
    return df


def load_raw_folder(folder: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "daily" not in payload:          # e.g. an error body - skip it
            print(f"  skipping {path.name}: no 'daily' block")
            continue
        frames.append(json_to_frame(payload, location_from_name(path)))
    if not frames:
        raise FileNotFoundError(f"No JSON responses found in {folder}")
    df = pd.concat(frames, ignore_index=True)
    n_before = len(df)
    # keep the last occurrence: later files (re-downloaded partial years) are newer
    df = (df.drop_duplicates(subset=["location", "date"], keep="last")
            .sort_values(["location", "date"])
            .reset_index(drop=True))
    print(f"  {len(frames)} files, {n_before} rows -> {len(df)} rows after removing "
          f"{n_before - len(df)} duplicate (location, date) rows")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create RainTomorrow inside each location and drop rows without a tomorrow."""
    df = df.sort_values(["location", "date"]).copy()
    nxt = df.groupby("location")["precipitation_sum"].shift(-1)
    nxt_date = df.groupby("location")["date"].shift(-1)
    consecutive = (nxt_date - df["date"]) == pd.Timedelta(days=1)
    df["RainTomorrow"] = (nxt >= RAIN_THRESHOLD_MM).astype("Int64")
    # no label when tomorrow is missing (last day of each location / gaps / NaN)
    df.loc[~consecutive | nxt.isna(), "RainTomorrow"] = pd.NA
    df = df.dropna(subset=["RainTomorrow"]).copy()
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading raw JSON files ...")
    sensed = load_raw_folder(RAW_DIR)
    sensed.to_csv(SENSED_CSV, index=False, date_format="%Y-%m-%d")
    print(f"  saved {SENSED_CSV.relative_to(ROOT)}  shape={sensed.shape}")

    final = add_target(sensed)
    final.to_csv(FINAL_CSV, index=False, date_format="%Y-%m-%d")
    print(f"  saved {FINAL_CSV.relative_to(ROOT)}  shape={final.shape}")
    print("  RainTomorrow distribution:",
          final["RainTomorrow"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
