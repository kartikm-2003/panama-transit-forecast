"""
Ingest daily precipitation for the Chagres watershed via Open-Meteo.

Open-Meteo's archive endpoint serves ERA5 / ERA5-Land reanalysis with no
API key and a generous free tier (10k requests/day), which sidesteps the
Copernicus CDS signup for this project. Same underlying data as CDS's ERA5.

We sample a small grid (~6 points) spanning the Chagres basin -- upper
watershed, Alajuela, mid-basin, Gatun Lake, lower Chagres -- and store both
the per-point series and their unweighted mean as the basin-average.

Endpoint: https://archive-api.open-meteo.com/v1/archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Six points spread across the Chagres watershed. Coordinates are approximate
# but cover the key hydrologic sub-basins that feed Gatun Lake.
BASIN_POINTS: list[dict] = [
    {"id": "upper_chagres", "lat": 9.45, "lon": -79.55},
    {"id": "alajuela_lake", "lat": 9.35, "lon": -79.60},
    {"id": "boqueron",      "lat": 9.40, "lon": -79.75},
    {"id": "gamboa",        "lat": 9.12, "lon": -79.70},
    {"id": "gatun_east",    "lat": 9.20, "lon": -79.85},
    {"id": "gatun_west",    "lat": 9.25, "lon": -80.00},
]


def fetch_point(lat: float, lon: float, start: date, end: date,
                session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    r = session.get(ARCHIVE_URL, params=params, timeout=90)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily") or {}
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])).date,
        "precip_mm": daily.get("precipitation_sum", []),
    })
    return df


def fetch_basin(start: date, end: date, points: list[dict] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long-form per-point df, basin-average daily df)."""
    points = points or BASIN_POINTS
    session = requests.Session()

    long_rows: list[pd.DataFrame] = []
    for p in points:
        df = fetch_point(p["lat"], p["lon"], start, end, session)
        df.insert(0, "lon", p["lon"])
        df.insert(0, "lat", p["lat"])
        df.insert(0, "point_id", p["id"])
        long_rows.append(df)
        time.sleep(0.1)  # be gentle on the free endpoint

    long_df = pd.concat(long_rows, ignore_index=True)
    basin = (long_df.groupby("date", as_index=False)["precip_mm"]
                    .mean().rename(columns={"precip_mm": "precip_mm_basin_avg"}))
    return long_df, basin


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(long_df: pd.DataFrame, basin: pd.DataFrame,
                   out_root: Path, start: date, end: date) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "precipitation" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    basin_path = out_dir / "chagres_daily.csv"
    points_path = out_dir / "chagres_points.csv"
    basin.to_csv(basin_path, index=False, encoding="utf-8")
    long_df.to_csv(points_path, index=False, encoding="utf-8")

    manifest = {
        "source": "Open-Meteo Archive API (ERA5 / ERA5-Land)",
        "endpoint": ARCHIVE_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid_points": BASIN_POINTS,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "rows_basin": int(len(basin)),
        "rows_points": int(len(long_df)),
        "basin_avg_sha256": _sha256(basin_path),
        "points_sha256": _sha256(points_path),
        "notes": (
            "Unweighted mean across 6 grid points covering upper Chagres, "
            "Alajuela, and Gatun. ERA5 tends to under-estimate convective "
            "tropical rainfall vs gauge networks; treat magnitudes as "
            "consistent-but-biased, and lean on anomalies rather than absolutes."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
    )
    parser.add_argument(
        "--start", type=lambda s: date.fromisoformat(s), default=date(2010, 1, 1),
        help="Earliest date to fetch (default 2010-01-01; ERA5 goes back to 1940)",
    )
    parser.add_argument(
        "--end", type=lambda s: date.fromisoformat(s), default=date.today(),
        help="Latest date to fetch (default today; ERA5 has ~5-day lag)",
    )
    args = parser.parse_args(argv)

    print(f"Fetching Chagres basin precipitation ({args.start} -> {args.end}), "
          f"{len(BASIN_POINTS)} grid points...")
    long_df, basin = fetch_basin(args.start, args.end)
    print(f"  basin rows: {len(basin):,}  "
          f"range: {basin['date'].min()} -> {basin['date'].max()}")
    print(f"  basin mean daily precip: {basin['precip_mm_basin_avg'].mean():.1f} mm")
    print(f"  basin annual precip:     {basin['precip_mm_basin_avg'].mean() * 365:.0f} mm")

    out = write_snapshot(long_df, basin, args.out, args.start, args.end)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
