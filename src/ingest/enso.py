"""
Ingest the NOAA Oceanic Nino Index (ONI) from NOAA/PSL.

Format: fixed-width text, header line with '<start_year> <end_year>', then
one line per year with 12 monthly values, terminated by a '-99.9' sentinel
and trailer metadata. Missing values are -99.9.

The plan uses ONI at 3-6 month lead as a leading indicator for Gatun Lake
level (El Nino -> Panama drought). Monthly frequency is fine; we'll forward-
fill onto the daily backbone during feature engineering.

Source is PSL's mirror of NOAA CPC:
  https://psl.noaa.gov/data/correlation/oni.data
  underlying: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
FILL_VALUE = -99.9


def fetch_oni() -> pd.DataFrame:
    """Return a long-form DataFrame: month_start (Timestamp), oni (float)."""
    r = requests.get(ONI_URL, timeout=60)
    r.raise_for_status()
    text = r.text

    lines = text.splitlines()
    # First data line has two ints: start_year, end_year. Anything after the
    # yearly block starts with the sentinel or non-year text.
    year_rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            break  # hit the trailer
        if year < 1900 or year > 2100:
            break
        vals = [float(x) for x in parts[1:13]]
        if len(vals) != 12:
            break
        year_rows.append((year, vals))

    records = []
    for year, vals in year_rows:
        for month, v in enumerate(vals, start=1):
            records.append({
                "month_start": pd.Timestamp(year=year, month=month, day=1),
                "oni": None if v == FILL_VALUE else v,
            })
    df = pd.DataFrame(records)
    df["oni"] = pd.to_numeric(df["oni"])
    df = df.dropna(subset=["oni"]).sort_values("month_start").reset_index(drop=True)
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(df: pd.DataFrame, out_root: Path) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "enso" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "oni.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    manifest = {
        "source": "NOAA/PSL — Oceanic Nino Index (ONI, ERSSTv6)",
        "url": ONI_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "date_min": str(df["month_start"].min().date()),
        "date_max": str(df["month_start"].max().date()),
        "latest_oni": float(df["oni"].iloc[-1]),
        "csv_sha256": _sha256(csv_path),
        "notes": (
            "Monthly index; expand to daily via forward-fill in feature "
            "engineering. Positive values = El Nino; associated with reduced "
            "Panama watershed rainfall at a 3-6 month lead."
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
    args = parser.parse_args(argv)

    print("Fetching NOAA ONI...")
    df = fetch_oni()
    print(f"  rows: {len(df):,}  range: {df['month_start'].min().date()} -> "
          f"{df['month_start'].max().date()}")
    print(f"  latest ONI: {df['oni'].iloc[-1]:+.2f}")
    out = write_snapshot(df, args.out)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
