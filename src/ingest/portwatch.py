"""
Ingest IMF PortWatch Daily Chokepoint Transit Calls.

Source: https://portwatch.imf.org (ArcGIS FeatureServer, updated weekly Tue 09:00 ET).

Panama Canal is the modelling target; Suez, Cape of Good Hope, and Bab
el-Mandeb are pulled as substitution / disruption signals (Suez blockages
and Red Sea attacks reroute traffic to the Cape and, indirectly, affect
Panama demand via Asia-US routing).

Writes per chokepoint under data/raw/portwatch/<snapshot_date>/<slug>/:
  transits.csv         -- one row per day, all columns from the service
  raw_response.json    -- full ArcGIS payload (audit trail)
  manifest.json        -- source URL, download timestamp, row count,
                          date range, SHA256 of the CSV (for versioning)

The manifest lets us detect the March-July 2024 AIS-coverage backfill
(and any future PortWatch revisions) between snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SERVICE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
PAGE_SIZE = 2000  # requested; server caps its own per-page limit

# Slug -> (portid, human name). Slugs are stable directory names; portids
# are what the PortWatch service actually indexes on.
CHOKEPOINTS: dict[str, tuple[str, str]] = {
    "panama": ("chokepoint2", "Panama Canal"),
    "suez": ("chokepoint1", "Suez Canal"),
    "cape_of_good_hope": ("chokepoint7", "Cape of Good Hope"),
    "bab_el_mandeb": ("chokepoint4", "Bab el-Mandeb Strait"),
}


def _fetch_page(session: requests.Session, portid: str, offset: int) -> dict:
    params = {
        "where": f"portid = '{portid}'",
        "outFields": "*",
        "orderByFields": "date ASC",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "returnGeometry": "false",
        "f": "json",
    }
    r = session.get(SERVICE_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error at offset {offset}: {payload['error']}")
    return payload


def fetch_chokepoint(portid: str, session: requests.Session | None = None
                     ) -> tuple[pd.DataFrame, list[dict]]:
    """Return (dataframe, raw feature list) for the full history of a chokepoint."""
    session = session or requests.Session()
    features: list[dict] = []
    offset = 0
    while True:
        payload = _fetch_page(session, portid, offset)
        batch = payload.get("features", [])
        if not batch:
            break
        features.extend(batch)
        offset += len(batch)
        if not payload.get("exceededTransferLimit"):
            break

    rows = [f["attributes"] for f in features]
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None).dt.date
        df = df.sort_values("date").reset_index(drop=True)
    return df, features


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(slug: str, portid: str, name: str, df: pd.DataFrame,
                   features: list[dict], out_root: Path) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "portwatch" / snapshot_date / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "transits.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    (out_dir / "raw_response.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8",
    )

    manifest = {
        "source": "IMF PortWatch — Daily Chokepoints Data",
        "service_url": SERVICE_URL,
        "portid": portid,
        "chokepoint_name": name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "date_min": str(df["date"].min()) if len(df) else None,
        "date_max": str(df["date"].max()) if len(df) else None,
        "csv_sha256": _sha256(csv_path),
        "notes": (
            "PortWatch AIS coverage revision (Mar-Jul 2024, backfilled Oct 2024) "
            "means older snapshots are not reproducible against newer downloads. "
            "Compare csv_sha256 across snapshots to detect revisions."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
    )
    parser.add_argument(
        "--chokepoints",
        default=",".join(CHOKEPOINTS.keys()),
        help=f"Comma-separated slugs from: {','.join(CHOKEPOINTS)}",
    )
    args = parser.parse_args(argv)

    session = requests.Session()
    slugs = [s.strip() for s in args.chokepoints.split(",") if s.strip()]
    unknown = set(slugs) - set(CHOKEPOINTS)
    if unknown:
        parser.error(f"Unknown chokepoint slugs: {sorted(unknown)}")

    for slug in slugs:
        portid, name = CHOKEPOINTS[slug]
        print(f"Fetching {name} ({portid})...")
        df, features = fetch_chokepoint(portid, session=session)
        out = write_snapshot(slug, portid, name, df, features, args.out)
        if len(df):
            print(f"  rows: {len(df):,}  "
                  f"range: {df['date'].min()} -> {df['date'].max()}  "
                  f"median n_total: {int(df['n_total'].median())}")
        print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
