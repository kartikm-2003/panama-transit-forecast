"""
Ingest IMF PortWatch Daily Ports Data for US East and West Coast ports.

The plan uses coastal port-call ratios (USWC vs USEC) as a demand-side
substitution signal: when Panama Canal capacity tightens, Asia-US ocean
routing shifts to West Coast + land-bridge; when Panama runs freely,
direct East Coast calls grow. Signal is lagged (we see the port call
after the ship arrives), so treat as historical exog only.

Writes per port under data/raw/ports/<snapshot_date>/<slug>/:
  daily.csv            -- one row per day
  raw_response.json    -- full ArcGIS payload (audit trail)
  manifest.json        -- source URL, download timestamp, hashes
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
    "Daily_Ports_Data/FeatureServer/0/query"
)
PAGE_SIZE = 2000

# Ten major container / mixed-cargo US ports split by coast, sufficient to
# construct a USWC/USEC ratio. IDs come from PortWatch_ports_database.
PORTS: dict[str, tuple[str, str, str]] = {
    # slug: (portid, portname, coast)
    "la_long_beach":     ("port664",  "Los Angeles-Long Beach", "west"),
    "oakland":           ("port839",  "Oakland",                "west"),
    "seattle":           ("port1175", "Seattle",                "west"),
    "tacoma":            ("port1248", "Tacoma",                 "west"),
    "new_york_new_jersey": ("port815",  "New York-New Jersey",    "east"),
    "savannah":          ("port1170", "Savannah",               "east"),
    "charleston":        ("port231",  "Charleston",             "east"),
    "norfolk":           ("port826",  "Norfolk",                "east"),
    "port_everglades":   ("port951",  "Port Everglades",        "east"),
    "houston":           ("port481",  "Houston (US-TX)",        "gulf"),
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


def fetch_port(portid: str, session: requests.Session | None = None
               ) -> tuple[pd.DataFrame, list[dict]]:
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


def write_snapshot(slug: str, portid: str, name: str, coast: str,
                   df: pd.DataFrame, features: list[dict], out_root: Path) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "ports" / snapshot_date / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "daily.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    (out_dir / "raw_response.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8",
    )

    manifest = {
        "source": "IMF PortWatch — Daily Ports Data",
        "service_url": SERVICE_URL,
        "portid": portid,
        "port_name": name,
        "coast": coast,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "date_min": str(df["date"].min()) if len(df) else None,
        "date_max": str(df["date"].max()) if len(df) else None,
        "csv_sha256": _sha256(csv_path),
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
        "--ports", default=",".join(PORTS.keys()),
        help=f"Comma-separated slugs from: {','.join(PORTS)}",
    )
    args = parser.parse_args(argv)

    session = requests.Session()
    slugs = [s.strip() for s in args.ports.split(",") if s.strip()]
    unknown = set(slugs) - set(PORTS)
    if unknown:
        parser.error(f"Unknown port slugs: {sorted(unknown)}")

    for slug in slugs:
        portid, name, coast = PORTS[slug]
        print(f"Fetching {name} ({portid}, {coast})...")
        df, features = fetch_port(portid, session=session)
        out = write_snapshot(slug, portid, name, coast, df, features, args.out)
        if len(df):
            median_calls = int(df["portcalls"].median()) if "portcalls" in df else -1
            print(f"  rows: {len(df):,}  "
                  f"range: {df['date'].min()} -> {df['date'].max()}  "
                  f"median portcalls: {median_calls}")
        print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
