"""
Ingest Gatun Lake water levels from ACP's public CSVs.

Two files, two roles in the model:

  history.csv     - daily observed level, feet, 1965-present. The hero
                    regressor: ACP announces cap changes based on this and
                    on projections derived from it.
  projection.csv  - ACP's own forward-looking (~15 days) lake level and
                    the corresponding max Neopanamax / Panamax drafts.
                    OVERWRITTEN on every publish. Snapshotting it daily is
                    the ONLY way to reconstruct the 'projections as known
                    at time t' series that the plan calls for -- without
                    it, backtests will silently peek at actuals.

Alajuela (Madden) has no equivalent public CSV at this endpoint (checked);
skip and revisit if it appears later.
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

HISTORY_URL = (
    "https://evtms-rpts.pancanal.com/eng/h2o/"
    "Download_Gatun_Lake_Water_Level_History.csv"
)
PROJECTION_URL = (
    "https://evtms-rpts.pancanal.com/eng/h2o/"
    "Gatun_Water_Level_Projection.csv"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_history(session: requests.Session) -> pd.DataFrame:
    r = session.get(HISTORY_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "gatun_level_ft"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def fetch_projection(session: requests.Session) -> pd.DataFrame:
    r = session.get(PROJECTION_URL, timeout=60)
    r.raise_for_status()
    # Two disclaimer lines precede the header row. skiprows=2 handles it,
    # but if ACP ever drops the disclaimer we fall back to scanning for the
    # column header explicitly.
    lines = r.text.splitlines()
    header_ix = next(
        (i for i, ln in enumerate(lines) if ln.lower().startswith("projected_date")),
        None,
    )
    if header_ix is None:
        raise RuntimeError("Projection CSV: could not locate header row")
    df = pd.read_csv(StringIO("\n".join(lines[header_ix:])))
    df.columns = [c.strip() for c in df.columns]
    df["projected_date"] = pd.to_datetime(df["projected_date"], format="%m/%d/%Y").dt.date
    df = df.sort_values("projected_date").reset_index(drop=True)
    return df


def write_history(df: pd.DataFrame, out_root: Path) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "gatun_lake" / "history" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "history.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    manifest = {
        "source": "ACP Gatun Lake Water Level History",
        "url": HISTORY_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "latest_level_ft": float(df["gatun_level_ft"].iloc[-1]),
        "csv_sha256": _sha256(csv_path),
        "notes": (
            "Full history is republished daily; compare csv_sha256 across "
            "snapshots to detect back-revisions to older observations."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return out_dir


def write_projection(df: pd.DataFrame, out_root: Path) -> Path:
    # Projections are per-run: file gets overwritten upstream, so each snapshot
    # goes into its own dated file, keyed by download date.
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "gatun_lake" / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"projection_{snapshot_date}.csv"

    # Record the snapshot date INSIDE the file so the accumulated corpus
    # trivially concats into a (as_of_date, projected_date) panel.
    df_out = df.copy()
    df_out.insert(0, "as_of_date", snapshot_date)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    # Append to (or create) an index so it's a one-file read for downstream.
    idx_path = out_dir / "_index.csv"
    idx_row = pd.DataFrame([{
        "as_of_date": snapshot_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_days": int(len(df)),
        "projected_date_min": str(df["projected_date"].min()),
        "projected_date_max": str(df["projected_date"].max()),
        "csv_sha256": _sha256(csv_path),
        "file": csv_path.name,
    }])
    if idx_path.exists():
        existing = pd.read_csv(idx_path)
        idx = pd.concat([existing, idx_row], ignore_index=True)
        idx = idx.drop_duplicates("as_of_date", keep="last")
    else:
        idx = idx_row
    idx.to_csv(idx_path, index=False, encoding="utf-8")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument("--projection-only", action="store_true")
    args = parser.parse_args(argv)

    session = _session()

    if not args.projection_only:
        print("Fetching Gatun Lake history...")
        hist = fetch_history(session)
        out = write_history(hist, args.out)
        print(f"  rows: {len(hist):,}  range: {hist['date'].min()} -> {hist['date'].max()}")
        print(f"  latest level: {hist['gatun_level_ft'].iloc[-1]:.2f} ft")
        print(f"  -> {out}")

    if not args.history_only:
        print("\nFetching Gatun Lake projection snapshot...")
        proj = fetch_projection(session)
        out = write_projection(proj, args.out)
        print(f"  horizon: {len(proj)} days  "
              f"({proj['projected_date'].min()} -> {proj['projected_date'].max()})")
        print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
