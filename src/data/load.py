"""
Canonical loaders. DuckDB warehouse is the primary source; if the DB has
not been built (or is missing), each loader transparently falls back to
reading the latest raw CSV snapshot. Same DataFrame either way, so the
notebook and downstream code don't care which path served the data.

Rebuild the warehouse after any new ingest:
    uv run python -m src.data.build_db
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_root() -> Path:
    return repo_root() / "data" / "raw"


def db_path() -> Path:
    return repo_root() / "data" / "processed" / "panama.db"


def connect(read_only: bool = True):
    """Open the DuckDB warehouse. Import here so duckdb stays optional."""
    import duckdb
    return duckdb.connect(str(db_path()), read_only=read_only)


def _db_available() -> bool:
    return db_path().exists()


def _q(sql: str) -> pd.DataFrame:
    with connect() as con:
        return con.execute(sql).df()


def _latest_snapshot(source_dir: Path) -> Path:
    """Return the most recent YYYY-MM-DD subdirectory under source_dir."""
    snaps = sorted(p for p in source_dir.iterdir() if p.is_dir() and p.name[:4].isdigit())
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {source_dir}")
    return snaps[-1]


# --------------------------------------------------------------------------
# File-based readers. Used as fallback and by build_db.py to populate the DB.
# --------------------------------------------------------------------------

def _read_panama_transits_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "portwatch")
    df = pd.read_csv(snap / "panama" / "transits.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _read_chokepoint_from_files(slug: str) -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "portwatch")
    df = pd.read_csv(snap / slug / "transits.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _read_all_chokepoints_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "portwatch")
    frames = []
    for sub in snap.iterdir():
        if not sub.is_dir():
            continue
        df = pd.read_csv(sub / "transits.csv", parse_dates=["date"])
        df = df[["date", "n_total"]].copy()
        df["chokepoint"] = sub.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["chokepoint", "date"])


def _read_gatun_history_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "gatun_lake" / "history")
    df = pd.read_csv(snap / "history.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _read_gatun_projections_from_files() -> pd.DataFrame:
    proj_dir = raw_root() / "gatun_lake" / "projections"
    files = sorted(proj_dir.glob("projection_*.csv"))
    if not files:
        raise FileNotFoundError(f"No projection snapshots under {proj_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["projected_date"] = pd.to_datetime(df["projected_date"])
    return df.sort_values(["as_of_date", "projected_date"]).reset_index(drop=True)


def _read_precipitation_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "precipitation")
    df = pd.read_csv(snap / "chagres_daily.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _read_precip_points_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "precipitation")
    return pd.read_csv(snap / "chagres_points.csv", parse_dates=["date"])


def _read_enso_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "enso")
    df = pd.read_csv(snap / "oni.csv", parse_dates=["month_start"])
    return df.sort_values("month_start").reset_index(drop=True)


def _read_advisories_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "acp_advisories")
    return pd.read_csv(snap / "parameters.csv")


def _read_monthly_ops_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "acp_monthly_summaries")
    df = pd.read_csv(snap / "monthly_ops.csv", parse_dates=["month_start"])
    return df.sort_values("month_start").reset_index(drop=True)


def _read_ports_from_files() -> pd.DataFrame:
    snap = _latest_snapshot(raw_root() / "ports")
    frames = []
    for sub in snap.iterdir():
        if not sub.is_dir():
            continue
        df = pd.read_csv(sub / "daily.csv", parse_dates=["date"])
        meta = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
        df["port_slug"] = sub.name
        df["coast"] = meta.get("coast")
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["port_slug", "date"])


# --------------------------------------------------------------------------
# Public loaders. DB when available, files otherwise. Same output either way.
# --------------------------------------------------------------------------

def load_panama_transits() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM panama_transits ORDER BY date")
    return _read_panama_transits_from_files()


def load_chokepoint(slug: str) -> pd.DataFrame:
    if _db_available():
        return _q(f"SELECT * FROM chokepoints WHERE chokepoint = '{slug}' ORDER BY date")
    return _read_chokepoint_from_files(slug)


def load_all_chokepoints() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM chokepoints ORDER BY chokepoint, date")
    return _read_all_chokepoints_from_files()


def load_gatun_history() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM gatun_history ORDER BY date")
    return _read_gatun_history_from_files()


def load_gatun_projections() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM gatun_projections ORDER BY as_of_date, projected_date")
    return _read_gatun_projections_from_files()


def load_precipitation() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM precip_daily ORDER BY date")
    return _read_precipitation_from_files()


def load_enso() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM enso ORDER BY month_start")
    return _read_enso_from_files()


def load_advisories() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM advisories")
    return _read_advisories_from_files()


def load_monthly_ops() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM monthly_ops ORDER BY month_start")
    return _read_monthly_ops_from_files()


def load_ports() -> pd.DataFrame:
    if _db_available():
        return _q("SELECT * FROM ports ORDER BY port_slug, date")
    return _read_ports_from_files()


def load_daily_panel(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Pre-joined daily backbone: transits, Gatun, precip, coast port-calls, ONI.

    Only available via the DB; raises if the warehouse hasn't been built.
    Optional (start, end) bracket the returned window (inclusive).
    """
    if not _db_available():
        raise FileNotFoundError(
            "Warehouse not built. Run: uv run python -m src.data.build_db"
        )
    where = ""
    if start:
        where = f"WHERE date >= '{start}'"
    if end:
        where += (" AND " if where else "WHERE ") + f"date <= '{end}'"
    return _q(f"SELECT * FROM daily_panel {where} ORDER BY date")


def load_all() -> dict[str, pd.DataFrame]:
    """One call to get every source. Handy for the EDA notebook."""
    return {
        "panama": load_panama_transits(),
        "chokepoints": load_all_chokepoints(),
        "gatun": load_gatun_history(),
        "precip": load_precipitation(),
        "enso": load_enso(),
        "advisories": load_advisories(),
        "ports": load_ports(),
    }
