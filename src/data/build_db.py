"""
Materialise the latest raw snapshot of every source into a single DuckDB file.

DuckDB (file-based, columnar, zero-config) is a much better fit than SQLite
here: our queries are analytical -- multi-source joins keyed on date, rolling
windows, group-by aggregations across years -- which is what DuckDB is
designed for. SQLite would work but slower and with worse ergonomics for
window functions.

Raw files remain the source of truth. This script rebuilds the DB from
scratch on every run so 'delete panama.db and re-run' is always safe.

Usage:
    uv run python -m src.data.build_db                 # write to default path
    uv run python -m src.data.build_db --db my.db      # write elsewhere
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Modelling window is 2019+ (PortWatch's coverage floor). Feature engineering
# needs ~5 years of pre-window data for climatology, same-day-of-year means,
# 90-day rolling anomalies, etc. Anything older is trimmed at DB-build time
# so the warehouse only carries what actually feeds the model. Raw files on
# disk are unchanged and remain the auditable source.
DAILY_CUTOFF = pd.Timestamp("2014-01-01")
MONTHLY_CUTOFF = pd.Timestamp("2013-01-01")  # ENSO leads by 3-6 months

from src.data.load import (
    _read_advisories_from_files,
    _read_all_chokepoints_from_files,
    _read_enso_from_files,
    _read_gatun_history_from_files,
    _read_gatun_projections_from_files,
    _read_monthly_ops_from_files,
    _read_panama_transits_from_files,
    _read_ports_from_files,
    _read_precip_points_from_files,
    _read_precipitation_from_files,
    raw_root,
    repo_root,
)


def _collect_manifests() -> pd.DataFrame:
    """Row per (source, snapshot_date) with hashes for provenance."""
    rows = []
    for m in sorted(raw_root().rglob("manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        rel = m.relative_to(raw_root()).parent
        # First path segment is the source name; remaining segments encode
        # snapshot layout (dates, coasts, chokepoints).
        parts = rel.parts
        rows.append({
            "source": parts[0],
            "subpath": "/".join(parts[1:]) if len(parts) > 1 else "",
            "downloaded_at_utc": d.get("downloaded_at_utc"),
            "rows": d.get("rows") or d.get("rows_basin") or d.get("advisories_indexed"),
            "date_min": d.get("date_min") or d.get("requested_start"),
            "date_max": d.get("date_max") or d.get("requested_end"),
            "sha256": (d.get("csv_sha256") or d.get("basin_avg_sha256")
                       or d.get("parameters_sha256") or d.get("index_sha256")),
            "url": d.get("url") or d.get("service_url") or d.get("endpoint")
                   or d.get("listing_url"),
        })
    return pd.DataFrame(rows)


def build(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    print(f"Building {db_path} ...")
    con = duckdb.connect(str(db_path))

    frames = {
        "panama_transits":    _read_panama_transits_from_files(),
        "chokepoints":        _read_all_chokepoints_from_files(),
        "gatun_history":      _read_gatun_history_from_files(),
        "gatun_projections":  _read_gatun_projections_from_files(),
        "precip_daily":       _read_precipitation_from_files(),
        "precip_points":      _read_precip_points_from_files(),
        "enso":               _read_enso_from_files(),
        "advisories":         _read_advisories_from_files(),
        "monthly_ops":        _read_monthly_ops_from_files(),
        "ports":              _read_ports_from_files(),
        "data_manifest":      _collect_manifests(),
    }

    # Trim to modelling-relevant window. Advisories, projections and the
    # provenance manifest are untouched (already scoped or forward-looking).
    daily_cut = DAILY_CUTOFF
    frames["panama_transits"] = frames["panama_transits"].loc[frames["panama_transits"]["date"] >= daily_cut]
    frames["chokepoints"]     = frames["chokepoints"].loc[frames["chokepoints"]["date"] >= daily_cut]
    frames["gatun_history"]   = frames["gatun_history"].loc[frames["gatun_history"]["date"] >= daily_cut]
    frames["precip_daily"]    = frames["precip_daily"].loc[frames["precip_daily"]["date"] >= daily_cut]
    frames["precip_points"]   = frames["precip_points"].loc[frames["precip_points"]["date"] >= daily_cut]
    frames["ports"]           = frames["ports"].loc[frames["ports"]["date"] >= daily_cut]
    frames["enso"]            = frames["enso"].loc[frames["enso"]["month_start"] >= MONTHLY_CUTOFF]

    for name, df in frames.items():
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"  {name:22} {n:>7,} rows")

    # Indexes -- DuckDB doesn't strictly need them for scans but they help
    # point lookups and PK-style joins that the notebook will do a lot of.
    con.execute("CREATE UNIQUE INDEX ix_panama_date         ON panama_transits(date)")
    con.execute("CREATE UNIQUE INDEX ix_chokepoint_date_ck  ON chokepoints(date, chokepoint)")
    con.execute("CREATE UNIQUE INDEX ix_gatun_date          ON gatun_history(date)")
    con.execute("CREATE UNIQUE INDEX ix_precip_date         ON precip_daily(date)")
    con.execute("CREATE UNIQUE INDEX ix_enso_month          ON enso(month_start)")
    con.execute("CREATE UNIQUE INDEX ix_ports_date_slug     ON ports(date, port_slug)")
    con.execute("CREATE UNIQUE INDEX ix_advisories_id       ON advisories(adv_id)")
    con.execute("CREATE UNIQUE INDEX ix_monthly_ops_month    ON monthly_ops(month_start)")

    # Convenience view: everything pre-joined on the daily backbone.
    # ENSO is monthly -> forward-filled to daily via asof-join.
    con.execute("""
        CREATE VIEW daily_panel AS
        WITH
          ch AS (
            SELECT date,
                   MAX(n_total) FILTER (WHERE chokepoint = 'panama')            AS transits_panama,
                   MAX(n_total) FILTER (WHERE chokepoint = 'suez')              AS transits_suez,
                   MAX(n_total) FILTER (WHERE chokepoint = 'cape_of_good_hope') AS transits_cape,
                   MAX(n_total) FILTER (WHERE chokepoint = 'bab_el_mandeb')     AS transits_bab
            FROM chokepoints GROUP BY date
          ),
          po AS (
            SELECT date,
                   SUM(portcalls_container) FILTER (WHERE coast = 'west') AS usa_west_container,
                   SUM(portcalls_container) FILTER (WHERE coast = 'east') AS usa_east_container,
                   SUM(portcalls_container) FILTER (WHERE coast = 'gulf') AS usa_gulf_container
            FROM ports GROUP BY date
          )
        SELECT
          g.date,
          g.gatun_level_ft,
          ch.transits_panama, ch.transits_suez, ch.transits_cape, ch.transits_bab,
          pd.precip_mm_basin_avg,
          po.usa_west_container, po.usa_east_container, po.usa_gulf_container,
          (SELECT oni FROM enso e
             WHERE e.month_start <= g.date ORDER BY e.month_start DESC LIMIT 1) AS oni,
          (SELECT wait_hours_avg FROM monthly_ops m
             WHERE m.month_start <= g.date ORDER BY m.month_start DESC LIMIT 1) AS wait_hours_avg
        FROM gatun_history g
        LEFT JOIN ch ON ch.date = g.date
        LEFT JOIN precip_daily pd ON pd.date = g.date
        LEFT JOIN po ON po.date = g.date
        ORDER BY g.date
    """)

    con.close()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Done. {db_path} ({size_mb:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=repo_root() / "data" / "processed" / "panama.db",
    )
    args = parser.parse_args(argv)
    build(args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
