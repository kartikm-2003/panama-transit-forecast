"""
Parse ACP 'Monthly Canal Operations Summary' PDFs into a clean monthly panel.

These advisories publish exactly the operational metrics the plan calls
'queue length / wait days' and flagged as needing paid AIS data to
reconstruct. They are already downloaded by src.ingest.acp_advisories --
we just need to extract them into structured form.

Every monthly summary contains this block:

    Daily Average High Low
    Arrivals               32.20  43  24
    Oceangoing Transits    32.50  37  26
    Canal Waters Time (h)  21.60  32.2  16.4
    In-Transit Time (h)    10.70  12.8   9.3

Key derived metric: wait_hours_avg = canal_waters_hours_avg - in_transit_hours_avg.
The plan's third forecast target ('queue length / wait days') becomes
available at monthly grain via this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pdfplumber

from src.data.load import raw_root, repo_root

MONTH_NAMES = {
    m.lower(): i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], start=1)
}

SUBJECT_MONTH_RE = re.compile(
    r"Monthly Canal Operations Summary\s*[–—\-]\s*"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s*(\d{4})",
    re.IGNORECASE,
)

# Each stat row: '<label> <avg> <high> <low>' on one line, numbers may be
# integers or decimals. The labels are stable across the 44-file series.
_STAT_ROW = r"[\s\S]{0,20}?(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
ARRIVALS_RE  = re.compile(rf"Arrivals{_STAT_ROW}")
TRANSITS_RE  = re.compile(rf"Oceangoing\s+Transits{_STAT_ROW}")
CWT_RE       = re.compile(rf"Canal\s+Waters\s+Time[^\n]*?{_STAT_ROW}")
INTRANSIT_RE = re.compile(rf"In[- ]?Transit\s+Time[^\n]*?{_STAT_ROW}")


def _find_monthly_pdfs(pdf_root: Path, advisories_csv: Path) -> list[tuple[str, Path]]:
    """Return [(adv_id, pdf_path), ...] for every Monthly Canal Ops Summary."""
    adv = pd.read_csv(advisories_csv, encoding="utf-8")
    monthly = adv[adv["subject"].fillna("").str.contains(
        "Monthly Canal Operations", case=False)]
    out = []
    for _, row in monthly.iterrows():
        pdf = pdf_root / str(row["year"]) / f"{row['adv_id']}.pdf"
        if pdf.exists():
            out.append((row["adv_id"], pdf))
    return sorted(out, key=lambda x: x[0])


def _parse_month_ref(text: str) -> pd.Timestamp | None:
    m = SUBJECT_MONTH_RE.search(text)
    if not m:
        return None
    month = MONTH_NAMES[m.group(1).lower()]
    return pd.Timestamp(year=int(m.group(2)), month=month, day=1)


def _triple(pattern: re.Pattern, text: str) -> tuple[float, float, float] | tuple[None, None, None]:
    m = pattern.search(text)
    if not m:
        return (None, None, None)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def parse_monthly(adv_id: str, pdf_path: Path) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    month_start = _parse_month_ref(text)

    a_avg, a_hi, a_lo = _triple(ARRIVALS_RE, text)
    t_avg, t_hi, t_lo = _triple(TRANSITS_RE, text)
    cwt_avg, cwt_hi, cwt_lo = _triple(CWT_RE, text)
    it_avg, it_hi, it_lo = _triple(INTRANSIT_RE, text)

    wait_avg = cwt_avg - it_avg if (cwt_avg is not None and it_avg is not None) else None
    wait_hi  = cwt_hi  - it_hi  if (cwt_hi  is not None and it_hi  is not None) else None

    # Sanity bounds: canal-waters-time is 5-200 hours in the record; anything
    # outside is a parse mis-fire.
    if cwt_avg is not None and not (5 <= cwt_avg <= 250):
        cwt_avg = cwt_hi = cwt_lo = wait_avg = wait_hi = None

    return {
        "adv_id": adv_id,
        "month_start": month_start,
        "arrivals_daily_avg": a_avg,
        "arrivals_daily_high": a_hi,
        "arrivals_daily_low":  a_lo,
        "transits_daily_avg":  t_avg,
        "transits_daily_high": t_hi,
        "transits_daily_low":  t_lo,
        "canal_waters_hours_avg":  cwt_avg,
        "canal_waters_hours_high": cwt_hi,
        "canal_waters_hours_low":  cwt_lo,
        "in_transit_hours_avg":  it_avg,
        "in_transit_hours_high": it_hi,
        "in_transit_hours_low":  it_lo,
        "wait_hours_avg":  wait_avg,
        "wait_hours_high": wait_hi,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_snapshot(source_dir: Path) -> Path:
    snaps = sorted(p for p in source_dir.iterdir() if p.is_dir() and p.name[:4].isdigit())
    return snaps[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root() / "data" / "raw")
    args = parser.parse_args(argv)

    adv_snap = _latest_snapshot(raw_root() / "acp_advisories")
    pdf_root = raw_root() / "acp_advisories" / "_pdfs"
    monthly_pdfs = _find_monthly_pdfs(pdf_root, adv_snap / "parameters.csv")
    print(f"Parsing {len(monthly_pdfs)} monthly summaries...")

    rows = [parse_monthly(adv_id, path) for adv_id, path in monthly_pdfs]
    df = pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = args.out / "acp_monthly_summaries" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "monthly_ops.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    (out_dir / "manifest.json").write_text(json.dumps({
        "source": "ACP Monthly Canal Operations Summary (parsed from advisory PDFs)",
        "input_pdfs": len(monthly_pdfs),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "month_min": str(df["month_start"].min().date()) if len(df) else None,
        "month_max": str(df["month_start"].max().date()) if len(df) else None,
        "wait_avg_parsed": int(df["wait_hours_avg"].notna().sum()),
        "csv_sha256": _sha256(csv_path),
        "notes": (
            "wait_hours_avg = canal_waters_hours_avg - in_transit_hours_avg. "
            "'high' and 'low' are the highest and lowest daily-average values "
            "within the reporting month, not individual vessel peaks."
        ),
    }, indent=2), encoding="utf-8")

    ok = df["wait_hours_avg"].notna().sum()
    print(f"  parsed OK: {ok}/{len(df)}  "
          f"range: {df['month_start'].min().date()} -> {df['month_start'].max().date()}")
    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
