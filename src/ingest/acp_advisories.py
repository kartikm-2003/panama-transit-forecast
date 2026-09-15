"""
Ingest ACP 'Advisory to Shipping' PDFs from pancanal.com.

Advisories are the source of the highest-value exogenous regressors for the
model: daily transit slot counts (Panamax + Neopanamax) and maximum
authorized draft. They are known-in-advance, which is what makes them
legitimate `known_future` covariates at forecast time.

This module:
  1. Scrapes the advisory listing page and extracts every 'ADV-NN-YYYY' PDF URL.
  2. Downloads the PDFs to a local cache (idempotent -- re-runs skip existing).
  3. Extracts text with pdfplumber and applies best-effort regex to pull:
        - advisory_date, effective_date, subject
        - max_draft_meters, max_draft_feet
        - neopanamax_slots, panamax_slots
  4. Writes an index (all advisories) and a parameters table (extracted fields).

The parser is deliberately conservative: fields it cannot confidently extract
are left NaN and the 'needs_review' flag is set. Do not skip the manual
sweep for the ~30-50 advisories that actually change operational parameters
-- the plan budgets two days for this, and the model's coefficient on the
cap regressor depends on getting these right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

LISTING_URL = "https://pancanal.com/en/maritime-services/advisory-to-shipping/"

# pancanal.com returns 403 to bare python-requests. A conventional browser UA
# passes.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

# Match /adv-NN-YYYY or /a-NN-YYYY (case-insensitive) in the PDF URL.
ADV_URL_RE = re.compile(
    r"/(?:adv|a)[-_]?(\d{1,3})[-_](\d{4})[^/]*\.pdf$",
    re.IGNORECASE,
)

# 'N.NN m (NN.N feet)' -- the canonical draft phrasing in ACP advisories.
DRAFT_RE = re.compile(
    r"(\d{1,2}\.\d{1,2})\s*m(?:eters?)?\s*\(\s*(\d{1,2}(?:\.\d)?)\s*(?:feet|ft)\s*\)",
    re.IGNORECASE,
)

# ACP advisories consistently disambiguate slot counts with a spelled-out
# number followed by the digit in parens: 'adjusted to nine (9)',
# 'to twenty-five (25)'. Anchoring on that exact form eliminates almost all
# false positives (dates, draft measurements, auction sub-details, etc.).
_NUMBER_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|"
    r"twenty(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"thirty(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?)"
)
_SPELLED_COUNT = rf"{_NUMBER_WORD}\s*\((\d{{1,2}})\)"
_SPELLED_COUNT_ANY_RE = re.compile(rf"{_NUMBER_WORD}\s*\((\d{{1,2}})\)", re.IGNORECASE)

# 'Neopanamax Locks ... X (N)' with . or ; ending the sentence window.
NEOPANAMAX_SLOTS_RE = re.compile(
    rf"(?:"
    rf"Neopanamax\s+Locks[^.;\n]{{0,300}}?{_SPELLED_COUNT}"
    rf"|"
    rf"{_SPELLED_COUNT}[^.;\n]{{0,300}}?Neopanamax\s+Locks"
    rf")",
    re.IGNORECASE,
)
PANAMAX_SLOTS_RE = re.compile(
    rf"(?:"
    rf"(?<!Neo)Panamax\s+Locks[^.;\n]{{0,300}}?{_SPELLED_COUNT}"
    rf"|"
    rf"{_SPELLED_COUNT}[^.;\n]{{0,300}}?(?<!Neo)Panamax\s+Locks"
    rf")",
    re.IGNORECASE,
)

# 'effective on Month DD, YYYY' or 'effective Month DD, YYYY'.
EFFECTIVE_DATE_RE = re.compile(
    r"effective\s+(?:on\s+|from\s+|as\s+of\s+)?"
    r"([A-Z][a-z]+\s+\d{1,2},\s*\d{4})",
    re.IGNORECASE,
)

# The advisory issue date is on its own line near the top, after the header block.
ISSUE_DATE_RE = re.compile(
    r"^([A-Z][a-z]+\s+\d{1,2},\s*\d{4})\s*$",
    re.MULTILINE,
)

SUBJECT_RE = re.compile(r"SUBJECT\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


@dataclass
class AdvisoryRow:
    year: int
    number: int
    adv_id: str  # 'A-33-2026'
    url: str
    pdf_path: str | None
    downloaded: bool


@dataclass
class AdvisoryParams:
    adv_id: str
    year: int
    number: int
    advisory_date: str | None
    effective_date: str | None
    subject: str | None
    max_draft_meters: float | None
    max_draft_feet: float | None
    neopanamax_slots: int | None
    panamax_slots: int | None
    slot_candidates: str  # 'nine=9;twenty-three=23' -- raw finds for review
    mentions_transits: bool
    mentions_draft: bool
    needs_review: bool


def scrape_index(min_year: int, session: requests.Session | None = None) -> list[AdvisoryRow]:
    """Fetch the listing page and return every advisory PDF at or after min_year."""
    session = session or _session()
    r = session.get(LISTING_URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    seen: dict[str, AdvisoryRow] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = ADV_URL_RE.search(href)
        if not m:
            continue
        number, year = int(m.group(1)), int(m.group(2))
        if year < min_year:
            continue
        adv_id = f"A-{number:02d}-{year}"
        # Deduplicate: the same advisory can be linked more than once.
        seen.setdefault(adv_id, AdvisoryRow(
            year=year, number=number, adv_id=adv_id, url=href,
            pdf_path=None, downloaded=False,
        ))

    return sorted(seen.values(), key=lambda x: (x.year, x.number))


def download_pdf(row: AdvisoryRow, cache_dir: Path, session: requests.Session) -> None:
    """Download the PDF into cache_dir/<year>/<adv_id>.pdf, skipping if present."""
    target = cache_dir / str(row.year) / f"{row.adv_id}.pdf"
    row.pdf_path = str(target)
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(row.url, timeout=90)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    row.downloaded = True


def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


def _first(pattern: re.Pattern, text: str, group: int = 1) -> str | None:
    m = pattern.search(text)
    return m.group(group).strip() if m else None


def _slot_count(pattern: re.Pattern, text: str) -> int | None:
    m = pattern.search(text)
    if not m:
        return None
    # The alternation has 3 capture groups; take the first non-None digit run.
    for g in m.groups():
        if g and g.isdigit():
            return int(g)
    return None


def parse_parameters(row: AdvisoryRow, text: str) -> AdvisoryParams:
    subject = _first(SUBJECT_RE, text) or ""
    mentions_draft = "draft" in text.lower()
    mentions_transits = any(k in text.lower() for k in ("slot", "transit", "booking"))

    draft_m = DRAFT_RE.search(text)
    draft_meters = float(draft_m.group(1)) if draft_m else None
    draft_feet = float(draft_m.group(2)) if draft_m else None

    neop = _slot_count(NEOPANAMAX_SLOTS_RE, text)
    pana = _slot_count(PANAMAX_SLOTS_RE, text)

    # Sanity bounds: Panamax historically 20-27, Neopanamax 6-10.
    if neop is not None and not (1 <= neop <= 12):
        neop = None
    if pana is not None and not (10 <= pana <= 30):
        pana = None

    issue_date = _first(ISSUE_DATE_RE, text)
    effective_date = _first(EFFECTIVE_DATE_RE, text)

    # Surface every 'spelled (N)' pair so multi-step changes (e.g. A-29-2026,
    # which staggers Panamax 25 -> 23) are visible even though we only pin
    # one number to each lock automatically.
    candidates = [
        f"{m.group(0).split('(')[0].strip().lower()}={m.group(1)}"
        for m in _SPELLED_COUNT_ANY_RE.finditer(text)
    ]
    slot_candidates = ";".join(candidates)

    # 'needs_review' == this advisory changes an operational parameter but the
    # parser could not extract at least one of the numeric fields it mentions.
    needs_review = False
    if mentions_draft and (draft_meters is None or draft_feet is None):
        needs_review = True
    if ("slot" in subject.lower() or "transit" in subject.lower()) and \
            neop is None and pana is None:
        needs_review = True

    return AdvisoryParams(
        adv_id=row.adv_id, year=row.year, number=row.number,
        advisory_date=issue_date, effective_date=effective_date, subject=subject,
        max_draft_meters=draft_meters, max_draft_feet=draft_feet,
        neopanamax_slots=neop, panamax_slots=pana,
        slot_candidates=slot_candidates,
        mentions_transits=mentions_transits, mentions_draft=mentions_draft,
        needs_review=needs_review,
    )


def write_outputs(
    rows: list[AdvisoryRow],
    params: list[AdvisoryParams],
    out_root: Path,
) -> Path:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = out_root / "acp_advisories" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_df = pd.DataFrame([asdict(r) for r in rows])
    par_df = pd.DataFrame([asdict(p) for p in params])
    idx_path = out_dir / "index.csv"
    par_path = out_dir / "parameters.csv"
    idx_df.to_csv(idx_path, index=False, encoding="utf-8")
    par_df.to_csv(par_path, index=False, encoding="utf-8")

    manifest = {
        "source": "ACP Advisory to Shipping",
        "listing_url": LISTING_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "advisories_indexed": int(len(idx_df)),
        "advisories_needing_review": int(par_df["needs_review"].sum()) if len(par_df) else 0,
        "index_sha256": _sha256(idx_path),
        "parameters_sha256": _sha256(par_path),
        "notes": (
            "Regex extraction is best-effort. Every advisory whose subject "
            "mentions 'slot', 'transit', or 'draft' should be spot-checked; "
            "rows flagged needs_review=True are the top priority."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return out_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument(
        "--pdf-cache", type=Path, default=repo_root / "data" / "raw" / "acp_advisories" / "_pdfs",
        help="Persistent PDF cache (reused across snapshots)",
    )
    parser.add_argument(
        "--min-year", type=int, default=2022,
        help="Skip advisories older than this year (default 2022, when drought-era ops began)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="For quick tests: only process the first N advisories",
    )
    args = parser.parse_args(argv)

    session = _session()
    print(f"Scraping advisory index (min_year={args.min_year})...")
    rows = scrape_index(args.min_year, session=session)
    if args.limit:
        rows = rows[: args.limit]
    print(f"  found {len(rows)} advisories")

    params: list[AdvisoryParams] = []
    for i, row in enumerate(rows, 1):
        try:
            download_pdf(row, args.pdf_cache, session)
        except requests.HTTPError as e:
            print(f"  [{i}/{len(rows)}] {row.adv_id}: download failed ({e})")
            continue
        try:
            text = extract_text(Path(row.pdf_path))
            params.append(parse_parameters(row, text))
        except Exception as e:
            print(f"  [{i}/{len(rows)}] {row.adv_id}: parse failed ({e})")
        if i % 25 == 0:
            print(f"  processed {i}/{len(rows)}")

    out_dir = write_outputs(rows, params, args.out)
    par_df = pd.DataFrame([asdict(p) for p in params])
    print(f"\nOutputs -> {out_dir}")
    print(f"  advisories parsed:      {len(par_df)}")
    if len(par_df):
        print(f"  mention transits/slots: {int(par_df['mentions_transits'].sum())}")
        print(f"  mention draft:          {int(par_df['mentions_draft'].sum())}")
        print(f"  flagged needs_review:   {int(par_df['needs_review'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
