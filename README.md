# Panama Canal Transit Forecast

30-day forecasts of daily Panama Canal transit throughput under drought-driven
capacity rationing.

## The framing

Panama Canal daily transits are **not demand-driven** — they are
administratively capped by the ACP based on projected Gatun Lake levels.
Naively fitting a time-series model to transit counts is modelling a step
function that a committee announces in advance.

This project instead:

- treats the announced daily slot cap as a **known-in-advance exogenous
  regressor** (ACP publishes advisories ahead of time),
- models the censoring explicitly (transits = min(demand, capacity)),
- targets the operationally meaningful quantities: **transits, capacity
  utilization, and monthly wait time**.

Framed as: *"Forecasting congestion at a capacity-constrained chokepoint,
where capacity is itself a hydrologically driven policy variable."*

## Data sources — all free, no API keys

| Source | Module | Grain | Coverage |
|---|---|---|---|
| IMF PortWatch chokepoint transits (Panama + Suez + Cape + Bab el-Mandeb) | [`src/ingest/portwatch.py`](src/ingest/portwatch.py) | daily × chokepoint | 2019-01-01 → present |
| ACP Advisories to Shipping (slot caps, drafts) | [`src/ingest/acp_advisories.py`](src/ingest/acp_advisories.py) | per advisory | 2022 → present |
| ACP Monthly Canal Operations Summary (wait time, arrivals) | [`src/ingest/acp_monthly_summaries.py`](src/ingest/acp_monthly_summaries.py) | monthly | Dec 2021 → present |
| ACP Gatun Lake water level (history + 15-day projection) | [`src/ingest/gatun_lake.py`](src/ingest/gatun_lake.py) | daily | 1965 → present |
| NOAA/PSL Oceanic Niño Index (ONI) | [`src/ingest/enso.py`](src/ingest/enso.py) | monthly | 1950 → present |
| Open-Meteo ERA5 archive (Chagres basin precipitation) | [`src/ingest/precipitation.py`](src/ingest/precipitation.py) | daily × grid | 2010 → present |
| IMF PortWatch daily port calls (10 major US ports) | [`src/ingest/ports.py`](src/ingest/ports.py) | daily × port | 2019 → present |

Deliberately omitted vs the original plan: Baltic Dry Index / Freightos FBX
freight rates — no free daily source. AIS-derived queue length — reconstructed
instead from ACP's own published monthly wait times.

## Architecture

```
data/raw/              # source-of-truth CSV/JSON snapshots + SHA256 manifests
data/processed/
  └─ panama.db         # DuckDB warehouse (derived, rebuildable)
src/
  ├─ ingest/           # one module per source
  └─ data/
     ├─ load.py        # DB-first loaders (fall back to raw files)
     └─ build_db.py    # materialise raw snapshots into DuckDB
notebooks/
  └─ 01_eda.ipynb      # 39-cell Phase 2 EDA
```

**Raw files are the source of truth.** Every ingest writes a `manifest.json`
with a SHA256 hash so silent revisions upstream (e.g. PortWatch's Mar–Jul
2024 AIS backfill) can be detected across snapshots. The DuckDB warehouse is
a derived, rebuildable analytics layer trimmed to the modelling window
(2014-01-01 onward) to keep the working data lean.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
# 1. install dependencies
uv sync

# 2. ingest every source (~2 minutes)
uv run python -m src.ingest.portwatch
uv run python -m src.ingest.acp_advisories
uv run python -m src.ingest.acp_monthly_summaries
uv run python -m src.ingest.gatun_lake
uv run python -m src.ingest.enso
uv run python -m src.ingest.precipitation
uv run python -m src.ingest.ports

# 3. build the DuckDB warehouse
uv run python -m src.data.build_db

# 4. open the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

## Querying the warehouse

```python
from src.data.load import connect

con = connect()
df = con.execute("""
    SELECT date, gatun_level_ft, transits_panama, wait_hours_avg
    FROM daily_panel
    WHERE date BETWEEN '2023-06-01' AND '2023-12-31'
""").df()
```

`daily_panel` is a pre-joined view over transits, Gatun, precipitation, coast
port calls, ONI, and wait time — the six-source merge already done for you.

## What's in the EDA notebook

See [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the full 39-cell
walk-through. Highlights:

- Panama transits vs Gatun: the relationship is a **floor, not a slope** —
  above ~84 ft transits sit at the cap; below, they step down.
- 2023 drought: Gatun troughs at **79.24 ft on 2023-07-25**; wait time peaks
  at **55 hrs (2.3 days) average, 91 hrs peak in August 2023**.
- CCF: precipitation leads Gatun by ~93 days; Gatun leads transits by ~26 days.
- CUSUM structural break peaks on **2023-09-18** — right at drought onset.
- Chokepoint substitution: Bab el-Mandeb traffic collapses Nov 2023 (Houthi
  attacks); Cape of Good Hope absorbs the diversion.

## Model roadmap

Phase 1 (data) and Phase 2 (EDA) are complete. Remaining phases per the
project plan:

- **Phase 3** — feature engineering (point-in-time safe, three-bucket
  known-future / forecastable / lagged design)
- **Phase 4** — model ladder: seasonal-naive → SARIMAX → censored/Tobit →
  LightGBM → N-BEATSx/TFT
- **Phase 5** — rolling-origin backtest with results split by regime
  (normal vs restricted operations)
- **Phase 6** — SHAP / counterfactual (drought-attributable throughput loss)
