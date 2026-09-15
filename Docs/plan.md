Good pick. Here's the full build plan. I've checked the data sources are live, and there's one conceptual trap in this dataset that you need to design around from day one, so I'll start there.

## The trap (and why it's actually the selling point)

Panama Canal daily transits are not demand-driven. They are **administratively capped**. The ACP sets a maximum number of daily slots based on projected Gatun Lake levels, and vessels queue for them. The current cap is 34 vessels per day, with a scheduled drop to 32 transits starting September 15, 2026, and the queue reached 131 vessels in late August 2026.

So if you naively fit SARIMAX to transit counts, you are modelling a step function that a committee announces in advance. Your model will look great and mean nothing.

The fix, which is the whole intellectual content of the project:

- **Treat the announced daily slot cap as a known-in-advance exogenous regressor.** ACP publishes advisories ahead of time, so this is legitimately available at forecast time. This is the cleanest real-world example of a `known_future` covariate you will ever get.
- **Model the censoring explicitly.** Transits are `min(demand, capacity)`. On uncapped days the series reflects demand; on capped days it is truncated. A censored regression or a two-part model here is a genuine differentiator.
- **Forecast the thing that isn't capped: the queue and wait time.** Queue length is demand minus capacity accumulating over time, which is a much more honest forecasting target.

Frame the project as: *"Forecasting congestion at a capacity-constrained chokepoint, where capacity is itself a hydrologically driven policy variable."* That sentence alone gets you a follow-up question in an interview.

## Phase 1: Data acquisition (week 1)

**Target and traffic data, IMF PortWatch**
The Daily Chokepoint Transit Calls and Trade Volume Estimates dataset covers 28 major chokepoints, is updated weekly on Tuesdays at 9 AM ET, and is derived from satellite AIS signals. Pull the Panama Canal chokepoint series. It gives daily transit counts plus estimated trade volume in metric tons, split by vessel class (container, dry bulk, tanker, general cargo, roro). The vessel-class split is valuable: container ships buy booking slots, bulkers queue, so the classes respond differently to the same constraint. Download via CSV or the ArcGIS GeoServices API.

One caveat you must handle and should mention in your README: PortWatch identified an AIS coverage drop affecting March to July 2024, backfilled and revised in October 2024. Pin your data version and note the revision, because revised data means any result you compute is not reproducible against an older download.

**Gatun Lake level, the hero regressor**
ACP publishes water level indicators at `evtms-rpts.pancanal.com/eng/h2o/index.html`. Daily Gatun Lake water level measurements have been made by the ACP since 1965, and the ACP page also publishes projections. Scrape daily, or reconstruct history from the BTS data story and the ACP archives. Also grab Lake Alajuela (Madden) if available.

**Precipitation in the canal watershed**
ACP runs a rain gauge network, but the accessible route is ERA5 reanalysis or CHIRPS gridded precipitation clipped to the Chagres basin. Both are free via Copernicus CDS and Climate Hazards Center respectively. Build a basin-average daily precipitation series plus 30/60/90 day cumulative anomalies.

**ENSO index**
NOAA ONI, monthly, free. The 2023 drought was considered the worst in a century, partly driven by El Niño. ONI at a 3 to 6 month lead is a genuine leading indicator for lake level and is available in advance via CPC forecast probabilities.

**Capacity and policy**
ACP advisories to shipping, published as PDFs. Extract: daily slot count, maximum authorized draft in feet, booking auction rules. This is manual work for the historical series but it is maybe 150 advisories since 2022 and it is the single highest-value feature in the model. Budget two days for this.

**Substitution and demand signals**
Suez Canal transits from the same PortWatch chokepoint file (the two are substitutes and both were disrupted simultaneously in 2024), Cape of Good Hope transits, Baltic Dry Index or Freightos FBX spot rates, US West Coast versus East Coast port call counts from PortWatch port data.

**Calendar**
Chinese New Year (large, moving, and it shifts Asia to US East Coast container flows), US holidays, Panamanian holidays, day of week, week of year.

## Phase 2: EDA and target definition (week 2)

Before any model:

1. Plot transits against Gatun level with the capped periods shaded. Establish visually that the relationship is a threshold, not a slope.
2. Decompose with MSTL for weekly and annual seasonality. Check whether weekly seasonality weakens during capped periods, which it should, because the cap flattens it.
3. Test for structural breaks with Bai-Perron or a simple CUSUM. You will find breaks around mid-2023 and mid-2024. Document the dates.
4. Cross-correlation function between precipitation and lake level, and between lake level and transits, to establish the lag structure. Expect precipitation to lead lake level by roughly 2 to 6 weeks and lake level to lead transit caps by 4 to 12 weeks, because the ACP announces based on projections.

**Define three targets** and forecast all three:
- `transits_total` (daily count, censored)
- `transit_capacity_utilization` = transits divided by announced slots
- `queue_length` or average wait days, if you can source it consistently (Portcast and Desteia publish some, average wait times went from 2.44 days in June 2023 to a peak of 9.09 days in August before settling to 3.1 days by year end, but consistent daily history is the hard part, so treat this as a stretch target)

Horizon: 30 days ahead, daily. Justify it: that is roughly the ocean transit time from Asia, so a 30-day forecast is actionable for a routing decision.

## Phase 3: Feature engineering

| Bucket | Features | Availability at forecast time |
|---|---|---|
| Known future | day of week, holidays, Chinese New Year offset, announced slot cap, announced max draft | Fully known |
| Forecastable | Gatun level (ACP publishes projections), precipitation forecast, ONI forecast | Use actual published forecasts, never actuals |
| Lagged only | Suez transits, freight rates, port call counts | Lag by at least the horizon |

Derived features worth building: days since rainy season onset, lake level deviation from the same calendar day's 5-year mean, lake level rate of change over 14 days, a binary `restriction_active` flag, days since last cap change, and a Gatun level spline basis to capture the threshold nonlinearity.

Write a function `make_features(df, cutoff_date)` that only ever uses information available at `cutoff_date`. Every leak in this project will come from forgetting to do this.

## Phase 4: The model ladder

1. **Seasonal naive** (value from 7 days ago) and **cap-aware naive** (predict the announced cap). The second one is a brutal baseline. If your fancy model cannot beat "just predict the announced quota", say so, because that finding is more interesting than a marginal RMSE win.
2. **SARIMAX** with exog. Start with `(p,d,q)(P,D,Q,7)`, select by AICc, and include the cap, draft, and lake level features. Show the coefficient on the cap should be close to 1, which validates your framing.
3. **Censored / two-part model**. Model `P(capped)` with a classifier, and latent demand with a Tobit or a quantile regression on uncapped days. Combine. This is the part almost no portfolio project has.
4. **LightGBM** with lag features, rolling means, and the full exog set. Use `MultiOutputRegressor` or direct multi-horizon (one model per horizon h, or a global model with horizon as a feature).
5. **Neural**: N-BEATSx or TFT via `neuralforecast`. Both accept `futr_exog_list`, `hist_exog_list`, and `stat_exog_list` explicitly, which maps perfectly onto your three-bucket feature design. Run the five vessel classes as five related series so the global model has something to learn across.

## Phase 5: Validation

Rolling-origin backtest, expanding window, origins every 14 days, 30-day horizon, minimum 3 years of training before the first origin. That gives roughly 50 to 60 evaluation windows.

**Critically, report results split by regime**: normal operations versus restricted operations. A model that is excellent on average and useless during the 2023-24 drought is useless, because the drought is the only period anyone cares about. This split table is the centrepiece chart of your README.

Metrics: MASE against seasonal naive, pinball loss at the 10th, 50th, and 90th percentiles, coverage of 80% prediction intervals, and a Diebold-Mariano test between your best two models.

## Phase 6: Interpretability and the narrative

- SHAP values on LightGBM versus SARIMAX coefficients versus TFT variable-selection weights. Do all three agree that Gatun level and the announced cap dominate?
- **Counterfactual scenario**: hold the 2023 lake level at its 5-year normal and re-forecast. Estimate how many transits the drought cost. This converts a forecasting exercise into an impact estimate, which is what makes it memorable.
- A short note on what an operator would do with the forecast: reroute versus queue versus bid in the slot auction.

## Deliverables

```
panama-chokepoint-forecast/
├── README.md              # the narrative, charts, results tables
├── data/
│   ├── raw/               # immutable, versioned downloads
│   └── processed/
├── src/
│   ├── ingest/            # one module per source
│   ├── features.py        # point-in-time safe
│   ├── models/
│   └── backtest.py
├── notebooks/             # 01_eda, 02_baselines, 03_models, 04_results
├── reports/figures/
└── Makefile
```

Add a `dvc` or simple hash-manifest step for data versioning given the PortWatch revision issue. A small Streamlit app showing the 30-day forecast with intervals and a Gatun level slider is a cheap, high-impact addition.

## Timeline

Weeks 1 to 2 data and EDA, week 3 baselines and SARIMAX, week 4 censored model and LightGBM, week 5 neural and backtesting, week 6 interpretability, writeup, app. Roughly 60 to 80 hours of real work if you are efficient.

## Pitfalls to avoid

- Feeding actual future lake levels into the test period. Use the ACP's published projections instead, and if you cannot source historical projections, build a naive projection model and use its output.
- Ignoring the AIS revision, which silently changes your 2024 results.
- Reporting a single aggregate RMSE that averages over the drought and normal periods.
- Claiming the LSTM won. With one series and around 2,000 daily observations, it probably will not, and saying that clearly is a stronger signal than a fabricated win.

Resume bullet: *"Forecast daily Panama Canal transit throughput at a 30-day horizon under drought-driven capacity rationing, combining SARIMAX with censored-demand modelling, LightGBM and TFT; leak-free point-in-time feature pipeline with regime-split rolling-origin backtests and a counterfactual estimate of drought-attributable throughput loss."*

Want me to write the PortWatch and ACP ingestion code to get you started, or draft the point-in-time feature function?