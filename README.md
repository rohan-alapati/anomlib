# anomlib

`anomlib` is a Python library for anomaly event detection in time-series data.

It follows a consistent pipeline:
1. baseline
2. robust score
3. event merge

For labeled energy datasets such as LEAD/Kaggle, the repo also includes a supervised path:
1. leakage-safe feature build
2. row-level anomaly probability
3. hysteresis event merge

## Design split: core vs detectors

The codebase is intentionally split into:

- `anomlib/core`: reusable, domain-agnostic mechanics
- `anomlib/detectors`: opinionated detector presets

### Core (generic mechanics)

Core contains only generic building blocks:

- schema normalization (`normalize_timeseries_df`)
- robust residual scoring (`robust_z_score`)
- generic event merging (`scores_to_events`)
- generic baseline (`EntityRobustBaseline`)

Core does **not** encode energy-specific assumptions by default.

### Detectors (opinionated behavior lives here)

#### `EnergyTimeSeriesDetector` (opinionated preset)

This detector owns energy-specific assumptions:

- weekly seasonality (`day-of-week`, `hour`)
- fixed lag features (`24`, `168`)
- local-history blend with clipping
- default directional/event behavior suitable for incident-like anomalies

Use this when your data looks like meter/utility series.

#### `GenericTimeSeriesDetector` (less-opinionated preset)

This detector uses only generic baseline behavior from core:

- trailing robust per-entity baseline
- no weekly cadence assumptions
- no fixed energy lags
- generic defaults (`direction="both"`, no persistence required by default)

Use this when you want a reusable starting point without domain assumptions.

#### `EnergySupervisedDetector` (labeled energy preset)

This detector uses supervised learning for datasets that include an `anomaly` label:

- normalizes to the internal schema, then aligns labels on `(entity_id, timestamp)`
- builds leakage-safe lag and rolling features using only past data
- adds a train-only seasonal expectation and residual features
- trains a row-level classifier (LightGBM when available)
- converts probabilities into events with hysteresis eventing
- tunes event parameters on a validation split instead of tuning on test

Use this when you have labeled building-meter anomalies and want stronger precision/recall than the unsupervised baseline can usually provide.

### Opinionated modules

Energy-specific baseline logic lives in:
- `anomlib/opinionated/energy_baseline.py`

This keeps `anomlib/core` generic and moves domain assumptions out of core.

## Quick usage

### Energy preset

```python
from anomlib.detectors import EnergyTimeSeriesDetector

det = EnergyTimeSeriesDetector(
    entity_col="building_id",
    time_col="timestamp",
    value_col="meter_reading",
    direction="low",            # "low" | "high" | "both"
    threshold=3.5,
    threshold_quantile=0.999,
    threshold_cap=3.5,
    threshold_end_ratio=0.6,
    min_duration="8h",
    gap_tolerance="1h",
)

det.fit(df_history)
events, scored = det.detect(df_new)
```

### Generic preset

```python
from anomlib.detectors import GenericTimeSeriesDetector

det = GenericTimeSeriesDetector(
    entity_col="entity_id",
    time_col="timestamp",
    value_col="value",
    direction="both",
    threshold=3.5,
    min_duration="0h",
    gap_tolerance="0h",
)

det.fit(df_history)
events, scored = det.detect(df_new)
```

### Supervised energy preset

```python
from anomlib.detectors import EnergySupervisedDetector

det = EnergySupervisedDetector(
    entity_col="building_id",
    time_col="timestamp",
    value_col="meter_reading",
    direction="high",
    threshold_strategy="evented_f1_grid",
    use_hysteresis=True,
    min_duration="3h",
    gap_tolerance="3h",
)

det.fit(train_df, val_df=val_df)   # requires train_df["anomaly"]
events, scored = det.detect(test_df)
```

## Current structure

- opinionated baseline logic lives outside `anomlib/core`
- `anomlib/opinionated/energy_baseline.py` holds energy-specific assumptions
- `EntityRobustBaseline` remains the domain-agnostic core baseline
- `GenericTimeSeriesDetector` provides a reusable non-domain preset
- `EnergyTimeSeriesDetector` remains the explicit unsupervised energy preset
- `EnergySupervisedDetector` adds a labeled energy path with probability scoring and event tuning
- `benchmarks/kaggle_train_eval.py` supports both supervised evaluation and unsupervised routing

This keeps core reusable while making modeling assumptions explicit at detector level.

## Benchmarking (kaggle_train_eval)

`benchmarks/kaggle_train_eval.py` supports two main flows:

- unsupervised routing flow:
  uses `EnergyTimeSeriesDetector` and optional routing to `GenericTimeSeriesDetector`
- supervised flow:
  uses `EnergySupervisedDetector` with a per-entity time-based `train/val/test` split

In supervised mode the benchmark:

- splits each building into contiguous train/val/test blocks
- fits the classifier on train only
- tunes hysteresis event parameters on val only
- reports point-level and event-level metrics on test only
- writes:
  - `out/kaggle_events_with_overlap.csv`
  - `out/kaggle_scores.parquet`

This makes the benchmark useful both for model iteration and for understanding the operational tradeoff between precision, recall, and event quality.

The unsupervised benchmark path still supports auto-routing between detectors:

- set `AUTO_ROUTE_GENERIC = True` to route worst baseline-mismatch buildings
- control how many via `AUTO_ROUTE_TOP_K`
- always route specific IDs via `MANUAL_GENERIC_BUILDINGS`

This lets you mix `EnergyTimeSeriesDetector` and `GenericTimeSeriesDetector` in one run.
