# anomlib (WIP)

`anomlib` is a Python library for anomaly event detection in time-series data.

It follows a consistent pipeline:
1. baseline
2. robust score
3. event merge

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

## What changed in this refactor

- moved opinionated baseline logic out of `anomlib/core`
- added `anomlib/opinionated/energy_baseline.py` for energy assumptions
- replaced core baseline with domain-agnostic `EntityRobustBaseline`
- added `GenericTimeSeriesDetector`
- kept `EnergyTimeSeriesDetector` as explicit opinionated preset
 - added auto-routing support in `benchmarks/kaggle_train_eval.py`

This keeps core reusable while making modeling assumptions explicit at detector level.

## Benchmark routing (kaggle_train_eval)

`benchmarks/kaggle_train_eval.py` supports auto-routing between detectors:

- set `AUTO_ROUTE_GENERIC = True` to route worst baseline-mismatch buildings
- control how many via `AUTO_ROUTE_TOP_K`
- always route specific IDs via `MANUAL_GENERIC_BUILDINGS`

This lets you mix `EnergyTimeSeriesDetector` and `GenericTimeSeriesDetector` in one run.
