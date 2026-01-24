# anomlib WIP

**anomlib** is an open-source Python library for detecting **anomaly events** in time-series data.

Instead of returning noisy point-level flags, anomlib focuses on identifying **persistent, explainable anomaly periods** (“events”) that are actionable in real systems like building energy, utilities, and infrastructure monitoring.

---

## Why anomlib?

Anomaly detection is hard because:
- “Anomaly” depends on context (time, entity, seasonality)
- Different datasets need different assumptions
- Fully automatic models are unreliable without guardrails
- Most tools return pointwise scores, not incidents

anomlib addresses this by:
- providing **opinionated preset detectors** for common data types
- enforcing a **baseline → score → event** workflow
- producing **interpretable anomaly events**
- remaining **extensible** for researchers and advanced users

This library is designed to be **useful first**, and **flexible second**.

---

## Core concepts

### 1) Events, not just points
anomlib groups anomalous points into events

---

### 2) Preset detectors with sane defaults
anomlib ships with detectors that encode **domain assumptions** so users don’t start from scratch.

Current presets:
- **EnergyTimeSeriesDetector** — for building / utility meter data (WIP)

Each detector:
- learns a notion of “normal”
- scores deviations
- applies persistence and merging rules
- returns anomaly events + pointwise scores

---

### 3) Simple interface, optional depth
Basic usage:

```python
from anomlib.detectors import EnergyTimeSeriesDetector

det = EnergyTimeSeriesDetector(
    direction="low",
    sensitivity="medium",
    min_duration="6H"
)

det.fit(df_history)
events = det.detect(df_new)
