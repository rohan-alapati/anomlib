from __future__ import annotations

import pandas as pd

from anomlib.core.schema import normalize_timeseries_df
from anomlib.core.scoring import robust_z_score
from anomlib.core.eventing import scores_to_events
from anomlib.opinionated.energy_baseline import EnergySeasonalBaseline


def _to_timedelta(x) -> pd.Timedelta:
    """Accept '6H'/'6h', '1H'/'1h', or pd.Timedelta."""
    if isinstance(x, pd.Timedelta):
        return x
    if isinstance(x, str):
        # normalize common pandas strings like "6H" -> "6h"
        return pd.Timedelta(x.lower())
    return pd.Timedelta(x)  # last resort (numbers, etc.)


class EnergyTimeSeriesDetector:
    def __init__(
        self,
        entity_col: str = "building_id",
        time_col: str = "timestamp",
        value_col: str = "meter_reading",
        direction: str = "low",
        threshold: float = 3.5,
        threshold_quantile: float | None = 0.999,
        threshold_cap: float | None = None,
        threshold_end_ratio: float = 1.0,
        seasonal_weight: float = 0.75,
        rolling_window: int = 36,
        local_clip_ratio: float = 0.2,
        min_duration: str | pd.Timedelta = "6h",
        gap_tolerance: str | pd.Timedelta = "1h",
    ):
        # user-facing column names (dataset-specific)
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col

        self.direction = direction
        self.threshold = float(threshold)
        self.threshold_quantile = threshold_quantile
        self.threshold_cap = float(threshold_cap) if threshold_cap is not None else None
        self.threshold_end_ratio = float(threshold_end_ratio)
        self.seasonal_weight = float(seasonal_weight)
        self.rolling_window = int(rolling_window)
        self.local_clip_ratio = float(local_clip_ratio)
        self.min_duration = _to_timedelta(min_duration)
        self.gap_tolerance = _to_timedelta(gap_tolerance)
        self._threshold_by_entity = None

        # Opinionated energy baseline (weekly seasonality + fixed lags).
        self.baseline = EnergySeasonalBaseline(
            entity_col="entity_id",
            time_col="timestamp",
            value_col="value",
            seasonal_weight=self.seasonal_weight,
            rolling_window=self.rolling_window,
            local_clip_ratio=self.local_clip_ratio,
        )

    def fit(self, df: pd.DataFrame):
        # normalize to internal schema: entity_id, timestamp, value
        d = normalize_timeseries_df(df, self.entity_col, self.time_col, self.value_col)
        self.baseline.fit(d)
        if self.threshold_quantile is not None:
            pred = self.baseline.predict(d)
            scores = robust_z_score(d, value_col="value", pred=pred).abs()
            by_ent = scores.groupby(d["entity_id"]).quantile(self.threshold_quantile)
            if self.threshold_cap is not None:
                by_ent = by_ent.clip(upper=self.threshold_cap)
            self._threshold_by_entity = by_ent.to_dict()
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        # normalize to internal schema
        d = normalize_timeseries_df(df, self.entity_col, self.time_col, self.value_col)
        pred = self.baseline.predict(d)
        return robust_z_score(d, value_col="value", pred=pred)

    def detect(self, df: pd.DataFrame):
        # normalize ONCE and compute everything off this same frame (prevents misalignment bugs)
        d = normalize_timeseries_df(df, self.entity_col, self.time_col, self.value_col)

        pred = self.baseline.predict(d)
        d["score"] = robust_z_score(d, value_col="value", pred=pred).to_numpy()

        if self._threshold_by_entity:
            threshold = {"__default__": self.threshold, **self._threshold_by_entity}
        else:
            threshold = self.threshold
        events = scores_to_events(
            df=d,
            score_col="score",
            entity_col="entity_id",
            time_col="timestamp",
            direction=self.direction,
            threshold=threshold,
            min_duration=self.min_duration,
            threshold_end_ratio=self.threshold_end_ratio,
            gap_tolerance=self.gap_tolerance,
            assume_sorted=True,
        )

        return events, d[["entity_id", "timestamp", "value", "score"]]
