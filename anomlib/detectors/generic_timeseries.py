from __future__ import annotations

import pandas as pd

from anomlib.core.baseline import EntityRobustBaseline
from anomlib.core.eventing import scores_to_events
from anomlib.core.schema import normalize_timeseries_df
from anomlib.core.scoring import robust_z_score


def _to_timedelta(x) -> pd.Timedelta:
    if isinstance(x, pd.Timedelta):
        return x
    if isinstance(x, str):
        return pd.Timedelta(x.lower())
    return pd.Timedelta(x)


class GenericTimeSeriesDetector:
    """
    Less-opinionated detector using only domain-agnostic core behavior.
    """

    def __init__(
        self,
        entity_col: str = "entity_id",
        time_col: str = "timestamp",
        value_col: str = "value",
        direction: str = "both",
        threshold: float = 3.5,
        threshold_quantile: float | None = None,
        threshold_cap: float | None = None,
        threshold_end_ratio: float = 1.0,
        min_duration: str | pd.Timedelta = "0h",
        gap_tolerance: str | pd.Timedelta = "0h",
        min_history: int = 5,
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col

        self.direction = direction
        self.threshold = float(threshold)
        self.threshold_quantile = threshold_quantile
        self.threshold_cap = float(threshold_cap) if threshold_cap is not None else None
        self.threshold_end_ratio = float(threshold_end_ratio)
        self.min_duration = _to_timedelta(min_duration)
        self.gap_tolerance = _to_timedelta(gap_tolerance)
        self._threshold_by_entity = None

        self.baseline = EntityRobustBaseline(
            entity_col="entity_id",
            time_col="timestamp",
            value_col="value",
            min_history=min_history,
        )

    def fit(self, df: pd.DataFrame):
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
        d = normalize_timeseries_df(df, self.entity_col, self.time_col, self.value_col)
        pred = self.baseline.predict(d)
        return robust_z_score(d, value_col="value", pred=pred)

    def detect(self, df: pd.DataFrame):
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
        )
        return events, d[["entity_id", "timestamp", "value", "score"]]
