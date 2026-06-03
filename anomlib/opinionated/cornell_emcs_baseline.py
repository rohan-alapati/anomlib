from __future__ import annotations

import numpy as np
import pandas as pd

from anomlib.core.types import PredictionFrame


class CornellEMCSSeasonalBaseline:
    """
    Robust seasonal baseline for Cornell EMCS electricity series.

    - hourly cadence: entity x day-of-week x hour
    - daily cadence: entity x month x day-of-week
    - fallback hierarchy when a seasonal bucket has too little history
    - per-entity MAD scale from training residuals
    """

    def __init__(
        self,
        *,
        cadence: str,
        entity_col: str = "entity_id",
        time_col: str = "timestamp",
        value_col: str = "value",
        min_group_history: int = 5,
        min_scale: float = 1e-6,
    ):
        if cadence not in {"hourly", "daily"}:
            raise ValueError("cadence must be 'hourly' or 'daily'.")

        self.cadence = cadence
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col
        self.min_group_history = int(min_group_history)
        self.min_scale = float(min_scale)

        self._entity_median: pd.Series | None = None
        self._global_median: float | None = None
        self._global_scale: float | None = None
        self._scale_by_entity: pd.Series | None = None
        self._profiles: list[tuple[list[str], pd.DataFrame]] = []

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[[self.entity_col, self.time_col, self.value_col]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col], errors="coerce")
        d[self.value_col] = pd.to_numeric(d[self.value_col], errors="coerce")
        d = d.dropna(subset=[self.entity_col, self.time_col, self.value_col])
        d = d.sort_values([self.entity_col, self.time_col]).reset_index(drop=True)
        d["dow"] = d[self.time_col].dt.dayofweek.astype(int)
        d["hour"] = d[self.time_col].dt.hour.astype(int)
        d["month"] = d[self.time_col].dt.month.astype(int)
        return d

    def _profile_groupings(self) -> list[list[str]]:
        if self.cadence == "hourly":
            return [
                [self.entity_col, "dow", "hour"],
                [self.entity_col, "dow"],
                [self.entity_col, "hour"],
                [self.entity_col],
            ]
        return [
            [self.entity_col, "month", "dow"],
            [self.entity_col, "month"],
            [self.entity_col, "dow"],
            [self.entity_col],
        ]

    def _fit_profiles(self, d: pd.DataFrame) -> None:
        self._profiles = []
        for grouping in self._profile_groupings():
            profile = (
                d.groupby(grouping, dropna=False)[self.value_col]
                .agg(expected="median", history_count="size")
                .reset_index()
            )
            self._profiles.append((grouping, profile))

    def _merge_profile(self, d: pd.DataFrame, grouping: list[str], profile: pd.DataFrame) -> pd.Series:
        merged = d[grouping].merge(profile, on=grouping, how="left")
        expected = merged["expected"].where(merged["history_count"] >= self.min_group_history)
        return pd.Series(expected.to_numpy(), index=d.index, dtype=float)

    def _expected(self, d: pd.DataFrame) -> pd.Series:
        if self._entity_median is None or self._global_median is None:
            raise ValueError("Call fit() before predict().")

        expected = pd.Series(np.nan, index=d.index, dtype="float64")
        for grouping, profile in self._profiles:
            candidate = self._merge_profile(d, grouping, profile)
            expected = expected.fillna(candidate)

        expected = expected.fillna(d[self.entity_col].map(self._entity_median))
        expected = expected.fillna(float(self._global_median))
        return expected.astype(float)

    def fit(self, df: pd.DataFrame):
        d = self._prepare(df)
        if len(d) == 0:
            raise ValueError("Training dataframe has no valid rows after normalization.")

        self._entity_median = d.groupby(self.entity_col)[self.value_col].median()
        self._global_median = float(d[self.value_col].median())
        self._fit_profiles(d)

        expected = self._expected(d)
        resid = d[self.value_col] - expected
        scale_by_entity = resid.groupby(d[self.entity_col]).apply(
            lambda s: float((s - s.median()).abs().median())
        )
        positive = scale_by_entity[scale_by_entity > self.min_scale]
        self._global_scale = float(positive.median()) if len(positive) else 1.0
        self._scale_by_entity = (
            scale_by_entity.clip(lower=self.min_scale)
            .replace(0.0, self._global_scale)
            .fillna(self._global_scale)
        )
        return self

    def predict(self, df: pd.DataFrame) -> PredictionFrame:
        if self._scale_by_entity is None or self._global_scale is None:
            raise ValueError("Call fit() before predict().")

        d = self._prepare(df)
        expected = self._expected(d)
        scale = d[self.entity_col].map(self._scale_by_entity).fillna(self._global_scale).astype(float)
        return PredictionFrame(expected=expected, scale=scale)
