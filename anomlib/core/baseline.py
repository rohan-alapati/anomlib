from __future__ import annotations

import pandas as pd

from .types import PredictionFrame


class EntityRobustBaseline:
    """
    Domain-agnostic baseline:
    - expected = trailing per-entity expanding median (shifted to avoid lookahead)
    - fallback expected for cold start = per-entity median learned in fit
    - scale = per-entity MAD of residuals learned in fit
    """

    def __init__(
        self,
        entity_col: str,
        time_col: str,
        value_col: str,
        min_history: int = 5,
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col
        self.min_history = int(min_history)

        self._entity_median = None
        self._scale_by_entity = None
        self._global_scale = None

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[[self.entity_col, self.time_col, self.value_col]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col])
        d = d.sort_values([self.entity_col, self.time_col]).reset_index(drop=True)
        return d

    def _expected_from_history(self, d: pd.DataFrame) -> pd.Series:
        expected = d.groupby(self.entity_col)[self.value_col].transform(
            lambda s: s.shift(1).expanding(min_periods=self.min_history).median()
        )
        return expected.astype("float64")

    def fit(self, df: pd.DataFrame):
        d = self._sorted(df)
        self._entity_median = d.groupby(self.entity_col)[self.value_col].median()

        expected = self._expected_from_history(d)
        mask = expected.isna()
        if mask.any():
            expected.loc[mask] = d.loc[mask, self.entity_col].map(self._entity_median)

        resid = d[self.value_col] - expected
        self._scale_by_entity = resid.groupby(d[self.entity_col]).apply(
            lambda s: (s - s.median()).abs().median()
        )

        positive = self._scale_by_entity[self._scale_by_entity > 0]
        self._global_scale = float(positive.median()) if len(positive) else 1.0
        self._scale_by_entity = (
            self._scale_by_entity.replace(0, self._global_scale)
            .fillna(self._global_scale)
        )
        return self

    def predict(self, df: pd.DataFrame) -> PredictionFrame:
        if self._entity_median is None or self._scale_by_entity is None:
            raise ValueError("The baseline must be fitted before predict().")

        d = self._sorted(df)
        expected = self._expected_from_history(d)

        mask = expected.isna()
        if mask.any():
            expected.loc[mask] = d.loc[mask, self.entity_col].map(self._entity_median)

        scale = d[self.entity_col].map(self._scale_by_entity.to_dict()).fillna(self._global_scale)
        return PredictionFrame(expected=expected, scale=scale)
