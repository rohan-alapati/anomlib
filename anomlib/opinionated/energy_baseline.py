from __future__ import annotations

import pandas as pd

from anomlib.core.types import PredictionFrame


class EnergySeasonalBaseline:
    """
    Opinionated energy baseline:
    - seasonal profile by (entity, day-of-week, hour)
    - local history via lag-24, lag-168, and rolling median
    - bounded blending to reduce anomaly pull on expected
    """

    def __init__(
        self,
        entity_col: str,
        time_col: str,
        value_col: str,
        seasonal_weight: float = 0.75,
        rolling_window: int = 36,
        local_clip_ratio: float = 0.2,
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col

        self.seasonal_weight = float(seasonal_weight)
        self.rolling_window = int(rolling_window)
        self.local_clip_ratio = float(local_clip_ratio)

        self._means = None
        self._entity_median = None
        self._scale_by_entity = None
        self._global_scale = None

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[[self.entity_col, self.time_col, self.value_col]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col])
        d = d.sort_values([self.entity_col, self.time_col]).reset_index(drop=True)
        d["dow"] = d[self.time_col].dt.dayofweek
        d["hour"] = d[self.time_col].dt.hour
        return d

    def _seasonal_expected(self, d: pd.DataFrame) -> pd.Series:
        expected = d.set_index([self.entity_col, "dow", "hour"]).index.map(self._means)
        return pd.Series(expected, index=d.index, dtype="float64")

    def _local_expected(self, d: pd.DataFrame) -> pd.Series:
        g = d.groupby(self.entity_col, sort=False)[self.value_col]
        lag_24 = g.shift(24)
        lag_168 = g.shift(168)
        roll_med = g.transform(
            lambda s: s.shift(1).rolling(self.rolling_window, min_periods=max(4, self.rolling_window // 3)).median()
        )
        return pd.concat([lag_24, lag_168, roll_med], axis=1).median(axis=1, skipna=True).astype("float64")

    def _hybrid_expected(self, d: pd.DataFrame) -> pd.Series:
        seasonal = self._seasonal_expected(d)
        local = self._local_expected(d)

        expected = seasonal.copy()
        have_local = local.notna()
        have_seasonal = seasonal.notna()

        mask_blend = have_local & have_seasonal
        local_blend = local.loc[mask_blend]
        seasonal_blend = seasonal.loc[mask_blend]
        max_shift = seasonal_blend.abs().clip(lower=1.0) * self.local_clip_ratio
        local_bounded = seasonal_blend + (local_blend - seasonal_blend).clip(
            lower=-max_shift, upper=max_shift
        )
        expected.loc[mask_blend] = (
            self.seasonal_weight * seasonal_blend
            + (1.0 - self.seasonal_weight) * local_bounded
        )

        mask_local_only = have_local & (~have_seasonal)
        expected.loc[mask_local_only] = local.loc[mask_local_only]
        return expected

    def fit(self, df: pd.DataFrame):
        d = self._sorted(df)
        self._means = (
            d.groupby([self.entity_col, "dow", "hour"])[self.value_col]
            .median()
            .rename("expected")
        )
        self._entity_median = d.groupby(self.entity_col)[self.value_col].median()

        expected = self._hybrid_expected(d)
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
        expected = self._hybrid_expected(d)
        mask = expected.isna()
        if mask.any():
            expected.loc[mask] = d.loc[mask, self.entity_col].map(self._entity_median)

        scale = d[self.entity_col].map(self._scale_by_entity.to_dict()).fillna(self._global_scale)
        return PredictionFrame(expected=expected, scale=scale)
