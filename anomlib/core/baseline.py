import pandas as pd
from .types import PredictionFrame

class CalendarMeanBaseline:
    def __init__(self, entity_col: str, time_col: str, value_col: str):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col
        self._means = None

    def fit(self, df: pd.DataFrame):
        d = df[[self.entity_col, self.time_col, self.value_col]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col])
        d["hour"] = d[self.time_col].dt.hour
        # mean per (entity, hour)
        self._means = (
            d.groupby([self.entity_col, "hour"])[self.value_col]
            .mean()
            .rename("expected")
        )
        # global fallback mean per entity (if hour missing)
        self._entity_mean = d.groupby(self.entity_col)[self.value_col].mean()
        return self

    def predict(self, df: pd.DataFrame) -> PredictionFrame:
        d = df[[self.entity_col, self.time_col, self.value_col]].copy()
        d[self.time_col] = pd.to_datetime(d[self.time_col])
        d["hour"] = d[self.time_col].dt.hour

        expected = d.set_index([self.entity_col, "hour"]).index.map(self._means)
        expected = pd.Series(expected, index=df.index, dtype="float64")

        # fallback to entity mean
        mask = expected.isna()
        if mask.any():
            expected.loc[mask] = df.loc[mask, self.entity_col].map(self._entity_mean)

        # robust-ish scale per entity (MAD)
        scale = df.groupby(self.entity_col)[self.value_col].transform(
            lambda s: (s - s.median()).abs().median()
        )
        # avoid zero scale
        scale = scale.replace(0, scale[scale > 0].median())
        return PredictionFrame(expected=expected, scale=scale)
