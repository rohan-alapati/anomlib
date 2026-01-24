import pandas as pd
from anomlib.core.baseline import CalendarMeanBaseline
from anomlib.core.scoring import robust_z_score
from anomlib.core.eventing import scores_to_events

class EnergyTimeSeriesDetector:
    def __init__(
        self,
        entity_col="building_id",
        time_col="timestamp",
        value_col="meter_reading",
        direction="low",              # Kaggle-like start
        threshold=3.5,                # tune later
        min_duration="6H",
        gap_tolerance="1H",
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col
        self.direction = direction
        self.threshold = float(threshold)
        self.min_duration = pd.Timedelta(min_duration)
        self.gap_tolerance = pd.Timedelta(gap_tolerance)
        self.baseline = CalendarMeanBaseline(entity_col, time_col, value_col)

    def fit(self, df: pd.DataFrame):
        self.baseline.fit(df)
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        pred = self.baseline.predict(df)
        return robust_z_score(df, self.value_col, pred)

    def detect(self, df: pd.DataFrame):
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df["score"] = self.score(df)
        events = scores_to_events(
            df=df,
            score_col="score",
            entity_col=self.entity_col,
            time_col=self.time_col,
            direction=self.direction,
            threshold=self.threshold,
            min_duration=self.min_duration,
            gap_tolerance=self.gap_tolerance,
        )
        return events, df[["score"]]
