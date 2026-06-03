from dataclasses import dataclass
from typing import Literal
import pandas as pd

Direction = Literal["low", "high", "both"]

@dataclass(frozen=True)
class PredictionFrame:
    expected: pd.Series        # y_hat aligned to df rows
    scale: pd.Series           # robust scale aligned to df rows

@dataclass(frozen=True)
class Event:
    entity_id: str | int
    start: pd.Timestamp
    end: pd.Timestamp
    direction: Literal["low", "high", "both"]
    severity: float
    score_peak: float
    score_mean: float
    reason: str
    duration: pd.Timedelta | None = None
    max_abs_score: float | None = None
    mean_abs_score: float | None = None
    observed_mean: float | None = None
    expected_mean: float | None = None
    residual_mean: float | None = None
