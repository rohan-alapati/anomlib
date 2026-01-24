from dataclasses import dataclass
from typing import Literal, Optional
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
    direction: Literal["low", "high"]
    severity: float
    score_peak: float
    score_mean: float
    reason: str
