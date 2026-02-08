# anomlib/core/schema.py
from __future__ import annotations
import pandas as pd

REQUIRED_STD_COLS = ("entity_id", "timestamp", "value")

def normalize_timeseries_df(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Normalize a user-provided dataframe into the library's internal schema:
      entity_id, timestamp, value

    - renames columns
    - parses timestamp
    - coerces value to numeric
    - drops rows missing essentials
    - sorts by (entity_id, timestamp)
    """
    d = df[[entity_col, time_col, value_col]].copy()

    d = d.rename(columns={
        entity_col: "entity_id",
        time_col: "timestamp",
        value_col: "value",
    })

    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    d = d.dropna(subset=list(REQUIRED_STD_COLS))
    d = d.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return d
