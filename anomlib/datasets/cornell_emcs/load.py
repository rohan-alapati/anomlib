from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

TIME_COLUMN_CANDIDATES = ("time", "timestamp", "date")
ELECTRICITY_COLUMN_PATTERN = re.compile(
    r"(elec|electric|power|kw\b|kw/|kwh)",
    flags=re.IGNORECASE,
)


def infer_cornell_emcs_time_column(df: pd.DataFrame) -> str:
    for candidate in df.columns:
        if candidate.strip().lower() in TIME_COLUMN_CANDIDATES:
            return candidate
    raise ValueError(
        "Could not infer a timestamp column. Expected one of "
        f"{TIME_COLUMN_CANDIDATES}, found {list(df.columns)}."
    )


def infer_cornell_emcs_meter_columns(
    df: pd.DataFrame,
    time_col: str = "Time",
) -> list[str]:
    candidates = [col for col in df.columns if col != time_col]
    if not candidates:
        raise ValueError("No meter columns were found in the Cornell EMCS export.")

    electricity_cols = [col for col in candidates if ELECTRICITY_COLUMN_PATTERN.search(col)]
    return electricity_cols if electricity_cols else candidates


def load_cornell_emcs_csv(
    path: str | Path,
    *,
    time_col: str | None = None,
    meter_columns: list[str] | None = None,
    drop_invalid_timestamps: bool = True,
    drop_missing_values: bool = True,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    resolved_time_col = time_col or infer_cornell_emcs_time_column(raw)
    resolved_meter_columns = meter_columns or infer_cornell_emcs_meter_columns(raw, time_col=resolved_time_col)

    missing_cols = [col for col in [resolved_time_col, *resolved_meter_columns] if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing expected Cornell EMCS columns: {missing_cols}")

    melted = raw[[resolved_time_col, *resolved_meter_columns]].melt(
        id_vars=[resolved_time_col],
        value_vars=resolved_meter_columns,
        var_name="entity_id",
        value_name="value",
    )
    melted = melted.rename(columns={resolved_time_col: "timestamp"})
    melted["timestamp"] = pd.to_datetime(melted["timestamp"], errors="coerce").astype("datetime64[ns]")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

    if drop_invalid_timestamps:
        melted = melted.dropna(subset=["timestamp"])
    if drop_missing_values:
        melted = melted.dropna(subset=["value"])

    melted = melted.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return melted[["entity_id", "timestamp", "value"]]


def infer_timeseries_cadence(
    df: pd.DataFrame,
    *,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
) -> str:
    if len(df) < 2:
        return "daily"

    d = df[[entity_col, time_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values([entity_col, time_col])
    if len(d) < 2:
        return "daily"

    deltas = (
        d.groupby(entity_col, sort=False)[time_col]
        .diff()
        .dropna()
    )
    if len(deltas) == 0:
        return "daily"

    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(hours=2):
        return "hourly"
    return "daily"
