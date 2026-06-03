from __future__ import annotations

from math import ceil

import pandas as pd

from anomlib.core.types import Event
from anomlib.datasets.cornell_emcs import infer_timeseries_cadence
from anomlib.opinionated import CornellEMCSSeasonalBaseline

MAD_TO_ROBUST_STD = 1.4826


def _to_timedelta(x: str | pd.Timedelta) -> pd.Timedelta:
    if isinstance(x, pd.Timedelta):
        return x
    normalized = x
    if isinstance(x, str) and x.endswith("d"):
        normalized = f"{x[:-1]}D"
    return pd.to_timedelta(normalized)


def _cadence_defaults(cadence: str) -> dict[str, object]:
    if cadence == "hourly":
        return {
            "threshold": 4.0,
            "min_duration": pd.Timedelta(hours=3),
            "max_gap": pd.Timedelta(hours=1),
        }
    return {
        "threshold": 3.75,
        "min_duration": pd.Timedelta(days=2),
        "max_gap": pd.Timedelta(days=1),
    }


class CornellEMCSElectricityDetector:
    def __init__(
        self,
        *,
        cadence: str | None = None,
        direction: str = "both",
        threshold: float | None = None,
        min_duration: str | pd.Timedelta | None = None,
        max_gap: str | pd.Timedelta | None = None,
        threshold_quantile: float | None = None,
        threshold_floor: float = 3.5,
        min_group_history: int = 5,
    ):
        self.cadence = cadence
        self.direction = direction
        self.threshold = threshold
        self.min_duration = _to_timedelta(min_duration) if min_duration is not None else None
        self.max_gap = _to_timedelta(max_gap) if max_gap is not None else None
        self.threshold_quantile = threshold_quantile
        self.threshold_floor = float(threshold_floor)
        self.min_group_history = int(min_group_history)

        self._baseline: CornellEMCSSeasonalBaseline | None = None
        self._threshold_by_entity: dict[str | int, float] = {}
        self._cadence_delta: pd.Timedelta | None = None
        self._resolved_threshold: float | None = None
        self._resolved_min_duration: pd.Timedelta | None = None
        self._resolved_max_gap: pd.Timedelta | None = None

    def fit(self, df: pd.DataFrame):
        d = self._normalize(df)
        inferred_cadence = self.cadence or infer_timeseries_cadence(d)
        defaults = _cadence_defaults(inferred_cadence)
        self.cadence = inferred_cadence
        self._resolved_threshold = float(self.threshold if self.threshold is not None else defaults["threshold"])
        self._resolved_min_duration = (
            self.min_duration if self.min_duration is not None else defaults["min_duration"]
        )
        self._resolved_max_gap = self.max_gap if self.max_gap is not None else defaults["max_gap"]
        self._cadence_delta = self._infer_step(d)

        self._baseline = CornellEMCSSeasonalBaseline(
            cadence=self.cadence,
            min_group_history=self.min_group_history,
        ).fit(d)

        if self.threshold_quantile is not None:
            train_scores = self.score(d)
            calibrated = (
                train_scores.groupby("entity_id", sort=False)["absolute_score"]
                .quantile(self.threshold_quantile)
                .clip(lower=self.threshold_floor)
            )
            self._threshold_by_entity = calibrated.to_dict()
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._normalize(df)
        if self._baseline is None:
            raise RuntimeError("Call fit() before score().")

        pred = self._baseline.predict(d)
        scored = d.copy()
        scored["expected"] = pred.expected.to_numpy()
        scored["residual"] = scored["value"] - scored["expected"]
        scored["scale"] = pred.scale.clip(lower=1e-6).to_numpy()
        scored["signed_score"] = scored["residual"] / (scored["scale"] * MAD_TO_ROBUST_STD)
        scored["absolute_score"] = scored["signed_score"].abs()
        scored["direction"] = "normal"
        scored.loc[scored["signed_score"] > 0, "direction"] = "high"
        scored.loc[scored["signed_score"] < 0, "direction"] = "low"
        return scored[
            [
                "entity_id",
                "timestamp",
                "value",
                "expected",
                "residual",
                "scale",
                "signed_score",
                "absolute_score",
                "direction",
            ]
        ]

    def detect(self, df: pd.DataFrame) -> tuple[list[Event], pd.DataFrame]:
        scored = self.score(df)
        events = self._build_events(scored)
        return events, scored

    def events_to_frame(self, events: list[Event]) -> pd.DataFrame:
        rows = []
        for event in events:
            rows.append(
                {
                    "entity_id": event.entity_id,
                    "start_time": event.start,
                    "end_time": event.end,
                    "duration": event.duration,
                    "direction": event.direction,
                    "max_abs_score": event.max_abs_score if event.max_abs_score is not None else event.severity,
                    "mean_abs_score": event.mean_abs_score if event.mean_abs_score is not None else event.score_mean,
                    "observed_mean": event.observed_mean,
                    "expected_mean": event.expected_mean,
                    "residual_mean": event.residual_mean,
                    "reason": event.reason,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "entity_id",
                "start_time",
                "end_time",
                "duration",
                "direction",
                "max_abs_score",
                "mean_abs_score",
                "observed_mean",
                "expected_mean",
                "residual_mean",
                "reason",
            ],
        )

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"entity_id", "timestamp", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "CornellEMCSElectricityDetector expects normalized columns "
                f"{sorted(required)}. Missing: {sorted(missing)}."
            )
        d = df[["entity_id", "timestamp", "value"]].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["entity_id", "timestamp", "value"])
        return d.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    def _infer_step(self, d: pd.DataFrame) -> pd.Timedelta:
        deltas = d.groupby("entity_id", sort=False)["timestamp"].diff().dropna()
        if len(deltas) == 0:
            if self.cadence == "hourly":
                return pd.Timedelta(hours=1)
            return pd.Timedelta(days=1)
        return deltas.median()

    def _threshold_for_entity(self, entity_id: str | int) -> float:
        default = self._resolved_threshold if self._resolved_threshold is not None else self.threshold_floor
        if not self._threshold_by_entity:
            return float(default)
        return float(
            self._threshold_by_entity.get(
                entity_id,
                self._threshold_by_entity.get(str(entity_id), default),
            )
        )

    def _direction_for_row(self, score: float, threshold: float) -> str | None:
        if self.direction == "high":
            return "high" if score >= threshold else None
        if self.direction == "low":
            return "low" if score <= -threshold else None
        if score >= threshold:
            return "high"
        if score <= -threshold:
            return "low"
        return None

    def _build_events(self, scored: pd.DataFrame) -> list[Event]:
        if self._resolved_min_duration is None or self._resolved_max_gap is None or self._cadence_delta is None:
            raise RuntimeError("Call fit() before detect().")

        events: list[Event] = []
        min_points = max(1, ceil(self._resolved_min_duration / self._cadence_delta))
        allowed_gap = self._cadence_delta + self._resolved_max_gap

        for entity_id, g in scored.groupby("entity_id", sort=False):
            threshold = self._threshold_for_entity(entity_id)
            open_rows: list[pd.Series] = []
            event_direction: str | None = None
            last_flagged_time: pd.Timestamp | None = None

            def flush() -> None:
                nonlocal open_rows, event_direction, last_flagged_time
                flagged_rows = [row for row in open_rows if self._direction_for_row(row["signed_score"], threshold) == event_direction]
                if event_direction is not None and len(flagged_rows) >= min_points:
                    event_df = pd.DataFrame(open_rows)
                    events.append(self._summarize_event(entity_id, event_direction, threshold, event_df))
                open_rows = []
                event_direction = None
                last_flagged_time = None

            for _, row in g.iterrows():
                row_direction = self._direction_for_row(float(row["signed_score"]), threshold)
                row_time = row["timestamp"]

                if event_direction is None:
                    if row_direction is None:
                        continue
                    event_direction = row_direction
                    open_rows = [row]
                    last_flagged_time = row_time
                    continue

                if row_direction == event_direction:
                    open_rows.append(row)
                    last_flagged_time = row_time
                    continue

                if row_direction is None and last_flagged_time is not None and (row_time - last_flagged_time) <= allowed_gap:
                    open_rows.append(row)
                    continue

                flush()
                if row_direction is not None:
                    event_direction = row_direction
                    open_rows = [row]
                    last_flagged_time = row_time

            flush()
        return events

    def _summarize_event(
        self,
        entity_id: str | int,
        direction: str,
        threshold: float,
        event_df: pd.DataFrame,
    ) -> Event:
        keep_mask = event_df["signed_score"].apply(
            lambda score: self._direction_for_row(float(score), threshold) == direction
        )
        flagged = event_df.loc[keep_mask].copy()
        start = event_df["timestamp"].min()
        end = event_df["timestamp"].max()
        observed_mean = float(flagged["value"].mean())
        expected_mean = float(flagged["expected"].mean())
        residual_mean = float(flagged["residual"].mean())
        max_abs_score = float(flagged["absolute_score"].max())
        mean_abs_score = float(flagged["absolute_score"].mean())
        duration = end - start + (self._cadence_delta or pd.Timedelta(0))
        reason = self._format_reason(
            direction=direction,
            start=start,
            end=end,
            observed_mean=observed_mean,
            expected_mean=expected_mean,
        )
        return Event(
            entity_id=entity_id,
            start=start,
            end=end,
            direction=direction,  # type: ignore[arg-type]
            severity=max_abs_score,
            score_peak=max_abs_score,
            score_mean=mean_abs_score,
            reason=reason,
            duration=duration,
            max_abs_score=max_abs_score,
            mean_abs_score=mean_abs_score,
            observed_mean=observed_mean,
            expected_mean=expected_mean,
            residual_mean=residual_mean,
        )

    def _format_reason(
        self,
        *,
        direction: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        observed_mean: float,
        expected_mean: float,
    ) -> str:
        span = end - start + (self._cadence_delta or pd.Timedelta(0))
        if self.cadence == "hourly":
            units = max(1, int(round(span / pd.Timedelta(hours=1))))
            unit_label = "hours"
        else:
            units = max(1, int(round(span / pd.Timedelta(days=1))))
            unit_label = "days"

        if abs(expected_mean) >= 1e-6:
            pct = abs((observed_mean - expected_mean) / expected_mean) * 100.0
            return (
                f"Electricity usage was persistently {direction} for {units} {unit_label}: "
                f"observed mean {pct:.0f}% {'above' if direction == 'high' else 'below'} expected baseline."
            )
        return (
            f"Electricity usage was persistently {direction} for {units} {unit_label}: "
            "expected baseline was near zero."
        )
