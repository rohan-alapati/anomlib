import pandas as pd
import numpy as np
from .types import Event

def scores_to_events(
    df: pd.DataFrame,
    score_col: str,
    entity_col: str,
    time_col: str,
    direction: str,              # "low", "high", or "both"
    threshold: float | dict | None,  # Allow None explicitly
    min_duration: pd.Timedelta,
    threshold_end_ratio: float = 1.0,
    gap_tolerance: pd.Timedelta = pd.Timedelta(0),
    assume_sorted: bool = False,
) -> list[Event]:
    """
    Convert pointwise scores into merged anomaly events.
    - direction="low": score <= -threshold
    - direction="high": score >= threshold
    - direction="both": abs(score) >= threshold
    Uses hysteresis when threshold_end_ratio < 1.0:
    start on threshold, continue while above threshold*ratio.
    """
    out: list[Event] = []
    df = df[[entity_col, time_col, score_col]].dropna()
    if not assume_sorted:
        df = df.sort_values([entity_col, time_col])

    def is_flag(s: float, thr: float) -> bool:
        if direction == "both":
            return abs(s) >= thr
        return (s <= -thr) if direction == "low" else (s >= thr)

    for eid, g in df.groupby(entity_col, sort=False):
        # Ensure eid is of type str or int
        if not isinstance(eid, (str, int)):
            eid = str(eid)

        times = g[time_col].to_list()
        scores = g[score_col].to_list()

        start_i = None
        last_flagged_i = None

        def flush(end_i: int | None):
            nonlocal start_i
            if start_i is None or end_i is None:  # Ensure end_i is not None
                return
            start_t = times[start_i]
            end_t = times[end_i]
            if end_t - start_t >= min_duration:
                seg_scores = scores[start_i:end_i+1]
                out.append(Event(
                    entity_id=str(eid) if not isinstance(eid, (str, int)) else eid,
                    start=start_t,
                    end=end_t,
                    direction=direction,  # type: ignore
                    severity=float(max(abs(x) for x in seg_scores)),
                    score_peak=float(
                        max(abs(x) for x in seg_scores)
                        if direction == "both"
                        else (max(seg_scores) if direction == "high" else -min(seg_scores))
                    ),
                    score_mean=float(sum(abs(x) for x in seg_scores) / len(seg_scores)),
                    reason=f"{direction} deviation beyond {thr_start} (end_ratio={threshold_end_ratio})",
                ))
            start_i = None

        thr = threshold
        if isinstance(threshold, dict):
            default_thr = threshold.get("__default__", None)
            thr = threshold.get(eid, threshold.get(str(eid), default_thr))
            if thr is None:
                thr = 0.0
            thr = float(thr)
        # Ensure thr is resolved to a float
        if isinstance(thr, dict):
            default_thr = thr.get("__default__", 0.0)
            thr = thr.get(eid, thr.get(str(eid), default_thr))
        if thr is None:
            thr = 0.0
        thr_start = float(thr)
        thr_end = max(0.0, thr_start * float(threshold_end_ratio))
        for i, (t, s) in enumerate(zip(times, scores)):
            flagged_start = is_flag(s, thr_start)
            flagged_continue = is_flag(s, thr_end)
            if start_i is None:
                if flagged_start:
                    start_i = i
                    last_flagged_i = i
            else:
                # currently in an event; decide if it continues
                if flagged_continue:
                    last_flagged_i = i
                else:
                    # allow small gaps
                    if (
                        gap_tolerance > pd.Timedelta(0)
                        and last_flagged_i is not None
                        and (t - times[last_flagged_i]) <= gap_tolerance
                    ):
                        continue
                    else:
                        flush(last_flagged_i)
        # flush tail
        if start_i is not None:
            flush(last_flagged_i)

    return out


def scores_to_events_hysteresis(
    df: pd.DataFrame,
    score_col: str,
    entity_col: str,
    time_col: str,
    direction: str,  # "high" only for now
    start_threshold: float,
    continue_threshold: float,
    min_duration: pd.Timedelta,
    gap_tolerance: pd.Timedelta = pd.Timedelta(0),
    assume_sorted: bool = False,
) -> list[Event]:
    """
    Convert pointwise scores to events using explicit hysteresis thresholds.
    - start event when score >= start_threshold
    - continue event while score >= continue_threshold
    """
    if direction != "high":
        raise ValueError("scores_to_events_hysteresis currently supports direction='high' only.")

    out: list[Event] = []
    d = df[[entity_col, time_col, score_col]].dropna()
    if not assume_sorted:
        d = d.sort_values([entity_col, time_col])
    thr_start = float(start_threshold)
    thr_continue = float(min(continue_threshold, start_threshold))

    for eid, g in d.groupby(entity_col, sort=False):
        if not isinstance(eid, (str, int)):
            eid = str(eid)

        times = g[time_col].to_list()
        scores = g[score_col].to_list()
        start_i = None
        last_kept_i = None

        def flush(end_i: int | None):
            nonlocal start_i
            if start_i is None or end_i is None:
                return
            start_t = times[start_i]
            end_t = times[end_i]
            if end_t - start_t >= min_duration:
                seg_scores = scores[start_i:end_i + 1]
                out.append(
                    Event(
                        entity_id=eid,
                        start=start_t,
                        end=end_t,
                        direction="high",
                        severity=float(max(seg_scores)),
                        score_peak=float(max(seg_scores)),
                        score_mean=float(sum(seg_scores) / len(seg_scores)),
                        reason=(
                            "high probability hysteresis "
                            f"(start>={thr_start:.3f}, continue>={thr_continue:.3f})"
                        ),
                    )
                )
            start_i = None

        for i, (t, s) in enumerate(zip(times, scores)):
            start_flag = s >= thr_start
            continue_flag = s >= thr_continue
            if start_i is None:
                if start_flag:
                    start_i = i
                    last_kept_i = i
            else:
                if continue_flag:
                    last_kept_i = i
                else:
                    if (
                        gap_tolerance > pd.Timedelta(0)
                        and last_kept_i is not None
                        and (t - times[last_kept_i]) <= gap_tolerance
                    ):
                        continue
                    flush(last_kept_i)

        if start_i is not None:
            flush(last_kept_i)

    return out


def events_to_point_labels(
    df: pd.DataFrame,
    events,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    assume_sorted: bool = False,
) -> np.ndarray:
    """
    Convert event intervals into pointwise boolean labels aligned to df rows.
    pred[i] = True iff df row i falls within any event interval for the same entity.
    """
    pred = np.zeros(len(df), dtype=bool)
    if len(df) == 0 or not events:
        return pred

    d = df[[entity_col, time_col]].copy()
    # Keep positional index so assignment is aligned even when df.index is non-consecutive.
    d["__pos"] = np.arange(len(d), dtype=int)

    ev_by_ent: dict[object, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for e in events:
        ev_by_ent.setdefault(e.entity_id, []).append((pd.to_datetime(e.start), pd.to_datetime(e.end)))

    for ent, g in d.groupby(entity_col, sort=False):
        intervals = ev_by_ent.get(ent)
        if not intervals:
            intervals = ev_by_ent.get(str(ent))
        if not intervals:
            continue

        intervals = sorted(intervals, key=lambda x: x[0])
        gg = g if assume_sorted else g.sort_values(time_col)
        t = pd.to_datetime(gg[time_col]).to_numpy()
        pos = gg["__pos"].to_numpy()
        m = np.zeros(len(gg), dtype=bool)
        j = 0
        for i in range(len(gg)):
            ti = t[i]
            while j < len(intervals) and intervals[j][1] < ti:
                j += 1
            if j < len(intervals):
                s, e = intervals[j]
                if s <= ti <= e:
                    m[i] = True
        pred[pos] = m
    return pred
