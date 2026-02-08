import pandas as pd
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
    df = df[[entity_col, time_col, score_col]].dropna().sort_values([entity_col, time_col])

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
