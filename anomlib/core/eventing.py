import pandas as pd
from .types import Event

def scores_to_events(
    df: pd.DataFrame,
    score_col: str,
    entity_col: str,
    time_col: str,
    direction: str,              # "low" or "high"
    threshold: float,
    min_duration: pd.Timedelta,
    gap_tolerance: pd.Timedelta = pd.Timedelta(0),
) -> list[Event]:
    """
    Convert pointwise scores into merged anomaly events.
    - direction="low": score <= -threshold
    - direction="high": score >= threshold
    """
    out: list[Event] = []
    df = df[[entity_col, time_col, score_col]].dropna().sort_values([entity_col, time_col])

    def is_flag(s: float) -> bool:
        return (s <= -threshold) if direction == "low" else (s >= threshold)

    for eid, g in df.groupby(entity_col, sort=False):
        # Ensure eid is of type str or int
        if not isinstance(eid, (str, int)):
            eid = str(eid)

        times = g[time_col].to_list()
        scores = g[score_col].to_list()

        start_i = None
        last_time = None

        def flush(end_i: int):
            nonlocal start_i
            if start_i is None:
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
                    score_peak=float(max(seg_scores) if direction=="high" else -min(seg_scores)),
                    score_mean=float(sum(abs(x) for x in seg_scores) / len(seg_scores)),
                    reason=f"{direction} deviation beyond {threshold}",
                ))
            start_i = None

        for i, (t, s) in enumerate(zip(times, scores)):
            flagged = is_flag(s)
            if start_i is None:
                if flagged:
                    start_i = i
                    last_time = t
            else:
                # currently in an event; decide if it continues
                if flagged:
                    last_time = t
                else:
                    # allow small gaps
                    if gap_tolerance > pd.Timedelta(0) and (t - last_time) <= gap_tolerance:
                        pass
                    else:
                        flush(i-1)
        # flush tail
        if start_i is not None:
            flush(len(times)-1)

    return out
