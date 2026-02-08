# benchmarks/kaggle_train_eval.py
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from anomlib.datasets.kaggle_energy.load import load_train
from anomlib.detectors import EnergyTimeSeriesDetector, GenericTimeSeriesDetector
from anomlib.core.schema import normalize_timeseries_df


DATA_PATH = "data/kaggle/train.csv"
OUT_DIR = "out"

# Detector knobs
DIRECTION = "both"
THRESHOLD = 3.5  # fallback only if per-entity thresholds missing
THRESHOLD_QUANTILE = 0.9995
THRESHOLD_CAP = 3.5
THRESHOLD_END_RATIO = 0.6
MIN_DURATION = "8h"      # use lowercase to avoid pandas deprecation warnings
GAP_TOLERANCE = "1h"
TRAIN_FRAC = 0.8
USE_GENERIC_DETECTOR = False
# Optional hybrid routing when USE_GENERIC_DETECTOR is False:
# auto-route worst baseline-mismatch buildings to GenericTimeSeriesDetector.
AUTO_ROUTE_GENERIC = True
AUTO_ROUTE_TOP_K = 10
MANUAL_GENERIC_BUILDINGS: set[int] = {32, 55, 144, 149, 173, 174, 183, 238, 240, 248, 1068}


def per_entity_time_split(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for _, g in df.groupby(entity_col, sort=False):
        g = g.sort_values(time_col)
        k = int(len(g) * frac)
        parts.append((g.iloc[:k], g.iloc[k:]))
    train = pd.concat([a for a, _ in parts]).sort_values([entity_col, time_col])
    test = pd.concat([b for _, b in parts]).sort_values([entity_col, time_col])
    return train, test


def label_points_from_events(
    df: pd.DataFrame,
    events,
    entity_col: str = "building_id",
    time_col: str = "timestamp",
) -> pd.Series:
    """
    Convert event intervals -> pointwise 0/1 predictions for each row in df.
    Runs in ~O(N + E) by sweeping time-ordered rows against time-ordered events per entity.
    """
    out = pd.Series(0, index=df.index, dtype=np.int8)

    # group events per entity
    ev_by_ent = defaultdict(list)
    for e in events:
        ev_by_ent[e.entity_id].append((pd.to_datetime(e.start), pd.to_datetime(e.end)))

    # sweep per entity
    for ent, g in df.groupby(entity_col, sort=False):
        intervals = ev_by_ent.get(ent)
        if not intervals:
            continue

        intervals.sort(key=lambda x: x[0])
        gg = g.sort_values(time_col)
        t = gg[time_col].to_numpy()

        j = 0
        m = np.zeros(len(gg), dtype=np.int8)

        for i in range(len(gg)):
            ti = t[i]
            # advance interval pointer while interval ends before ti
            while j < len(intervals) and intervals[j][1] < ti:
                j += 1
            if j < len(intervals):
                s, e = intervals[j]
                if s <= ti <= e:
                    m[i] = 1

        out.loc[gg.index] = m

    return out


def event_overlap_table(
    df: pd.DataFrame,
    events,
    entity_col: str = "building_id",
    time_col: str = "timestamp",
    label_col: str = "anomaly",
) -> pd.DataFrame:
    rows = []
    for k, e in enumerate(events):
        m = (
            (df[entity_col] == e.entity_id)
            & (df[time_col] >= pd.to_datetime(e.start))
            & (df[time_col] <= pd.to_datetime(e.end))
        )
        seg = df.loc[m]
        if len(seg) == 0:
            frac = np.nan
            n = 0
        else:
            frac = float((seg[label_col] == 1).mean()) if label_col in seg.columns else np.nan
            n = int(len(seg))

        rows.append(
            {
                "event_idx": k,
                "entity_id": e.entity_id,
                "start": pd.to_datetime(e.start),
                "end": pd.to_datetime(e.end),
                "n_points": n,
                "label_frac": frac,
                "direction": getattr(e, "direction", None),
                "severity": getattr(e, "severity", np.nan),
                "score_peak": getattr(e, "score_peak", np.nan),
                "score_mean": getattr(e, "score_mean", np.nan),
                "reason": getattr(e, "reason", ""),
            }
        )
    return pd.DataFrame(rows)


def detect_with_router(
    energy_det: EnergyTimeSeriesDetector,
    generic_det: GenericTimeSeriesDetector | None,
    df: pd.DataFrame,
    generic_buildings: set[int],
) -> tuple[list, pd.DataFrame]:
    if not generic_buildings or generic_det is None:
        return energy_det.detect(df)

    generic_mask = df["building_id"].isin(generic_buildings)
    df_generic = df.loc[generic_mask]
    df_energy = df.loc[~generic_mask]

    all_events = []
    score_parts = []

    if len(df_energy) > 0:
        e_events, e_scores = energy_det.detect(df_energy)
        all_events.extend(e_events)
        score_parts.append(e_scores)

    if len(df_generic) > 0:
        g_events, g_scores = generic_det.detect(df_generic)
        all_events.extend(g_events)
        score_parts.append(g_scores)

    if score_parts:
        scores = (
            pd.concat(score_parts, axis=0)
            .sort_values(["entity_id", "timestamp"])
            .reset_index(drop=True)
        )
    else:
        scores = pd.DataFrame(columns=["entity_id", "timestamp", "value", "score"])

    return all_events, scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load
    df = load_train(DATA_PATH)
    df = df.copy()

    # Ensure timestamp dtype
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Drop NaN readings (common in this dataset)
    before = len(df)
    df = df.dropna(subset=["meter_reading"]).reset_index(drop=True)
    after = len(df)
    if after != before:
        print(f"dropped NaN meter_reading: {before - after:,} rows (kept {after:,})")

    # 2) Time-based train/test split per entity
    train_df, test_df = per_entity_time_split(
        df, entity_col="building_id", time_col="timestamp", frac=TRAIN_FRAC
    )
    print(
        f"per-entity split  "
        f"train rows: {len(train_df):,} ({len(train_df)/len(df):.1%})  "
        f"test rows: {len(test_df):,} ({len(test_df)/len(df):.1%})"
    )

    if len(test_df) == 0:
        raise ValueError("Test split is empty. Adjust TRAIN_FRAC or inspect timestamp distribution.")

    # 3) Fit + detect
    det_cls = GenericTimeSeriesDetector if USE_GENERIC_DETECTOR else EnergyTimeSeriesDetector
    det = det_cls(
        entity_col="building_id",
        time_col="timestamp",
        value_col="meter_reading",
        direction=DIRECTION,
        threshold=THRESHOLD,
        threshold_quantile=THRESHOLD_QUANTILE,
        threshold_cap=THRESHOLD_CAP,
        threshold_end_ratio=THRESHOLD_END_RATIO,
        min_duration=MIN_DURATION,
        gap_tolerance=GAP_TOLERANCE,
    )
    print("detector:", det_cls.__name__)
    generic_buildings = set(MANUAL_GENERIC_BUILDINGS)
    if not USE_GENERIC_DETECTOR:
        print("router auto_route_generic:", AUTO_ROUTE_GENERIC)
        print("router auto_route_top_k:", AUTO_ROUTE_TOP_K)
        print("router manual_generic_buildings:", sorted(MANUAL_GENERIC_BUILDINGS))

    if "anomaly" in train_df.columns:
        fit_df = train_df[train_df["anomaly"] == 0]
        print(f"fit on train normal-only rows: {len(fit_df):,} / {len(train_df):,}")
    else:
        fit_df = train_df
        print(f"fit on all train rows (no anomaly labels present): {len(train_df):,}")

    # Auto-select routed buildings using energy-baseline mismatch on normal rows.
    if not USE_GENERIC_DETECTOR and AUTO_ROUTE_GENERIC and len(fit_df) > 0:
        det.fit(fit_df)
        fit_sorted = fit_df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        fit_norm = normalize_timeseries_df(fit_sorted, "building_id", "timestamp", "meter_reading")
        fit_pred = det.baseline.predict(fit_norm)
        fit_sorted["expected"] = fit_pred.expected.to_numpy()
        fit_sorted["abs_err"] = (fit_sorted["meter_reading"] - fit_sorted["expected"]).abs()
        mismatch = fit_sorted.groupby("building_id")["abs_err"].median().sort_values(ascending=False)
        auto_buildings = {int(x) for x in mismatch.head(AUTO_ROUTE_TOP_K).index.tolist()}
        generic_buildings.update(auto_buildings)
        print("router auto_routed_buildings:", sorted(auto_buildings))

    # Fit final detector(s) on routed subsets.
    if USE_GENERIC_DETECTOR:
        det.fit(fit_df)
    else:
        fit_energy = fit_df[~fit_df["building_id"].isin(generic_buildings)]
        if len(fit_energy) > 0:
            det.fit(fit_energy)
        else:
            det.fit(fit_df)
        print("router generic_buildings(final):", sorted(generic_buildings))

    generic_det = None
    if not USE_GENERIC_DETECTOR and generic_buildings:
        generic_fit = fit_df[fit_df["building_id"].isin(generic_buildings)]
        if len(generic_fit) > 0:
            generic_det = GenericTimeSeriesDetector(
                entity_col="building_id",
                time_col="timestamp",
                value_col="meter_reading",
                direction=DIRECTION,
                threshold=THRESHOLD,
                threshold_quantile=THRESHOLD_QUANTILE,
                threshold_cap=THRESHOLD_CAP,
                threshold_end_ratio=THRESHOLD_END_RATIO,
                min_duration=MIN_DURATION,
                gap_tolerance=GAP_TOLERANCE,
            )
            generic_det.fit(generic_fit)

    if USE_GENERIC_DETECTOR:
        events, scores = det.detect(test_df)
    else:
        if isinstance(det, EnergyTimeSeriesDetector):
            events, scores = detect_with_router(det, generic_det, test_df, generic_buildings)
        else:
            raise TypeError("det must be an instance of EnergyTimeSeriesDetector to use detect_with_router.")
    print("threshold_quantile:", THRESHOLD_QUANTILE)
    print("threshold_cap:", THRESHOLD_CAP)
    print("threshold_end_ratio:", THRESHOLD_END_RATIO)
    print("fallback_threshold:", THRESHOLD)
    if det._threshold_by_entity:
        thr_series = pd.Series(det._threshold_by_entity)
        print("learned thresholds summary:\n", thr_series.describe().to_string())
    else:
        print("learned thresholds: none")

    print("num events:", len(events))
    print("sample event:", events[0] if events else None)

    # 4) Attach scores + pointwise predictions
    # scores is expected to align to test_df rows; if it's a Series, align by index.
    df_eval = test_df.copy()
    if isinstance(scores, pd.Series):
        df_eval["score"] = scores.values
    elif isinstance(scores, pd.DataFrame) and "score" in scores.columns:
        df_eval["score"] = scores["score"].values
    else:
        # last resort: try attribute access
        try:
            df_eval["score"] = scores
        except Exception:
            df_eval["score"] = np.nan

    df_eval["pred"] = label_points_from_events(df_eval, events)

    # 5) Pointwise metrics (test split only)
    if "anomaly" in df_eval.columns:
        tp = int(((df_eval["pred"] == 1) & (df_eval["anomaly"] == 1)).sum())
        fp = int(((df_eval["pred"] == 1) & (df_eval["anomaly"] == 0)).sum())
        fn = int(((df_eval["pred"] == 0) & (df_eval["anomaly"] == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print("tp fp fn:", tp, fp, fn)
        print(f"precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}")
    else:
        print("No 'anomaly' column found — skipping TP/FP/FN metrics.")

    # 6) Event overlap summary (test split only)
    if "anomaly" in df_eval.columns and len(events) > 0:
        evt_df = event_overlap_table(df_eval, events)

        # Bucket events by how “label-aligned” they are
        buckets = pd.cut(
            evt_df["label_frac"],
            bins=[-0.01, 0.3, 0.7, 1.01],
            labels=["weak (<30%)", "mixed (30–70%)", "clean (>70%)"],
        )
        summary = buckets.value_counts(dropna=False)

        print("\nEvent label-overlap buckets (fraction of true anomaly points inside event):")
        for k, v in summary.items():
            print(f"  {k}: {int(v)}")

        print("\nEvent metrics (test split only):")
        print("events per building (median):", evt_df.groupby("entity_id").size().median())
        durations = (evt_df["end"] - evt_df["start"]).dt.total_seconds() / 3600
        print("event duration hours (median):", durations.median())
        print("label_frac median:", evt_df["label_frac"].median())
        print("label_frac > 0.5:", (evt_df["label_frac"] > 0.5).mean())

        evt_path = os.path.join(OUT_DIR, "kaggle_events_with_overlap.csv")
        evt_df.to_csv(evt_path, index=False)
        print("\nwrote:", evt_path)
    else:
        evt_df = None
        print("\nSkipping event-level overlap table (no labels or no events).")

    # 7) Save test scores
    score_path = os.path.join(OUT_DIR, "kaggle_scores.parquet")
    save_cols = ["building_id", "timestamp", "meter_reading", "pred", "score"]
    if "anomaly" in df_eval.columns:
        save_cols.insert(3, "anomaly")
    df_eval[save_cols].to_parquet(score_path, index=False)
    print("wrote:", score_path)


if __name__ == "__main__":
    main()
