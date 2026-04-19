# benchmarks/kaggle_train_eval.py
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from anomlib.datasets.kaggle_energy.load import load_train
from anomlib.detectors import (
    EnergySupervisedDetector,
    EnergyTimeSeriesDetector,
    GenericTimeSeriesDetector,
)
from anomlib.core.eventing import events_to_point_labels
from anomlib.core.schema import normalize_timeseries_df


DATA_PATH = "data/kaggle/train.csv"
OUT_DIR = "out"

# Legacy detector knobs (non-supervised path)
DIRECTION = "both"
THRESHOLD = 3.5  # fallback only if per-entity thresholds missing
THRESHOLD_QUANTILE = 0.9995
THRESHOLD_CAP = 3.5
THRESHOLD_END_RATIO = 0.6
MIN_DURATION = "8h"      # use lowercase to avoid pandas deprecation warnings
GAP_TOLERANCE = "1h"
TRAIN_FRAC = 0.8
USE_GENERIC_DETECTOR = False

# Supervised detector + split knobs
USE_SUPERVISED_DETECTOR = True
SPLIT_MODE = "3way"
TRAIN_FRAC_3WAY = 0.7
VAL_FRAC_3WAY = 0.1
TEST_FRAC_3WAY = 0.2
SUP_THRESHOLD_STRATEGY = "evented_f1_grid"  # "evented_f1_grid", "recall_target", or "f1"
SUP_RECALL_TARGET = 0.65
SUP_USE_HYSTERESIS = True
SUP_CONTINUE_THRESHOLD = 0.85
SUP_MIN_DURATION = "3h"
SUP_GAP_TOLERANCE = "3h"
SUP_THRESHOLD_END_RATIO = 0.75

# Optional hybrid routing when USE_GENERIC_DETECTOR is False:
# auto-route worst baseline-mismatch buildings to GenericTimeSeriesDetector.
AUTO_ROUTE_GENERIC = False
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


def per_entity_time_split_3way(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for _, g in df.groupby(entity_col, sort=False):
        g = g.sort_values(time_col)
        n = len(g)
        k_train = int(n * train_frac)
        k_val = int(n * val_frac)
        a = g.iloc[:k_train]
        b = g.iloc[k_train:k_train + k_val]
        c = g.iloc[k_train + k_val:]
        parts.append((a, b, c))

    train_df = pd.concat([a for a, _, _ in parts]).sort_values([entity_col, time_col])
    val_df = pd.concat([b for _, b, _ in parts]).sort_values([entity_col, time_col])
    test_df = pd.concat([c for _, _, c in parts]).sort_values([entity_col, time_col])
    return train_df, val_df, test_df


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
    pred = events_to_point_labels(df, events, entity_col=entity_col, time_col=time_col)
    return pd.Series(pred.astype(np.int8), index=df.index)


def build_true_events(
    df: pd.DataFrame,
    entity_col: str = "building_id",
    time_col: str = "timestamp",
    label_col: str = "anomaly",
) -> list[dict]:
    out: list[dict] = []
    for ent, g in df.groupby(entity_col, sort=False):
        gg = g.sort_values(time_col)
        times = pd.to_datetime(gg[time_col]).to_numpy()
        labels = pd.to_numeric(gg[label_col], errors="coerce").fillna(0).astype(int).to_numpy()

        start_i = None
        for i, y in enumerate(labels):
            if y == 1 and start_i is None:
                start_i = i
            elif y == 0 and start_i is not None:
                end_i = i - 1
                out.append(
                    {
                        "entity_id": ent,
                        "start": pd.to_datetime(times[start_i]),
                        "end": pd.to_datetime(times[end_i]),
                        "n_true": int(end_i - start_i + 1),
                    }
                )
                start_i = None

        if start_i is not None:
            end_i = len(labels) - 1
            out.append(
                {
                    "entity_id": ent,
                    "start": pd.to_datetime(times[start_i]),
                    "end": pd.to_datetime(times[end_i]),
                    "n_true": int(end_i - start_i + 1),
                }
            )

    return out


def event_overlap_precision_recall(
    df: pd.DataFrame,
    pred_events,
    true_events: list[dict],
    tau: float,
    entity_col: str = "building_id",
    time_col: str = "timestamp",
    label_col: str = "anomaly",
) -> tuple[float, float]:
    if len(true_events) == 0:
        return 0.0, 0.0

    true_by_ent: dict[object, list[dict]] = defaultdict(list)
    for te in true_events:
        true_by_ent[te["entity_id"]].append(te)

    detected_true = 0
    for te in true_events:
        ent = te["entity_id"]
        n_true = max(1, int(te["n_true"]))
        best = 0.0
        for pe in pred_events:
            if pe.entity_id != ent:
                continue
            m = (
                (df[entity_col] == ent)
                & (pd.to_datetime(df[time_col]) >= te["start"])
                & (pd.to_datetime(df[time_col]) <= te["end"])
                & (pd.to_datetime(df[time_col]) >= pd.to_datetime(pe.start))
                & (pd.to_datetime(df[time_col]) <= pd.to_datetime(pe.end))
                & (pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int) == 1)
            )
            overlap = int(m.sum()) / n_true
            if overlap > best:
                best = overlap
        if best >= tau:
            detected_true += 1

    correct_pred = 0
    if len(pred_events) > 0:
        for pe in pred_events:
            ent = pe.entity_id
            best = 0.0
            for te in true_by_ent.get(ent, []):
                n_true = max(1, int(te["n_true"]))
                m = (
                    (df[entity_col] == ent)
                    & (pd.to_datetime(df[time_col]) >= te["start"])
                    & (pd.to_datetime(df[time_col]) <= te["end"])
                    & (pd.to_datetime(df[time_col]) >= pd.to_datetime(pe.start))
                    & (pd.to_datetime(df[time_col]) <= pd.to_datetime(pe.end))
                    & (pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int) == 1)
                )
                overlap = int(m.sum()) / n_true
                if overlap > best:
                    best = overlap
            if best >= tau:
                correct_pred += 1

    event_recall = detected_true / len(true_events) if len(true_events) else 0.0
    event_precision = correct_pred / len(pred_events) if len(pred_events) else 0.0
    return event_precision, event_recall


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

    # 2) Split
    if USE_SUPERVISED_DETECTOR and SPLIT_MODE == "3way":
        train_df, val_df, test_df = per_entity_time_split_3way(
            df,
            entity_col="building_id",
            time_col="timestamp",
            train_frac=TRAIN_FRAC_3WAY,
            val_frac=VAL_FRAC_3WAY,
            test_frac=TEST_FRAC_3WAY,
        )
        print(
            f"per-entity 3-way split  "
            f"train rows: {len(train_df):,} ({len(train_df)/len(df):.1%})  "
            f"val rows: {len(val_df):,} ({len(val_df)/len(df):.1%})  "
            f"test rows: {len(test_df):,} ({len(test_df)/len(df):.1%})"
        )
    else:
        train_df, test_df = per_entity_time_split(
            df, entity_col="building_id", time_col="timestamp", frac=TRAIN_FRAC
        )
        val_df = train_df.iloc[0:0].copy()
        print(
            f"per-entity split  "
            f"train rows: {len(train_df):,} ({len(train_df)/len(df):.1%})  "
            f"test rows: {len(test_df):,} ({len(test_df)/len(df):.1%})"
        )

    if len(test_df) == 0:
        raise ValueError("Test split is empty. Adjust split fractions or inspect timestamp distribution.")

    # 3) Fit + detect
    if USE_SUPERVISED_DETECTOR:
        det = EnergySupervisedDetector(
            entity_col="building_id",
            time_col="timestamp",
            value_col="meter_reading",
            direction="high",
            threshold=0.9,
            threshold_strategy=SUP_THRESHOLD_STRATEGY,
            recall_target=SUP_RECALL_TARGET,
            use_hysteresis=SUP_USE_HYSTERESIS,
            continue_threshold=SUP_CONTINUE_THRESHOLD,
            threshold_end_ratio=SUP_THRESHOLD_END_RATIO,
            min_duration=SUP_MIN_DURATION,
            gap_tolerance=SUP_GAP_TOLERANCE,
        )
        print("detector:", EnergySupervisedDetector.__name__)
        if "anomaly" not in train_df.columns:
            raise ValueError("USE_SUPERVISED_DETECTOR=True requires an 'anomaly' label column.")
        print(f"fit on train rows with labels: {len(train_df):,}")
        print(f"calibrate on val rows: {len(val_df):,}")
        det.fit(train_df, val_df=val_df if len(val_df) > 0 else None)
        events, scores = det.detect(test_df)
        print("threshold_strategy:", SUP_THRESHOLD_STRATEGY)
        print("recall_target:", SUP_RECALL_TARGET)
        print("calibrated_threshold:", getattr(det, "_threshold", np.nan))
        if SUP_USE_HYSTERESIS:
            print(
                "hysteresis thresholds:",
                getattr(det, "_start_threshold", np.nan),
                getattr(det, "_continue_threshold", np.nan),
            )
    else:
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

    # 4) Attach scores + pointwise predictions (TEST only)
    df_eval = test_df.copy()
    if isinstance(scores, pd.Series):
        df_eval["score"] = scores.values
    elif isinstance(scores, pd.DataFrame) and "score" in scores.columns:
        df_eval["score"] = scores["score"].values
    else:
        try:
            df_eval["score"] = scores
        except Exception:
            df_eval["score"] = np.nan

    df_eval["pred"] = label_points_from_events(df_eval, events)

    # 5) Pointwise metrics (TEST only)
    if "anomaly" in df_eval.columns:
        tp = int(((df_eval["pred"] == 1) & (df_eval["anomaly"] == 1)).sum())
        fp = int(((df_eval["pred"] == 1) & (df_eval["anomaly"] == 0)).sum())
        fn = int(((df_eval["pred"] == 0) & (df_eval["anomaly"] == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print("\nPointwise metrics (test):")
        print("tp fp fn:", tp, fp, fn)
        print(f"precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}")

        true_events = build_true_events(df_eval)
        ep10, er10 = event_overlap_precision_recall(df_eval, events, true_events, tau=0.1)
        ep50, er50 = event_overlap_precision_recall(df_eval, events, true_events, tau=0.5)
        print("\nEvent-level metrics (test):")
        print(f"Event Precision@0.1: {ep10:.3f}  Event Recall@0.1: {er10:.3f}")
        print(f"Event Precision@0.5: {ep50:.3f}  Event Recall@0.5: {er50:.3f}")
    else:
        print("No 'anomaly' column found — skipping TP/FP/FN and event metrics.")

    # 6) Event overlap summary (TEST only)
    if "anomaly" in df_eval.columns and len(events) > 0:
        evt_df = event_overlap_table(df_eval, events)

        buckets = pd.cut(
            evt_df["label_frac"],
            bins=[-0.01, 0.3, 0.7, 1.01],
            labels=["weak (<30%)", "mixed (30–70%)", "clean (>70%)"],
        )
        summary = buckets.value_counts(dropna=False)

        print("\nEvent label-overlap buckets (fraction of true anomaly points inside event):")
        for k, v in summary.items():
            print(f"  {k}: {int(v)}")

        print("\nEvent summary stats (test):")
        print("events per building (median):", evt_df.groupby("entity_id").size().median())
        durations = (evt_df["end"] - evt_df["start"]).dt.total_seconds() / 3600
        print("event duration hours (median):", durations.median())
        print("label_frac median:", evt_df["label_frac"].median())
        print("label_frac > 0.5:", (evt_df["label_frac"] > 0.5).mean())

        # evt_path = os.path.join(OUT_DIR, "kaggle_events_with_overlap.csv")
        # evt_df.to_csv(evt_path, index=False)
        # print("\nwrote:", evt_path)
    else:
        print("\nSkipping event-level overlap table (no labels or no events).")

    # # 7) Save test scores
    # score_path = os.path.join(OUT_DIR, "kaggle_scores.parquet")
    # save_cols = ["building_id", "timestamp", "meter_reading", "pred", "score"]
    # if "anomaly" in df_eval.columns:
    #     save_cols.insert(3, "anomaly")
    # df_eval[save_cols].to_parquet(score_path, index=False)
    # print("wrote:", score_path)


if __name__ == "__main__":
    main()
