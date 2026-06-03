# benchmarks/kaggle_train_eval.py
import io
import os
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from anomlib.core.eventing import events_to_point_labels
from anomlib.core.schema import normalize_timeseries_df
from anomlib.datasets.kaggle_energy.load import load_train
from anomlib.detectors import (
    EnergySupervisedDetector,
    EnergyTimeSeriesDetector,
    GenericTimeSeriesDetector,
)


DATA_PATH = "data/kaggle/train.csv"
OUT_DIR = "out"

# Legacy detector knobs (non-supervised path)
DIRECTION = "both"
THRESHOLD = 3.5  # fallback only if per-entity thresholds missing
THRESHOLD_QUANTILE = 0.9995
THRESHOLD_CAP = 3.5
THRESHOLD_END_RATIO = 0.6
MIN_DURATION = "8h"  # use lowercase to avoid pandas deprecation warnings
GAP_TOLERANCE = "1h"
TRAIN_FRAC = 0.8
USE_GENERIC_DETECTOR = False

# Supervised detector + split knobs
USE_SUPERVISED_DETECTOR = True
SPLIT_MODE = "3way"
TRAIN_FRAC_3WAY = 0.7
VAL_FRAC_3WAY = 0.1
TEST_FRAC_3WAY = 0.2
SUP_CONTINUE_THRESHOLD = 0.85
SUP_MIN_DURATION = "3h"
SUP_GAP_TOLERANCE = "3h"

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
    pred = events_to_point_labels(
        df,
        events,
        entity_col=entity_col,
        time_col=time_col,
        assume_sorted=True,
    )
    return pd.Series(pred.astype(np.int8), index=df.index)


def _build_entity_eval_index(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    label_col: str,
) -> dict[object, dict[str, np.ndarray]]:
    tmp = df[[entity_col, time_col]].copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    if label_col in df.columns:
        tmp[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(np.int8)
    else:
        tmp[label_col] = np.zeros(len(df), dtype=np.int8)
    tmp = tmp.sort_values([entity_col, time_col])

    out: dict[object, dict[str, np.ndarray]] = {}
    for ent, g in tmp.groupby(entity_col, sort=False):
        times_ns = g[time_col].to_numpy(dtype="datetime64[ns]").astype("int64")
        labels = g[label_col].to_numpy(dtype=np.int8)
        anomaly_times_ns = times_ns[labels == 1]
        payload = {
            "times_ns": times_ns,
            "anomaly_times_ns": anomaly_times_ns,
        }
        out[ent] = payload
        out[str(ent)] = payload
    return out


def _entity_keys(entity_id) -> tuple[object, ...]:
    keys: list[object] = [entity_id]
    str_key = str(entity_id)
    if str_key != entity_id:
        keys.append(str_key)
    return tuple(keys)


def _lookup_entity_payload(
    entity_index: dict[object, dict[str, np.ndarray]],
    entity_id,
) -> dict[str, np.ndarray] | None:
    for key in _entity_keys(entity_id):
        payload = entity_index.get(key)
        if payload is not None:
            return payload
    return None


def _count_points_in_interval(times_ns: np.ndarray, start, end) -> int:
    if len(times_ns) == 0:
        return 0
    start_ns = int(pd.Timestamp(start).value)
    end_ns = int(pd.Timestamp(end).value)
    if end_ns < start_ns:
        return 0
    left = int(np.searchsorted(times_ns, start_ns, side="left"))
    right = int(np.searchsorted(times_ns, end_ns, side="right"))
    return max(0, right - left)


def _group_events_by_entity(events) -> dict[object, list]:
    grouped: dict[object, list] = defaultdict(list)
    for event in events:
        for key in _entity_keys(event.entity_id):
            grouped[key].append(event)
    return grouped


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

    entity_index = _build_entity_eval_index(df, entity_col, time_col, label_col)
    true_by_ent: dict[object, list[dict]] = defaultdict(list)
    for te in true_events:
        for key in _entity_keys(te["entity_id"]):
            true_by_ent[key].append(te)
    pred_by_ent = _group_events_by_entity(pred_events)

    detected_true = 0
    for te in true_events:
        ent = te["entity_id"]
        n_true = max(1, int(te["n_true"]))
        ent_index = _lookup_entity_payload(entity_index, ent)
        if ent_index is None:
            continue
        anomaly_times_ns = ent_index["anomaly_times_ns"]
        best = 0.0
        for pe in pred_by_ent.get(ent, pred_by_ent.get(str(ent), [])):
            start = max(pd.Timestamp(te["start"]), pd.Timestamp(pe.start))
            end = min(pd.Timestamp(te["end"]), pd.Timestamp(pe.end))
            overlap = _count_points_in_interval(anomaly_times_ns, start, end) / n_true
            if overlap > best:
                best = overlap
        if best >= tau:
            detected_true += 1

    correct_pred = 0
    if len(pred_events) > 0:
        for pe in pred_events:
            ent = pe.entity_id
            ent_index = _lookup_entity_payload(entity_index, ent)
            if ent_index is None:
                continue
            anomaly_times_ns = ent_index["anomaly_times_ns"]
            best = 0.0
            for te in true_by_ent.get(ent, true_by_ent.get(str(ent), [])):
                n_true = max(1, int(te["n_true"]))
                start = max(pd.Timestamp(te["start"]), pd.Timestamp(pe.start))
                end = min(pd.Timestamp(te["end"]), pd.Timestamp(pe.end))
                overlap = _count_points_in_interval(anomaly_times_ns, start, end) / n_true
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
    columns = [
        "event_idx",
        "entity_id",
        "start",
        "end",
        "n_points",
        "label_frac",
        "direction",
        "severity",
        "score_peak",
        "score_mean",
        "reason",
    ]
    entity_index = _build_entity_eval_index(df, entity_col, time_col, label_col)
    rows = []
    for k, e in enumerate(events):
        ent_index = _lookup_entity_payload(entity_index, e.entity_id)
        if ent_index is None:
            frac = np.nan
            n = 0
        else:
            n = _count_points_in_interval(ent_index["times_ns"], e.start, e.end)
            if n == 0:
                frac = np.nan
            else:
                n_anom = _count_points_in_interval(ent_index["anomaly_times_ns"], e.start, e.end)
                frac = float(n_anom / n) if label_col in df.columns else np.nan

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
    return pd.DataFrame(rows, columns=columns)


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


def run_silently(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def format_timedelta_compact(value) -> str:
    td = pd.Timedelta(value)
    total_seconds = int(td.total_seconds())
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


def format_metric(value: float, decimals: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.{decimals}f}"


def summarize_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }


def summarize_detector(det, use_supervised: bool) -> dict:
    summary = {"name": type(det).__name__}
    if use_supervised:
        summary.update(
            {
                "start_threshold": float(getattr(det, "_start_threshold", np.nan)),
                "continue_threshold": float(getattr(det, "_continue_threshold", np.nan)),
                "min_duration": format_timedelta_compact(det.min_duration),
                "gap_tolerance": format_timedelta_compact(det.gap_tolerance),
            }
        )
    else:
        summary.update(
            {
                "threshold": float(getattr(det, "threshold", np.nan)),
                "threshold_end_ratio": float(getattr(det, "threshold_end_ratio", np.nan)),
                "min_duration": format_timedelta_compact(det.min_duration),
                "gap_tolerance": format_timedelta_compact(det.gap_tolerance),
            }
        )
    return summary


def compute_point_metrics(df_eval: pd.DataFrame) -> dict:
    if "anomaly" not in df_eval.columns or "pred" not in df_eval.columns:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan}

    y_true = pd.to_numeric(df_eval["anomaly"], errors="coerce").fillna(0).astype(int)
    pred = pd.to_numeric(df_eval["pred"], errors="coerce").fillna(0).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_event_metrics(df_eval: pd.DataFrame, events) -> dict:
    if "anomaly" not in df_eval.columns:
        return {
            "event_precision_01": np.nan,
            "event_recall_01": np.nan,
            "event_precision_05": np.nan,
            "event_recall_05": np.nan,
        }

    true_events = build_true_events(df_eval)
    ep10, er10 = event_overlap_precision_recall(df_eval, events, true_events, tau=0.1)
    ep50, er50 = event_overlap_precision_recall(df_eval, events, true_events, tau=0.5)
    return {
        "event_precision_01": ep10,
        "event_recall_01": er10,
        "event_precision_05": ep50,
        "event_recall_05": er50,
    }


def compute_ops_summary(evt_df: pd.DataFrame, events) -> dict:
    num_events = int(len(events))

    if num_events > 0 and "entity_id" in evt_df.columns:
        median_events_per_building = float(evt_df.groupby("entity_id").size().median())
    else:
        median_events_per_building = 0.0

    if num_events > 0 and {"start", "end"}.issubset(evt_df.columns):
        durations = (evt_df["end"] - evt_df["start"]).dt.total_seconds() / 3600.0
        median_event_duration_h = float(durations.median()) if len(durations) else 0.0
    else:
        median_event_duration_h = 0.0

    if "label_frac" in evt_df.columns and len(evt_df) > 0:
        median_label_frac = float(evt_df["label_frac"].median())
    else:
        median_label_frac = np.nan

    return {
        "num_events": num_events,
        "median_events_per_building": median_events_per_building,
        "median_event_duration_h": median_event_duration_h,
        "median_label_frac": median_label_frac,
    }


def print_report(
    split_summary: dict,
    detector_summary: dict,
    point_metrics: dict,
    event_metrics: dict,
    ops_summary: dict,
) -> None:
    print("Split")
    print(f"train={split_summary['train_rows']}")
    if split_summary["val_rows"] > 0:
        print(f"val={split_summary['val_rows']}")
    print(f"test={split_summary['test_rows']}")
    print()

    print("Detector")
    print(detector_summary["name"])
    if "start_threshold" in detector_summary:
        print(
            "start="
            f"{format_metric(detector_summary['start_threshold'], 2)} "
            "continue="
            f"{format_metric(detector_summary['continue_threshold'], 2)} "
            f"min_duration={detector_summary['min_duration']} "
            f"gap_tolerance={detector_summary['gap_tolerance']}"
        )
    else:
        print(
            "threshold="
            f"{detector_summary['threshold']:g} "
            "threshold_end_ratio="
            f"{detector_summary['threshold_end_ratio']:g} "
            f"min_duration={detector_summary['min_duration']} "
            f"gap_tolerance={detector_summary['gap_tolerance']}"
        )
    print()

    print("Pointwise")
    print(
        "precision="
        f"{format_metric(point_metrics['precision'])} "
        "recall="
        f"{format_metric(point_metrics['recall'])} "
        "f1="
        f"{format_metric(point_metrics['f1'])}"
    )
    print()

    print("Event-level")
    print(
        "P@0.1="
        f"{format_metric(event_metrics['event_precision_01'])} "
        "R@0.1="
        f"{format_metric(event_metrics['event_recall_01'])}"
    )
    print(
        "P@0.5="
        f"{format_metric(event_metrics['event_precision_05'])} "
        "R@0.5="
        f"{format_metric(event_metrics['event_recall_05'])}"
    )
    print()

    print("Operations")
    print(f"num_events={ops_summary['num_events']}")
    print(
        "median_events_per_building="
        f"{format_metric(ops_summary['median_events_per_building'], 1)}"
    )
    print(
        "median_event_duration_h="
        f"{format_metric(ops_summary['median_event_duration_h'], 1)}"
    )
    print(f"median_label_frac={format_metric(ops_summary['median_label_frac'])}")


def load_dataset() -> pd.DataFrame:
    df = load_train(DATA_PATH).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna(subset=["meter_reading"]).reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if USE_SUPERVISED_DETECTOR and SPLIT_MODE == "3way":
        return per_entity_time_split_3way(
            df,
            entity_col="building_id",
            time_col="timestamp",
            train_frac=TRAIN_FRAC_3WAY,
            val_frac=VAL_FRAC_3WAY,
            test_frac=TEST_FRAC_3WAY,
        )

    train_df, test_df = per_entity_time_split(
        df,
        entity_col="building_id",
        time_col="timestamp",
        frac=TRAIN_FRAC,
    )
    val_df = train_df.iloc[0:0].copy()
    return train_df, val_df, test_df


def build_supervised_detector() -> EnergySupervisedDetector:
    return EnergySupervisedDetector(
        entity_col="building_id",
        time_col="timestamp",
        value_col="meter_reading",
        direction="high",
        threshold=0.9,
        threshold_strategy="evented_f1_grid",
        continue_threshold=SUP_CONTINUE_THRESHOLD,
        min_duration=SUP_MIN_DURATION,
        gap_tolerance=SUP_GAP_TOLERANCE,
        explain_events=False,
    )


def fit_detect_supervised(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[EnergySupervisedDetector, list, pd.DataFrame]:
    if "anomaly" not in train_df.columns:
        raise ValueError("USE_SUPERVISED_DETECTOR=True requires an 'anomaly' label column.")

    det = build_supervised_detector()
    run_silently(det.fit, train_df, val_df=val_df if len(val_df) > 0 else None)
    events, scores = run_silently(det.detect, test_df)
    return det, events, scores


def build_unsupervised_detector(detector_cls):
    return detector_cls(
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


def fit_detect_unsupervised(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[EnergyTimeSeriesDetector | GenericTimeSeriesDetector, list, pd.DataFrame]:
    det_cls = GenericTimeSeriesDetector if USE_GENERIC_DETECTOR else EnergyTimeSeriesDetector
    det = build_unsupervised_detector(det_cls)
    generic_buildings = set(MANUAL_GENERIC_BUILDINGS)

    if "anomaly" in train_df.columns:
        fit_df = train_df[train_df["anomaly"] == 0]
    else:
        fit_df = train_df

    if not USE_GENERIC_DETECTOR and AUTO_ROUTE_GENERIC and len(fit_df) > 0:
        run_silently(det.fit, fit_df)
        fit_sorted = fit_df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        fit_norm = normalize_timeseries_df(fit_sorted, "building_id", "timestamp", "meter_reading")
        fit_pred = det.baseline.predict(fit_norm)
        fit_sorted["expected"] = fit_pred.expected.to_numpy()
        fit_sorted["abs_err"] = (fit_sorted["meter_reading"] - fit_sorted["expected"]).abs()
        mismatch = fit_sorted.groupby("building_id")["abs_err"].median().sort_values(ascending=False)
        auto_buildings = {int(x) for x in mismatch.head(AUTO_ROUTE_TOP_K).index.tolist()}
        generic_buildings.update(auto_buildings)

    if USE_GENERIC_DETECTOR:
        run_silently(det.fit, fit_df)
    else:
        fit_energy = fit_df[~fit_df["building_id"].isin(generic_buildings)]
        run_silently(det.fit, fit_energy if len(fit_energy) > 0 else fit_df)

    generic_det = None
    if not USE_GENERIC_DETECTOR and generic_buildings:
        generic_fit = fit_df[fit_df["building_id"].isin(generic_buildings)]
        if len(generic_fit) > 0:
            generic_det = build_unsupervised_detector(GenericTimeSeriesDetector)
            run_silently(generic_det.fit, generic_fit)

    if USE_GENERIC_DETECTOR:
        events, scores = run_silently(det.detect, test_df)
    else:
        if not isinstance(det, EnergyTimeSeriesDetector):
            raise TypeError("det must be an instance of EnergyTimeSeriesDetector to use detect_with_router.")
        events, scores = run_silently(detect_with_router, det, generic_det, test_df, generic_buildings)

    return det, events, scores


def fit_and_detect(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    if USE_SUPERVISED_DETECTOR:
        return fit_detect_supervised(train_df, val_df, test_df)
    return fit_detect_unsupervised(train_df, test_df)


def build_eval_frame(test_df: pd.DataFrame, scores, events) -> pd.DataFrame:
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
    return df_eval


def write_artifacts(df_eval: pd.DataFrame, evt_df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    evt_path = os.path.join(OUT_DIR, "kaggle_events_with_overlap.csv")
    evt_df.to_csv(evt_path, index=False)

    score_path = os.path.join(OUT_DIR, "kaggle_scores.parquet")
    save_cols = ["building_id", "timestamp", "meter_reading", "pred", "score"]
    if "anomaly" in df_eval.columns:
        save_cols.insert(3, "anomaly")
    df_eval[save_cols].to_parquet(score_path, index=False)


def main():
    df = load_dataset()
    train_df, val_df, test_df = split_dataset(df)
    if len(test_df) == 0:
        raise ValueError("Test split is empty. Adjust split fractions or inspect timestamp distribution.")

    det, events, scores = fit_and_detect(train_df, val_df, test_df)
    df_eval = build_eval_frame(test_df, scores, events)
    evt_df = event_overlap_table(df_eval, events)

    split_summary = summarize_split(train_df, val_df, test_df)
    detector_summary = summarize_detector(det, use_supervised=USE_SUPERVISED_DETECTOR)
    point_metrics = compute_point_metrics(df_eval)
    event_metrics = compute_event_metrics(df_eval, events)
    ops_summary = compute_ops_summary(evt_df, events)

    write_artifacts(df_eval, evt_df)
    print_report(split_summary, detector_summary, point_metrics, event_metrics, ops_summary)


if __name__ == "__main__":
    main()
