from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

_ORIG_STDERR_FD = os.dup(2)
_DEVNULL_FD = os.open(os.devnull, os.O_WRONLY)
os.dup2(_DEVNULL_FD, 2)

import pandas as pd

from anomlib.datasets import infer_timeseries_cadence, load_cornell_emcs_csv
from anomlib.detectors import CornellEMCSElectricityDetector

DEFAULT_CSV_PATH = "/Users/rohanalapati/anomlib/data/emcs/Electric-data-2026-06-01 21_13_10.csv"


def rolling_score(
    df: pd.DataFrame,
    *,
    cadence: str,
    initial_train_frac: float,
) -> tuple[CornellEMCSElectricityDetector, pd.DataFrame]:
    scored_parts: list[pd.DataFrame] = []
    last_detector: CornellEMCSElectricityDetector | None = None

    for _, group in df.groupby("entity_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            continue

        initial_rows = max(1, int(round(len(group) * initial_train_frac)))
        initial_rows = min(initial_rows, len(group) - 1)
        detector = CornellEMCSElectricityDetector(
            cadence=cadence,
            threshold_quantile=None,
        )

        entity_scores: list[pd.DataFrame] = []
        for i in range(initial_rows, len(group)):
            detector.fit(group.iloc[:i].copy())
            scored_row = detector.score(group.iloc[i : i + 1].copy())
            entity_scores.append(scored_row)

        if entity_scores:
            scored_parts.append(pd.concat(entity_scores, ignore_index=True))
        last_detector = detector

    if last_detector is None:
        raise ValueError("Rolling mode requires at least two valid rows per entity.")

    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else df.iloc[0:0].copy()
    return last_detector, scored


def restore_stderr() -> None:
    os.dup2(_ORIG_STDERR_FD, 2)


def run_emcs_demo(
    csv_path: str,
    *,
    output_dir: str = "out",
    mode: str = "rolling",
    initial_train_frac: float = 0.6,
) -> dict[str, object]:
    normalized = load_cornell_emcs_csv(csv_path)
    cadence = infer_timeseries_cadence(normalized)

    if mode == "all":
        detector = CornellEMCSElectricityDetector(
            cadence=cadence,
            threshold_quantile=None,
        )
        fit_df = normalized
        detector.fit(fit_df)
        scored = detector.score(normalized)
    else:
        fit_df = normalized.groupby("entity_id", sort=False).apply(
            lambda g: g.iloc[: max(1, min(len(g) - 1, int(round(len(g) * initial_train_frac))))],
            include_groups=False,
        ).reset_index(drop=True)
        detector, scored = rolling_score(
            normalized,
            cadence=cadence,
            initial_train_frac=initial_train_frac,
        )

    events = detector._build_events(scored)
    events_df = detector.events_to_frame(events).sort_values(
        ["max_abs_score", "start_time"],
        ascending=[False, True],
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir_path / f"cornell_emcs_scored_points_{mode}.csv"
    events_path = output_dir_path / f"cornell_emcs_events_{mode}.csv"
    scored.to_csv(scored_path, index=False)
    events_df.to_csv(events_path, index=False)

    return {
        "mode": mode,
        "cadence": cadence,
        "normalized": normalized,
        "fit_df": fit_df,
        "scored": scored,
        "events": events,
        "events_df": events_df,
        "scored_path": scored_path,
        "events_path": events_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cornell EMCS electricity anomaly detection.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=f"Path to a Cornell EMCS CSV export. Defaults to {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Directory for scored points and event outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "rolling"),
        default="rolling",
        help="Use 'all' to fit+score the full file or 'rolling' for history-only walk-forward scoring.",
    )
    parser.add_argument(
        "--initial-train-frac",
        type=float,
        default=0.6,
        help="Initial history fraction to use before rolling one-step-ahead scoring begins.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-entity counts and top events in addition to the summary.",
    )
    args = parser.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        result = run_emcs_demo(
            args.csv_path,
            output_dir=args.output_dir,
            mode=args.mode,
            initial_train_frac=args.initial_train_frac,
        )
    restore_stderr()
    normalized = result["normalized"]
    fit_df = result["fit_df"]
    scored = result["scored"]
    events_df = result["events_df"]
    scored_path = result["scored_path"]
    events_path = result["events_path"]
    fit_start = fit_df["timestamp"].min()
    fit_end = fit_df["timestamp"].max()
    score_start = scored["timestamp"].min()
    score_end = scored["timestamp"].max()

    print(
        f"mode={args.mode} cadence={result['cadence']} "
        f"entities={normalized['entity_id'].nunique()} "
        f"fit_rows={len(fit_df)} scored_rows={len(scored)} events={len(events_df)}"
    )
    print(f"fit_window={fit_start} -> {fit_end}")
    print(f"score_window={score_start} -> {score_end}")
    print(f"saved={scored_path} | {events_path}")

    if args.verbose and len(events_df) > 0:
        print("\nevent count per entity:")
        print(events_df.groupby("entity_id").size().sort_values(ascending=False).to_string())
        print("\ntop 10 events by severity:")
        print(events_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
