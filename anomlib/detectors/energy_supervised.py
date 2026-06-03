from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomlib.core.eventing import (
    events_to_point_labels,
    scores_to_events_hysteresis,
)
from anomlib.core.schema import normalize_timeseries_df
from anomlib.core.types import Event


def _to_timedelta(x) -> pd.Timedelta:
    if isinstance(x, pd.Timedelta):
        return x
    if isinstance(x, str):
        return pd.Timedelta(x.lower())
    return pd.Timedelta(x)


@dataclass
class _ConstantProbModel:
    prob: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = float(np.clip(self.prob, 0.0, 1.0))
        out = np.empty((len(x), 2), dtype=float)
        out[:, 0] = 1.0 - p
        out[:, 1] = p
        return out


class EnergySupervisedDetector:
    def __init__(
        self,
        entity_col: str = "building_id",
        time_col: str = "timestamp",
        value_col: str = "meter_reading",
        direction: str = "both",
        threshold: float = 0.9,
        threshold_strategy: str = "evented_f1_grid",
        start_threshold: float | None = None,
        continue_threshold: float = 0.85,
        min_duration: str | pd.Timedelta = "3h",
        gap_tolerance: str | pd.Timedelta = "3h",
        explain_events: bool = True,
        feature_windows: tuple[int, ...] = (24, 168),
        lags: tuple[int, ...] = (1, 24, 168),
        model_type: str = "lightgbm",
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.value_col = value_col

        self.direction = direction
        self.threshold = float(threshold)
        self.threshold_strategy = str(threshold_strategy)
        self.use_hysteresis = True
        self.start_threshold = float(start_threshold) if start_threshold is not None else None
        self.continue_threshold = float(continue_threshold)
        self.min_duration = _to_timedelta(min_duration)
        self.gap_tolerance = _to_timedelta(gap_tolerance)
        self.explain_events = bool(explain_events)
        self.feature_windows = tuple(int(w) for w in feature_windows)
        self.lags = tuple(int(k) for k in lags)
        self.model_type = str(model_type).lower()

        self.label_col = "anomaly"

        self.model = None
        self._threshold = float(threshold)
        self._start_threshold = float(threshold)
        self._continue_threshold = float(min(self.continue_threshold, self._start_threshold))
        self._feature_cols: list[str] = []
        self._impute_values: dict[str, float] = {}
        self._feat_median: dict[str, float] = {}
        self._feat_iqr: dict[str, float] = {}
        self._seasonal_profile = pd.DataFrame(
            columns=["entity_id", "dow", "hour", "seasonal_expected"]
        )
        self._seasonal_profile_map = pd.Series(dtype=float)
        self._entity_median = pd.Series(dtype=float)
        self._global_median = 0.0

    def fit(self, df_train: pd.DataFrame, val_df: pd.DataFrame | None = None):
        d_train, y_train = self._normalize_with_labels(df_train)
        if len(d_train) == 0:
            raise ValueError("Training dataframe has no valid rows after normalization.")

        if val_df is None:
            val_mask = self._entity_tail_mask(d_train, frac=0.15)
            train_inner = d_train.loc[~val_mask].reset_index(drop=True)
            y_inner = y_train.loc[~val_mask].reset_index(drop=True)
            d_val = d_train.loc[val_mask].reset_index(drop=True)
            y_val = y_train.loc[val_mask].reset_index(drop=True)
            if len(train_inner) == 0:
                train_inner = d_train.copy()
                y_inner = y_train.copy()
                d_val = d_train.iloc[0:0].copy()
                y_val = y_train.iloc[0:0].copy()
        else:
            train_inner = d_train
            y_inner = y_train
            d_val, y_val = self._normalize_with_labels(val_df)

        self._fit_profiles(train_inner)
        x_inner = self._make_features(train_inner, fit_mode=True)
        calib_model = self._fit_model(x_inner, y_inner)

        if len(d_val) > 0 and y_val.nunique() > 0:
            x_val = self._make_features(d_val, fit_mode=False)
            y_val_arr = y_val.to_numpy()
            val_prob = self._predict_prob(calib_model, x_val)
            self._tune_event_params_on_val(d_val, val_prob, y_val_arr)
        else:
            self._threshold = self.threshold
            self._start_threshold = (
                float(self.start_threshold) if self.start_threshold is not None else float(self._threshold)
            )
            self._continue_threshold = float(min(self.continue_threshold, self._start_threshold))

        # Final model for inference (fit on train only for leakage safety).
        self._fit_profiles(d_train)
        x_train = self._make_features(d_train, fit_mode=True)
        self.model = self._fit_model(x_train, y_train)
        return self

    def detect(self, df_test: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Call fit() before detect().")

        d = normalize_timeseries_df(df_test, self.entity_col, self.time_col, self.value_col)
        x = self._make_features(d, fit_mode=False)
        prob = self._predict_prob(self.model, x)

        d["score"] = prob
        events = self._events_from_scores(d, prob, float(self._threshold))
        if self.explain_events and len(events) > 0:
            events = self._annotate_events(events, d, x)
        return events, d[["entity_id", "timestamp", "value", "score"]]

    def _events_from_scores(self, d: pd.DataFrame, prob: np.ndarray, threshold: float):
        tmp = d[["entity_id", "timestamp"]].copy()
        tmp["score"] = prob
        return scores_to_events_hysteresis(
            df=tmp,
            score_col="score",
            entity_col="entity_id",
            time_col="timestamp",
            direction="high",
            start_threshold=float(self._start_threshold),
            continue_threshold=float(self._continue_threshold),
            min_duration=self.min_duration,
            gap_tolerance=self.gap_tolerance,
            assume_sorted=True,
        )

    def _tune_event_params_on_val(
        self,
        d_val: pd.DataFrame,
        val_prob: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        val_base = d_val[["entity_id", "timestamp"]].copy()
        val_base["score"] = val_prob
        val_points = d_val[["entity_id", "timestamp"]]

        starts = [0.60, 0.70, 0.80, 0.90]
        deltas = [0.10, 0.15, 0.20]
        min_ds = ["2h", "3h", "6h"]
        gaps = ["1h", "3h"]
        best: dict[str, float | int | str] | None = None

        for start in starts:
            for delta in deltas:
                cont = max(0.30, start - delta)
                for min_d in min_ds:
                    for gap in gaps:
                        events = scores_to_events_hysteresis(
                            df=val_base,
                            score_col="score",
                            entity_col="entity_id",
                            time_col="timestamp",
                            direction="high",
                            start_threshold=float(start),
                            continue_threshold=float(cont),
                            min_duration=_to_timedelta(min_d),
                            gap_tolerance=_to_timedelta(gap),
                            assume_sorted=True,
                        )
                        pred = events_to_point_labels(
                            val_points,
                            events,
                            entity_col="entity_id",
                            time_col="timestamp",
                            assume_sorted=True,
                        ).astype(int)
                        m = self._point_metrics(pred, y_val)
                        candidate = {
                            "start": float(start),
                            "continue": float(cont),
                            "min_duration": str(min_d),
                            "gap_tolerance": str(gap),
                            "precision": float(m["precision"]),
                            "recall": float(m["recall"]),
                            "f1": float(m["f1"]),
                            "n_events": int(len(events)),
                        }
                        if self._is_better_event_candidate(candidate, best):
                            best = candidate

        if best is None:
            best = {
                "start": float(self.threshold),
                "continue": float(min(self.continue_threshold, self.threshold)),
                "min_duration": str(self.min_duration),
                "gap_tolerance": str(self.gap_tolerance),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "n_events": 0,
            }

        self._start_threshold = float(best["start"])
        self._continue_threshold = float(best["continue"])
        self.min_duration = _to_timedelta(str(best["min_duration"]))
        self.gap_tolerance = _to_timedelta(str(best["gap_tolerance"]))
        self._threshold = self._start_threshold

        print(
            "val grid best: "
            f"start={self._start_threshold:.2f} continue={self._continue_threshold:.2f} "
            f"min_d={self.min_duration} gap={self.gap_tolerance} "
            f"p={float(best['precision']):.3f} r={float(best['recall']):.3f} "
            f"f1={float(best['f1']):.3f} events={int(best['n_events'])}"
        )

    def _point_metrics(self, pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def _is_better_event_candidate(
        self,
        candidate: dict[str, float | int | str],
        incumbent: dict[str, float | int | str] | None,
    ) -> bool:
        if incumbent is None:
            return True

        cand_eligible = float(candidate["precision"]) >= 0.50
        inc_eligible = float(incumbent["precision"]) >= 0.50
        if cand_eligible != inc_eligible:
            return cand_eligible

        cand_key = (
            float(candidate["f1"]),
            float(candidate["precision"]),
            -int(candidate["n_events"]),
            float(candidate["start"]),
        )
        inc_key = (
            float(incumbent["f1"]),
            float(incumbent["precision"]),
            -int(incumbent["n_events"]),
            float(incumbent["start"]),
        )
        return cand_key > inc_key

    def _normalize_with_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.label_col not in df.columns:
            raise ValueError(f"Supervised detector requires '{self.label_col}' in fit dataframe.")

        d = normalize_timeseries_df(df, self.entity_col, self.time_col, self.value_col)
        labels = (
            df[[self.entity_col, self.time_col, self.label_col]]
            .rename(columns={self.entity_col: "entity_id", self.time_col: "timestamp"})
            .copy()
        )
        labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")
        labels[self.label_col] = pd.to_numeric(labels[self.label_col], errors="coerce")
        labels = labels.dropna(subset=["entity_id", "timestamp", self.label_col])

        merged = d.merge(labels, on=["entity_id", "timestamp"], how="left")
        merged[self.label_col] = merged[self.label_col].fillna(0).astype(int).clip(0, 1)
        return merged[["entity_id", "timestamp", "value"]], merged[self.label_col]

    def _expected_for_rows(self, d: pd.DataFrame) -> pd.Series:
        keys = pd.MultiIndex.from_arrays(
            [
                d["entity_id"].to_numpy(),
                d["timestamp"].dt.dayofweek.astype(int).to_numpy(),
                d["timestamp"].dt.hour.astype(int).to_numpy(),
            ]
        )
        exp = pd.Series(keys.map(self._seasonal_profile_map), index=d.index, dtype=float)
        exp = exp.fillna(d["entity_id"].map(self._entity_median))
        exp = exp.fillna(self._global_median)
        return pd.to_numeric(exp, errors="coerce").fillna(self._global_median).astype(float)

    def _annotate_events(self, events, d: pd.DataFrame, x: pd.DataFrame):
        d2 = d.copy()
        d2["expected"] = self._expected_for_rows(d2)

        idx_by_entity: dict[object, np.ndarray] = {}
        time_ns_by_entity: dict[object, np.ndarray] = {}
        for ent, g in d2.groupby("entity_id", sort=False):
            idx = g.index.to_numpy()
            t_ns = pd.to_datetime(g["timestamp"]).to_numpy(dtype="datetime64[ns]").astype("int64")
            idx_by_entity[ent] = idx
            idx_by_entity[str(ent)] = idx
            time_ns_by_entity[ent] = t_ns
            time_ns_by_entity[str(ent)] = t_ns

        exclude_features = {"hour", "dow", "month"}
        preferred_prefixes = ("lag_", "delta_", "roll_", "resid", "abs_resid")
        eps = 1e-9
        explained_events: list[Event] = []

        for e in events:
            ent = e.entity_id
            idx = idx_by_entity.get(ent)
            t_ns = time_ns_by_entity.get(ent)
            if idx is None or t_ns is None or len(idx) == 0:
                explained_events.append(e)
                continue

            start_ns = int(pd.to_datetime(e.start).value)
            end_ns = int(pd.to_datetime(e.end).value)
            left = int(np.searchsorted(t_ns, start_ns, side="left"))
            right = int(np.searchsorted(t_ns, end_ns, side="right"))
            if right <= left:
                explained_events.append(e)
                continue

            idx_window = idx[left:right]
            scores = pd.to_numeric(d2.loc[idx_window, "score"], errors="coerce").to_numpy(dtype=float)
            if len(scores) == 0 or np.all(np.isnan(scores)):
                explained_events.append(e)
                continue

            rel_peak = int(np.nanargmax(scores))
            idx_peak = int(idx_window[rel_peak])
            prob_peak = float(d2.at[idx_peak, "score"])
            prob_mean = float(np.nanmean(scores))
            expected_peak = float(d2.at[idx_peak, "expected"])
            value_peak = float(d2.at[idx_peak, "value"])
            residual_peak = value_peak - expected_peak

            ranked: list[tuple[float, str, float]] = []
            for feat in self._feature_cols:
                if feat in exclude_features:
                    continue
                if feat not in x.columns:
                    continue
                val = pd.to_numeric(pd.Series([x.at[idx_peak, feat]]), errors="coerce").iloc[0]
                if pd.isna(val):
                    continue
                med = float(self._feat_median.get(feat, 0.0))
                iqr = float(self._feat_iqr.get(feat, 1.0))
                scale = iqr if abs(iqr) > eps else 1.0
                nm = abs(float(val) - med) / scale
                pref = 1 if feat.startswith(preferred_prefixes) else 0
                ranked.append((nm + 1e-6 * pref, feat, float(val)))

            ranked.sort(key=lambda t: t[0], reverse=True)
            top3 = ranked[:3]
            top_s = ", ".join([f"{f}={v:.3g}" for _, f, v in top3]) if top3 else "n/a"

            thresh_s = (
                f"thr_start={self._start_threshold:.3f},"
                f"thr_cont={self._continue_threshold:.3f}"
            )

            reason = (
                f"prob_peak={prob_peak:.3f} ({thresh_s}); "
                f"expected={expected_peak:.3g}; value={value_peak:.3g}; "
                f"resid={residual_peak:+.3g}; top: {top_s}"
            )

            explained_events.append(
                Event(
                    entity_id=e.entity_id,
                    start=e.start,
                    end=e.end,
                    direction=e.direction,
                    severity=prob_peak,
                    score_peak=prob_peak,
                    score_mean=prob_mean,
                    reason=reason,
                )
            )

        return explained_events

    def _entity_tail_mask(self, d: pd.DataFrame, frac: float = 0.2) -> pd.Series:
        mask = pd.Series(False, index=d.index)
        for _, g in d.groupby("entity_id", sort=False):
            n = len(g)
            if n <= 1:
                continue
            k = max(1, int(n * frac))
            idx = g.index[-k:]
            mask.loc[idx] = True
        return mask

    def _fit_profiles(self, d: pd.DataFrame) -> None:
        tmp = d.copy()
        tmp["hour"] = tmp["timestamp"].dt.hour
        tmp["dow"] = tmp["timestamp"].dt.dayofweek

        prof = (
            tmp.groupby(["entity_id", "dow", "hour"], observed=True)["value"]
            .median()
            .rename("seasonal_expected")
            .reset_index()
        )
        self._seasonal_profile = prof
        self._seasonal_profile_map = (
            prof.set_index(["entity_id", "dow", "hour"])["seasonal_expected"]
            if len(prof) > 0
            else pd.Series(dtype=float)
        )
        self._entity_median = tmp.groupby("entity_id", observed=True)["value"].median()
        self._global_median = float(tmp["value"].median()) if len(tmp) else 0.0

    def _make_features(self, d: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        x = d.copy()
        x["hour"] = x["timestamp"].dt.hour.astype(float)
        x["dow"] = x["timestamp"].dt.dayofweek.astype(float)
        x["month"] = x["timestamp"].dt.month.astype(float)

        g = x.groupby("entity_id", sort=False)["value"]
        for k in self.lags:
            lag_col = f"lag_{k}"
            x[lag_col] = g.shift(k)
            x[f"delta_{k}"] = x["value"] - x[lag_col]
            x[f"{lag_col}_missing"] = x[lag_col].isna().astype(float)

        v_past = g.shift(1)
        for w in self.feature_windows:
            min_periods = max(8, w // 3)
            rg = v_past.groupby(x["entity_id"], sort=False).rolling(
                window=w,
                min_periods=min_periods,
            )
            x[f"roll_mean_{w}"] = rg.mean().reset_index(level=0, drop=True)
            x[f"roll_std_{w}"] = rg.std().reset_index(level=0, drop=True)
            x[f"roll_median_{w}"] = rg.median().reset_index(level=0, drop=True)

        seasonal_keys = pd.MultiIndex.from_arrays(
            [
                x["entity_id"].to_numpy(),
                x["dow"].astype(int).to_numpy(),
                x["hour"].astype(int).to_numpy(),
            ]
        )
        x["seasonal_expected"] = seasonal_keys.map(self._seasonal_profile_map).astype(float)
        x["seasonal_expected"] = x["seasonal_expected"].fillna(x["entity_id"].map(self._entity_median))
        x["seasonal_expected"] = x["seasonal_expected"].fillna(self._global_median)
        x["resid"] = x["value"] - x["seasonal_expected"]
        x["abs_resid"] = x["resid"].abs()

        drop_cols = {"entity_id", "timestamp", "value"}
        if fit_mode:
            self._feature_cols = [c for c in x.columns if c not in drop_cols]
            med = x[self._feature_cols].median(numeric_only=True)
            self._impute_values = {c: float(med.get(c, 0.0)) for c in self._feature_cols}

        if not self._feature_cols:
            self._feature_cols = [c for c in x.columns if c not in drop_cols]

        out = x.reindex(columns=self._feature_cols)
        for c in self._feature_cols:
            fill_v = self._impute_values.get(c, 0.0)
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill_v)
        if fit_mode:
            med = out.median(numeric_only=True)
            q75 = out.quantile(0.75, numeric_only=True)
            q25 = out.quantile(0.25, numeric_only=True)
            iqr = q75 - q25
            std = out.std(numeric_only=True)
            self._feat_median = {c: float(med.get(c, 0.0)) for c in self._feature_cols}
            self._feat_iqr = {}
            for c in self._feature_cols:
                v = float(iqr.get(c, np.nan))
                if not np.isfinite(v) or abs(v) < 1e-9:
                    s = float(std.get(c, np.nan))
                    v = s if np.isfinite(s) and abs(s) > 1e-9 else 1.0
                self._feat_iqr[c] = v
        return out

    def _fit_model(self, x: pd.DataFrame, y: pd.Series):
        y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
        pos = int((y_arr == 1).sum())
        neg = int((y_arr == 0).sum())

        if len(np.unique(y_arr)) < 2:
            return _ConstantProbModel(prob=float(y_arr.mean()) if len(y_arr) else 0.0)

        pos_weight = float(neg / max(pos, 1))

        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                model = LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=0,
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
                )
                model.fit(x, y_arr)
                return model
            except ImportError:
                pass

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as e:
            raise ImportError(
                "No supported model backend found. Install lightgbm or scikit-learn."
            ) from e

        sample_weight = np.ones(len(y_arr), dtype=float)
        sample_weight[y_arr == 1] = pos_weight
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=63,
            random_state=0,
        )
        model.fit(x, y_arr, sample_weight=sample_weight)
        return model

    def _predict_prob(self, model, x: pd.DataFrame) -> np.ndarray:
        p = model.predict_proba(x)
        if p.ndim != 2 or p.shape[1] < 2:
            raise ValueError("Model predict_proba returned unexpected shape.")
        return p[:, 1].astype(float)
