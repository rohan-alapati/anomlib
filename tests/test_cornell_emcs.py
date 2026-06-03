from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from anomlib.datasets import (
    infer_cornell_emcs_meter_columns,
    infer_timeseries_cadence,
    load_cornell_emcs_csv,
)
from anomlib.detectors import CornellEMCSElectricityDetector


def _write_csv(base_dir: str, contents: str) -> Path:
    path = Path(base_dir) / "cornell.csv"
    path.write_text(contents, encoding="utf-8")
    return path


def _daily_frame(start: str, periods: int, values: list[float], entity_id: str = "meter_a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": entity_id,
            "timestamp": pd.date_range(start, periods=periods, freq="D"),
            "value": values,
        }
    )


def _hourly_training_frame(weeks: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=weeks * 7 * 24, freq="h")
    values = [100.0 + (t.dayofweek * 10.0) + t.hour for t in ts]
    return pd.DataFrame({"entity_id": "meter_a", "timestamp": ts, "value": values})


class CornellEMCSTestCase(unittest.TestCase):
    def test_cornell_csv_loader_parses_wide_format_and_drops_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _write_csv(
                tmp_dir,
                "\n".join(
                    [
                        '"Time","AliceCookHouse.Elec.PowerScout18/kWsystem","SteamFlow"',
                        "2024-06-01 00:00:00,10.5,1",
                        "bad-timestamp,12.0,2",
                        "2024-06-03 00:00:00,not-a-number,3",
                    ]
                ),
            )

            raw = pd.read_csv(csv_path)
            self.assertEqual(
                infer_cornell_emcs_meter_columns(raw),
                ["AliceCookHouse.Elec.PowerScout18/kWsystem"],
            )

            loaded = load_cornell_emcs_csv(csv_path)
            self.assertEqual(list(loaded.columns), ["entity_id", "timestamp", "value"])
            self.assertEqual(
                loaded["entity_id"].unique().tolist(),
                ["AliceCookHouse.Elec.PowerScout18/kWsystem"],
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.iloc[0]["value"], 10.5)
            self.assertTrue(pd.api.types.is_datetime64_ns_dtype(loaded["timestamp"]))

    def test_infer_timeseries_cadence_detects_daily_and_hourly(self) -> None:
        daily = _daily_frame("2024-01-01", 5, [1, 2, 3, 4, 5])
        hourly = pd.DataFrame(
            {
                "entity_id": "meter_a",
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

        self.assertEqual(infer_timeseries_cadence(daily), "daily")
        self.assertEqual(infer_timeseries_cadence(hourly), "hourly")

    def test_hourly_baseline_uses_day_of_week_and_hour_grouping(self) -> None:
        train = _hourly_training_frame()
        test_times = pd.date_range("2024-01-29", periods=24, freq="h")
        test = pd.DataFrame(
            {
                "entity_id": "meter_a",
                "timestamp": test_times,
                "value": [100.0 + (t.dayofweek * 10.0) + t.hour for t in test_times],
            }
        )

        detector = CornellEMCSElectricityDetector(
            cadence="hourly",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        scored = detector.score(test)

        self.assertEqual(scored["expected"].round(6).tolist(), test["value"].round(6).tolist())
        self.assertLess(scored["signed_score"].abs().max(), 1e-6)

    def test_daily_baseline_uses_month_and_day_of_week_grouping(self) -> None:
        ts = pd.date_range("2024-01-01", periods=181, freq="D")
        train = pd.DataFrame(
            {
                "entity_id": "meter_a",
                "timestamp": ts,
                "value": [50.0 + (t.month * 3.0) + (t.dayofweek * 2.0) for t in ts],
            }
        )
        test_times = pd.date_range("2024-06-15", periods=14, freq="D")
        test = pd.DataFrame(
            {
                "entity_id": "meter_a",
                "timestamp": test_times,
                "value": [50.0 + (t.month * 3.0) + (t.dayofweek * 2.0) for t in test_times],
            }
        )

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        scored = detector.score(test)

        self.assertEqual(scored["expected"].round(6).tolist(), test["value"].round(6).tolist())

    def test_scoring_does_not_use_future_test_values(self) -> None:
        train = _daily_frame("2024-01-01", 60, [50.0] * 60)
        test = _daily_frame("2024-03-01", 2, [200.0, 50.0])

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        scored = detector.score(test)

        self.assertEqual(scored.loc[0, "expected"], 50.0)
        self.assertEqual(scored.loc[1, "expected"], 50.0)

    def test_isolated_spike_does_not_become_event(self) -> None:
        train = _daily_frame("2024-01-01", 90, [50.0] * 90)
        test = _daily_frame("2024-04-01", 5, [50.0, 50.0, 150.0, 50.0, 50.0])

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            threshold=3.5,
            min_duration="2d",
            max_gap="0d",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        events, _ = detector.detect(test)

        self.assertEqual(events, [])

    def test_sustained_high_deviation_becomes_event(self) -> None:
        train = _daily_frame("2024-01-01", 90, [50.0] * 90)
        test = _daily_frame("2024-04-01", 4, [150.0, 155.0, 152.0, 151.0])

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            direction="high",
            threshold=3.5,
            min_duration="2d",
            max_gap="0d",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        events, _ = detector.detect(test)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.direction, "high")
        self.assertIsNotNone(event.observed_mean)
        assert event.observed_mean is not None
        self.assertGreater(event.observed_mean, 150.0)
        self.assertIn("persistently high", event.reason)

    def test_sustained_low_deviation_becomes_event(self) -> None:
        train = _daily_frame("2024-01-01", 90, [50.0] * 90)
        test = _daily_frame("2024-04-01", 4, [5.0, 7.0, 6.0, 8.0])

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            direction="low",
            threshold=3.5,
            min_duration="2d",
            max_gap="0d",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        events, _ = detector.detect(test)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "low")
        self.assertIn("persistently low", events[0].reason)

    def test_small_gap_can_be_merged_into_one_event(self) -> None:
        train = _daily_frame("2024-01-01", 90, [50.0] * 90)
        test = _daily_frame("2024-04-01", 5, [150.0, 151.0, 50.0, 152.0, 153.0])

        detector = CornellEMCSElectricityDetector(
            cadence="daily",
            direction="high",
            threshold=3.5,
            min_duration="3d",
            max_gap="1d",
            threshold_quantile=None,
            min_group_history=1,
        ).fit(train)
        events, _ = detector.detect(test)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start, pd.Timestamp("2024-04-01"))
        self.assertEqual(events[0].end, pd.Timestamp("2024-04-05"))


if __name__ == "__main__":
    unittest.main()
