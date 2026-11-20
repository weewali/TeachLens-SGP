"""Lesson fusion (window table) and lag analysis on synthetic events with planted lags."""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.fusion import Feature, build_windows, load_events, onsets, overlap_range  # noqa: E402
from src.analysis.lag_analysis import (analyze, benjamini_hochberg, cross_correlation,  # noqa: E402
                                       lagged_model, peri_event, shift_test)

ZONE = timezone(timedelta(hours=4))
BASE = datetime(2026, 11, 3, 19, 16, 0, tzinfo=ZONE)


def event(indicator, role, value, start, end=None, track=1):
    """A saved event with real times start/end seconds after BASE (t0/t1 filled as load_events does)."""
    end = start if end is None else end
    t0, t1 = BASE + timedelta(seconds=start), BASE + timedelta(seconds=end)
    return {"video_id": "x", "track_id": track, "role": role, "indicator_name": indicator,
            "start_time": start, "end_time": end, "duration": end - start, "value": value,
            "start_real_time": t0.isoformat(timespec="milliseconds"),
            "end_real_time": t1.isoformat(timespec="milliseconds"),
            "t0": t0.timestamp(), "t1": t1.timestamp()}


def positions(role, start, end, track=1, hips=None, shot=0):
    rows = []
    for i, t in enumerate(range(start, end)):
        hip = hips(i) if hips else None
        rows.append(event("position", role, {"hip_center_in_shot": hip, "camera_shot": shot}, t, t, track))
    return rows


class WindowTableTest(unittest.TestCase):
    def test_shares_align_on_real_time_and_skip_unknown(self):
        # Students recorded from 0 s, the instructor camera from 3 s: the table starts at 3 s.
        student = positions("student", 0, 30, 1) + positions("student", 0, 30, 2) + [
            event("orientation_proxy", "student", "on_target", 0, 30, 1),
            event("orientation_proxy", "student", "off_target", 0, 13, 2),
            event("orientation_proxy", "student", "on_target", 13, 30, 2)]
        teacher = positions("teacher", 3, 30) + [
            event("facing_direction", "teacher", "toward_class", 3, 13),
            event("facing_direction", "teacher", "sideways_left", 13, 18),
            event("facing_direction", "teacher", "unknown", 18, 23),
            event("facing_direction", "teacher", "toward_class", 23, 29)]
        features = [Feature("T_facing_class", "teacher", "facing_direction", ("toward_class",)),
                    Feature("S_on_target", "students", "orientation_proxy", ("on_target",), desirable=1)]
        table, meta = build_windows(student, teacher, window_seconds=5.0, features=features)
        self.assertEqual(meta["windows"], 5)  # 3..28 s
        self.assertEqual(meta["start_real_time"], (BASE + timedelta(seconds=3)).isoformat(timespec="milliseconds"))
        facing = table["T_facing_class"].tolist()
        self.assertEqual(facing[:3], [1.0, 1.0, 0.0])
        self.assertTrue(math.isnan(facing[3]))  # only unknown evidence in 18-23 s
        self.assertAlmostEqual(facing[4], 1.0)
        # Window 3-8 s: one student on target, the other off -> class share 0.5.
        self.assertAlmostEqual(table["S_on_target"].iloc[0], 0.5)
        self.assertAlmostEqual(table["S_on_target"].iloc[2], 1.0)

    def test_counts_and_teacher_path(self):
        student = positions("student", 0, 21) + [
            event("hand_raise_episode", "student", {"side": "right"}, 11, 13),
            event("hand_raise_outcome", "student", {"outcome": "answered_likely"}, 11, 13),
            event("hand_raise_outcome", "student", {"outcome": "no_answer_seen"}, 16, 17)]
        # Hips move 0.1 per second in shot 0; at 15 s a new shot starts (no step across it).
        teacher = (positions("teacher", 0, 15, hips=lambda i: [0.1 * i, 0.5])
                   + positions("teacher", 15, 21, hips=lambda i: [0.0, 0.5], shot=1))
        features = [Feature("S_hand_raises", "students", "hand_raise_episode", (), kind="count"),
                    Feature("S_answers", "students", "hand_raise_outcome", ("answered_likely",), kind="count"),
                    Feature("T_path", "teacher", "position", (), kind="path")]
        table, _ = build_windows(student, teacher, window_seconds=5.0, features=features)
        self.assertEqual(table["S_hand_raises"].tolist(), [0, 0, 1, 0])
        self.assertEqual(table["S_answers"].tolist(), [0, 0, 1, 0])
        self.assertAlmostEqual(table["T_path"].iloc[1], 0.5)  # steps landing at 5..9 s
        self.assertAlmostEqual(table["T_path"].iloc[3], 0.0)  # new shot, the hips stay put

    def test_needs_real_times_and_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.jsonl"
            path.write_text(json.dumps({"indicator_name": "position", "role": "student",
                                        "start_real_time": None, "end_real_time": None}) + "\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_events(path)
            good = event("position", "student", {}, 0, 0)
            path.write_text(json.dumps({k: v for k, v in good.items() if k not in ("t0", "t1")}) + "\n",
                            encoding="utf-8")
            self.assertAlmostEqual(load_events(path)[0]["t0"], good["t0"])
        with self.assertRaises(ValueError):
            overlap_range(positions("student", 0, 10), positions("teacher", 20, 30))

    def test_onsets_skip_close_restarts(self):
        events = [event("board_interaction", "teacher", "board", 5, 9),
                  event("board_interaction", "teacher", "board", 11, 14),   # 6 s after the last onset
                  event("board_interaction", "teacher", "none", 14, 30),
                  event("board_interaction", "teacher", "screen", 31, 35)]
        self.assertEqual(onsets(events, "board_interaction", "teacher", ("board", "screen"),
                                BASE.timestamp(), 5.0, 10, min_gap_seconds=10), [1, 6])


def planted(n=150, lag=2, seed=1):
    """Teacher bursts x and a student series y that follows x lag windows later."""
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < 0.3).astype(float)
    y = np.full(n, 0.6) + rng.normal(0, 0.05, n)
    y[lag:] += 0.3 * x[:n - lag]
    return x, y


class LagAnalysisTest(unittest.TestCase):
    def test_cross_correlation_finds_planted_lag(self):
        x, y = planted()
        r = cross_correlation(x, y, 6)
        self.assertEqual(max(r, key=lambda k: r[k]), 2)
        self.assertGreater(r[2], 0.8)
        result = shift_test(x, y, 6, 200, np.random.default_rng(0))
        self.assertEqual(result["lag_windows"], 2)
        self.assertLess(result["p_shift"], 0.01)
        self.assertEqual(result["null_shifts"], 150 - 2 * 6 - 1)  # every rotation once (exact)

    def test_short_lesson_cannot_reach_tiny_p(self):
        x, y = planted(n=55)
        result = shift_test(x, y, 6, 500, np.random.default_rng(0))
        self.assertEqual(result["null_shifts"], 42)
        self.assertAlmostEqual(result["p_shift"], round(1 / 43, 4))

    def test_shift_test_on_unrelated_series(self):
        rng = np.random.default_rng(5)
        x = (rng.random(150) < 0.3).astype(float)
        y = rng.normal(0.5, 0.1, 150)
        self.assertGreater(shift_test(x, y, 6, 200, np.random.default_rng(0))["p_shift"], 0.05)

    def test_constant_series_gives_no_correlation(self):
        self.assertEqual(cross_correlation(np.zeros(50), np.linspace(0, 1, 50), 3), {})

    def test_lagged_model_recovers_coefficient(self):
        rng = np.random.default_rng(2)
        n = 300
        x = (rng.random(n) < 0.4).astype(float)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.2 + 0.5 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(0, 0.05)
        model = lagged_model(pd.DataFrame({"S": y, "T": x}), "S", ["T"], 1)
        term = model["terms"]["T[t-1]"]
        self.assertAlmostEqual(term["coef"], 0.5, delta=0.05)
        self.assertLess(term["p"], 0.001)
        self.assertAlmostEqual(model["terms"]["S[t-1]"]["coef"], 0.3, delta=0.08)
        self.assertEqual(model["windows_used"], n - 1)

    def test_peri_event_change(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0.5, 0.02, 200)
        onset = [20, 60, 100, 140, 180 - 10]
        for w in onset:
            y[w + 1:w + 5] += 0.2
        result = peri_event(onset, y, 2, 4, 300, np.random.default_rng(0))
        self.assertEqual(result["events"], 5)
        self.assertGreater(result["mean_change"], 0.15)
        self.assertLess(result["p_random_onsets"], 0.01)
        self.assertEqual(result["curve_relative_windows"], [-2, -1, 0, 1, 2, 3, 4])

    def test_benjamini_hochberg(self):
        self.assertEqual(benjamini_hochberg([0.01, 0.04, 0.03, 0.5]), [0.04, 0.0533, 0.0533, 0.5])
        self.assertEqual(benjamini_hochberg([]), [])

    def test_analyze_guardrails_and_findings(self):
        x, y = planted(n=150, lag=2)
        rare = np.zeros(150)
        rare[40] = 1.0  # one window at the screen: too rare to correlate
        table = pd.DataFrame({"T_board": x, "T_screen": rare, "S_on_target": y,
                              "S_phone": np.random.default_rng(4).normal(0.2, 0.05, 150)})
        meta = {"window_seconds": 5.0, "features": {
            "T_board": {"camera": "teacher", "desirable": 0, "kind": "share"},
            "T_screen": {"camera": "teacher", "desirable": 0, "kind": "share"},
            "S_on_target": {"camera": "students", "desirable": 1, "kind": "share"},
            "S_phone": {"camera": "students", "desirable": -1, "kind": "share"}}}
        settings = {"permutations": 200, "model_lags": [2], "outcomes": ["S_on_target"],
                    "model_predictors": ["T_board", "T_screen"], "min_events": 3}
        result = analyze(table, meta, settings, {"screen_start": [40], "board_start": [10, 50, 90, 130]})
        self.assertEqual(result["tested_teacher_features"], ["T_board"])
        self.assertEqual(result["untested_teacher_features"], {"T_screen": 1})
        top = result["links"][0]
        self.assertEqual((top["teacher_feature"], top["student_feature"], top["lag_seconds"]),
                         ("T_board", "S_on_target", 10.0))
        self.assertLessEqual(top["q"], 0.05)
        self.assertEqual(list(result["lagged_models"][0]["terms"]), ["const", "T_board[t-2]", "S_on_target[t-1]"])
        finding = [f for f in result["findings"] if f["method"] == "exploratory_lag_correlation"][0]
        self.assertEqual((finding["kind"], finding["student_feature"]), ("strength", "S_on_target"))
        self.assertEqual(result["peri_event"][0], {"teacher_event": "screen_start", "events": 1,
                                                   "skipped": "too few onsets"})
        self.assertTrue(all("q" in e for e in result["peri_event"][1:]))


if __name__ == "__main__":
    unittest.main()
