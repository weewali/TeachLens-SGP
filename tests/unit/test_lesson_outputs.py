"""2026-11-19 additions: desk posture, on/off-task states, hypotheses, pooling, facts.json, side-by-side timing."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.fusion import FEATURES, Feature, build_windows, student_state  # noqa: E402
from src.analysis.lag_analysis import analyze, comparison, pooled_analyze  # noqa: E402
from src.analysis.visual_indicators import desk_posture, polygon_top  # noqa: E402
from src.inference.pose import Keypoint, PosePerson  # noqa: E402
from src.reporting.facts import lesson_facts  # noqa: E402
from src.reporting.side_by_side import ReviewTiming, header_lines  # noqa: E402

ZONE = timezone(timedelta(hours=4))
BASE = datetime(2026, 11, 3, 19, 16, 0, tzinfo=ZONE)
DESK = ((0.58, 0.80), (0.65, 0.77), (1.0, 0.76), (1.0, 0.88), (0.68, 0.92), (0.58, 0.93))
FEATURE_META = {f.name: {"camera": f.camera, "indicator": f.indicator, "labels": list(f.labels), "kind": f.kind,
                         "desirable": f.desirable, "description": f.description} for f in FEATURES}


def teacher_behind_desk(shoulder_y: float, legs: bool = False) -> PosePerson:
    """A 3840x2160 instructor-camera person whose shoulders (230 px apart) are at shoulder_y."""
    points = [(0.78, shoulder_y - 0.05)] * 17
    points[5], points[6] = (0.75, shoulder_y), (0.81, shoulder_y)
    confidence = [0.9] * 17
    for i in (13, 14, 15, 16):
        confidence[i] = 0.9 if legs else 0.0
    return PosePerson(1, (0.70, shoulder_y - 0.1, 0.85, 0.78), 0.9,
                      tuple(Keypoint(x, y, c) for (x, y), c in zip(points, confidence)), 3840, 2160)


class DeskPostureTest(unittest.TestCase):
    def test_polygon_top(self):
        self.assertAlmostEqual(polygon_top(DESK, 0.8), 0.7657, places=3)
        self.assertIsNone(polygon_top(DESK, 0.3))

    def test_seated_and_standing_behind_the_desk(self):
        # Desk top at x=0.78 is y=0.766; 230 px shoulder width = 0.106 of the image height.
        seated = teacher_behind_desk(0.766 - 0.7 * 230 / 2160)
        standing = teacher_behind_desk(0.766 - 1.8 * 230 / 2160)
        between = teacher_behind_desk(0.766 - 1.25 * 230 / 2160)
        self.assertEqual(desk_posture(seated, DESK, 0.35, 1.1, 1.4)[0], "sitting")
        self.assertAlmostEqual(desk_posture(seated, DESK, 0.35, 1.1, 1.4)[1], 0.7, places=1)
        self.assertEqual(desk_posture(standing, DESK, 0.35, 1.1, 1.4)[0], "standing")
        self.assertEqual(desk_posture(between, DESK, 0.35, 1.1, 1.4)[0], "unknown")
        # Visible knees or ankles: not hidden behind the desk.
        self.assertEqual(desk_posture(teacher_behind_desk(0.69, legs=True), DESK, 0.35, 1.1, 1.4),
                         ("unknown", None))


def event(indicator, value, start, end, track=1, role="student"):
    t0, t1 = BASE + timedelta(seconds=start), BASE + timedelta(seconds=end)
    return {"indicator_name": indicator, "role": role, "track_id": track, "value": value,
            "start_real_time": t0.isoformat(), "end_real_time": t1.isoformat(),
            "t0": t0.timestamp(), "t1": t1.timestamp()}


class StudentStateTest(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(student_state({"orientation_proxy": "on_target"}), "on_task")
        self.assertEqual(student_state({"orientation_proxy": "looking_down", "note_taking": "writing"}), "on_task")
        self.assertEqual(student_state({"orientation_proxy": "looking_down", "device_use": "phone_likely"}),
                         "off_task")
        self.assertEqual(student_state({"orientation_proxy": "on_target", "alertness": "dozing"}), "off_task")
        self.assertEqual(student_state({"head_target": "talking_peer"}), "off_task")
        self.assertEqual(student_state({"orientation_proxy": "looking_down"}), "unknown")
        self.assertEqual(student_state({}), "unknown")

    def test_note_taking_counts_as_on_task_in_windows(self):
        student = [event("position", {}, t, t) for t in range(0, 21)] + [
            event("orientation_proxy", "looking_down", 0, 20, track=1),
            event("note_taking", "writing", 0, 20, track=1),
            event("orientation_proxy", "looking_down", 0, 10, track=2),
            event("device_use", "phone", 0, 10, track=2),
            event("orientation_proxy", "on_target", 10, 20, track=2)]
        teacher = [event("position", {}, t, t, role="teacher") for t in range(0, 21)]
        features = [Feature("S_on_task", "students", "", (), kind="on_task"),
                    Feature("S_off_task", "students", "", (), kind="off_task")]
        table, _ = build_windows(student, teacher, window_seconds=5.0, features=features)
        self.assertEqual(table["S_on_task"].round(2).tolist(), [0.5, 0.5, 1.0, 1.0])
        self.assertEqual(table["S_off_task"].round(2).tolist(), [0.5, 0.5, 0.0, 0.0])


def lesson_table(n=120, seed=0, effect=0.3, seated_always=False):
    """Teacher seated in blocks; students go off task one window after the teacher sits."""
    rng = np.random.default_rng(seed)
    seated = np.ones(n) if seated_always else np.repeat(rng.random(n // 6) < 0.5, 6)[:n].astype(float)
    off = np.clip(0.2 + rng.normal(0, 0.05, n), 0, 1)
    off[1:] += effect * seated[:-1]
    return pd.DataFrame({"lesson_seconds": np.arange(n) * 5.0, "T_seated": seated, "S_off_task": off,
                         "S_on_task": 1 - off, "S_hand_raises": np.zeros(n)})


SETTINGS = {"permutations": 100, "hypotheses": [{"teacher": "T_seated", "student": "S_off_task", "expect": 1,
                                                 "code": "T-SIT"}]}


class HypothesisTest(unittest.TestCase):
    def test_planted_effect_is_found_as_expected(self):
        table = lesson_table()
        meta = {"window_seconds": 5.0, "features": {k: FEATURE_META[k] for k in ("T_seated", "S_off_task",
                                                                                 "S_on_task", "S_hand_raises")}}
        result = analyze(table, meta, SETTINGS)
        h = result["hypotheses"][0]
        self.assertEqual((h["code"], h["lag_windows"]), ("T-SIT", 1))
        self.assertTrue(h["matches_expectation"])
        self.assertLessEqual(h["q"], 0.05)
        self.assertGreater(h["comparison"]["difference"], 0.2)
        self.assertEqual(result["findings"][0]["kind"], "weak_point")  # more off task: unfavourable
        self.assertEqual(result["findings"][0]["method"], "hypothesis_lag_correlation")

    def test_no_variation_is_not_tested(self):
        meta = {"window_seconds": 5.0, "features": {k: FEATURE_META[k] for k in ("T_seated", "S_off_task")}}
        result = analyze(lesson_table(seated_always=True)[["lesson_seconds", "T_seated", "S_off_task"]], meta,
                         SETTINGS)
        self.assertIn("varied too little", result["hypotheses"][0]["not_tested"])
        self.assertEqual(result["findings"], [])

    def test_comparison_numbers(self):
        x = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], float)
        y = np.array([0, .9, .9, .9, .1, .1, .1, .9, .9, .9, .1, .1])
        result = comparison(x, y, 1, 0.5)
        self.assertEqual((result["windows_with"], result["windows_without"]), (6, 5))
        self.assertAlmostEqual(result["outcome_with"], 0.9)
        self.assertAlmostEqual(result["outcome_without"], 0.1)

    def test_pooled_within_lessons(self):
        tables = [("a", lesson_table(seed=1, effect=0.15)), ("b", lesson_table(seed=2, effect=0.15)),
                  ("c", lesson_table(seated_always=True))]
        pooled = pooled_analyze(tables, FEATURE_META, SETTINGS)
        h = pooled["hypotheses"][0]
        self.assertEqual(h["lessons_used"], ["a", "b"])
        self.assertEqual(h["lessons_without_variation"], ["c"])
        self.assertTrue(h["matches_expectation"])
        self.assertLessEqual(h["q"], 0.05)
        self.assertGreater(h["comparison"]["difference_within_lessons"], 0.1)


class FactsTest(unittest.TestCase):
    def test_lesson_facts_follow_the_llm_schema(self):
        table = lesson_table()
        table["S_hand_raises"] = 0.0
        table.loc[[3, 40], "S_hand_raises"] = 1.0
        table["S_phone"] = 0.1
        meta = {"window_seconds": 5.0, "features": FEATURE_META}
        report = {**meta, "start_real_time": BASE.isoformat(), **analyze(table, meta, SETTINGS)}
        facts = lesson_facts(table, report, lesson="l", videos=["s.MOV", "i.MOV"],
                             student_summary={"student_hand_raises": {"episodes_by_max_level": {"low": 2},
                                                                      "outcomes": {"answered_likely": 1}},
                                              "visible_student_count": {"median": 6}},
                             instructor_summary={"teacher_visible_fraction_of_valid_samples": 0.98})
        self.assertEqual(facts["schema_version"], 1)
        for key in ("lecture", "data_quality", "indicator_validation", "glossary", "alternative_explanations",
                    "facts", "report_note"):
            self.assertIn(key, facts)
        ids = [f["id"] for f in facts["facts"]]
        self.assertEqual(ids, [f"F{i}" for i in range(1, len(ids) + 1)])
        kinds = {f["kind"] for f in facts["facts"]}
        self.assertTrue({"comparison", "hand_raises", "phone_use", "highest_on_task_stretch"} <= kinds)
        comparison_fact = next(f for f in facts["facts"] if f["kind"] == "comparison")
        self.assertEqual(comparison_fact["direction"], "unfavourable")  # seated -> more off task
        self.assertEqual(comparison_fact["evidence"]["strength"], "suggestive")
        for fact in facts["facts"]:
            self.assertIn(fact["direction"], ("favourable", "unfavourable"))
            self.assertIn(fact["evidence"]["strength"], ("suggestive", "weak", "descriptive"))
        # The LLM stage rejects internal-state words; the facts avoid them too.
        text = json.dumps(facts["facts"])
        self.assertIsNone(re.search(r"\b(attention|attentive|engag\w*|bored)\b", text, re.I))


class SideBySideTest(unittest.TestCase):
    def test_frame_timing_and_header(self):
        timing = ReviewTiming(Path("x.mp4"), BASE, 8.0, 1.0)
        self.assertEqual(timing.frame_at(BASE + timedelta(seconds=8)), 0)
        self.assertEqual(timing.frame_at(BASE + timedelta(seconds=20.4)), 12)
        left, middle, right = header_lines({"T_seated": 1.0, "T_at_desk": 1.0, "T_facing_class": 0.2,
                                            "S_on_task": 0.75, "S_off_task": 0.25, "S_phone": float("nan")},
                                           BASE, 65, "sgp_A3")
        self.assertIn("19:16:00", left)
        self.assertIn("01:05", left)
        self.assertEqual(middle, "instructor: seated, at desk")
        self.assertEqual(right, "class: on task 75%  off task 25%  phone ?")


if __name__ == "__main__":
    unittest.main()
