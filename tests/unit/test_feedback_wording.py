"""The positive and negative points of main code/Main Code.py are plain English for the instructor: they speak to
"you", avoid statistics words, and keep the numbers behind them in facts.json and the technical details.

Uses the made-up demo lecture (tests/fixtures/demo_lecture.py) and small hand-made tables; no video or model."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from demo_lecture import _main_code, make_demo_lecture  # noqa: E402

mc = _main_code()
JARGON = re.compile(r"\bwindows?\b|\bobservable\b|\busable\b|\bp [=<]|\bdescribes\b|comparisons made|\bseat_\d|"
                    r"\bfocus(?:ed)?\b|\bdetected\b", re.I)


def points_of(text):
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]


class HelperTests(unittest.TestCase):
    def test_plain_pieces(self):
        self.assertEqual(mc.length_words(120), "2 minutes")
        self.assertEqual(mc.length_words(90), "1 minute 30 seconds")
        self.assertEqual(mc.length_words(40), "40 seconds")
        self.assertEqual(mc.seat_name("seat_06"), "seat 6")
        self.assertEqual(mc.share_of_time(1.0), "all of the time")
        self.assertEqual(mc.share_of_time(0.004), "0% of the time")
        self.assertEqual(mc.share_of_time(0.81), "about 81% of the time")
        self.assertEqual(mc.times_words(["00:14", "00:14", "03:02"]), "twice at 00:14 and once at 03:02")
        self.assertEqual(mc.times_words(["01:00", "02:00"]), "at 01:00 and 02:00")
        self.assertEqual(mc.times_words(["00:10", "00:20", "00:30", "00:40"]), "for example at 00:10, 00:20 and 00:30")
        self.assertEqual(mc.you_were("the instructor was explaining to the class"), "you were explaining to the class")
        self.assertEqual(mc.you_were("the instructor was mostly out of view"), "you were mostly out of the camera's view")


class DemoLectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        make_demo_lecture(cls.temp.name)
        reports = Path(cls.temp.name) / "sessions" / "demo_lecture" / "reports"
        cls.text = (reports / "final_feedback_report.txt").read_text(encoding="utf-8")
        cls.details = (reports / "analysis_details.txt").read_text(encoding="utf-8")
        cls.facts = json.loads((reports / "facts.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_points_are_plain_english(self):
        points = points_of(self.text)
        self.assertTrue(2 <= len(points) <= 6, points)
        for point in points:
            self.assertIsNone(JARGON.search(point), point)
        self.assertTrue(any("you were" in p for p in points))
        self.assertIn("\"heads up\" means", self.text)

    def test_numbers_and_statistics_stay_in_the_facts_and_details(self):
        kinds = {f["kind"] for f in self.facts["facts"]}
        self.assertGreaterEqual(kinds, {"behaviour_moments", "highest_focus_stretch", "hand_raises"})
        self.assertIn("heads up", self.facts["glossary"])
        comparisons = [f for f in self.facts["facts"] if f["kind"] == "comparison"]
        self.assertTrue(comparisons)
        for fact in comparisons:
            self.assertIn(f"Comparison: {fact['student_behaviour']} {fact['student_share_pct']}%", self.details)
        self.assertRegex(self.details, r"Comparison: .*; p [=<] ")


class SmallCaseTests(unittest.TestCase):
    """One phone moment and one hand raise: the points say 'once' and do not read a pattern into one moment."""

    def test_single_moments(self):
        rows = []
        for i in range(12):
            rows.append({"window_id": i, "start_s": 10.0 * i, "end_s": 10.0 * (i + 1), "instructor_present": 1.0,
                         "explanation_ratio": 1.0, "board_work_ratio": float("nan"), "interaction_count": 0,
                         "movement_amount": 0.1, "teacher_stationary_ratio": 0.0, "teacher_seated_ratio": 0.0,
                         "pose_coverage": 1.0, "seats_observed": 5.0, "attention_level": 0.9 if i != 6 else 0.5,
                         "phone_rate": 0.0, "head_down_still_rate": 0.0, "hand_raise_count": 1 if i == 3 else 0})
        windows = pd.DataFrame(rows)
        events = pd.DataFrame([
            {"entity": "student", "entity_id": "seat_02", "label": "phone_use", "start_s": 44.0, "end_s": 46.0, "duration_s": 2.0},
            {"entity": "student", "entity_id": "seat_03", "label": "hand_raise", "start_s": 32.0, "end_s": 36.0, "duration_s": 4.0},
        ])
        positive, negative, _, _, facts = mc.build_feedback_points(windows, events)
        phone = next(p for p in negative if p.startswith("A phone"))
        self.assertEqual(phone, "A phone was seen once: at 00:44 in seat 2, while you were explaining to the class. A single "
                                "moment is not enough to link it to what you were doing. Phones can also be used for course work.")
        hand = next(p for p in positive if "hand" in p)
        self.assertEqual(hand, "A student raised their hand once (at 00:32), while you were explaining to the class.")
        for point in positive + negative:
            self.assertIsNone(JARGON.search(point), point)
        self.assertEqual(next(f for f in facts if f["kind"] == "hand_raises")["count"], 1)


if __name__ == "__main__":
    unittest.main()
