"""Event filtering and value text for scripts/show_indicator.py; no outputs needed."""

import argparse
import importlib.util
import unittest

from src.utils.paths import project_path


spec = importlib.util.spec_from_file_location("show_indicator",
                                              project_path("scripts/show_indicator.py"))
show = importlib.util.module_from_spec(spec)
spec.loader.exec_module(show)


def event(name, start, end, value, track=1, role="student", coverage="adequate"):
    return {"indicator_name": name, "start_time": start, "end_time": end, "duration": end - start,
            "value": value, "track_id": track, "role": role, "coverage": coverage}


EVENTS = [
    event("sitting_standing", 20.0, 95.0, "sitting", track=3),
    event("sitting_standing", 95.0, 105.0, "standing", track=3),
    event("sitting_standing", 0.0, 3.0, "unknown", track=1, coverage="limited"),
    event("visible_student_count", 10.0, 10.0, 6, track=None, role="classroom"),
    event("role_classification", 1.0, 1.0, {"label": "teacher", "source": "camera_context",
                                           "verified_by_model": False}, role="teacher"),
]


class SelectTests(unittest.TestCase):
    def test_filters_by_track_value_and_coverage(self):
        self.assertEqual(len(show.select(EVENTS, "sitting_standing")), 3)
        self.assertEqual([e["value"] for e in show.select(EVENTS, "sitting_standing", track=3)],
                         ["sitting", "standing"])
        self.assertEqual(len(show.select(EVENTS, "sitting_standing", value="standing")), 1)
        self.assertEqual(len(show.select(EVENTS, "sitting_standing", coverage="limited")), 1)
        self.assertEqual(len(show.select(EVENTS, "visible_student_count", value="6")), 1)

    def test_time_range_keeps_overlapping_intervals_in_order(self):
        rows = show.select(EVENTS, "sitting_standing", start=90.0, end=100.0)
        self.assertEqual([e["value"] for e in rows], ["sitting", "standing"])
        self.assertEqual(show.select(EVENTS, "sitting_standing", start=200.0), [])


class TextTests(unittest.TestCase):
    def test_compact_values(self):
        self.assertEqual(show.compact("role_classification", EVENTS[4]["value"]), "teacher")
        self.assertEqual(show.compact("motion_speed", {"body_heights_per_second": 0.075,
                                                       "assumes_fixed_camera": True}), "0.075 bh/s")
        self.assertEqual(show.compact("shoulder_rise",
                                      {"shoulder_widths_above_seated_level": 1.04}), "+1.04 widths")
        self.assertIn("occluded?", show.compact("visibility_occlusion", {
            "face_landmarks_visible": True, "upper_body_visible": False,
            "possible_occlusion": True, "maximum_box_overlap_fraction": 0.3}))

    def test_times(self):
        self.assertEqual(show.parse_time("1:30"), 90.0)
        self.assertEqual(show.clock(95.0), "1:35.0")
        with self.assertRaises(argparse.ArgumentTypeError):
            show.parse_time("abc")


if __name__ == "__main__":
    unittest.main()
