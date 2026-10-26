"""Summary reduction from saved events, including --start/--end segments."""

import tempfile
import unittest
from pathlib import Path

from src.analysis.aggregates import summarize_list_a
from src.analysis.events import make_event, write_jsonl


def counts(path, times):
    write_jsonl(path, [make_event("v1", "visible_student_count", t, t, 4) for t in times])


class SummaryTests(unittest.TestCase):
    def test_full_run_reports_whole_recording_as_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.jsonl"
            counts(path, [0, .2, .4])
            summary = summarize_list_a(path, "student_view", "v1", 60.0, .2)
            self.assertEqual((summary["analyzed_start_seconds"], summary["analyzed_end_seconds"],
                              summary["analyzed_duration_seconds"]), (0.0, 60.0, 60.0))
            self.assertFalse(any("analyzed range" in item for item in summary["limitations"]))

    def test_segment_caps_valid_seconds_and_adds_limitation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.jsonl"
            counts(path, [30 + .2 * i for i in range(10)])
            summary = summarize_list_a(path, "student_view", "v1", 60.0, .5, 30.0, 33.0)
            self.assertEqual(summary["video_duration_seconds"], 60.0)
            self.assertEqual(summary["analyzed_duration_seconds"], 3.0)
            self.assertEqual(summary["student_count_valid_sampled_seconds"], 3.0)  # 10 x .5 capped
            self.assertTrue(any("analyzed range" in item for item in summary["limitations"]))

    def test_camera_motion_steps_and_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.jsonl"
            steps = [(0, None), (0, True), (1, False), (1, True)]
            write_jsonl(path, [make_event("v1", "camera_motion", t * .2, t * .2,
                                          {"shot": shot, "step_reliable": reliable},
                                          role="recording")
                               for t, (shot, reliable) in enumerate(steps)])
            summary = summarize_list_a(path, "instructor_view", "v1", 1.0, .2)
            self.assertEqual(summary["camera_motion"],
                             {"sample_steps": {"first": 1, "measured": 2, "unmeasured": 1}, "shots": 2})
            counts(path.with_name("fixed.jsonl"), [0])
            self.assertIsNone(summarize_list_a(path.with_name("fixed.jsonl"), "student_view",
                                               "v1", 1.0, .2)["camera_motion"])


if __name__ == "__main__":
    unittest.main()
