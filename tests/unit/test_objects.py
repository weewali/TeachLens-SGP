"""COCO object detection without poses: adapter, summary, overlay, and the run itself.

A fake detector stands in for YOLO26, so no weights, GPU, or download are needed.
"""

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.analysis.aggregates import summarize_objects
from src.analysis.events import make_event, write_jsonl
from src.inference.objects import DetectedObject, ObjectDetector, adapt_detections
from src.pipeline import detect_objects
from src.reporting.annotated_video import ObjectOverlay
from src.utils.config import load_config, validate_config
from src.utils.paths import project_path


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

    def int(self):
        return FakeTensor([int(v) for v in self.values])


class FakeBoxes:
    def __init__(self, rows):
        self.xyxyn = FakeTensor([row[:4] for row in rows])
        self.conf = FakeTensor([row[4] for row in rows])
        self.cls = FakeTensor([float(row[5]) for row in rows])
        self.count = len(rows)

    def __len__(self):
        return self.count


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


NAMES = {63: "laptop", 67: "cell phone", 73: "book"}


class AdapterTests(unittest.TestCase):
    def test_converts_boxes_scores_and_class_names(self):
        found = adapt_detections(FakeResult([[.1, .2, .3, .4, .91234, 67]]), NAMES)
        self.assertEqual(found, (DetectedObject("cell phone", .9123, (.1, .2, .3, .4)),))
        self.assertEqual(adapt_detections(FakeResult([]), NAMES), ())

    def test_missing_weights_never_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                ObjectDetector(Path(directory) / "yolo26n.pt", ["laptop"], .25)


class SummaryTests(unittest.TestCase):
    def test_per_class_presence_with_explicit_zeros(self):
        events = []
        for time, phones in ((0.0, 0), (1.0, 2), (2.0, 1)):
            for _ in range(phones):
                events.append(make_event("v", "object", time, time,
                                         {"label": "cell phone", "box_xyxy": [0, 0, .1, .1]},
                                         confidence=.6))
            events.append(make_event("v", "object_count", time, time,
                                     {"cell phone": phones, "laptop": 0}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v.objects.jsonl"
            write_jsonl(path, events)
            summary = summarize_objects(path, "student_view", "v", 3.0, 1.0, ["cell phone", "laptop"])
        phone, laptop = summary["objects"]["cell phone"], summary["objects"]["laptop"]
        self.assertEqual(summary["analyzed_samples"], 3)
        self.assertEqual((phone["samples_seen"], phone["max_count_in_one_sample"]), (2, 2))
        self.assertEqual((phone["first_seen_seconds"], phone["last_seen_seconds"]), (1.0, 2.0))
        self.assertAlmostEqual(phone["fraction_of_samples"], .6667)
        self.assertAlmostEqual(phone["mean_confidence"], .6)
        self.assertEqual((laptop["samples_seen"], laptop["fraction_of_samples"]), (0, 0.0))
        self.assertIsNone(laptop["mean_confidence"])


class OverlayTests(unittest.TestCase):
    def test_counts_in_panel_and_frame_left_unchanged(self):
        overlay = ObjectOverlay("v | student_view | objects", ["cell phone", "laptop"], duration=60)
        found = (DetectedObject("cell phone", .8, (.1, .1, .3, .3)),
                 DetectedObject("cell phone", .5, (.5, .5, .7, .7)))
        self.assertIn("objects: cell phone 2", overlay.hud_lines(1.0, found))
        self.assertIn("objects: none detected", overlay.hud_lines(1.0, ()))
        frame = np.zeros((120, 160, 3), np.uint8)
        drawn = overlay.draw(frame, found, 1.0)
        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(drawn.sum()), 0)


class ConfigTests(unittest.TestCase):
    def test_default_classes_and_validation(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertIn("cell phone", config["objects"]["classes"])
        self.assertEqual(config["objects"]["detector"], "models/detection/yolo26n.pt")
        for bad in ({"classes": "laptop"}, {"classes": ["laptop"], "confidence_threshold": 2},
                    {"classes": ["laptop"], "image_size": 1000}):
            broken = {**config, "objects": bad}
            with self.assertRaises(ValueError):
                validate_config(broken)


class FakeDetector:
    """Sees a phone in every frame after the first and a laptop in the last."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        found = []
        if self.calls > 1:
            found.append(DetectedObject("cell phone", .7, (.1, .1, .2, .2)))
        if self.calls == 4:
            found.append(DetectedObject("laptop", .9, (.4, .4, .8, .8)))
        return tuple(found)


class RunTests(unittest.TestCase):
    def test_writes_events_summary_and_video_with_real_times(self):
        config = load_config(project_path("configs/default.yaml"))
        config["review"]["video_codec"] = "mp4v"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.mp4"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
            for index in range(40):  # 4 s at 10 fps -> 4 samples at 1 per second
                writer.write(np.full((48, 64, 3), index * 6, np.uint8))
            writer.release()
            start = datetime(2026, 11, 3, 19, 0, tzinfo=timezone(timedelta(hours=4)))
            with mock.patch("src.pipeline.ObjectDetector", FakeDetector):
                events, summary, video = detect_objects(
                    clip, "v_objects", "student_view", config, output_dir=root,
                    save_video=True, labels=["cell phone", "laptop"], real_start=start)
                with self.assertRaises(FileExistsError):  # no silent overwrite
                    detect_objects(clip, "v_objects", "student_view", config, output_dir=root,
                                   labels=["laptop"])
            # One dated run folder: runs/<YYYY-MM-DD_HHMM>_v_objects/ with objects.* and objects_run.json.
            self.assertRegex(events.parent.name, r"^\d{4}-\d{2}-\d{2}_\d{4}_v_objects$")
            self.assertEqual(events.parent.parent.name, "runs")
            self.assertEqual(json.loads((events.parent / "objects_run.json").read_text(encoding="utf-8"))["run_id"],
                             "v_objects")
            rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            counts = [row for row in rows if row["indicator_name"] == "object_count"]
            self.assertEqual([row["value"] for row in counts],
                             [{"cell phone": 0, "laptop": 0}, {"cell phone": 1, "laptop": 0},
                              {"cell phone": 1, "laptop": 0}, {"cell phone": 1, "laptop": 1}])
            laptop = [row for row in rows if row["indicator_name"] == "object"
                      and row["value"]["label"] == "laptop"]
            self.assertEqual(laptop[0]["start_time"], 3.0)
            self.assertEqual(laptop[0]["start_real_time"], "2026-11-03T19:00:03.000+04:00")
            report = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(report["objects"]["cell phone"]["samples_seen"], 3)
            self.assertEqual(report["recording_start_source"], "given")
            capture = cv2.VideoCapture(str(video))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            capture.release()


if __name__ == "__main__":
    unittest.main()
