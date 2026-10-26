"""Segment reading on a tiny synthetic clip; frame brightness encodes its index."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.video import frames


class SegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "clip.mp4"
        writer = cv2.VideoWriter(str(cls.path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for index in range(40):  # 4 s at 10 fps
            writer.write(np.full((48, 64, 3), index * 6, np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def read(self, start=0.0, end=None):
        return [(round(time, 3), int(frame.mean())) for _, time, frame
                in frames(self.path, .2, start, end)]

    def test_segment_keeps_source_times_and_seeks_to_correct_frames(self):
        samples = self.read(1.0, 2.0)
        self.assertEqual([time for time, _ in samples], [1.0, 1.2, 1.4, 1.6, 1.8])
        for time, brightness in samples:
            self.assertAlmostEqual(brightness, round(time * 10) * 6, delta=3)

    def test_segment_uses_full_run_sample_grid(self):
        full = self.read()
        self.assertEqual(self.read(1.05, 1.7), [s for s in full if 1.05 <= s[0] < 1.7])
        self.assertEqual(self.read(1.05, 1.7)[0][0], 1.2)

    def test_start_after_video_end_yields_nothing(self):
        self.assertEqual(self.read(10.0), [])


if __name__ == "__main__":
    unittest.main()
