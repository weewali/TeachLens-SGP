"""Detecting outputs held open by another program; Windows-only behaviour is skipped elsewhere."""

import os
from pathlib import Path
import tempfile
import unittest

from src.pipeline import analyze_video, tracker_settings
from src.utils.config import load_config
from src.utils.paths import file_in_use, project_path


class TrackerSettingsTests(unittest.TestCase):
    def test_repository_tracker_file_becomes_absolute_and_builtin_names_stay(self):
        config = load_config(project_path("configs/default.yaml"))
        resolved = Path(tracker_settings(config["models"]["tracker"]))
        self.assertTrue(resolved.is_absolute() and resolved.is_file())
        self.assertIn("match_thresh: 0.95", resolved.read_text(encoding="utf-8"))
        self.assertEqual(tracker_settings("bytetrack.yaml"), "bytetrack.yaml")


class FileInUseTests(unittest.TestCase):
    def test_missing_file_is_not_in_use(self):
        self.assertFalse(file_in_use(Path(tempfile.gettempdir()) / "teachlens_missing.mp4"))

    @unittest.skipUnless(os.name == "nt", "Windows refuses to replace open files")
    def test_open_file_is_in_use_until_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.mp4"
            path.write_bytes(b"x")
            with open(path, "rb"):  # like a video player showing the review MP4
                self.assertTrue(file_in_use(path))
            self.assertFalse(file_in_use(path))

    @unittest.skipUnless(os.name == "nt", "Windows refuses to replace open files")
    def test_overwrite_stops_before_analysis_when_an_output_is_open(self):
        config = load_config(project_path("configs/default.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"")  # never decoded: the check comes first
            (root / "runs" / "2026-11-19_1200_v1").mkdir(parents=True)
            review = root / "runs" / "2026-11-19_1200_v1" / "review.mp4"
            review.write_bytes(b"x")
            with open(review, "rb"):
                with self.assertRaisesRegex(PermissionError, "open in another program"):
                    analyze_video(video, "v1", "student_view", config, output_dir=root,
                                  overwrite=True, save_video=True)


if __name__ == "__main__":
    unittest.main()
