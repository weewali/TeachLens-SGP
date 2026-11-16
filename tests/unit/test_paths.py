"""Detecting outputs held open by another program; Windows-only behaviour is skipped elsewhere."""

import os
from pathlib import Path
import tempfile
import unittest

from src.pipeline import analyze_video
from src.utils.config import load_config
from src.utils.paths import file_in_use, project_path


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
            (root / "debug").mkdir()
            review = root / "debug" / "v1.list_a.mp4"
            review.write_bytes(b"x")
            with open(review, "rb"):
                with self.assertRaisesRegex(PermissionError, "open in another program"):
                    analyze_video(video, "v1", "student_view", config, output_dir=root,
                                  overwrite=True, save_video=True)


if __name__ == "__main__":
    unittest.main()
