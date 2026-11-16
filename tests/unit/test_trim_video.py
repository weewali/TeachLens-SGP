"""Time stamping and ffmpeg command for scripts/trim_video.py; no ffmpeg or video needed."""

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import unittest

from src.preprocessing.recording_time import from_probe
from src.utils.paths import project_path


spec = importlib.util.spec_from_file_location("trim_video", project_path("scripts/trim_video.py"))
trim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trim)

PLUS_4 = timezone(timedelta(hours=4))


class TrimTests(unittest.TestCase):
    def test_cut_snaps_back_to_a_keyframe(self):
        keyframes = [0.0, 2.966667, 5.933333, 8.9, 11.866667]
        self.assertEqual(trim.keyframe_before(keyframes, 10.0), 8.9)
        self.assertEqual(trim.keyframe_before(keyframes, 8.9), 8.9)
        self.assertEqual(trim.keyframe_before(keyframes, 0.5), 0.0)
        self.assertEqual(trim.keyframe_before([], 4.0), 0.0)

    def test_stamped_time_is_original_start_plus_cut(self):
        tags = trim.stamp_tags(datetime(2026, 11, 3, 19, 16, 1, tzinfo=PLUS_4), 8.9)
        self.assertEqual(tags, {trim.APPLE_DATE: "2026-11-03T19:16:09.900+0400",
                                "creation_time": "2026-11-03T15:16:09.900000Z"})
        # TeachLens reads the new start back from those tags.
        recording = from_probe({"format": {"duration": "60", "tags": tags}, "streams": []})
        self.assertEqual(recording.start_source, trim.APPLE_DATE)
        self.assertEqual(recording.local_start,
                         datetime(2026, 11, 3, 19, 16, 9, 900000, tzinfo=PLUS_4))

    def test_command_copies_streams_and_drops_old_metadata(self):
        tags = {"creation_time": "2026-11-03T15:16:09.900000Z"}
        command = trim.ffmpeg_command(Path("in.MP4"), Path("out.MP4"), 8.9, 51.1, tags)
        self.assertEqual(command[command.index("-ss") + 1], "8.900000")
        self.assertEqual(command[command.index("-t") + 1], "51.100000")
        self.assertIn("copy", command)
        self.assertEqual(command[command.index("-map_metadata") + 1], "-1")
        self.assertIn("creation_time=2026-11-03T15:16:09.900000Z", command)
        to_end = trim.ffmpeg_command(Path("in.MP4"), Path("out.MP4"), 0.0, None, tags)
        self.assertNotIn("-t", to_end)

    def test_time_arguments(self):
        self.assertEqual(trim.parse_time("4:37"), 277.0)
        self.assertEqual(trim.parse_time("8.5"), 8.5)


if __name__ == "__main__":
    unittest.main()
