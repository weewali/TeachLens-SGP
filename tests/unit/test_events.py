"""JSONL must be complete and must not overwrite earlier evidence by default."""

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.events import make_event, with_real_time, write_jsonl


class EventTests(unittest.TestCase):
    def test_atomic_write_and_no_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            event = make_event("video_01", "position", .2, .2, {"x": .5})
            write_jsonl(path, [event])
            original = path.read_text(encoding="utf-8")
            self.assertIn('"indicator_name": "position"', original)
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [event])
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_failed_stream_does_not_publish_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            def events():
                yield make_event("video_01", "position", .2, .2, .5)
                raise RuntimeError("simulated inference error")
            with self.assertRaises(RuntimeError):
                write_jsonl(path, events())
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_real_time_is_recording_start_plus_video_time(self):
        started = datetime(2026, 10, 26, 19, 8, 27, tzinfo=timezone(timedelta(hours=4)))
        event = make_event("video_01", "sitting_standing", 58.2, 62.8, "standing")
        timed = with_real_time(event, started)
        self.assertEqual((timed.start_real_time, timed.end_real_time),
                         ("2026-10-26T19:09:25.200+04:00", "2026-10-26T19:09:29.800+04:00"))
        self.assertEqual((timed.start_time, timed.duration), (58.2, 4.6))  # video times unchanged
        untagged = with_real_time(event, None)
        self.assertIsNone(untagged.start_real_time)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_jsonl(path, [timed, untagged])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["start_real_time"], "2026-10-26T19:09:25.200+04:00")
            self.assertIsNone(rows[1]["end_real_time"])


if __name__ == "__main__":
    unittest.main()
