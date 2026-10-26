"""JSONL must be complete and must not overwrite earlier evidence by default."""

import tempfile
import unittest
from pathlib import Path

from src.analysis.events import make_event, write_jsonl


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


if __name__ == "__main__":
    unittest.main()
