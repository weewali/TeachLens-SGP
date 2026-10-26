"""Recording-time parsing from ffprobe-shaped JSON; no ffprobe or video needed."""

from datetime import datetime, timedelta, timezone
import unittest

from src.preprocessing.recording_time import (
    format_duration, format_offset, from_probe, parse_timestamp,
)

PLUS_4 = timezone(timedelta(hours=4))

# Shaped like an iPhone .MOV; the location value is a dummy.
IPHONE = {
    "streams": [{"codec_type": "video", "nb_frames": "200", "avg_frame_rate": "40000/1667",
                 "duration": "8.333333",
                 "tags": {"creation_time": "2026-10-26T14:31:20.000000Z"}},
                {"codec_type": "audio", "nb_frames": "393"}],
    "format": {"duration": "8.4", "tags": {  # container longer than the video track
        "creation_time": "2026-10-26T14:31:20.000000Z",
        "com.apple.quicktime.creationdate": "2026-10-26T18:31:19+0400",
        "com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 16 Pro",
        "com.apple.quicktime.location.ISO6709": "+00.0000+000.0000/"}},
}


class TimestampTests(unittest.TestCase):
    def test_tag_formats(self):
        self.assertEqual(parse_timestamp("2026-10-26T14:31:20.000000Z", naive_is_utc=True),
                         datetime(2026, 10, 26, 14, 31, 20, tzinfo=timezone.utc))
        self.assertEqual(parse_timestamp("2026-10-26T18:31:19+0400", naive_is_utc=False),
                         datetime(2026, 10, 26, 18, 31, 19, tzinfo=PLUS_4))
        self.assertEqual(parse_timestamp("2026-10-26 14:31:20", naive_is_utc=True).tzinfo,
                         timezone.utc)
        self.assertIsNone(parse_timestamp("2026-10-26 14:31:20", naive_is_utc=False).tzinfo)
        self.assertEqual(parse_timestamp("2026-10-26T14:31:20.5Z", naive_is_utc=True).microsecond,
                         500000)
        self.assertIsNone(parse_timestamp("not a date", naive_is_utc=True))
        self.assertIsNone(parse_timestamp("1970-01-01T00:00:00Z", naive_is_utc=True))  # unset clock

    def test_formatting(self):
        self.assertEqual(format_duration(8.333333), "00:00:08.333")
        self.assertEqual(format_duration(3725.5), "01:02:05.500")
        self.assertEqual(format_offset(datetime(2026, 1, 1, tzinfo=PLUS_4)), "UTC+04:00")


class ProbeTests(unittest.TestCase):
    def test_iphone_prefers_local_creationdate(self):
        recording = from_probe(IPHONE)
        self.assertEqual(recording.start_source, "com.apple.quicktime.creationdate")
        self.assertEqual(recording.local_start, datetime(2026, 10, 26, 18, 31, 19, tzinfo=PLUS_4))
        self.assertEqual(recording.other_times["creation_time"],
                         datetime(2026, 10, 26, 14, 31, 20, tzinfo=timezone.utc))
        self.assertEqual((recording.frames, round(recording.fps, 2)), (200, 24.0))
        self.assertEqual(recording.camera, "Apple iPhone 16 Pro")
        self.assertTrue(recording.has_location)
        self.assertEqual(recording.clock_at(8.125),
                         datetime(2026, 10, 26, 18, 31, 27, 125000, tzinfo=PLUS_4))
        self.assertAlmostEqual((recording.end - recording.local_start).total_seconds(), 8.333333)

    def test_utc_only_and_missing_tags(self):
        utc_only = from_probe({"format": {"duration": "10", "tags": {
            "creation_time": "2026-10-26T14:31:20Z"}}, "streams": []})
        self.assertEqual(utc_only.start_source, "creation_time")
        self.assertEqual(utc_only.local_start, utc_only.start.astimezone())  # this computer's zone
        bare = from_probe({"format": {"duration": "10"}, "streams": [
            {"codec_type": "video", "avg_frame_rate": "30000/1001"}]})
        self.assertIsNone(bare.start)
        self.assertIsNone(bare.clock_at(1.0))
        self.assertAlmostEqual(bare.fps, 29.97, places=2)
        self.assertFalse(bare.has_location)


if __name__ == "__main__":
    unittest.main()
