"""Recording-time parsing from ffprobe-shaped JSON; no ffprobe or video needed."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import struct
import tempfile
import unittest

from src.preprocessing.recording_time import (
    UDTA_DATE, format_duration, format_offset, from_probe, parse_timestamp, read_udta_date,
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

    def test_udta_date_beats_export_creation_time(self):
        # A Photos re-export: creation_time is the export, udta/date the recording.
        recording = from_probe({"format": {"duration": "10", "tags": {
            "creation_time": "2026-11-05T22:00:28.000000Z",
            UDTA_DATE: "2026-11-03T19:16:01+0400"}}, "streams": []})
        self.assertEqual(recording.start_source, UDTA_DATE)
        self.assertEqual(recording.local_start, datetime(2026, 11, 3, 19, 16, 1, tzinfo=PLUS_4))
        self.assertIn("creation_time", recording.other_times)
        both = dict(IPHONE, format=dict(IPHONE["format"], tags=dict(
            IPHONE["format"]["tags"], **{UDTA_DATE: "2026-10-26T18:31:00+0400"})))
        self.assertEqual(from_probe(both).start_source, "com.apple.quicktime.creationdate")

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


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


class UdtaDateTests(unittest.TestCase):
    def write(self, data: bytes) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        handle.write(data)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_reads_date_box_after_media_data(self):
        moov = box(b"moov", box(b"mvhd", bytes(100)) + box(b"udta", box(b"loci", bytes(20))
                                                             + box(b"date", b"2026-11-03T19:16:01+0400")))
        # 64-bit mdat size, as in files over 4 GB.
        mdat = struct.pack(">I4sQ", 1, b"mdat", 16 + 64) + bytes(64)
        path = self.write(box(b"ftyp", b"mp42" + bytes(4)) + mdat + moov)
        self.assertEqual(read_udta_date(path), "2026-11-03T19:16:01+0400")

    def test_missing_or_broken_boxes_give_none(self):
        no_udta = box(b"ftyp", b"qt  ") + box(b"moov", box(b"mvhd", bytes(100)))
        self.assertIsNone(read_udta_date(self.write(no_udta)))
        truncated = struct.pack(">I4s", 5000, b"moov") + bytes(20)
        self.assertIsNone(read_udta_date(self.write(truncated)))
        self.assertIsNone(read_udta_date(self.write(b"not a video")))


if __name__ == "__main__":
    unittest.main()
