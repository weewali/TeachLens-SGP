"""Show when a video was recorded (HH:MM:SS) from its metadata, using ffprobe."""

import argparse
from pathlib import Path
import sys

# Allow direct `python scripts/recording_time.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.recording_time import (  # noqa: E402
    RecordingTime, format_clock, format_duration, format_offset, read_recording_time,
)


def describe(path: Path, recording: RecordingTime, at: list[float]) -> list[str]:
    lines = [str(path)]
    if recording.camera:
        lines.append(f"  Camera            {recording.camera}")
    start = recording.local_start
    if start is None:
        lines.append("  Recording start   not found (no usable time tag; edited, re-encoded, "
                     "or downloaded files often lose it)")
    else:
        lines.append(f"  Recording start   {format_clock(start)}  ({start:%Y-%m-%d}, "
                     f"{format_offset(start)})  from {recording.start_source}")
        if recording.end is not None:
            lines.append(f"  Recording end     {format_clock(recording.end)}  (start + duration)")
    if recording.duration is not None:
        rate = (f"{recording.frames} frames, " if recording.frames else "") + (
            f"{recording.fps:.2f} fps" if recording.fps else "frame rate unknown")
        lines.append(f"  Duration          {format_duration(recording.duration)}  ({rate})")
    for tag, moment in recording.other_times.items():
        shown = moment.astimezone(start.tzinfo) if start is not None and start.tzinfo else moment
        lines.append(f"  Also tagged       {tag} {format_clock(shown)}")
    if start is not None:
        lines.append("  Precision         whole seconds from the camera clock; about 1 s from the "
                     "true first frame")
    for seconds in at:
        clock = recording.clock_at(seconds)
        lines.append(f"  At {seconds:9.3f} s    {format_clock(clock, True) if clock else 'unknown'}")
    if recording.has_location:
        lines.append("  Location          GPS location tag present (not shown); strip it before sharing")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+", help="One or more video files")
    parser.add_argument("--at", type=float, nargs="+", default=[],
                        help="Also show the clock time at these video positions (seconds)")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable or path")
    args = parser.parse_args()
    starts = []
    for path in args.videos:
        if not path.is_file():
            print(f"{path}: file not found", file=sys.stderr)
            return 1
        try:
            recording = read_recording_time(path, args.ffprobe)
        except RuntimeError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        print("\n".join(describe(path, recording, args.at)) + "\n")
        starts.append((path, recording.start))
    if len(starts) > 1 and all(start is not None and start.tzinfo for _, start in starts):
        first_path, first = starts[0]
        print(f"Start times relative to {first_path.name} (metadata only, about 1 s precision):")
        for path, start in starts[1:]:
            print(f"  {path.name}: {(start - first).total_seconds():+.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
