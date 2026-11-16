"""Cut a recording without re-encoding and keep its real recording time.

Video editors (Photos, Clipchamp, FilmCraft, ...) re-export a clip with the
export time or no time at all, so TeachLens can no longer line the two
cameras up. This copies the video and first audio stream unchanged with
ffmpeg and writes the cut's own start time (the original's recording start
plus the seconds cut from the front) into the new file, as Apple's
creationdate and as creation_time.

Without re-encoding a cut can only begin on a keyframe (about every 1 s on
the student iPhone files and every 3 s on the instructor files), so the new
file starts at the last keyframe at or before --start; the stamped and
printed times are those of its actual first frame. Location and other
metadata are not copied.

Examples (repository root, TeachLens environment):
    python scripts/trim_video.py data/raw_instructor_videos/sgp_A1_instructor.MP4 --end 4:37
    python scripts/trim_video.py data/raw_student_videos/sgp_A1_student.MOV --start 0:08 --end 4:45
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import sys

# Allow direct `python scripts/trim_video.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.recording_time import read_recording_time  # noqa: E402

APPLE_DATE = "com.apple.quicktime.creationdate"


def parse_time(value: str) -> float:
    """Seconds from 90, 90.5, 1:30, or 1:02:03."""
    seconds = 0.0
    try:
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds or [HH:]MM:SS, got {value!r}") from None
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"time must not be negative: {value!r}")
    return seconds


def keyframe_before(keyframes: list[float], start: float) -> float:
    """The last keyframe time at or before start (0 if there is none)."""
    earlier = [time for time in keyframes if time <= start + 1e-3]
    return max(earlier) if earlier else 0.0


def keyframe_times(video: Path, start: float, ffprobe: str = "ffprobe") -> list[float]:
    """Keyframe times of the first video stream from 10 s before start to just after it."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-read_intervals", f"{max(0.0, start - 10):.3f}%{start + 1:.3f}",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read keyframes: {result.stderr.strip()}")
    return [float(line.strip().strip(",")) for line in result.stdout.splitlines()
            if line.strip().strip(",")]


def stamp_tags(recording_start: datetime, offset: float) -> dict[str, str]:
    """Metadata for a cut that begins offset seconds into the recording."""
    start = recording_start + timedelta(seconds=offset)
    if start.tzinfo is None:
        start = start.astimezone()  # a naive tag is this computer's local time
    local = start.isoformat(timespec="milliseconds")
    local = local[:-3] + local[-2:]  # +04:00 -> +0400, as Apple writes it
    utc = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return {APPLE_DATE: local, "creation_time": utc}


def ffmpeg_command(source: Path, output: Path, start: float, duration: float | None,
                   tags: dict[str, str], ffmpeg: str = "ffmpeg") -> list[str]:
    command = [ffmpeg, "-v", "error", "-ss", f"{start:.6f}", "-i", str(source)]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-map_metadata", "-1"]
    for key, value in tags.items():
        command += ["-metadata", f"{key}={value}"]
    return command + ["-movflags", "use_metadata_tags", str(output)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Original recording (with its recording time)")
    parser.add_argument("--start", type=parse_time, default=0.0,
                        help="Cut from about here (seconds or [HH:]MM:SS); default: video start")
    parser.add_argument("--end", type=parse_time, help="Cut until here; default: video end")
    parser.add_argument("--output", type=Path,
                        help="New file; default: <name>_cut<extension> next to the original")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    source = args.video.resolve()
    output = (args.output or source.with_name(f"{source.stem}_cut{source.suffix}")).resolve()
    if not source.is_file():
        print(f"Video not found: {source}", file=sys.stderr)
        return 1
    if output == source:
        print("The output must be a new file; the original is never replaced", file=sys.stderr)
        return 1
    if output.exists() and not args.overwrite:
        print(f"Output already exists: {output} (use --overwrite)", file=sys.stderr)
        return 1
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        print(f"ffmpeg not found ({args.ffmpeg})", file=sys.stderr)
        return 1
    try:
        recording = read_recording_time(source, args.ffprobe)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    if recording.local_start is None:
        print(f"{source.name} has no recording time to carry over; cut the original file instead",
              file=sys.stderr)
        return 1
    if args.end is not None and (args.end <= args.start
                                 or (recording.duration and args.start >= recording.duration)):
        print("Expected --start before --end and before the video's end", file=sys.stderr)
        return 1

    try:
        start = keyframe_before(keyframe_times(source, args.start, args.ffprobe), args.start)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    duration = args.end - start if args.end is not None else None
    tags = stamp_tags(recording.local_start, start)
    command = ffmpeg_command(source, output, start, duration, tags, args.ffmpeg)
    if args.overwrite:
        command.insert(1, "-y")
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"ffmpeg failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    cut = read_recording_time(output, args.ffprobe)
    print(f"Wrote {output}")
    print(f"  starts at {start:.3f} s of the original (asked for {args.start:g} s; "
          f"cuts begin on a keyframe)")
    print(f"  recording start {cut.local_start.isoformat(timespec='milliseconds')} "
          f"(original {recording.local_start.isoformat(timespec='seconds')} "
          f"from {recording.start_source}, + {start:.3f} s)")
    if cut.duration is not None:
        print(f"  duration {cut.duration:.3f} s, {cut.frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
