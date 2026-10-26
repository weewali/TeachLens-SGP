"""Recording start time and duration from a video's container metadata.

Reads the tags with ffprobe (FFmpeg); OpenCV cannot see them. Recording
times are whole seconds written by the camera's clock: a clock-test video
from an iPhone 16 Pro showed Apple's creationdate about 0.7 s and
creation_time about 1.7 s after the first frame. They are good for the time
of day and for coarse alignment of two cameras, not for frame-level sync.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

# Preferred first: Apple's tag is the local start time with its UTC offset;
# creation_time (MP4/MOV header) is UTC.
START_TAGS = ("com.apple.quicktime.creationdate", "creation_time")
CAMERA_TAGS = (("com.apple.quicktime.make", "com.apple.quicktime.model"),
               ("com.android.manufacturer", "com.android.model"),
               ("make", "model"))
EARLIEST_VALID_YEAR = 2000  # earlier dates come from unset camera clocks


def parse_timestamp(text: str, *, naive_is_utc: bool) -> datetime | None:
    """Parse an ISO-like tag ("...Z", "...+0400", "... 14:31:20"); None if unusable.

    A time without a zone is UTC for creation_time (the MP4 header
    convention) and left naive otherwise.
    """
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    value = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", value)
    value = re.sub(r"\.(\d+)", lambda match: "." + (match.group(1) + "000000")[:6], value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.year < EARLIEST_VALID_YEAR:
        return None
    if parsed.tzinfo is None and naive_is_utc:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class RecordingTime:
    """What the container says about when and how long a video was recorded."""
    start: datetime | None
    start_source: str | None  # the tag the start came from
    other_times: dict[str, datetime] = field(default_factory=dict)  # remaining time tags
    duration: float | None = None  # seconds
    frames: int | None = None
    fps: float | None = None
    camera: str | None = None
    has_location: bool = False  # a GPS tag exists; its value is never kept

    @property
    def local_start(self) -> datetime | None:
        """Start in the camera's own offset, or a UTC tag in this computer's zone."""
        if self.start is None or self.start.tzinfo is None:
            return self.start
        return self.start.astimezone() if self.start_source == "creation_time" else self.start

    @property
    def end(self) -> datetime | None:
        if self.local_start is None or self.duration is None:
            return None
        return self.local_start + timedelta(seconds=self.duration)

    def clock_at(self, seconds: float) -> datetime | None:
        """Clock time of a position in the video (seconds from its start)."""
        if self.local_start is None:
            return None
        return self.local_start + timedelta(seconds=seconds)


def from_probe(data: dict[str, Any]) -> RecordingTime:
    """Build a RecordingTime from ffprobe JSON (format and stream sections)."""
    container = data.get("format", {})
    tags = {key.lower(): value for key, value in container.get("tags", {}).items()}
    video = next((stream for stream in data.get("streams", [])
                  if stream.get("codec_type") == "video"), {})
    if "creation_time" not in tags and "creation_time" in video.get("tags", {}):
        tags["creation_time"] = video["tags"]["creation_time"]
    times = {}
    for tag in START_TAGS:
        if tag in tags:
            parsed = parse_timestamp(tags[tag], naive_is_utc=tag == "creation_time")
            if parsed is not None:
                times[tag] = parsed
    source = next(iter(times), None)
    start = times.pop(source) if source is not None else None
    # The video track's own duration: the container can run longer (e.g. audio).
    duration = next((float(source["duration"]) for source in (video, container)
                     if source.get("duration")), None)
    frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None
    fps = None
    if frames and duration:
        fps = frames / duration
    elif "/" in str(video.get("avg_frame_rate", "")):
        numerator, denominator = (float(part) for part in video["avg_frame_rate"].split("/"))
        fps = numerator / denominator if denominator else None
    camera = next((" ".join(tags[key] for key in pair if key in tags)
                   for pair in CAMERA_TAGS if any(key in tags for key in pair)), None)
    return RecordingTime(start, source, times, duration, frames, fps, camera,
                         any("location" in key or "gps" in key for key in tags))


def probe(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Run ffprobe and return the JSON needed by from_probe()."""
    if shutil.which(ffprobe) is None and not Path(ffprobe).is_file():
        raise RuntimeError(f"ffprobe not found ({ffprobe}); install FFmpeg or pass its path")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration:format_tags:stream=codec_type,nb_frames,avg_frame_rate,duration"
         ":stream_tags=creation_time",
         "-of", "json", str(path)],
        capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def read_recording_time(path: Path, ffprobe: str = "ffprobe") -> RecordingTime:
    return from_probe(probe(path, ffprobe))


def format_clock(moment: datetime, milliseconds: bool = False) -> str:
    text = moment.strftime("%H:%M:%S")
    return text + f".{moment.microsecond // 1000:03d}" if milliseconds else text


def format_offset(moment: datetime) -> str:
    """UTC offset as UTC+04:00, or a note when the tag had no time zone."""
    if moment.utcoffset() is None:
        return "no time zone in tag"
    minutes = int(moment.utcoffset().total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def format_duration(seconds: float) -> str:
    """Seconds as HH:MM:SS.mmm."""
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours:02d}:{minutes:02d}:{rest // 1000:02d}.{rest % 1000:03d}"
