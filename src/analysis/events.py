"""Canonical timestamped evidence records; no report prose lives here."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


@dataclass(frozen=True)
class Event:
    video_id: str
    track_id: int | None
    role: str
    indicator_name: str
    start_time: float
    end_time: float
    duration: float
    value: Any
    confidence: float | None
    coverage: str
    zone: str | None = None
    # Wall-clock times (ISO 8601 with UTC offset); null without a recording-time tag.
    start_real_time: str | None = None
    end_real_time: str | None = None

    def __post_init__(self) -> None:
        if not self.video_id or not self.indicator_name:
            raise ValueError("Events require a video_id and indicator_name")
        if self.role not in {"teacher", "student", "unknown", "both", "classroom", "recording"}:
            raise ValueError(f"Invalid role: {self.role}")
        if self.coverage not in {"adequate", "limited", "unknown"}:
            raise ValueError(f"Invalid coverage: {self.coverage}")
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError("Invalid event timestamps")
        if abs(self.duration - (self.end_time - self.start_time)) > 0.001:
            raise ValueError("Event duration must match timestamps")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between zero and one")


def make_event(
    video_id: str,
    indicator_name: str,
    start: float,
    end: float,
    value: Any,
    *,
    track_id: int | None = None,
    role: str = "classroom",
    confidence: float | None = None,
    coverage: str = "adequate",
    zone: str | None = None,
) -> Event:
    start, end = round(start, 3), round(end, 3)
    return Event(video_id, track_id, role, indicator_name, start, end,
                 round(end - start, 3), value, confidence, coverage, zone)


def with_real_time(event: Event, recording_start: datetime | None) -> Event:
    """Add wall-clock times: the recording start plus the event's video times.

    recording_start comes from the video's metadata, which holds whole seconds
    and was 0.7-0.95 s after the first frame on tested iPhones, so real times
    are good to about a second. Without it the fields stay null.
    """
    if recording_start is None:
        return event

    def clock(seconds: float) -> str:
        return (recording_start + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")
    return replace(event, start_real_time=clock(event.start_time), end_real_time=clock(event.end_time))


def write_jsonl(path: Path, events: Iterable[Event], *, overwrite: bool = False) -> None:
    """Stream to a temporary file, then publish only a complete JSONL run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; choose a new video ID or --overwrite")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            for event in events:
                stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
