"""Link the two cameras of one lesson by real time, from saved evidence only.

There is no cross-camera identity: the cameras see different people from
different sides, and IDs reset per recording. What can be linked is timing.
A person who leaves one view and shows up in the other within a few seconds
is very probably the same person. Both runs' events carry real (wall-clock)
times, accurate to about 1 s (docs/scenario_test_videos.md), so a link
window of several seconds covers the clock error plus the walk.

Two links are reported:
- **student_to_front**: a student track leaves the student view at a frame
  edge, and an extra, non-teacher person appears in the instructor view
  within the window. How long that person stays at the board (box centre
  inside the board target's x range, below its top) gives at_board_seconds:
  a student presenting or writing at the board.
- **teacher_in_student_view**: the teacher leaves the instructor view (a
  teacher_presence interval ends), and a new track enters the student view
  within the window. That student-view track was counted as a student by the
  all-tracks role policy; the link says it is probably the teacher, with
  its time in view, so summaries can exclude it.

Nothing in either JSONL is changed. Ambiguous cases (several candidates in
the window) are reported with every candidate, not resolved.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class _Seen:
    track_id: int
    first: datetime
    last: datetime
    role: str


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def read_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _tracks(events: Iterable[dict[str, Any]]) -> dict[int, _Seen]:
    """First and last real time each track was seen (position events), with its role."""
    seen: dict[int, _Seen] = {}
    for event in events:
        if event["indicator_name"] != "position" or event.get("start_real_time") is None:
            continue
        when, track = _time(event["start_real_time"]), event["track_id"]
        old = seen.get(track)
        if old is None:
            seen[track] = _Seen(track, when, when, event["role"])
        else:
            seen[track] = _Seen(track, min(old.first, when), max(old.last, when),
                                event["role"] if event["role"] != "unknown" else old.role)
    return seen


def _at_board(events: Iterable[dict[str, Any]], track_id: int,
              board: tuple[tuple[float, float], ...] | None, period: float) -> float | None:
    """Seconds the track's box centre was in front of the board (None without a board)."""
    if not board:
        return None
    xs = [x for x, _ in board]
    top = min(y for _, y in board)
    count = 0
    for event in events:
        if event["indicator_name"] == "position" and event["track_id"] == track_id:
            cx, cy = event["value"]["box_center"]
            count += min(xs) <= cx <= max(xs) and cy >= top
    return round(count * period, 1)


def link_cameras(student_events: list[dict[str, Any]], instructor_events: list[dict[str, Any]], *,
                 window_seconds: float = 20.0, board: tuple[tuple[float, float], ...] | None = None,
                 sample_period: float = 1.0) -> dict[str, Any]:
    """Cross-camera links for one lesson (both runs need real times)."""
    students = _tracks(student_events)
    instructor = _tracks(instructor_events)
    if not students or not instructor:
        return {"links": [], "limitations": ["A run has no real times or no tracks; nothing to link."]}
    exits = [(_time(e["start_real_time"]), e["track_id"]) for e in student_events
             if e["indicator_name"] == "entry_exit" and e["value"] == "apparent_exit_at_frame_edge"
             and e.get("start_real_time")]
    entries = [(_time(e["start_real_time"]), e["track_id"]) for e in student_events
               if e["indicator_name"] == "entry_exit" and e["value"] == "apparent_entry_at_frame_edge"
               and e.get("start_real_time")]
    teacher_ends = [_time(e["end_real_time"]) for e in instructor_events
                    if e["indicator_name"] == "teacher_presence" and e.get("end_real_time")]
    instructor_start = min(seen.first for seen in instructor.values())
    links = []
    for when, track in exits:
        candidates = [seen for seen in instructor.values() if seen.role != "teacher"
                      and seen.first > instructor_start  # not already there when recording began
                      and -3 <= (seen.first - when).total_seconds() <= window_seconds]
        if candidates:
            links.append({
                "link": "student_to_front", "student_view_track": track,
                "left_student_view": when.isoformat(timespec="milliseconds"),
                "instructor_view_tracks": [{
                    "track_id": c.track_id, "appeared": c.first.isoformat(timespec="milliseconds"),
                    "last_seen": c.last.isoformat(timespec="milliseconds"),
                    "seconds_after_leaving": round((c.first - when).total_seconds(), 1),
                    "at_board_seconds": _at_board(instructor_events, c.track_id, board, sample_period),
                } for c in sorted(candidates, key=lambda c: c.first)],
                "ambiguous": len(candidates) > 1})
    for when in teacher_ends:
        candidates = [(entered, track) for entered, track in entries
                      if -3 <= (entered - when).total_seconds() <= window_seconds]
        if candidates:
            links.append({
                "link": "teacher_in_student_view", "left_instructor_view": when.isoformat(timespec="milliseconds"),
                "student_view_tracks": [{
                    "track_id": track, "entered": entered.isoformat(timespec="milliseconds"),
                    "last_seen": students[track].last.isoformat(timespec="milliseconds") if track in students else None,
                    "seconds_in_view": (round((students[track].last - students[track].first).total_seconds()
                                              + sample_period, 1) if track in students else None),
                    "seconds_after_leaving": round((entered - when).total_seconds(), 1),
                } for entered, track in sorted(candidates)],
                "ambiguous": len(candidates) > 1})
    return {"links": links, "window_seconds": window_seconds,
            "limitations": [
                "Links are timing coincidences within the window, not identity matching.",
                "Real times are whole-second recording tags plus video time: about +/-1 s per camera.",
                "Only apparent exits/entries at a frame edge are used; a person leaving through the "
                "middle of the image, or a track that fragments, is missed.",
            ]}
