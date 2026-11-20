"""Line up a lesson's two recordings by real time, before any analysis.

A lesson is filmed by two cameras (student_view, instructor_view) that start
at different moments. From each file's recording-start tag and duration (or
a scenario manifest's corrected start and sync window) this module works out
the stretch both cameras recorded and which seconds of each file to analyze.
The result is a pair of CameraRun plans; src/pipeline.analyze_lesson() runs
them.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.preprocessing.recording_time import parse_timestamp, read_recording_time
from src.utils.paths import project_path

LOGGER = logging.getLogger(__name__)
CAMERAS = ("student_view", "instructor_view")
DEFAULT_MANIFEST = project_path("data/manifests/sgp_scenario_videos.csv")


@dataclass(frozen=True)
class CameraRun:
    """One camera of a lesson: source file, run ID, analyzed range, real start."""

    camera: str
    video: Path
    video_id: str
    start: float = 0.0                  # seconds from the file's start
    end: float | None = None            # None = the file's end
    real_start: datetime | None = None  # None = read the file's recording-start tag

    @property
    def path(self) -> str:
        """Overlay trail for the review video: the teacher's path on the instructor camera."""
        return "teacher" if self.camera == "instructor_view" else "off"


def valid_name(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "-_" for c in name)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Rows of a scenario manifest (data/manifests/sgp_scenario_videos.csv), checked."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for number, row in enumerate(rows, start=2):
        missing = [key for key in ("video_id", "scenario", "camera", "file") if not row.get(key)]
        if missing:
            raise ValueError(f"{path} line {number}: missing {', '.join(missing)}")
        if row["camera"] not in CAMERAS:
            raise ValueError(f"{path} line {number}: unknown camera {row['camera']!r}")
    return rows


def _duration_from_file(video: Path) -> float | None:
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    return count / fps if fps > 0 and count > 0 else None


def recording_window(video: Path, real_start: datetime | None = None) -> tuple[datetime, float, str]:
    """Real start, duration in seconds, and where the start came from ("given" or the tag)."""
    recording = read_recording_time(video)
    start = real_start or recording.local_start
    if start is None:
        raise ValueError(f"{video.name} has no recording-time tag; give its recording start "
                         "(--student-start/--instructor-start)")
    duration = recording.duration or _duration_from_file(video)
    if not duration:
        raise ValueError(f"Cannot read the duration of {video.name}")
    return start, float(duration), "given" if real_start else str(recording.start_source)


def shared_range(starts: dict[str, datetime], durations: dict[str, float]) -> tuple[datetime, float]:
    """Real start and length (seconds) of the stretch every camera recorded."""
    begin = max(starts.values())
    finish = min(starts[c] + timedelta(seconds=durations[c]) for c in starts)
    length = (finish - begin).total_seconds()
    if length <= 0:
        raise ValueError("The two recordings do not overlap in real time; check their recording starts")
    return begin, length


def limit_range(start: float, end: float | None, lesson_start: float = 0.0,
                lesson_end: float | None = None) -> tuple[float, float | None]:
    """Keep lesson seconds [lesson_start, lesson_end), counted from start, of a file's [start, end)."""
    new_start = start + lesson_start
    new_end = start + lesson_end if lesson_end is not None else end
    if end is not None and new_end is not None:
        new_end = min(new_end, end)
    if new_end is not None and new_end <= new_start:
        raise ValueError("The requested --start is at or after the end of the shared recording")
    return round(new_start, 3), (round(new_end, 3) if new_end is not None else None)


def plan_from_videos(student_video: Path, instructor_video: Path, name: str, *,
                     student_start: datetime | None = None, instructor_start: datetime | None = None,
                     synced: bool = True, lesson_start: float = 0.0,
                     lesson_end: float | None = None) -> list[CameraRun]:
    """Both cameras' runs for two video files; synced keeps only the shared real-time stretch.

    Starts come from the files' recording-start tags unless given (an iPhone
    clip trimmed on the phone keeps the uncut start, so give the real one).
    """
    if not valid_name(name):
        raise ValueError("The lesson name may contain only letters, digits, '-' and '_'")
    if not synced and (lesson_start or lesson_end is not None):
        raise ValueError("--start/--end count from the shared start; leave out --full to use them")
    videos = {"student_view": student_video, "instructor_view": instructor_video}
    given = {"student_view": student_start, "instructor_view": instructor_start}
    starts, durations = {}, {}
    for camera, video in videos.items():
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        starts[camera], durations[camera], source = recording_window(video, given[camera])
        LOGGER.info("%s: %s starts %s (%s), %.1f s", camera, video.name,
                    starts[camera].isoformat(timespec="milliseconds"), source, durations[camera])
    begin, length = shared_range(starts, durations) if synced else (None, None)
    runs = []
    for camera in CAMERAS:
        if synced:
            offset = (begin - starts[camera]).total_seconds()
            start, end = offset, offset + length
            if end >= durations[camera] - 1e-3:
                end = None
        else:
            start, end = 0.0, None
        start, end = limit_range(start, end, lesson_start, lesson_end)
        runs.append(CameraRun(camera, videos[camera], f"{name}_{camera.split('_')[0]}", start, end,
                              given[camera]))
    return runs


def plan_from_manifest(scenario: str, *, tag: str | None = None, take: str | None = None,
                       manifest: Path = DEFAULT_MANIFEST, data_root: Path | None = None,
                       synced: bool = True, lesson_start: float = 0.0,
                       lesson_end: float | None = None) -> tuple[str, list[CameraRun]]:
    """Lesson name and both cameras' runs for a manifest scenario.

    Run IDs are the manifest video IDs plus _<tag>; the lesson name is the
    student ID without "_student" (e.g. sgp_A1, sgp_A4_t1) plus _<tag>.
    synced uses each row's sync window and corrected recording start.
    """
    if not synced and (lesson_start or lesson_end is not None):
        raise ValueError("--start/--end count from the shared start; leave out --full to use them")
    takes: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in read_manifest(manifest):
        if row["scenario"].upper() == scenario.upper() and (take is None or row.get("take") == str(take)):
            takes.setdefault(row.get("take") or "", {}).setdefault(row["camera"], []).append(row)
    # A take is a lesson when both cameras recorded it (A4 take 2 has no student video).
    complete = {key: cams for key, cams in takes.items() if set(cams) == set(CAMERAS)}
    if not complete:
        raise ValueError(f"Scenario {scenario} has no take with both a student_view and an instructor_view "
                         "recording in the manifest")
    if len(complete) > 1:
        raise ValueError(f"Scenario {scenario} has several complete takes ({', '.join(sorted(complete))}); "
                         "choose one with --take")
    by_camera = next(iter(complete.values()))
    if any(len(found) > 1 for found in by_camera.values()):
        raise ValueError(f"Scenario {scenario} has several recordings per camera; give each a take number")
    root = data_root or project_path(".")
    runs = []
    for camera in CAMERAS:
        row = by_camera[camera][0]
        file = Path(row["file"])
        start = float(row.get("sync_start_seconds") or 0) if synced else 0.0
        end = float(row["sync_end_seconds"]) if synced and row.get("sync_end_seconds") else None
        start, end = limit_range(start, end, lesson_start, lesson_end)
        real_start = (parse_timestamp(row["recording_start"], naive_is_utc=False)
                      if row.get("recording_start") else None)
        runs.append(CameraRun(camera, file if file.is_absolute() else root / file,
                              f"{row['video_id']}_{tag}" if tag else row["video_id"], start, end, real_start))
    student_id = by_camera["student_view"][0]["video_id"]
    base = student_id[:-len("_student")] if student_id.endswith("_student") else f"sgp_{scenario}"
    return (f"{base}_{tag}" if tag else base), runs
