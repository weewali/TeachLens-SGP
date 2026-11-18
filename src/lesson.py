"""Lesson pipeline: one command from a lesson's two recordings to fused results.

Steps, each reading only what the step before saved:
1. Line the two recordings up by real time and keep the stretch both cameras
   filmed: from the files' recording-start tags and durations, or from a
   manifest row's sync window and corrected start.
2. Analyze the student camera and the instructor camera with every switched-on
   indicator (src/pipeline.analyze_video): JSONL evidence, JSON summary, and a
   review MP4 for each.
3. Optionally detect classroom objects on both cameras (detect_objects).
4. Link the cameras by timing (src/analysis/cross_camera.py).
5. Fuse the two runs into windows and run the lag analysis
   (src/analysis/fusion.py, src/analysis/lag_analysis.py), with a timeline chart.
6. Write outputs/reports/<lesson>.lesson.json listing every step and output.
Steps 4 and 5 also run alone on saved runs (scripts/link_cameras.py,
scripts/fuse_lesson.py). The pose models are imported only when a camera is
analyzed, so those scripts stay light.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from src.preprocessing.recording_time import parse_timestamp, read_recording_time
from src.utils.paths import file_in_use, project_path

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


def lesson_paths(root: Path, name: str) -> dict[str, Path]:
    """Lesson-level outputs (the camera runs keep their own paths)."""
    reports = root / "reports"
    return {"cross_camera": reports / f"{name}.cross_camera.json",
            "fusion": reports / f"{name}.fusion.json",
            "fusion_windows": reports / f"{name}.fusion_windows.csv",
            "fusion_links": reports / f"{name}.fusion_links.csv",
            "fusion_plot": root / "visualizations" / f"{name}.fusion.png",
            "lesson": reports / f"{name}.lesson.json"}


def check_free(paths: Sequence[Path], overwrite: bool) -> None:
    """Refuse to replace outputs without overwrite, or ones another program holds open."""
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite or another name")
        if path.exists() and file_in_use(path):
            raise PermissionError(f"{path} is open in another program; close it and run again")


def _publish(path: Path, write) -> None:
    """Write through a same-directory temporary file, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, data: dict[str, Any]) -> None:
    _publish(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8"))


def link_runs(student_events: Path, instructor_events: Path, config: dict[str, Any],
              window_seconds: float = 20.0) -> dict[str, Any]:
    """Cross-camera timing links of two saved runs (board target from the instructor camera's zones)."""
    from src.analysis.cross_camera import link_cameras, read_events
    from src.utils.config import load_targets
    board = load_targets(config["cameras"]["instructor_view"].get("zones"), "instructor_view").get("board")
    return link_cameras(read_events(student_events), read_events(instructor_events),
                        window_seconds=window_seconds, board=board,
                        sample_period=float(config["video"]["sample_every_seconds"]))


def fuse_and_write(student_events: Path, instructor_events: Path, paths: dict[str, Path],
                   settings: dict[str, Any], *, title: str, header: dict[str, Any] | None = None,
                   window_seconds: float | None = None, plot: bool = True) -> tuple[Any, dict[str, Any]]:
    """Fuse two saved runs, run the lag analysis, and write the fusion JSON, CSVs, and chart."""
    import pandas as pd
    from src.analysis.fusion import fuse_runs, load_events
    table, report = fuse_runs(load_events(student_events), load_events(instructor_events), settings,
                              window_seconds)
    report = {**(header or {}), **report}
    write_json(paths["fusion"], report)
    _publish(paths["fusion_windows"], lambda tmp: table.to_csv(tmp, index=False, float_format="%.4f"))
    _publish(paths["fusion_links"], lambda tmp: pd.DataFrame(report["links"]).to_csv(tmp, index=False))
    if plot:
        from src.reporting.charts import plot_fusion_timelines
        _publish(paths["fusion_plot"], lambda tmp: plot_fusion_timelines(table, title, tmp))
    return table, report


def _stamp(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="milliseconds") if moment else None


def analyze_lesson(runs: Sequence[CameraRun], name: str, config: dict[str, Any], *,
                   output_dir: Path | None = None, weight: Path | None = None, device: str | None = None,
                   overwrite: bool = False, reuse: bool = False, save_video: bool | None = None,
                   objects: bool = False, plot: bool = True) -> dict[str, Any]:
    """Run every step for one lesson and return what <name>.lesson.json records.

    reuse keeps a camera run (or object run) whose JSONL and summary already
    exist instead of analyzing it again; overwrite replaces existing outputs.
    """
    if not valid_name(name):
        raise ValueError("The lesson name may contain only letters, digits, '-' and '_'")
    cameras = {run.camera: run for run in runs}
    if set(cameras) != set(CAMERAS) or len(runs) != 2:
        raise ValueError("A lesson needs exactly one student_view and one instructor_view run")
    root = output_dir or project_path(config["paths"]["outputs"])
    paths = lesson_paths(root, name)
    # Fail before hours of analysis, not after.
    check_free([path for key, path in paths.items() if plot or key != "fusion_plot"], overwrite)
    for run in runs:
        if not run.video.is_file():
            raise FileNotFoundError(f"Video not found: {run.video}")
    began = time.perf_counter()
    recorded = []
    for number, camera in enumerate(CAMERAS, start=1):
        run = cameras[camera]
        events = root / "events" / f"{run.video_id}.list_a.jsonl"
        summary = root / "reports" / f"{run.video_id}.list_a.json"
        review = root / "debug" / f"{run.video_id}.list_a.mp4"
        started = time.perf_counter()
        if reuse and events.is_file() and summary.is_file():
            status = "reused"
            LOGGER.info("[%d/2] %s: saved run %s reused", number, camera, run.video_id)
        else:
            from src.pipeline import analyze_video
            LOGGER.info("[%d/2] %s: analyzing %s, %g-%s s -> %s", number, camera, run.video.name, run.start,
                        "end" if run.end is None else f"{run.end:g}", run.video_id)
            analyze_video(run.video, run.video_id, camera, config, weight=weight, output_dir=root,
                          device=device, overwrite=overwrite, save_video=save_video, live_preview=False,
                          start=run.start, end=run.end, path=run.path, real_start=run.real_start)
            status = "analyzed"
        entry = {"camera": camera, "video": str(run.video), "video_id": run.video_id,
                 "analyzed_start_seconds": run.start, "analyzed_end_seconds": run.end,
                 "real_start_given": _stamp(run.real_start), "status": status,
                 "seconds": round(time.perf_counter() - started, 1),
                 "events": str(events), "summary": str(summary),
                 "review_video": str(review) if review.is_file() else None}
        if objects:
            object_id = f"{run.video_id}_objects"
            object_events = root / "events" / f"{object_id}.objects.jsonl"
            object_summary = root / "reports" / f"{object_id}.objects.json"
            if reuse and object_events.is_file() and object_summary.is_file():
                object_status = "reused"
            else:
                from src.pipeline import detect_objects
                LOGGER.info("[%d/2] %s: detecting objects -> %s", number, camera, object_id)
                detect_objects(run.video, object_id, camera, config, output_dir=root, device=device,
                               overwrite=overwrite, save_video=save_video, start=run.start, end=run.end,
                               real_start=run.real_start)
                object_status = "analyzed"
            object_video = root / "debug" / f"{object_id}.objects.mp4"
            entry["objects"] = {"video_id": object_id, "status": object_status, "events": str(object_events),
                                "summary": str(object_summary),
                                "review_video": str(object_video) if object_video.is_file() else None}
        recorded.append(entry)
    student_events = root / "events" / f"{cameras['student_view'].video_id}.list_a.jsonl"
    instructor_events = root / "events" / f"{cameras['instructor_view'].video_id}.list_a.jsonl"
    header = {"lesson": name, "student_run": cameras["student_view"].video_id,
              "instructor_run": cameras["instructor_view"].video_id}
    LOGGER.info("Linking the cameras by real time")
    links = {**header, **link_runs(student_events, instructor_events, config)}
    write_json(paths["cross_camera"], links)
    LOGGER.info("Fusing the runs and running the lag analysis")
    _, fusion = fuse_and_write(student_events, instructor_events, paths, config.get("fusion") or {},
                               title=name, header=header, plot=plot)
    lesson = {
        **header,
        "finished_real_time": _stamp(datetime.now().astimezone()),
        "seconds": round(time.perf_counter() - began, 1),
        "runs": recorded,
        "cross_camera_links": len(links["links"]),
        "fusion": {key: fusion[key] for key in ("window_seconds", "windows", "start_real_time", "end_real_time",
                                                "tested_teacher_features", "untested_teacher_features",
                                                "findings")},
        "outputs": {key: str(path) for key, path in paths.items() if plot or key != "fusion_plot"},
        "limitations": fusion["limitations"],
    }
    write_json(paths["lesson"], lesson)
    return lesson
