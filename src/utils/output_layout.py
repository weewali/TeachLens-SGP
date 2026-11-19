"""Where every output goes: one folder per run and per lesson, named by date, time, and name.

User request 2026-11-19: the flat events/, reports/, debug/ folders made it
impossible to tell which files belong together. Layout under outputs/:

    lessons/<YYYY-MM-DD_HHMM>_<lesson>/     everything about one lesson (scripts/analyze_lesson.py)
        lesson.json                           what ran, on which ranges, timings, every path
        student/  instructor/                 the two camera runs (RUN_FILES)
        cross_camera.json                     timing links between the cameras
        fusion/fusion.json, windows.csv, links.csv, timeline.png
        facts.json                            input for the LLM report stage (llm_report.py)
        side_by_side.mp4                      both review videos lined up by real time
    runs/<YYYY-MM-DD_HHMM>_<run>/           a camera run made on its own (analyze.py,
                                            run_scenarios.py, detect_objects.py)
    pooled/<YYYY-MM-DD_HHMM>_<name>/        analyses over several lessons
    logs/                                   batch logs
    archive/                                old files kept for reference

The date and time are when the folder was created (local time). A run ID or
lesson name is found again by its folder name (or, for a camera run inside a
lesson, by the run_id in its run.json); the newest folder wins.
"""

from datetime import datetime
import json
from pathlib import Path
import re

STAMP_FORMAT = "%Y-%m-%d_%H%M"
FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{4})_(.+)$")
# A camera run's files (pose analysis and, optionally, object detection).
RUN_FILES = {"events": "events.jsonl", "summary": "summary.json", "review": "review.mp4", "info": "run.json",
             "objects_events": "objects.jsonl", "objects_summary": "objects_summary.json",
             "objects_review": "objects_review.mp4", "objects_info": "objects_run.json"}
LESSON_FILES = {"lesson": "lesson.json", "cross_camera": "cross_camera.json",
                "fusion": "fusion/fusion.json", "fusion_windows": "fusion/windows.csv",
                "fusion_links": "fusion/links.csv", "fusion_plot": "fusion/timeline.png",
                "facts": "facts.json", "side_by_side": "side_by_side.mp4"}
CAMERA_FOLDERS = {"student_view": "student", "instructor_view": "instructor"}
KINDS = ("lessons", "runs", "pooled")


def stamp(moment: datetime | None = None) -> str:
    """Local date and time as YYYY-MM-DD_HHMM."""
    return (moment or datetime.now()).strftime(STAMP_FORMAT)


def folder_name(name: str, moment: datetime | None = None) -> str:
    return f"{stamp(moment)}_{name}"


def run_paths(folder: Path) -> dict[str, Path]:
    return {key: folder / name for key, name in RUN_FILES.items()}


def lesson_paths(folder: Path) -> dict[str, Path]:
    return {key: folder / name for key, name in LESSON_FILES.items()}


def read_info(path: Path) -> dict:
    """A run.json (or any small JSON file) as a dict; {} when missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _named(parent: Path, name: str) -> list[tuple[str, Path]]:
    """(stamp, folder) for folders under parent named <stamp>_<name>."""
    if not parent.is_dir():
        return []
    found = []
    for child in parent.iterdir():
        match = FOLDER_RE.match(child.name)
        if child.is_dir() and match and match.group(2) == name:
            found.append((match.group(1), child))
    return sorted(found)


def find_lesson(root: Path, name: str, kind: str = "lessons") -> Path | None:
    """Newest lessons/<stamp>_<name> (or pooled/...) folder, or None."""
    found = _named(root / kind, name)
    return found[-1][1] if found else None


def find_run(root: Path, run_id: str) -> Path | None:
    """Newest folder holding camera run run_id: runs/<stamp>_<id>, or a lesson's
    student/instructor folder whose run.json (or objects_run.json) names it."""
    candidates = _named(root / "runs", run_id)
    lessons = root / "lessons"
    if lessons.is_dir():
        for lesson in lessons.iterdir():
            match = FOLDER_RE.match(lesson.name)
            if not (lesson.is_dir() and match):
                continue
            for sub in CAMERA_FOLDERS.values():
                for info_name in (RUN_FILES["info"], RUN_FILES["objects_info"]):
                    if read_info(lesson / sub / info_name).get("run_id") == run_id:
                        candidates.append((match.group(1), lesson / sub))
    return sorted(candidates)[-1][1] if candidates else None


def events_file(root: Path, run_id: str) -> Path | None:
    """The JSONL of a run: events.jsonl, or objects.jsonl for an object-detection run."""
    folder = find_run(root, run_id)
    if folder is None:
        return None
    files = run_paths(folder)
    if read_info(files["objects_info"]).get("run_id") == run_id and files["objects_events"].is_file():
        return files["objects_events"]
    for key in ("events", "objects_events"):
        if files[key].is_file():
            return files[key]
    return None


def new_folder(root: Path, kind: str, name: str, moment: datetime | None = None) -> Path:
    """root/<kind>/<stamp>_<name> (not created yet)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown output kind {kind}; choose from {KINDS}")
    return root / kind / folder_name(name, moment)


def run_folder(root: Path, run_id: str, overwrite: bool) -> Path:
    """Where a stand-alone camera run goes: its existing folder (only with
    overwrite) or a new runs/<stamp>_<id>."""
    existing = find_run(root, run_id)
    if existing is not None and not overwrite:
        raise FileExistsError(f"Run {run_id} already exists in {existing}; choose a new ID or --overwrite")
    return existing or new_folder(root, "runs", run_id)
