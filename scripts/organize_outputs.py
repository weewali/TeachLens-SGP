"""Move outputs from the old flat folders into one dated folder per run and per lesson.

Before 2026-11-19 every run wrote to outputs/events/<id>.list_a.jsonl,
outputs/reports/<id>.list_a.json, and outputs/debug/<id>.list_a.mp4, and
lessons to outputs/reports/<lesson>.* and outputs/visualizations/. This
script moves those files into the layout of src/utils/output_layout.py:

    lessons/<stamp>_<lesson>/   lesson.json, cross_camera.json, fusion/ (fusion.json, windows.csv, links.csv,
                                timeline.png), student/ and instructor/ (the two camera runs)
    runs/<stamp>_<run>/         events.jsonl, summary.json, review.mp4, run.json (and objects.* files)
    logs/                       batch logs
    archive/                    everything else (old prototypes, backups, stale temporary files)

The stamp is the old files' modification time. Files are only moved (renamed
on the same disk), never deleted or rewritten; a run.json is added to each
camera run. Without --apply it only prints the plan. Afterwards, refresh a
lesson's fusion, facts.json, and side-by-side video with
`python scripts/analyze_lesson.py --scenario <X> --tag <tag> --reuse --overwrite`.

Examples (repository root, TeachLens environment):
    python scripts/organize_outputs.py            # show the plan
    python scripts/organize_outputs.py --apply
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.output_layout import CAMERA_FOLDERS, folder_name  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

RUN_SUFFIXES = {".list_a.jsonl": ("events", "events.jsonl"), ".list_a.json": ("reports", "summary.json"),
                ".list_a.mp4": ("debug", "review.mp4"), ".objects.jsonl": ("events", "objects.jsonl"),
                ".objects.json": ("reports", "objects_summary.json"), ".objects.mp4": ("debug", "objects_review.mp4")}
LESSON_SUFFIXES = {".lesson.json": ("reports", "lesson.json"), ".cross_camera.json": ("reports", "cross_camera.json"),
                   ".fusion.json": ("reports", "fusion/fusion.json"),
                   ".fusion_windows.csv": ("reports", "fusion/windows.csv"),
                   ".fusion_links.csv": ("reports", "fusion/links.csv"),
                   ".fusion.png": ("visualizations", "fusion/timeline.png")}
OLD_FOLDERS = ("events", "reports", "debug", "visualizations")
NEW_FOLDERS = ("lessons", "runs", "pooled", "logs", "archive")


def mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def run_files(root: Path, run_id: str) -> dict[str, Path]:
    """Old files of one run: new name -> old path."""
    found = {}
    for suffix, (folder, new_name) in RUN_SUFFIXES.items():
        path = root / folder / f"{run_id}{suffix}"
        if path.is_file():
            found[new_name] = path
    return found


def sample_timing(events: Path) -> tuple[float | None, float | None, int]:
    """First sample time, sample period, and samples, from the per-sample events of a run."""
    times = set()
    with events.open(encoding="utf-8") as stream:
        for line in stream:
            if '"teacher_presence_sample"' in line or '"visible_student_count"' in line \
                    or '"object_count"' in line:
                times.add(round(float(json.loads(line)["start_time"]), 3))
    times = sorted(times)
    if len(times) < 2:
        return (times[0] if times else None), None, len(times)
    return times[0], round((times[-1] - times[0]) / (len(times) - 1), 6), len(times)


def run_info(run_id: str, files: dict[str, Path], video: str | None) -> dict[str, dict]:
    """run.json (and objects_run.json) contents for a migrated run."""
    infos = {}
    for kind, events_name, summary_name, review_name, info_name in (
            ("pose", "events.jsonl", "summary.json", "review.mp4", "run.json"),
            ("objects", "objects.jsonl", "objects_summary.json", "objects_review.mp4", "objects_run.json")):
        if events_name not in files:
            continue
        summary = json.loads(files[summary_name].read_text(encoding="utf-8")) if summary_name in files else {}
        first, period, samples = sample_timing(files[events_name])
        infos[info_name] = {
            "run_id": run_id, "analysis": kind, "camera": summary.get("camera_id"), "video": video,
            "analyzed_start_seconds": summary.get("analyzed_start_seconds"),
            "analyzed_end_seconds": summary.get("analyzed_end_seconds"),
            "recording_start_real_time": summary.get("recording_start_real_time"),
            "recording_start_source": summary.get("recording_start_source"),
            "created_real_time": mtime(files[events_name]).astimezone().isoformat(timespec="seconds"),
            "review_video": review_name if review_name in files else None,
            "first_sample_seconds": first, "sample_period": period, "samples": samples,
            "migrated": "scripts/organize_outputs.py (files moved from the old flat folders)"}
    return infos


def plan(root: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, dict]]]:
    """(moves old -> new, run.json files to write) for everything under root."""
    moves, infos = [], []
    claimed: set[Path] = set()
    run_ids = sorted({re.sub(r"\.(list_a|objects)\.(jsonl|json|mp4)$", "", p.name)
                      for folder in ("events", "reports", "debug") if (root / folder).is_dir()
                      for p in (root / folder).iterdir()
                      if p.is_file() and re.search(r"\.(list_a|objects)\.(jsonl|json|mp4)$", p.name)})
    # Lessons first, so their camera runs move into the lesson folder.
    for lesson_file in sorted((root / "reports").glob("*.lesson.json")) if (root / "reports").is_dir() else []:
        name = lesson_file.name[:-len(".lesson.json")]
        lesson = json.loads(lesson_file.read_text(encoding="utf-8"))
        target = root / "lessons" / folder_name(name, mtime(lesson_file))
        for suffix, (folder, new_name) in LESSON_SUFFIXES.items():
            path = root / folder / f"{name}{suffix}"
            if path.is_file():
                moves.append((path, target / new_name))
                claimed.add(path)
        for run in lesson.get("runs", []):
            files = run_files(root, run["video_id"])
            sub = target / CAMERA_FOLDERS[run["camera"]]
            for new_name, path in files.items():
                if path not in claimed:
                    moves.append((path, sub / new_name))
                    claimed.add(path)
            for info_name, info in run_info(run["video_id"], files, run.get("video")).items():
                infos.append((sub / info_name, info))
    for run_id in run_ids:
        files = {k: v for k, v in run_files(root, run_id).items() if v not in claimed}
        if not files:
            continue
        stamp_source = files.get("events.jsonl") or files.get("objects.jsonl") or next(iter(files.values()))
        target = root / "runs" / folder_name(run_id, mtime(stamp_source))
        for new_name, path in files.items():
            moves.append((path, target / new_name))
            claimed.add(path)
        for info_name, info in run_info(run_id, files, None).items():
            infos.append((target / info_name, info))
    # Everything else: logs, old fusion prototypes, backups, stale temporary files.
    for folder in OLD_FOLDERS:
        for path in sorted((root / folder).iterdir()) if (root / folder).is_dir() else []:
            if path in claimed or path.name == ".gitkeep":
                continue
            if path.suffix == ".log":
                moves.append((path, root / "logs" / path.name))
            elif path.name.endswith(".tmp"):
                moves.append((path, root / "archive" / "stale_temporary_files" / path.name))
            else:
                moves.append((path, root / "archive" / f"old_{folder}" / path.name))
    for path in sorted(root.iterdir()):
        if path.name in OLD_FOLDERS + NEW_FOLDERS or path.name == ".gitkeep":
            continue
        moves.append((path, root / "archive" / path.name))
    return moves, infos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--apply", action="store_true", help="Move the files (default: only print the plan)")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    moves, infos = plan(root)
    targets = [new for _, new in moves]
    clashes = [new for new in targets if new.exists() or targets.count(new) > 1]
    for old, new in moves:
        print(f"{old.relative_to(root)}  ->  {new.relative_to(root)}")
    print(f"{len(moves)} moves, {len(infos)} run.json files to add")
    if clashes:
        print(f"Refusing: {len(set(clashes))} targets already exist or repeat, e.g. {clashes[0]}", file=sys.stderr)
        return 1
    if not args.apply:
        print("Nothing moved; run again with --apply")
        return 0
    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    for path, info in infos:
        if not path.exists():
            path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for folder in NEW_FOLDERS:
        (root / folder).mkdir(exist_ok=True)
        (root / folder / ".gitkeep").touch()
    for folder in OLD_FOLDERS:
        old = root / folder
        if old.is_dir() and all(p.name == ".gitkeep" for p in old.iterdir()):
            try:
                for p in old.iterdir():
                    p.unlink()
                old.rmdir()
            except OSError as exc:  # e.g. OneDrive holding the folder; it is empty, remove it later
                print(f"Could not remove the empty folder {old}: {exc}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
