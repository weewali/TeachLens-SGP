"""Analyze whole lessons with one command: both cameras, camera links, and lag analysis.

Input: a lesson's two recordings (student camera and instructor camera), as
two video files or as scenarios of data/manifests/sgp_scenario_videos.csv.
Steps per lesson (src/lesson.py): line the cameras up by real time and keep
the stretch both filmed; analyze each camera with every switched-on
indicator; optionally detect objects; link the cameras by timing; fuse the
two runs and run the lag analysis.

Outputs (under outputs/, local and ignored by Git):
    events/<run>.list_a.jsonl, reports/<run>.list_a.json, debug/<run>.list_a.mp4  per camera
    reports/<lesson>.cross_camera.json   timing links between the cameras
    reports/<lesson>.fusion.json, .fusion_windows.csv, .fusion_links.csv,
    visualizations/<lesson>.fusion.png   lag analysis and timelines
    reports/<lesson>.lesson.json         what ran, on which ranges, and every output path
The review MP4s hold original classroom footage: keep them local.

Examples (repository root, TeachLens environment):
    python scripts/analyze_lesson.py --student-video data/raw_student_videos/sgp_A1_student.MOV ^
        --instructor-video data/raw_instructor_videos/sgp_A1_instructor.MOV --name lesson_01
    python scripts/analyze_lesson.py --scenario A1 --tag lesson
    python scripts/analyze_lesson.py --all-scenarios --tag lesson
    python scripts/analyze_lesson.py --scenario B1 --tag try --start 2:00 --end 4:00 --no-save-video
"""

import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import parse_real_time, parse_seconds  # noqa: E402
from src.lesson import (DEFAULT_MANIFEST, analyze_lesson, plan_from_manifest,  # noqa: E402
                        plan_from_videos, read_manifest)
from src.pipeline import AnalysisStopped  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

LOGGER = logging.getLogger("analyze_lesson")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    files = parser.add_argument_group("a lesson from two video files")
    files.add_argument("--student-video", type=Path, help="Student-camera recording")
    files.add_argument("--instructor-video", type=Path, help="Instructor-camera recording")
    files.add_argument("--name", help="Lesson name (letters, digits, - and _); runs are <name>_student "
                                      "and <name>_instructor")
    files.add_argument("--student-start", type=parse_real_time,
                       help="Real recording start of the student video (ISO 8601) if its tag is wrong")
    files.add_argument("--instructor-start", type=parse_real_time,
                       help="Real recording start of the instructor video, e.g. for a clip trimmed on an iPhone")
    scenarios = parser.add_argument_group("lessons from the scenario manifest")
    scenarios.add_argument("--scenario", action="append", help="Scenario to run (repeatable, e.g. A1)")
    scenarios.add_argument("--all-scenarios", action="store_true",
                           help="Every scenario with both cameras in the manifest")
    scenarios.add_argument("--take", help="Take, when a scenario has several complete takes")
    scenarios.add_argument("--tag", help="Suffix for the run IDs and lesson names, e.g. lesson")
    scenarios.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    scenarios.add_argument("--data-root", type=Path, default=project_path("."),
                           help="Folder the manifest's file paths start from (default: repository root)")
    parser.add_argument("--start", type=parse_seconds, default=0.0,
                        help="Analyze from this many seconds after the shared start ([HH:]MM:SS)")
    parser.add_argument("--end", type=parse_seconds, help="Stop this many seconds after the shared start")
    parser.add_argument("--full", action="store_true",
                        help="Analyze whole files instead of only the stretch both cameras recorded")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", default=str(project_path("configs/sgp_room.yaml")),
                        help="Room override with zones/targets (default: the scenario room, "
                             "configs/sgp_room.yaml); 'none' for no override")
    parser.add_argument("--weight", type=Path, help="Local YOLO26 pose .pt; never auto-downloaded")
    parser.add_argument("--output-dir", type=Path, help="Root for events/, reports/, debug/, visualizations/")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None,
                        help="Write the review MP4s (default: review.save_video, on)")
    parser.add_argument("--objects", action="store_true",
                        help="Also run the isolated COCO object detector on both cameras")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the fusion timeline PNG")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    existing.add_argument("--reuse", action="store_true",
                          help="Keep camera runs that already exist and redo only the later steps")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned runs and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from_files = args.student_video or args.instructor_video
    if from_files and (args.scenario or args.all_scenarios):
        parser.error("give either two video files or scenarios, not both")
    if from_files and not (args.student_video and args.instructor_video and args.name):
        parser.error("two video files need --student-video, --instructor-video, and --name")
    if not from_files and not (args.scenario or args.all_scenarios):
        parser.error("give --student-video/--instructor-video/--name, --scenario, or --all-scenarios")
    lesson_range = {"synced": not args.full, "lesson_start": args.start, "lesson_end": args.end}
    try:
        if from_files:
            lessons = [(args.name, plan_from_videos(args.student_video.resolve(), args.instructor_video.resolve(),
                                                    args.name, student_start=args.student_start,
                                                    instructor_start=args.instructor_start, **lesson_range))]
        else:
            rows = read_manifest(args.manifest)
            wanted = args.scenario or sorted({row["scenario"] for row in rows
                                              if row["scenario"].lower() != "unknown"})
            lessons = [plan_from_manifest(scenario, tag=args.tag, take=args.take, manifest=args.manifest,
                                          data_root=args.data_root.resolve(), **lesson_range)
                       for scenario in wanted]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    for name, runs in lessons:
        for run in runs:
            window = f"{run.start:g}-{'end' if run.end is None else f'{run.end:g}'} s"
            print(f"{name:<22} {run.camera:<16} {run.video_id:<34} {window:<18} {run.video.name}")
    missing = [str(run.video) for _, runs in lessons for run in runs if not run.video.is_file()]
    if missing:
        LOGGER.error("Missing video files (check --data-root): %s", ", ".join(missing))
        return 1
    if args.dry_run:
        return 0

    local = None if str(args.local_config).lower() == "none" else Path(args.local_config)
    config = load_config(args.config, local)
    output_dir = args.output_dir.resolve() if args.output_dir else None
    results = []
    for number, (name, runs) in enumerate(lessons, start=1):
        LOGGER.info("Lesson %d/%d: %s", number, len(lessons), name)
        started = time.perf_counter()
        try:
            lesson = analyze_lesson(runs, name, config, output_dir=output_dir,
                                    weight=args.weight.resolve() if args.weight else None, device=args.device,
                                    overwrite=args.overwrite, reuse=args.reuse, save_video=args.save_video,
                                    objects=args.objects, plot=args.plot)
        except AnalysisStopped as exc:
            LOGGER.warning("%s stopped: %s", name, exc)
            return 1
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError, RuntimeError) as exc:
            LOGGER.error("%s failed: %s", name, exc)
            results.append((name, f"failed: {exc}", None))
            continue
        results.append((name, "done", lesson))
        LOGGER.info("%s done in %.1f min", name, (time.perf_counter() - started) / 60)
    print()
    for name, status, lesson in results:
        if lesson is None:
            print(f"{name:<22} {status}")
            continue
        fusion = lesson["fusion"]
        kinds = ", ".join(f"{f['kind']} {f['teacher_feature']} -> {f['student_feature']}"
                          for f in fusion["findings"][:3]) or "none"
        print(f"{name:<22} {fusion['windows']:>4} windows, {lesson['cross_camera_links']} camera links, "
              f"{len(fusion['findings'])} findings ({kinds}); {lesson['seconds'] / 60:.1f} min")
        print(f"{'':<22} {lesson['outputs']['lesson']}")
    return 1 if any(lesson is None for _, _, lesson in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
