"""Fuse a lesson's student-view and instructor-view runs and run the lag analysis.

Reads the two saved runs (outputs/events/<id>.list_a.jsonl; no video or
model), lines them up by real time, cuts the lesson into windows, and tests
which student changes follow which teacher actions (SGP1 proposal FR 6).
scripts/analyze_lesson.py runs this step automatically after analyzing both
cameras; use this script to redo only the fusion on saved runs.

Writes outputs/reports/<name>.fusion.json (links, lagged models, peri-event
changes, findings), <name>.fusion_windows.csv, <name>.fusion_links.csv, and
outputs/visualizations/<name>.fusion.png.

Examples (repository root, TeachLens environment):
    python scripts/fuse_lesson.py --scenario A1 --tag full
    python scripts/fuse_lesson.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lesson import check_free, fuse_and_write, lesson_paths, plan_from_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

LOGGER = logging.getLogger("fuse_lesson")


def print_results(report: dict, paths: dict[str, Path]) -> None:
    """Short console view of a fusion report."""
    print(f"{report['windows']} windows of {report['window_seconds']:g} s, "
          f"{report['start_real_time'][11:19]}-{report['end_real_time'][11:19]}")
    untested = ", ".join(f"{k} ({v})" for k, v in report["untested_teacher_features"].items())
    print(f"tested teacher features: {', '.join(report['tested_teacher_features']) or 'none'}")
    print(f"too rare to test (windows present): {untested or 'none'}")
    print("strongest links (teacher -> later students):")
    for link in report["links"][:8]:
        print(f"  {link['teacher_feature']:<24} -> {link['student_feature']:<16} lag {link['lag_seconds']:>4g} s "
              f"r {link['r']:+.2f}  p {link['p_shift']:.3f}  q {link['q']:.3f}")
    print(f"findings (q <= {report['settings']['alpha']}): {len(report['findings'])}")
    for finding in report["findings"][:8]:
        print(f"  {finding['kind']:<11} {finding['teacher_feature']} -> {finding['student_feature']} "
              f"({finding['method']}, effect {finding['effect']:+.3f}, q {finding['q']:.3f})")
    for key, path in paths.items():
        if key.startswith("fusion") and path.exists():
            print(f"{key}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Manifest scenario (e.g. A1); run IDs come from the manifest")
    parser.add_argument("--tag", help="Tag the runs were made with (run_scenarios.py / analyze_lesson.py --tag)")
    parser.add_argument("--take", help="Manifest take when a scenario has several (e.g. 1)")
    parser.add_argument("--student", help="Student-view run ID (instead of --scenario)")
    parser.add_argument("--instructor", help="Instructor-view run ID (instead of --scenario)")
    parser.add_argument("--name", help="Report name (default: sgp_<scenario>[_<take>]_<tag>, or the student run ID)")
    parser.add_argument("--window", type=float, help="Window seconds (default fusion.window_seconds)")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.scenario:
            name, runs = plan_from_manifest(args.scenario, tag=args.tag, take=args.take)
            student_id, instructor_id = runs[0].video_id, runs[1].video_id
        elif args.student and args.instructor:
            name, student_id, instructor_id = args.student, args.student, args.instructor
        else:
            parser.error("give --scenario (with --tag) or both --student and --instructor")
        name = args.name or name
        paths = lesson_paths(args.output_dir, name)
        check_free([path for key, path in paths.items() if key.startswith("fusion")
                    and (args.plot or key != "fusion_plot")], args.overwrite)
        events = {run: args.output_dir / "events" / f"{run}.list_a.jsonl" for run in (student_id, instructor_id)}
        for path in events.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing run {path}; run that camera first")
        _, report = fuse_and_write(events[student_id], events[instructor_id], paths,
                                   load_config(args.config).get("fusion") or {}, title=name,
                                   header={"lesson": name, "student_run": student_id,
                                           "instructor_run": instructor_id},
                                   window_seconds=args.window, plot=args.plot)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print_results(report, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
