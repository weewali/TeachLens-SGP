"""Redo only the fusion, lag analysis, and LLM facts of a lesson from its two saved runs.

scripts/analyze_lesson.py runs this step automatically after analyzing both
cameras; use this script (or analyze_lesson.py --reuse) after changing the
fusion settings or hypotheses. No video or model is needed.

The runs are found by run ID (outputs/runs/<stamp>_<id>/ or a lesson's
student/ and instructor/ folders). Results go to the lesson's folder,
outputs/lessons/<stamp>_<name>/ (the newest one with that name, or a new
one): fusion/fusion.json, fusion/windows.csv, fusion/links.csv,
fusion/timeline.png, and facts.json.

Examples (repository root, TeachLens environment):
    python scripts/fuse_lesson.py --scenario A3 --tag lesson --overwrite
    python scripts/fuse_lesson.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1_full
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import fuse_and_write  # noqa: E402
from src.preprocessing.alignment import plan_from_manifest  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.output_layout import find_lesson, find_run, lesson_paths, new_folder  # noqa: E402
from src.utils.paths import file_in_use, project_path  # noqa: E402

LOGGER = logging.getLogger("fuse_lesson")


def print_results(report: dict, folder: Path) -> None:
    """Short console view of a fusion report."""
    print(f"{report['windows']} windows of {report['window_seconds']:g} s, "
          f"{report['start_real_time'][11:19]}-{report['end_real_time'][11:19]}")
    print("hypotheses (teacher -> later students):")
    for h in report["hypotheses"]:
        if "r" in h:
            comp = h.get("comparison") or {}
            print(f"  {h.get('code') or '':<8} {h['teacher_feature']:<22} -> {h['student_feature']:<14} "
                  f"lag {h['lag_seconds']:>4g} s r {h['r']:+.2f} q {h['q']:.3f} "
                  f"{'as expected' if h['matches_expectation'] else 'opposite'}"
                  + (f"; after {comp['outcome_with']:.2f} vs {comp['outcome_without']:.2f} otherwise"
                     if comp else ""))
        else:
            print(f"  {h.get('code') or '':<8} {h['teacher_feature']:<22} -> {h['student_feature']:<14} "
                  f"not tested: {h['not_tested']}")
    print(f"findings (q <= {report['settings']['alpha']}): {len(report['findings'])}")
    for finding in report["findings"][:8]:
        print(f"  {finding['kind']:<11} {finding['teacher_feature']} -> {finding['student_feature']} "
              f"({finding['method']}, effect {finding['effect']:+.3f}, q {finding['q']:.3f})")
    print(f"folder: {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Manifest scenario (e.g. A1); run IDs come from the manifest")
    parser.add_argument("--tag", help="Tag the runs were made with (run_scenarios.py / analyze_lesson.py --tag)")
    parser.add_argument("--take", help="Manifest take when a scenario has several complete takes")
    parser.add_argument("--student", help="Student-view run ID (instead of --scenario)")
    parser.add_argument("--instructor", help="Instructor-view run ID (instead of --scenario)")
    parser.add_argument("--name", help="Lesson name (default: sgp_<scenario>[_t<take>]_<tag>, or the student run ID)")
    parser.add_argument("--window", type=float, help="Window seconds (default fusion.window_seconds)")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace the lesson's existing fusion files")
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
        folders = {}
        for run in (student_id, instructor_id):
            folders[run] = find_run(args.output_dir, run)
            if folders[run] is None:
                raise FileNotFoundError(f"No saved run {run} under {args.output_dir}; run that camera first")
        folder = find_lesson(args.output_dir, name) or new_folder(args.output_dir, "lessons", name)
        paths = lesson_paths(folder)
        for key in ("fusion", "fusion_windows", "fusion_links", "fusion_plot", "facts"):
            if paths[key].exists() and not args.overwrite:
                raise FileExistsError(f"{paths[key]} exists; use --overwrite or another --name")
            if paths[key].exists() and file_in_use(paths[key]):
                raise PermissionError(f"{paths[key]} is open in another program; close it and run again")
        _, report = fuse_and_write(folders[student_id], folders[instructor_id], folder, load_config(args.config),
                                   title=name, header={"lesson": name, "student_run": student_id,
                                                       "instructor_run": instructor_id},
                                   window_seconds=args.window, plot=args.plot)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print_results(report, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
