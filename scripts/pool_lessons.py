"""Test the fusion hypotheses over several analyzed lessons at once (within lessons).

One short lesson rarely has enough windows, and a teacher who sits for a
whole lesson cannot be compared with themselves standing in that lesson.
Pooling compares within each lesson and adds the lessons up (src/analysis/
lag_analysis.pooled_analyze). Reads each lesson's fusion/windows.csv (no
video or model); writes outputs/pooled/<YYYY-MM-DD_HHMM>_<name>/pooled.json and
facts.json (for the LLM report stage).

Examples (repository root, TeachLens environment):
    python scripts/pool_lessons.py --name scenarios --lessons sgp_A2_lesson sgp_A3_lesson sgp_A5_lesson
    python scripts/pool_lessons.py --name all_lessons --all
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import pool_lessons  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.output_layout import FOLDER_RE, find_lesson, lesson_paths  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

LOGGER = logging.getLogger("pool_lessons")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Name of the pooled analysis")
    parser.add_argument("--lessons", nargs="+", metavar="LESSON", help="Lesson names (newest folder of each)")
    parser.add_argument("--all", action="store_true", help="Every lesson under outputs/lessons/ (newest of each name)")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if bool(args.lessons) == args.all:
        parser.error("give --lessons or --all")
    names = args.lessons or sorted({FOLDER_RE.match(p.name).group(2) for p in (args.output_dir / "lessons").iterdir()
                                    if p.is_dir() and FOLDER_RE.match(p.name)})
    folders = []
    for name in names:
        folder = find_lesson(args.output_dir, name)
        if folder is None or not lesson_paths(folder)["fusion_windows"].is_file():
            LOGGER.warning("%s: no fused lesson found, left out", name)
            continue
        folders.append(folder)
    try:
        report = pool_lessons(folders, args.name, load_config(args.config), output_dir=args.output_dir,
                              overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print(f"{len(folders)} lessons: {', '.join(report['lessons'])}")
    for h in report["hypotheses"]:
        if "r" in h:
            comp = h.get("comparison") or {}
            print(f"  {h.get('code') or '':<8} {h['teacher_feature']:<22} -> {h['student_feature']:<14} "
                  f"{len(h['lessons_used'])} lessons, lag {h['lag_seconds']:g} s, r {h['r']:+.2f}, q {h['q']:.3f}, "
                  f"{'as expected' if h['matches_expectation'] else 'opposite'}"
                  + (f"; within-lesson gap {comp['difference_within_lessons']:+.2f}" if comp else ""))
        else:
            print(f"  {h.get('code') or '':<8} {h['teacher_feature']:<22} -> {h['student_feature']:<14} "
                  f"not tested: {h['not_tested']}")
    print(f"findings: {len(report['findings'])}; folder: {report['folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
