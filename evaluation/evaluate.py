"""Accuracy evaluation of TeachLens (report/evaluation_plan.md). Run from this checkout's root:

    python -m evaluation.evaluate make --name test_set --lessons sgp_A4_t1_lesson sgp_A5_lesson sgp_A6_lesson sgp_B1_lesson
    python -m evaluation.evaluate score --folder outputs/evaluation/<stamp>_test_set
    python -m evaluation.evaluate stitch --name all_scenarios --all
    python -m evaluation.evaluate simulate --name power

make     blind labelling frames (frames/, items.csv); no pipeline labels are shown
score    compares labels.csv with the saved pipeline events -> metrics.json and tables
stitch   puts several lessons on one real-time line (gaps left empty) and runs the lag analysis on it
simulate false-positive rate and power of the lag test on simulated lessons of several lengths
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

from evaluation import OUTPUT_ROOT, PIPELINE_ROOT

LOGGER = logging.getLogger("evaluate")


def new_folder(name: str) -> Path:
    folder = OUTPUT_ROOT / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def lesson_folders(names: list[str] | None, use_all: bool) -> list[tuple[str, Path]]:
    from src.utils.output_layout import FOLDER_RE, find_lesson
    root = PIPELINE_ROOT / "outputs"
    if use_all:
        names = sorted({FOLDER_RE.match(p.name).group(2) for p in (root / "lessons").iterdir()
                        if p.is_dir() and FOLDER_RE.match(p.name)})
    found = []
    for name in names or []:
        folder = find_lesson(root, name)
        if folder is None:
            LOGGER.warning("%s: no lesson folder", name)
            continue
        found.append((name, folder))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--name", required=True)
    make.add_argument("--lessons", nargs="+", required=True)
    make.add_argument("--every", type=float, default=10.0, help="seconds between sampled moments")
    score = sub.add_parser("score")
    score.add_argument("--folder", type=Path, required=True)
    stitch = sub.add_parser("stitch")
    stitch.add_argument("--name", required=True)
    stitch.add_argument("--lessons", nargs="+")
    stitch.add_argument("--all", action="store_true")
    simulate = sub.add_parser("simulate")
    simulate.add_argument("--name", required=True)
    simulate.add_argument("--reps", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "make":
        from evaluation.frames import make_frames, write_csv
        lessons = lesson_folders(args.lessons, False)
        folder = new_folder(args.name)
        items = make_frames(lessons, folder, args.every)
        write_csv(folder / "items.csv", items)
        (folder / "evaluation.json").write_text(json.dumps(
            {"lessons": {n: str(f) for n, f in lessons}, "every_seconds": args.every, "frames": len(items)},
            indent=2), encoding="utf8")
        print(f"{len(items)} frames in {folder}")
    elif args.command == "score":
        from evaluation.metrics import score_folder
        print(json.dumps(score_folder(args.folder), indent=2))
    elif args.command == "stitch":
        from evaluation.stitch import stitch_lessons
        lessons = lesson_folders(args.lessons, args.all)
        print(json.dumps(stitch_lessons(lessons, new_folder(args.name)), indent=2))
    elif args.command == "simulate":
        from evaluation.simulate import run_simulation
        print(json.dumps(run_simulation(new_folder(args.name), args.reps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
