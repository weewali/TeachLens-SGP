"""Human review of a lesson's detections (src/reporting/review_sheets.py).

    python scripts/review_sheets.py make  --name lesson_01            # images + sheet.csv to fill in
    python scripts/review_sheets.py score --name lesson_01            # precision per indicator; updates facts.json

The sheets go to outputs/validation/<lesson name>/. Their images come from the review videos and show real people:
keep them local. After scoring, run scripts/report_lesson.py again so the report uses the checked detections.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_lesson import find_lesson  # noqa: E402
from src.reporting.review_sheets import make_sheets, score, update_facts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["make", "score"])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--lesson", type=Path, help="the lesson folder")
    which.add_argument("--name", help="the lesson name given to analyze_lesson.py (newest folder)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--samples", type=int, default=20, help="moments per camera (make)")
    args = parser.parse_args()
    lesson = args.lesson or find_lesson(args.name, args.output_dir)
    name = args.name or lesson.name.split("_", 2)[-1]
    folder = args.output_dir / "validation" / name
    if args.action == "make":
        sheet = make_sheets(lesson, folder, args.samples)
        print(f"Fill in the 'human' column of {sheet} (see README.txt there), then run the score step.")
        return 0
    results = score(folder / "sheet.csv")
    (folder / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    update_facts(lesson / "facts.json", results)
    for indicator, entry in sorted(results.items()):
        print(f"{indicator:<20} reviewed {entry['reviewed']:>3}  agreement {entry['agreement']:.2f}  "
              f"precision {entry['precision'] if entry['precision'] is not None else '-'}")
    print("facts.json updated. Run scripts/report_lesson.py again to rewrite the report with these results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
