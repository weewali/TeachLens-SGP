"""Write the instructor report of an analysed lesson (after scripts/analyze_lesson.py).

Adds the "during each activity" comparisons to the lesson's facts.json, writes tables/ (events.csv,
activity_comparisons.csv) and the rule-based points, then always asks Claude to rewrite the points
(src/reporting/feedback.py, src/reporting/llm_report.py). If the rewrite fails (no API key, no network, text
that fails the checks), the rule-based points are kept and report_status.json says why.

A real rewrite sends the lesson's facts.json (numbers only: no video, images or names) to Anthropic and costs
about 4 US cents with the default model.

Examples (repository root):
    python scripts/report_lesson.py --name lesson_01
    python scripts/report_lesson.py --lesson outputs/lessons/2026-11-20_1700_lesson_01 --model haiku
    python scripts/report_lesson.py --name lesson_01 --fake-response answer.json   # test the checks, no call
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import llm_report  # noqa: E402
from src.reporting.feedback import report_lesson  # noqa: E402


def find_lesson(name: str, output_dir: Path) -> Path:
    """The newest outputs/lessons/<stamp>_<name> folder."""
    found = sorted((output_dir / "lessons").glob(f"*_{name}"), key=lambda p: p.name)
    if not found:
        raise SystemExit(f"No lesson folder ending in _{name} under {output_dir / 'lessons'}.")
    return found[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--lesson", type=Path, help="the lesson folder")
    which.add_argument("--name", help="the lesson name given to analyze_lesson.py (newest folder)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", choices=sorted(llm_report.MODELS), default=llm_report.DEFAULT_MODEL)
    parser.add_argument("--fake-response", type=Path, help="a saved JSON answer to check instead of calling Claude")
    parser.add_argument("--no-cache", action="store_true", help="ignore a saved answer for identical facts")
    args = parser.parse_args()
    lesson = args.lesson or find_lesson(args.name, args.output_dir)
    fake = json.loads(args.fake_response.read_text(encoding="utf-8")) if args.fake_response else None
    status = report_lesson(lesson, model=args.model, fake_response=fake, use_cache=not args.no_cache,
                           log=lambda line: print(line, flush=True))
    print(f"Report: {lesson / 'final_feedback_report.txt'} (written by {'the AI' if status['source'] == 'llm' else 'the rules'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
