"""CLI: turns a lagged-analysis output into the instructor's positive / negative points. No video is processed.

The input is normally the team's fusion layer report (scripts/fuse_lesson.py -> outputs/reports/<name>.fusion.json);
its <name>.fusion_links.csv and the older provisional formats are accepted too (src/reporting/lag_facts.py).

    python scripts/report.py --lag-results outputs/reports/sgp_A1.fusion.json --dry-run          # prompt size and cost, calls nothing
    python scripts/report.py --lag-results tests/fixtures/fusion_example.json --rules-only       # no language model
    python scripts/report.py --lag-results outputs/reports/sgp_A1.fusion.json --model opus       # real call (about 4 cents)
"""

import argparse
from pathlib import Path
import sys

# Allow direct `python scripts/report.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import lag_facts, llm_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lag-results", required=True, type=Path,
                        help="the fusion layer's report (<name>.fusion.json or .fusion_links.csv) or an older results file")
    parser.add_argument("--lecture-id", help="name for this lecture's report folder (default: the results file name "
                                             "without .fusion)")
    parser.add_argument("--out-dir", type=Path, help="where to write the report points (default: outputs/reports/<lecture-id>)")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "configs" / "report_vocabulary.yaml")
    parser.add_argument("--model", choices=sorted(llm_text.MODELS), default="opus",
                        help="opus (default, about 4 cents per report), sonnet (about 2 cents) or haiku (about 0.1 cent)")
    parser.add_argument("--dry-run", action="store_true", help="show the prompt size and cost and write a preview; call nothing")
    parser.add_argument("--rules-only", action="store_true", help="write the rule-based points without a language model")
    parser.add_argument("--fake-response", type=Path, help="use this saved JSON answer instead of calling Claude (tests the checks)")
    parser.add_argument("--no-cache", action="store_true", help="ignore a saved answer for identical facts")
    parser.add_argument("--keep-facts", action="store_true", help="also save the facts the model saw (lag_facts.json)")
    parser.add_argument("--you", action="store_true",
                        help="write the rule-based points to the instructor as 'you' (as the web report does)")
    args = parser.parse_args()

    stem = args.lag_results.stem                      # sgp_A1.fusion -> sgp_A1, sgp_A1.fusion_links -> sgp_A1
    for suffix in (".fusion_links", ".fusion"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    lecture_id = args.lecture_id or stem
    out_dir = args.out_dir or ROOT / "outputs" / "reports" / lecture_id
    try:
        rows, metadata = lag_facts.load_results(args.lag_results)
        facts = lag_facts.build_facts(rows, lag_facts.load_vocabulary(args.vocabulary), metadata)
    except (OSError, ValueError) as error:
        sys.exit(f"Could not read the lagged-analysis results: {error}")
    if facts["not_interpreted"]:
        print(f"Left out (no plain-language name in {args.vocabulary.name}): {', '.join(facts['not_interpreted'])}")

    if args.dry_run:
        user = llm_text.build_user_message(facts)
        tokens, dollars = llm_text.estimate_cost(user, args.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        preview = out_dir / "llm_prompt_preview.txt"
        preview.write_text(llm_text.SYSTEM_PROMPT + "\n\n---\n\n" + user, encoding="utf-8")
        print(f"Dry run: {len(facts['facts'])} facts, about {tokens} input tokens; one call with {args.model} costs about "
              f"${dollars:.3f} at list price. Nothing was sent. Prompt preview: {preview}")
        return

    fake = None
    if args.fake_response:
        import json
        fake = json.loads(args.fake_response.read_text(encoding="utf-8"))
    points = llm_text.generate(facts, out_dir, model_alias=args.model, use_llm=not args.rules_only, fake_response=fake,
                               use_cache=not args.no_cache, keep_facts=args.keep_facts,
                               subject="you" if args.you else "the instructor")
    for attempt in points["attempts"]:
        print(f"Attempt ({attempt['source']}): " + ("passed every check" if not attempt["problems"] else
                                                     f"{len(attempt['problems'])} problem(s): " + "; ".join(attempt["problems"])))
    print(f"Report points ({points['source']}) written to {out_dir / llm_text.TEXT_FILE}")
    print()
    print(llm_text.render_text(points))


if __name__ == "__main__":
    main()
