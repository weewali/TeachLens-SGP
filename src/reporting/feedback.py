"""The instructor report of one analysed lesson folder (outputs/lessons/<stamp>_<lesson>/).

Runs after src.pipeline.analyze_lesson (which writes facts.json, fusion/, student/, instructor/):

1. adds the "during each activity" comparisons (src/analysis/activity_comparison.py) to facts.json as
   kind "activity_comparison" facts (replacing any added by an earlier run);
2. writes tables/: events.csv (both cameras' events, flattened) and activity_comparisons.csv;
3. has Claude read the facts and write the positive and negative points in plain language for the instructor
   (src/reporting/llm_report.py). Claude writes the feedback; there is no rule-based text (user decision
   2026-11-20). If Claude cannot write it (no API key, no network, an answer that fails the checks twice), no
   feedback is written and report_status.json says why, so the web app can show the reason and offer to try again.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.activity_comparison import compare
from src.reporting import llm_report
from src.reporting.facts import STUDENT_WORDS, TEACHER_WORDS, pct

ADDED_KIND = "activity_comparison"
MAX_ACTIVITY_FACTS = 6
MIN_GAP_POINTS = 5            # smaller gaps in a share are not worth a point


# ---------------------------------------------------------------- facts

def activity_facts(rows: list[dict[str, Any]], window_seconds: float) -> list[dict[str, Any]]:
    """facts.json entries (without ids) for the comparisons worth reporting: suggestive ones first, then the
    largest gaps; at most MAX_ACTIVITY_FACTS."""
    facts = []
    for r in rows:
        if r["p_hac"] is None:
            continue
        fact = {"direction": "favourable" if r["favourable"] else "unfavourable", "kind": ADDED_KIND,
                "instructor_activity": TEACHER_WORDS.get(r["teacher_feature"], r["teacher_feature"]),
                "student_measure": STUDENT_WORDS.get(r["measure"], r["measure"]),
                "measured": "at the same time (not later)",
                "windows": r["windows_with"], "seconds": int(r["windows_with"] * window_seconds),
                "other_windows": r["windows_without"]}
        if r["count"]:
            per_minute = 60 / window_seconds
            during, other = round(r["with"] * per_minute, 1), round(r["without"] * per_minute, 1)
            if during == other:
                continue
            fact.update({"per_minute_during": during, "per_minute_other_times": other})
            gap = abs(during - other)
        else:
            gap = abs(round(100 * r["difference"]))
            if gap < MIN_GAP_POINTS:
                continue
            fact.update({"pct_during": pct(r["with"]), "pct_other_times": pct(r["without"]), "gap_points": gap})
        p = r["p_hac"]
        fact["evidence"] = {"strength": "suggestive" if r["suggestive"] else "weak",
                            "test_p": "< 0.01" if p < 0.01 else f"= {p:.2f}",
                            "notes": ["The test allows for neighbouring windows being alike and for the "
                                      f"{r['tests']} comparisons made."]}
        fact["_rank"] = (0 if r["suggestive"] else 1, -gap)
        facts.append(fact)
    facts.sort(key=lambda f: f["_rank"])
    for fact in facts:
        del fact["_rank"]
    return facts[:MAX_ACTIVITY_FACTS]


def add_activity_facts(facts: dict[str, Any], added: list[dict[str, Any]]) -> dict[str, Any]:
    """facts.json with the activity comparisons replaced by `added`; ids are renumbered F1, F2, ..."""
    kept = [f for f in facts.get("facts", []) if f.get("kind") != ADDED_KIND]
    merged = []
    for number, fact in enumerate(kept + added, start=1):
        merged.append({"id": f"F{number}", **{k: v for k, v in fact.items() if k != "id"}})
    return {**facts, "facts": merged}


# ---------------------------------------------------------------- tables

def events_table(lesson: Path) -> pd.DataFrame:
    """Both cameras' events.jsonl as one table (value kept as text; dict values as JSON)."""
    rows = []
    for camera in ("instructor", "student"):
        path = lesson / camera / "events.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            value = event.get("value")
            rows.append({"camera": camera, "indicator": event.get("indicator_name"), "track_id": event.get("track_id"),
                         "role": event.get("role"), "zone": event.get("zone"),
                         "value": json.dumps(value) if isinstance(value, (dict, list)) else value,
                         "start_time": event.get("start_time"), "end_time": event.get("end_time"),
                         "start_real_time": event.get("start_real_time"), "coverage": event.get("coverage")})
    return pd.DataFrame(rows, columns=["camera", "indicator", "track_id", "role", "zone", "value", "start_time",
                                       "end_time", "start_real_time", "coverage"])


# ---------------------------------------------------------------- the whole stage

def lesson_windows(lesson: Path) -> tuple[pd.DataFrame | None, float]:
    path = lesson / "fusion" / "windows.csv"
    fusion_path = lesson / "fusion" / "fusion.json"
    window = 5.0
    if fusion_path.exists():
        window = float(json.loads(fusion_path.read_text(encoding="utf-8")).get("window_seconds", window))
    return (pd.read_csv(path) if path.exists() else None), window


def prepare_facts(lesson: Path, log=print) -> dict[str, Any]:
    """Steps 1 and 2: the comparisons added to facts.json, and the CSV tables. Returns the facts."""
    facts_path = lesson / "facts.json"
    if not facts_path.exists():
        raise FileNotFoundError(f"{facts_path} is missing: run scripts/analyze_lesson.py first.")
    log("[6/7] Preparing the facts and comparisons")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    table, window = lesson_windows(lesson)
    rows = compare(table) if table is not None and not table.empty else []
    facts = add_activity_facts(facts, activity_facts(rows, window))
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    tables = lesson / "tables"
    tables.mkdir(exist_ok=True)
    events_table(lesson).to_csv(tables / "events.csv", index=False)
    pd.DataFrame(rows).to_csv(tables / "activity_comparisons.csv", index=False)
    return facts


def report_lesson(lesson: Path, *, model: str = llm_report.DEFAULT_MODEL, fake_response: dict | None = None,
                  use_cache: bool = True, caller=llm_report.call_claude, log=print) -> dict[str, Any]:
    """Writes the report files of one lesson folder; returns report_status.json's content."""
    lesson = Path(lesson)
    facts = prepare_facts(lesson, log)
    log("[7/7] Writing the feedback with Claude")
    result = llm_report.rewrite(facts, lesson, model=model, fake_response=fake_response, use_cache=use_cache,
                                caller=caller)
    final = lesson / "final_feedback_report.txt"
    (lesson / "final_feedback_report_rules.txt").unlink(missing_ok=True)   # left by older versions
    if result["source"] == "llm":
        final.write_text(result["text"], encoding="utf-8")
        if result.get("left_out"):
            log(f"Claude wrote the feedback; {result['left_out']} point(s) that failed the automatic checks were left "
                "out, and every point shown passed.")
        else:
            log("Claude wrote the feedback and it passed every automatic check.")
    else:
        final.unlink(missing_ok=True)   # never show feedback from an earlier run as if it were this one's
        if result.get("code"):   # key or billing problem: the log is on the progress page, so only the code
            log(f"Error, Code={result['code']}")
        else:
            log(f"Claude could not write the feedback: {result['reason']}")
    status = {"source": result["source"], "reason": result["reason"], "code": result.get("code"),
              "left_out": result.get("left_out", 0), "model": llm_report.MODELS[model][0], "facts": len(facts["facts"])}
    (lesson / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
