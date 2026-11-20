"""The instructor report of one analysed lesson folder (outputs/lessons/<stamp>_<lesson>/).

Runs after src.pipeline.analyze_lesson (which writes facts.json, fusion/, student/, instructor/):

1. adds the "during each activity" comparisons (src/analysis/activity_comparison.py) to facts.json as
   kind "activity_comparison" facts (replacing any added by an earlier run);
2. writes tables/: events.csv (both cameras' events, flattened) and activity_comparisons.csv;
3. writes the rule-based positive / negative points (final_feedback_report_rules.txt), spoken to the
   instructor as "you", from the facts;
4. always asks Claude to rewrite them (src/reporting/llm_report.py). The checked AI text becomes
   final_feedback_report.txt; when the rewrite fails, the rule-based text is used and report_status.json
   says why, so the web app can show it.
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
MAX_POINTS = 3
# How each student measure reads in a sentence ("..., <phrase> 72% of the time").
MEASURE_PHRASES = {
    "S_on_task": "students were on task", "S_off_task": "students were off task",
    "S_phone": "students had a phone", "S_head_down": "students had their heads down",
    "S_on_target": "students faced the front", "S_off_target": "students faced away",
    "S_talking": "students were talking", "S_sleep": "students were dozing",
    "S_writing": "students were writing", "S_looking_down": "students were looking down",
}
WORD_TO_COLUMN = {words: column for column, words in STUDENT_WORDS.items()}
STRENGTH_WORDS = {"suggestive": "This gap is unlikely to be chance.",
                  "weak": "This gap could easily be chance."}


def youify(phrase: str) -> str:
    """'the instructor was seated' -> 'you were seated' (the report speaks to the instructor)."""
    return (phrase.replace("the instructor was", "you were").replace("the instructor had their", "you had your")
            .replace("the instructor", "you"))


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


# ---------------------------------------------------------------- rule-based points

def _measure_phrase(fact: dict[str, Any]) -> str:
    column = WORD_TO_COLUMN.get(fact.get("student_measure", ""))
    return MEASURE_PHRASES.get(column, fact.get("student_measure", "the students"))


def sentence(fact: dict[str, Any]) -> str | None:
    """One plain sentence (or two) for a fact, spoken to the instructor; None for kinds not worth a point."""
    kind = fact.get("kind")
    strength = STRENGTH_WORDS.get(fact.get("evidence", {}).get("strength"), "")
    act = youify(fact.get("instructor_activity", "you were teaching"))
    if kind in ("comparison", ADDED_KIND):
        if "pct_after" in fact or "pct_during" in fact:
            during = fact.get("pct_after", fact.get("pct_during"))
            other = fact.get("pct_other_windows", fact.get("pct_other_times"))
            when = f"In the {fact['measured']} after {act}" if kind == "comparison" else f"While {act}"
            return f"{when}, {_measure_phrase(fact)} {during}% of the time, compared with {other}% otherwise. {strength}"
        if "per_minute_during" in fact:
            return (f"While {act}, students raised a hand {fact['per_minute_during']} times a minute, compared with "
                    f"{fact['per_minute_other_times']} otherwise. {strength}")
        if "per_window_after" in fact:
            return (f"In the {fact['measured']} after {act}, there were {fact['per_window_after']} hand raises per "
                    f"5 seconds, compared with {fact['per_window_other']} otherwise. {strength}")
        return None
    if kind == "instructor_behaviour_share":
        return f"{act[0].upper() + act[1:]} for {fact['share_of_lesson_pct']}% of the lecture time that could be judged."
    if kind in ("highest_on_task_stretch", "lowest_on_task_stretch"):
        word = "highest" if kind.startswith("highest") else "lowest"
        text = (f"Between {fact['span']}, {fact['students_on_task_pct']}% of the students in view were on task, the "
                f"{word} for any 30 seconds of the lecture")
        if fact.get("instructor_activity"):
            text += f", while {youify(fact['instructor_activity'])}"
        text += "."
        if word == "lowest":
            text += " Watching this part of the video may show what changed."
        return text
    if kind == "hand_raises":
        times = fact.get("example_times") or []
        example = f" (for example at {', '.join(times)})" if times else ""
        return f"Students raised their hands {fact['count']} times{example}."
    if kind == "no_participation_stretch":
        return f"Between {fact['span']} ({fact['stretch_seconds']} seconds), no student raised a hand."
    if kind == "phone_use":
        return (f"A phone was seen with {fact['students_with_phone_pct']}% of the students in view on average, and "
                f"with {fact['peak_pct']}% at {fact['peak_window']}. Phones can also be used for course work.")
    return None


PRIORITY = {"comparison": 0, ADDED_KIND: 0, "hand_raises": 1, "phone_use": 1, "highest_on_task_stretch": 2,
            "lowest_on_task_stretch": 2, "no_participation_stretch": 3, "instructor_behaviour_share": 4}


def rule_points(facts: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Up to 3 positive and 3 negative sentences: suggestive comparisons first, then counts and stretches, then
    shares; weak comparisons only to fill the list."""
    def rank(fact: dict[str, Any]) -> tuple[int, int]:
        weak = fact.get("evidence", {}).get("strength") == "weak"
        return (5 if weak else PRIORITY.get(fact.get("kind"), 6), -int(fact.get("gap_points", 0) or 0))

    chosen: dict[str, list[str]] = {"favourable": [], "unfavourable": []}
    for fact in sorted(facts.get("facts", []), key=rank):
        side = chosen.get(fact.get("direction"))
        text = sentence(fact)
        if side is not None and text and len(side) < MAX_POINTS:
            side.append(" ".join(text.split()))
    return chosen["favourable"], chosen["unfavourable"]


def render_rules(facts: dict[str, Any]) -> str:
    positive, negative = rule_points(facts)
    return llm_report.render({"positive_points": [{"text": t} for t in positive],
                              "negative_points": [{"text": t} for t in negative]}, facts)


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


def report_lesson(lesson: Path, *, model: str = llm_report.DEFAULT_MODEL, fake_response: dict | None = None,
                  use_cache: bool = True, caller=llm_report.call_claude, log=print) -> dict[str, Any]:
    """Writes the report files of one lesson folder; returns report_status.json's content."""
    lesson = Path(lesson)
    facts_path = lesson / "facts.json"
    if not facts_path.exists():
        raise FileNotFoundError(f"{facts_path} is missing: run scripts/analyze_lesson.py first.")
    log("[6/7] Writing the report")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    table, window = lesson_windows(lesson)
    rows = compare(table) if table is not None and not table.empty else []
    facts = add_activity_facts(facts, activity_facts(rows, window))
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    tables = lesson / "tables"
    tables.mkdir(exist_ok=True)
    events_table(lesson).to_csv(tables / "events.csv", index=False)
    pd.DataFrame(rows).to_csv(tables / "activity_comparisons.csv", index=False)
    rules = render_rules(facts)
    (lesson / "final_feedback_report_rules.txt").write_text(rules, encoding="utf-8")

    log("[7/7] Plain-language rewrite (AI)")
    result = llm_report.rewrite(facts, lesson, model=model, fake_response=fake_response, use_cache=use_cache,
                                caller=caller)
    text = result["text"] if result["source"] == "llm" else rules
    (lesson / "final_feedback_report.txt").write_text(text, encoding="utf-8")
    status = {"source": result["source"], "reason": result["reason"], "model": llm_report.MODELS[model][0],
              "facts": len(facts["facts"]), "activity_comparisons": len(rows)}
    (lesson / "report_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if result["source"] == "llm":
        log("The AI rewrite passed every check.")
    else:
        log(f"The AI rewrite was not used: {result['reason']} The rule-based points were kept.")
    return status
