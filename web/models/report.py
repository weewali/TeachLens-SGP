"""Report model: everything the instructor's report page shows, gathered from a lesson folder's saved files.

Nothing is recomputed from video, so the page can be rebuilt at any time. Sources (outputs/lessons/<stamp>_<id>/):
    final_feedback_report.txt   positive and negative points (the checked AI rewrite, or the rule-based points)
    report_status.json          who wrote them, and why the AI rewrite was not used when it failed
    facts.json                  the key terms (its glossary) and the lesson length
    fusion/windows.csv, student/events.jsonl, student/summary.json   the charts and the moments
Not shown (user decisions): the statistics (fusion/fusion.json, lesson.json) and the limitations, which stay in the
lesson folder for the team.
"""
import json

from web.models import stat_charts
from web.models.charts import ensure_charts
from web.models.lecture import describe_activity

MOMENT_LABELS = {"hand_raise": "Hand raised", "phone_use": "Phone seen", "dozing": "Dozing posture"}
# glossary entries that describe the data, not words the page uses
DATA_TERMS = {"window", "later", "lesson time", "students on task (percent)"}


def fmt_time(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def parse_feedback(text):
    """{"overview", "positive", "negative", "note"} from the report text ('positive points:' / 'negative points:')."""
    parts = {"overview": [], "positive": [], "negative": [], "note": []}
    section = "overview"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "positive points:":
            section = "positive"
        elif stripped.lower() == "negative points:":
            section = "negative"
        elif stripped.startswith("Note:"):
            section = "note"
            parts["note"].append(stripped)
        elif stripped.startswith("- ") and section in ("positive", "negative"):
            parts[section].append(stripped[2:])
        elif stripped and section in ("overview", "note"):
            parts[section].append(stripped)
    empty = ("No clear positive point", "No clear negative point")
    return {"overview": " ".join(parts["overview"]) or None,
            "positive": [p for p in parts["positive"] if not p.startswith(empty)],
            "negative": [p for p in parts["negative"] if not p.startswith(empty)],
            "note": " ".join(parts["note"]) or None}


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def report_source(lesson):
    """Who wrote the points. AI text is disclosed; when the AI rewrite was not used, the reason is shown (user
    decision 2026-11-20: tell me if an error occurred, with why)."""
    status = _json(lesson / "report_status.json")
    if status.get("source") == "llm":
        return {"kind": "llm", "label": "Feedback written with AI from the measured results and checked automatically",
                "problem": None}
    return {"kind": "rules", "label": None,
            "problem": status.get("reason") or "The AI step has not run for this lecture."}


def moments(lecture, windows_table=None, window=5.0):
    """Every hand-raise, phone and dozing moment, with what the instructor was doing then."""
    table, window = (windows_table, window) if windows_table is not None else stat_charts.windows(lecture)
    by_start = {}
    if table is not None:
        for row in table.to_dict("records"):
            by_start[int(round(float(row["lesson_seconds"]) / window))] = stat_charts.activity_of(row, window)
    rows = []
    for m in stat_charts.moments(lecture):
        key = by_start.get(int(m["start"] // window))
        rows.append({"time": fmt_time(m["start"]), "end": fmt_time(m["end"]), "seconds": m["start"],
                     "duration": int(round(m["end"] - m["start"])), "seat": m["seat"], "kind": m["kind"],
                     "behaviour": MOMENT_LABELS[m["kind"]],
                     "activity": stat_charts.PHRASES[key].removeprefix("you were ") if key else "unknown"})
    return rows


def build_report(lecture):
    """Everything report.html needs, as plain Python data."""
    lesson = lecture.lesson_dir
    feedback = parse_feedback((lesson / "final_feedback_report.txt").read_text(encoding="utf-8"))
    facts = _json(lesson / "facts.json")
    found = moments(lecture)
    context = lecture.context()
    return {
        "lecture": lecture,
        "context": context,
        "activities": [describe_activity(a) for a in context.get("activities") or []
                       if isinstance(a, dict) and isinstance(a.get("minutes"), int)],
        "analysed_at": lecture.analysed_at(),
        "source": report_source(lesson),
        "feedback": feedback,
        "length": (facts.get("lecture") or {}).get("duration"),
        "timeline": stat_charts.timeline(lecture),
        "comparison": stat_charts.comparison(lecture),
        "charts": ensure_charts(lecture),
        "moments": found,
        "moment_kinds": [(key, label) for key, label in MOMENT_LABELS.items() if any(m["kind"] == key for m in found)],
        # the key terms: words the page uses, in plain language (keys like "evidence.strength" are left out)
        "glossary": {term: meaning for term, meaning in facts.get("glossary", {}).items()
                     if "." not in term and term not in DATA_TERMS},
    }
