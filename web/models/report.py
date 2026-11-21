"""Report model: everything the instructor's report page shows, gathered from a lesson folder's saved files.

Nothing is recomputed from video, so the page can be rebuilt at any time. Sources (outputs/lessons/<stamp>_<id>/):
    final_feedback_report.txt   positive and negative points, written by Claude and checked (absent if it failed)
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
from src.reporting.llm_report import PROMPT_VERSION  # noqa: E402  (stat_charts puts the project root on the path)

# Feedback written with earlier wording rules (with notes on how the results were measured) is not shown; the page
# offers "Try again" instead (user decision 2026-11-21: plain teaching advice only).
OUTDATED = "This feedback was written in an earlier style. Press \"Try again\" to have Claude write it again."

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
    # the line that stands for an empty side: nothing found, or every point failed the automatic checks
    empty = ("No clear positive point", "No clear negative point", "No positive point passed",
             "No negative point passed")
    return {"overview": " ".join(parts["overview"]) or None,
            "positive": [p for p in parts["positive"] if not p.startswith(empty)],
            "negative": [p for p in parts["negative"] if not p.startswith(empty)],
            "positive_empty": next((p for p in parts["positive"] if p.startswith(empty)), None),
            "negative_empty": next((p for p in parts["negative"] if p.startswith(empty)), None),
            "note": " ".join(parts["note"]) or None}


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def report_source(lesson):
    """Whether Claude wrote the feedback (disclosed with a label), or why there is none (user decisions 2026-11-20:
    Claude writes the feedback, no rule-based text, and an error is shown with its reason, except an API key problem,
    which is shown only as its code, e.g. "Code=01")."""
    status = _json(lesson / "report_status.json")
    if status.get("source") == "llm" and (lesson / "final_feedback_report.txt").exists():
        written_with = _json(lesson / "llm_output.json").get("prompt_version")
        if written_with and written_with != PROMPT_VERSION:
            return {"kind": "none", "label": None, "code": None, "outdated": True, "problem": OUTDATED}
        return {"kind": "llm", "label": "Feedback written by Claude from the measured results and checked automatically",
                "problem": None, "code": None}
    if status.get("code"):
        return {"kind": "none", "label": None, "problem": f"Code={status['code']}", "code": status["code"]}
    return {"kind": "none", "label": None, "code": None,
            "problem": status.get("reason") or "Claude has not written the feedback for this lecture yet."}


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
    final = lesson / "final_feedback_report.txt"
    feedback = parse_feedback(final.read_text(encoding="utf-8")) if final.exists() else         {"overview": None, "positive": [], "negative": [], "positive_empty": None, "negative_empty": None, "note": None}
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
        "glossary": key_terms(facts),
    }


# Words the report page and the PDF use, in plain language (user request 2026-11-20: more key terms). Observable
# postures and objects only; none of them measures attention, understanding or sleep.
REPORT_TERMS = {
    "students in view": "The students the student camera could see clearly at that moment. Percentages are shares of "
                        "them, not of everyone in the room.",
    "facing the front": "A student whose head and face point toward the front of the room, or toward a classmate who "
                        "is answering.",
    "phone seen": "A phone detected in a student's hand, or a head held down long enough that a phone is likely. A "
                  "phone can also be used for course work.",
    "head down": "A student's head tilted down, for example to read, write or look at a phone.",
    "dozing posture": "Eyes closed or the head resting down for a while, as the camera sees it. Someone thinking with "
                      "their eyes closed looks the same.",
    "hand raised": "A hand lifted with an open palm, or a wrist held clearly above the head.",
    "facing the class": "You, the instructor, turned toward the students, as seen by the instructor camera.",
    "at the board or screen": "You working at the board or the screen. Only measured when the room's board and screen "
                              "areas have been marked.",
    "out of the camera's view": "A 5-second part of the lecture in which the instructor camera saw you for less than "
                                "half of the time: you had walked out of its view, were hidden, or could not be told "
                                "apart from someone else.",
    "moment": "A stretch of time in which the student camera saw a hand raised, a phone or a dozing posture, with the "
              "seat where it happened.",
    "seat": "An anonymous place where a student usually sat, learnt from the video. Seat numbers are not names; row 1 "
            "is the row nearest the student camera.",
    "30-second average": "A share averaged over 30 seconds (six 5-second parts), so short jumps do not hide the trend.",
    "95% range": "Where the average would most likely fall if the same lecture were recorded again. A wide range means "
                 "the average is uncertain.",
}
# "unlikely to be chance" / "could easily be chance" were removed on 2026-11-21: the feedback no longer uses them
# (user decision: plain teaching advice, no statistical caveats on the report page).


def key_terms(facts):
    """The analysis's own definitions (facts.json glossary: on task, off task, seated ...) followed by the words the
    report uses (REPORT_TERMS); the data-only entries ("window", "evidence.strength" ...) are left out."""
    terms = {term: meaning for term, meaning in facts.get("glossary", {}).items()
             if "." not in term and term not in DATA_TERMS}
    for term, meaning in REPORT_TERMS.items():
        terms.setdefault(term, meaning)
    return terms
