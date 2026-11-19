"""Report model: everything the instructor's report page shows, gathered from a lecture's saved results.

Nothing is recomputed from video, so the page can be rebuilt at any time. Sources (all written by Main Code.py,
except llm_output.json, which main code/llm_report.py writes when the Claude step ran):
    reports/final_feedback_report.txt   positive and negative points (rule-based, or Claude's checked rewrite)
    reports/evaluation_result.json      lecture facts (duration, alignment, visibility, counts)
    reports/facts.json                  the key terms (its glossary)
    reports/fusion.json                 the team's fusion layer results (only the parts the Links tab needs)
    reports/links/report_points.json    the links over time in plain language (rule-based, or the checked AI rewrite)
    tables/events.csv, window_features.csv
Not shown (user decision): analysis_details.txt (indicators, linkage statistics, suggestions, limitations), which
stays for the team.
"""
import json

import pandas as pd

from web.models import stat_charts
from web.models.charts import ensure_charts

WINDOW_SECONDS = 10
MOMENT_LABELS = {"phone_use": "Phone in view", "head_down_still": "Head down and still (30 s or more)",
                 "hand_raise": "Hand raised", "reading": "Book in view"}
DATA_TERMS = {"window", "observable students", "observable student time"}


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


def report_source(reports_dir):
    """Who wrote the points: the AI rewrite (only if its text passed every check) or the standard feedback. AI-written
    text is disclosed to the instructor; the standard feedback needs no label."""
    path = reports_dir / "llm_output.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        attempts = record.get("attempts") or []
        if record.get("output") and attempts and not attempts[-1].get("problems"):
            return {"kind": "llm", "label": "Feedback written with AI from the measured results and checked automatically"}
    return {"kind": "rules", "label": None}


def activity_phrase(window):
    """What the instructor was doing in a 10-second window (the same wording as the report)."""
    return stat_charts.PHRASES[stat_charts.activity_of(window)]


def moments(tables_dir):
    """Every phone, head-down-and-still, hand-raise and book moment, with what the instructor was doing then."""
    events_path, windows_path = tables_dir / "events.csv", tables_dir / "window_features.csv"
    if not events_path.exists():
        return []
    events = pd.read_csv(events_path)
    windows = pd.read_csv(windows_path).set_index("window_id") if windows_path.exists() else pd.DataFrame()
    rows = []
    for e in events[events["label"].isin(MOMENT_LABELS)].sort_values("start_s").itertuples():
        window_id = int(e.start_s // WINDOW_SECONDS)
        activity = activity_phrase(windows.loc[window_id].to_dict()) if window_id in windows.index else "unknown"
        rows.append({"time": fmt_time(e.start_s), "end": fmt_time(e.end_s), "seconds": float(e.start_s),
                     "duration": int(round(e.duration_s)), "seat": stat_charts.seat_name(e.entity_id), "kind": e.label,
                     "behaviour": MOMENT_LABELS[e.label], "activity": activity.removeprefix("the instructor was ")})
    return rows


def links(lecture):
    """The "Links over time" tab: do student changes follow what the instructor did a few seconds earlier (the team's
    fusion layer), as positive and negative points, or why there are none for this lecture."""
    reports = lecture.reports_dir
    fusion_path, points_path = reports / "fusion.json", reports / "links" / "report_points.json"
    if not fusion_path.exists():
        return {"available": False, "reason": "This lecture was analysed before this part of the report existed. "
                                                "Run its analysis again to add it."}
    with open(fusion_path, encoding="utf-8") as f:
        fusion = json.load(f)
    if fusion.get("error") or not points_path.exists():
        return {"available": False, "reason": "This part of the analysis could not run for this lecture, for "
                                                "example because the two cameras overlap too little."}
    with open(points_path, encoding="utf-8") as f:
        points = json.load(f)
    return {
        "available": True,
        "overview": points.get("overview"),
        "positive": [p["text"] for p in points.get("positive_points", [])],
        "negative": [p["text"] for p in points.get("negative_points", [])],
        "note": points.get("note"),
        "ai": points.get("source") == "llm",     # only text that passed every check is saved as "llm"
        "window_seconds": f"{fusion['window_seconds']:g}",
        "analysed": fmt_time(fusion["windows"] * fusion["window_seconds"]),   # the time both cameras covered
    }


def lecture_length(result):
    """The analysed length, shown beside the analysis date."""
    duration = result.get("alignment", {}).get("duration_s")
    return fmt_time(duration) if duration else None


def build_report(lecture):
    """Everything report.html needs, as plain Python data."""
    reports = lecture.reports_dir
    feedback = parse_feedback((reports / "final_feedback_report.txt").read_text(encoding="utf-8"))
    facts = {}
    if (reports / "facts.json").exists():
        with open(reports / "facts.json", encoding="utf-8") as f:
            facts = json.load(f)
    result = lecture.result() or {}
    found = moments(lecture.tables_dir)
    return {
        "lecture": lecture,
        "context": lecture.context(),
        "analysed_at": lecture.analysed_at(),
        "source": report_source(reports),
        "feedback": feedback,
        "links": links(lecture),
        "length": lecture_length(result),
        "timeline": stat_charts.timeline(lecture),
        "comparison": stat_charts.comparison(lecture),
        "charts": ensure_charts(lecture),
        "moments": found,
        "moment_kinds": [(key, label) for key, label in MOMENT_LABELS.items() if any(m["kind"] == key for m in found)],
        # the key terms: words the page uses, in plain language. Terms that describe the data the AI step reads
        # (keys like "evidence.strength", "window", "observable ...") are left out; the page never uses them.
        "glossary": {term: meaning for term, meaning in facts.get("glossary", {}).items()
                     if "." not in term and term not in DATA_TERMS},
    }
