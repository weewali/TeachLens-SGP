"""Data for the two statistical charts on the report page, built from a lesson's saved files (no video is reprocessed).

- timeline:   the share of the students in view who were on task in every 5-second window of the lesson
              (fusion/windows.csv), what the instructor was doing then, and the hand-raise, phone and dozing moments.
              report.js draws it as an SVG; the same numbers are also shown as a table.
- comparison: the average of each student measure during each instructor activity, with a 95% range for the average
              that allows for neighbouring windows being alike (HAC errors), drawn as HTML bars.

The activity of a window and the 95% ranges come from src/analysis/activity_comparison.py, the same code that writes
the report's "during each activity" facts, so the charts and the feedback cannot disagree.
"""
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.activity_comparison import (ACTIVITIES, MEASURES, activity_of, mean_range,  # noqa: E402
                                              student_known)

LABELS = {key: label for key, label, _, _ in ACTIVITIES}
YOU = {key: you for key, _, you, _ in ACTIVITIES}
PHRASES = {key: phrase for key, _, _, phrase in ACTIVITIES}
LANES = [("hand_raise", "Hand raised"), ("phone_use", "Phone seen"), ("dozing", "Dozing posture")]
NICE_MAXIMA = (5, 10, 20, 25, 50, 100)


def fmt_time(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def seat_name(seat_id):
    match = re.fullmatch(r"seat_0*(\d+)", str(seat_id))
    return f"seat {match.group(1)}" if match else "a student"


def fmt_pct(value):
    """A percentage for display: '0%', '0.4%' (below 1, one decimal) or '95%'."""
    if value == 0:
        return "0%"
    return f"{value:.1f}%" if value < 1 else f"{value:.0f}%"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def windows(lecture):
    """(fusion/windows.csv as a table, window seconds), or (None, 5) before the analysis."""
    lesson = lecture.lesson_dir
    path = lesson / "fusion" / "windows.csv" if lesson else None
    if path is None or not path.exists():
        return None, 5.0
    window = float(_json(lesson / "fusion" / "fusion.json").get("window_seconds", 5.0))
    return pd.read_csv(path).sort_values("lesson_seconds").reset_index(drop=True), window


def _seconds(text):
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def moments(lecture):
    """Hand raises, phones and dozing postures seen by the student camera, on the lesson's time line (seconds from the
    start of the stretch both cameras recorded), each with the student's anonymous seat."""
    lesson = lecture.lesson_dir
    if not lesson or not (lesson / "student" / "events.jsonl").exists():
        return []
    fusion = _json(lesson / "fusion" / "fusion.json")
    start = _seconds(fusion.get("start_real_time"))
    seats = {}
    for seat in (_json(lesson / "student" / "summary.json").get("seats") or {}).get("seats", []):
        for track in seat.get("tracks", []):
            seats[track] = seat.get("seat_id")
    found = []
    for line in (lesson / "student" / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        name, value = event.get("indicator_name"), event.get("value")
        kind = ("hand_raise" if name == "hand_raise" and value not in ("none", "unknown", "down", None) else
                "phone_use" if name == "device_use" and value in ("phone", "phone_likely") else
                "dozing" if name == "alertness" and value == "dozing" else None)
        begin, end = _seconds(event.get("start_real_time")), _seconds(event.get("end_real_time"))
        if kind is None or start is None or begin is None:
            continue
        found.append({"kind": kind, "start": max(0.0, begin - start), "end": max(0.0, (end or begin) - start),
                      "seat": seat_name(event.get("zone") or seats.get(event.get("track_id")))})
    return sorted(found, key=lambda m: m["start"])


def timeline(lecture):
    """{"data": what report.js draws, "rows": the same numbers as table rows}, or None without saved windows."""
    table, window = windows(lecture)
    if table is None or table.empty:
        return None
    usable = student_known(table)
    lane_kinds = dict(LANES)
    found = moments(lecture)
    parts, rows = [], []
    for ok, row in zip(usable, table.to_dict("records")):
        key = activity_of(row, window)
        on_task = round(100 * float(row["S_on_task"]), 1) if ok else None
        start = float(row["lesson_seconds"])
        end = start + window
        parts.append({"start": start, "end": end, "on_task": on_task, "activity": key})
        seen = [f"{lane_kinds[m['kind']]} ({m['seat']})" for m in found if start <= m["start"] < end]
        rows.append({"time": f"{fmt_time(start)}–{fmt_time(end)}",
                     "on_task": fmt_pct(on_task) if on_task is not None else "too little student evidence",
                     "activity": LABELS[key], "key": key, "moments": ", ".join(seen) or "–"})
    present = {p["activity"] for p in parts}
    data = {
        "duration": parts[-1]["end"],
        "window_seconds": window,
        "parts": parts,
        "moments": found,
        "activities": [{"key": key, "label": LABELS[key], "you": YOU[key]} for key, *_ in ACTIVITIES if key in present],
        "lanes": [{"kind": kind, "label": label, "count": sum(m["kind"] == kind for m in found)} for kind, label in LANES],
    }
    return {"data": data, "rows": rows}


def _scale_max(column, cells):
    """The right end of a measure's scale, in percent: 100 for on task, else a round number just above the data."""
    if column == "S_on_task":
        return 100
    top = max((100 * (c["range"][1] if c["range"] else c["mean"]) for c in cells), default=0)
    return next((m for m in NICE_MAXIMA if m >= top), 100)


def comparison(lecture):
    """Rows (one per instructor activity) of averages with their 95% ranges, or None when there is nothing to compare."""
    table, window = windows(lecture)
    if table is None or table.empty:
        return None
    activities = table.apply(lambda row: activity_of(row.to_dict(), window), axis=1)
    usable = student_known(table)
    if usable.sum() < 2:
        return None
    raw = []
    for key, label, you, _ in ACTIVITIES:
        group = table[usable & (activities == key)]
        if group.empty:
            continue
        measures = {}
        for column, *_ in MEASURES:
            values = group[column].dropna() if column in group else pd.Series(dtype=float)
            band = mean_range(values)
            if band:
                band = (max(0.0, band[0]), min(1.0, band[1]))
            measures[column] = {"mean": float(values.mean()) if len(values) else float("nan"), "range": band,
                                "parts": int(len(values))}
        raw.append({"key": key, "label": label, "you": you, "parts": int(len(group)),
                    "share": int(round(100 * float((activities == key).mean()))), "measures": measures})
    if not raw:
        return None
    measures = []
    for column, name, words, _ in MEASURES:
        cells = [r["measures"][column] for r in raw if not math.isnan(r["measures"][column]["mean"])]
        measures.append({"column": column, "label": name, "words": words, "max": _scale_max(column, cells)})
    rows = []
    for r in raw:
        cells = []
        for m in measures:
            cell = r["measures"][m["column"]]
            if math.isnan(cell["mean"]):
                cells.append({"measure": m["label"], "text": "–", "width": 0, "range": None,
                              "tip": f"{r['you']}: not measured."})
                continue
            value, scale = 100 * cell["mean"], m["max"]
            tip = f"{r['you']}: the students in view {m['words']} {fmt_pct(value)} of the time on average"
            out = {"measure": m["label"], "text": fmt_pct(value), "width": round(min(value / scale, 1) * 100, 2),
                   "range": None}
            if cell["range"]:
                low, high = (100 * v for v in cell["range"])
                out.update(range=f"{fmt_pct(low).rstrip('%')}–{fmt_pct(high)}", low=round(min(low / scale, 1) * 100, 2),
                           span=round(max(min(high / scale, 1) - min(low / scale, 1), 0) * 100, 2))
                tip += f" (95% range {out['range']})"
            out["tip"] = tip + f", over {cell['parts']} {int(window)}-second parts."
            cells.append(out)
        rows.append({"key": r["key"], "label": r["label"], "share": r["share"], "parts": r["parts"], "cells": cells})
    return {"measures": measures, "rows": rows}
