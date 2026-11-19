"""Data for the two statistical charts on the report page, built from a lecture's saved tables (no video is reprocessed).

- timeline:   the share of the students in view with their heads up in every 10-second part of the lecture, what the
              instructor was doing then, and the hand-raise, phone and head-down-and-still moments. report.js draws it
              as an SVG that follows the page's light or dark theme; the same numbers are also shown as a table.
- comparison: the average of each student behaviour during each instructor activity, with a 95% range for the average
              that allows for neighbouring parts of the lecture being alike (HAC errors), drawn as HTML bars.

A part counts for the student numbers only when enough students were readable: the same rule the feedback uses
(MIN_POSE_COVERAGE and MIN_SEATS_OBSERVED in main code/Main Code.py; keep the two in step).
"""
import math
import re

import numpy as np
import pandas as pd

WINDOW_SECONDS = 10
MIN_POSE_COVERAGE = 0.4
MIN_SEATS_OBSERVED = 3
MIN_PARTS_FOR_RANGE = 5          # fewer parts than this: the average is shown without a range (MIN_GROUP_WINDOWS)

# Fixed order = fixed colour in every lecture (style.css: .act-<key>). The first three take chart slots 1-3, which pass
# the colour-vision checks as a set in both themes; the last two are neutral greys for "not a teaching activity".
ACTIVITIES = [
    ("explaining", "Explaining to the class", "You were explaining to the class",
     "the instructor was explaining to the class"),
    ("board", "At the board, facing it", "You were at the board, facing it", "the instructor was at the board facing it"),
    ("not_facing", "Not facing the class", "You were not facing the class", "the instructor was not facing the class"),
    ("in_view", "In view (facing not measured)", "You were in the camera's view", "the instructor was in view"),
    ("out_of_view", "Out of the camera's view", "You were mostly out of the camera's view",
     "the instructor was mostly out of view"),
]
LABELS = {key: label for key, label, _, _ in ACTIVITIES}
YOU = {key: you for key, _, you, _ in ACTIVITIES}
PHRASES = {key: phrase for key, _, _, phrase in ACTIVITIES}

# (column in window_features.csv, chart label, words for the tooltip)
MEASURES = [
    ("attention_level", "Heads up", "had their heads up"),
    ("phone_rate", "Phone in view", "had a phone in view"),
    ("head_down_still_rate", "Head down and still", "kept their heads down and stayed still"),
]
LANES = [("hand_raise", "Hand raised"), ("phone_use", "Phone seen"), ("head_down_still", "Head down and still")]
NICE_MAXIMA = (5, 10, 20, 25, 50, 100)


def activity_of(window):
    """What the instructor was doing in a 10-second window; the same rule as instructor_phrase() in Main Code.py."""
    present = window.get("instructor_present")
    if present is None or not present >= 0.5:
        return "out_of_view"
    if window.get("explanation_ratio", float("nan")) >= 0.5:
        return "explaining"
    if window.get("board_work_ratio", float("nan")) >= 0.3:
        return "board"
    if pd.isna(window.get("explanation_ratio")):
        return "in_view"
    return "not_facing"


def fmt_time(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def seat_name(entity_id):
    match = re.fullmatch(r"seat_0*(\d+)", str(entity_id))
    return f"seat {match.group(1)}" if match else str(entity_id)


def fmt_pct(value):
    """A percentage for display: '0%', '0.4%' (below 1, one decimal) or '95%'."""
    if value == 0:
        return "0%"
    return f"{value:.1f}%" if value < 1 else f"{value:.0f}%"


def _tables(lecture):
    windows_path, events_path = lecture.tables_dir / "window_features.csv", lecture.tables_dir / "events.csv"
    if not windows_path.exists():
        return None, None
    windows = pd.read_csv(windows_path).sort_values("start_s").reset_index(drop=True)
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=["label", "start_s", "end_s", "entity_id"])
    return windows, events


def _usable(windows):
    return ((windows["pose_coverage"] >= MIN_POSE_COVERAGE) & (windows["seats_observed"] >= MIN_SEATS_OBSERVED)
            & windows["attention_level"].notna())


def timeline(lecture):
    """{"data": what report.js draws, "rows": the same numbers as table rows}, or None without saved tables."""
    windows, events = _tables(lecture)
    if windows is None or windows.empty:
        return None
    usable = _usable(windows)
    lane_kinds = dict(LANES)
    moments = [{"kind": e.label, "start": float(e.start_s), "end": float(e.end_s), "seat": seat_name(e.entity_id)}
               for e in events.sort_values("start_s").itertuples() if e.label in lane_kinds]
    parts, rows = [], []
    for ok, window in zip(usable, windows.to_dict("records")):
        key = activity_of(window)
        heads_up = round(100 * float(window["attention_level"]), 1) if ok else None
        start, end = float(window["start_s"]), float(window["end_s"])
        parts.append({"start": start, "end": end, "heads_up": heads_up, "activity": key})
        seen = [f"{lane_kinds[m['kind']]} ({m['seat']})" for m in moments if start <= m["start"] < end]
        rows.append({"time": f"{fmt_time(start)}–{fmt_time(end)}",
                     "heads_up": fmt_pct(heads_up) if heads_up is not None else "too few students readable",
                     "activity": LABELS[key], "key": key, "moments": ", ".join(seen) or "–"})
    present = {p["activity"] for p in parts}
    data = {
        "duration": float(windows["end_s"].max()),
        "parts": parts,
        "moments": moments,
        "activities": [{"key": key, "label": LABELS[key], "you": YOU[key]} for key, *_ in ACTIVITIES if key in present],
        "lanes": [{"kind": kind, "label": label, "count": sum(m["kind"] == kind for m in moments)} for kind, label in LANES],
    }
    return {"data": data, "rows": rows}


def mean_range(values):
    """95% range (low, high) for the average of `values` (in time order), allowing for neighbouring parts being alike
    (Newey-West / HAC errors, the same approach as the feedback's comparisons). None when there are too few parts or
    statsmodels is missing; a range of zero width when every part had the same value."""
    values = np.asarray(values, dtype=float)
    if len(values) < MIN_PARTS_FOR_RANGE:
        return None
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    fit = sm.OLS(values, np.ones(len(values))).fit(cov_type="HAC", cov_kwds={"maxlags": min(3, len(values) - 2)}, use_t=True)
    low, high = np.asarray(fit.conf_int(0.05))[0]
    return max(0.0, float(low)), min(1.0, float(high))


def _scale_max(column, rows):
    """The right end of a measure's scale, in percent: 100 for heads up, else a round number just above the data."""
    if column == "attention_level":
        return 100
    top = max((100 * (r["range"][1] if r["range"] else r["mean"]) for r in rows), default=0)
    return next((m for m in NICE_MAXIMA if m >= top), 100)


def comparison(lecture):
    """Rows (one per instructor activity) of averages with their 95% ranges, or None when there is nothing to compare."""
    windows, _ = _tables(lecture)
    if windows is None or windows.empty:
        return None
    activities = windows.apply(lambda w: activity_of(w.to_dict()), axis=1)
    usable = _usable(windows)
    if usable.sum() < 2:
        return None
    raw = []
    for key, label, you, _ in ACTIVITIES:
        group = windows[usable & (activities == key)]
        if group.empty:
            continue
        measures = {}
        for column, _, _ in MEASURES:
            values = group[column].dropna()
            measures[column] = {"mean": float(values.mean()) if len(values) else float("nan"), "range": mean_range(values),
                                "parts": int(len(values))}
        raw.append({"key": key, "label": label, "you": you, "parts": int(len(group)),
                    "share": int(round(100 * float((activities == key).mean()))), "measures": measures})
    if not raw:
        return None
    measures = []
    for column, name, words in MEASURES:
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
            out["tip"] = tip + f", over {cell['parts']} ten-second parts."
            cells.append(out)
        rows.append({"key": r["key"], "label": r["label"], "share": r["share"], "parts": r["parts"], "cells": cells})
    return {"measures": measures, "rows": rows}
