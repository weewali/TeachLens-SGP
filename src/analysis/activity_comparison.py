"""Student behaviour DURING each instructor activity, compared with the rest of the lesson.

The fusion layer (src/analysis/fusion.py, lag_analysis.py) asks whether the class changes a few seconds AFTER a
teacher action. This module asks the simpler question the instructor report also needs: while the instructor was
doing X, how did the students behave compared with the other parts of the lesson? It works on the fusion's 5 s
window table (fusion/windows.csv), so both analyses use the same evidence.

Method (from the pipeline_creation branch, 2026-11-17, now on the lesson windows): for each teacher feature, the
windows where it filled at least half the window are compared with the other windows where the teacher was seen.
The p-value comes from a regression on an indicator with Newey-West (HAC) errors, which allow for neighbouring
windows being alike; a difference is "suggestive" only if p < 0.05 divided by the number of comparisons made
(Bonferroni). Both groups need MIN_WINDOWS windows. Associations, not causes.
"""
from typing import Any

import numpy as np
import pandas as pd

MIN_WINDOWS = 5          # windows needed on each side of a comparison
MIN_PARTS_FOR_RANGE = 5  # fewer windows than this: an average is shown without a 95% range
ACTIVE_SHARE = 0.5       # a teacher feature "fills" a window from this share on
ALPHA = 0.05

# Teacher features compared (fusion columns) with the words used for them: (label, "you" phrase).
COMPARED = {
    "T_facing_class": ("Facing the class", "you faced the class"),
    "T_board_or_screen": ("At the board or screen", "you worked at the board or screen"),
    "T_seated": ("Seated", "you were seated"),
    "T_gesturing": ("Using hand gestures", "you used hand gestures"),
    "T_walking": ("Walking", "you walked around"),
    "T_reading_notes": ("Looking down at notes", "you looked down at notes or a laptop"),
    "T_back_to_class": ("Back to the class", "you had your back to the class"),
}
# Student measures: (column, chart label, words, more is better).
MEASURES = [
    ("S_on_task", "On task", "were on task", True),
    ("S_phone", "Phone", "had a phone", False),
    ("S_head_down", "Head down", "had their heads down", False),
]
COUNT_MEASURES = [("S_hand_raises", "Hand raises", "raised a hand", True)]

# One activity per window for the lecture timeline and the comparison chart. Fixed order = fixed colour in every
# lecture (web/views/static/style.css: .act-<key>; the first three are the colour-checked chart slots 1-3).
ACTIVITIES = [
    ("explaining", "Facing the class", "You were facing the class", "you were facing the class"),
    ("board", "At the board or screen", "You were at the board or screen", "you were at the board or screen"),
    ("not_facing", "Not facing the class", "You were not facing the class", "you were not facing the class"),
    ("out_of_view", "Out of the camera's view", "You were mostly out of the camera's view",
     "you were mostly out of the camera's view"),
]
ACTIVITY_LABELS = {key: label for key, label, _, _ in ACTIVITIES}


def _value(window: dict[str, Any], column: str) -> float:
    value = window.get(column)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def activity_of(window: dict[str, Any], window_seconds: float) -> str:
    """The instructor's activity in one window of fusion/windows.csv."""
    known = _value(window, "teacher_known_seconds")
    if not known >= window_seconds / 2:
        return "out_of_view"
    if _value(window, "T_board_or_screen") >= 0.3 or _value(window, "T_writing") >= 0.3:
        return "board"
    if _value(window, "T_facing_class") >= ACTIVE_SHARE:
        return "explaining"
    return "not_facing"


def student_known(table: pd.DataFrame) -> pd.Series:
    """Windows with enough student evidence for the student measures."""
    known = table["students_state_known_seconds"] if "students_state_known_seconds" in table else \
        pd.Series(np.nan, index=table.index)
    return (known > 0) & table.get("S_on_task", pd.Series(np.nan, index=table.index)).notna()


def hac_fit(y: np.ndarray, x: np.ndarray | None = None):
    """OLS of y on a constant (and x) with Newey-West errors; None when statsmodels is missing."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    design = np.ones((len(y), 1)) if x is None else sm.add_constant(x.astype(float), has_constant="add")
    lags = max(0, min(3, len(y) - 2))
    return sm.OLS(y.astype(float), design).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)


def mean_range(values) -> tuple[float, float] | None:
    """95% range for the average of values (in time order), allowing for neighbouring windows being alike."""
    values = np.asarray(values, dtype=float)
    if len(values) < MIN_PARTS_FOR_RANGE:
        return None
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    fit = hac_fit(values)
    if fit is None:
        return None
    low, high = np.asarray(fit.conf_int(0.05))[0]
    return float(low), float(high)


def compare(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Every comparison that has enough windows on both sides: the student measure while a teacher feature filled the
    window versus the other windows where the teacher was seen. Each row: teacher_feature, label, you, measure,
    measure_label, words, count (per window, not a share), with/without windows and averages, difference, p_hac,
    suggestive (Bonferroni over the comparisons made), favourable."""
    rows = []
    usable = student_known(table)
    for feature, (label, you) in COMPARED.items():
        if feature not in table:
            continue
        values = table[feature].astype(float)
        seen = values.notna() & usable
        active = seen & (values >= ACTIVE_SHARE)
        other = seen & (values < ACTIVE_SHARE)
        if active.sum() < MIN_WINDOWS or other.sum() < MIN_WINDOWS:
            continue
        for column, measure_label, words, better_high in MEASURES + COUNT_MEASURES:
            if column not in table:
                continue
            y = table[column].astype(float)
            keep = (active | other) & y.notna()
            if (keep & active).sum() < MIN_WINDOWS or (keep & other).sum() < MIN_WINDOWS:
                continue
            with_mean = float(y[keep & active].mean())
            without_mean = float(y[keep & other].mean())
            p = None
            if y[keep].nunique() > 1:
                fit = hac_fit(y[keep].to_numpy(), active[keep].to_numpy().astype(float))
                if fit is not None:
                    p = float(np.asarray(fit.pvalues)[1])
            difference = with_mean - without_mean
            rows.append({"teacher_feature": feature, "label": label, "you": you, "measure": column,
                         "measure_label": measure_label, "words": words,
                         "count": column in dict((c, 0) for c, *_ in COUNT_MEASURES),
                         "windows_with": int((keep & active).sum()), "windows_without": int((keep & other).sum()),
                         "with": with_mean, "without": without_mean, "difference": difference, "p_hac": p,
                         "favourable": (difference > 0) == better_high})
    tests = sum(r["p_hac"] is not None for r in rows)
    for r in rows:
        r["tests"] = tests
        r["suggestive"] = r["p_hac"] is not None and tests > 0 and r["p_hac"] < ALPHA / tests
    return rows
