"""Lag analysis of a fused window table: do student changes follow teacher actions?

Proposal FR 6.3-6.4 and section 4.1 (SGP1 report): shift the teacher timeline
forward against the student timeline, explore lags with cross-correlation,
then quantify with a lagged model

    Y_t = b0 + b1 X_(t-k) + ... + a Y_(t-1) + e_t

where Y is a student outcome (e.g. on-target share) and X teacher features k
windows earlier; the Y_(t-1) term absorbs the slow drift of attention. Each
link records the lag, the variables, and the result (FR 6.4).

Guardrails (from the first A1 trial, 2026-11-18):
- A teacher feature that is present in fewer than min_active_windows windows
  has no variation to test (A1: one window at the screen gave r = 0.82 and
  absurd model t-values); such actions get peri-event averages instead,
  and only with at least min_events onsets.
- Significance uses a circular-shift test: the teacher series is rotated
  against the student series, which keeps both series' autocorrelation but
  breaks their alignment; p = how often a rotation gives as strong a lag
  correlation. When there are fewer rotations than permutations (a short
  lesson: 55 windows give 42), every rotation is used once, so the smallest
  possible p is 1 / (rotations + 1) and a short lesson cannot look more
  certain than it is. Many pairs are tested, so q-values (Benjamini-Hochberg)
  are reported and only q <= alpha links become findings.
- Model standard errors are Newey-West (HAC) because windows are not
  independent.
- Hypotheses fixed in advance (fusion.hypotheses, from the scenario
  booklet's effective / ineffective teaching codes, 2026-11-19) are tested
  and corrected among themselves; all other pairs are exploratory and
  corrected separately, so a handful of planned tests is not drowned by
  dozens of exploratory ones.
- A teacher feature needs at least min_active_windows windows with it AND
  without it: a teacher seated for the whole lesson (A3) cannot be compared
  with themselves standing. Such contrasts need several lessons:
  pooled_analyze() tests the hypotheses within lessons over a set of them.
- Each hypothesis also gets a plain comparison: the student outcome in the
  windows after the teacher feature was present versus the other windows
  (Newey-West p-value), which reads as "X% of students on task after ...
  versus Y% otherwise".
These are associations, not causes: a scripted scenario or a short lesson
cannot show that a teacher behaviour works.
"""

from typing import Any, Sequence

import numpy as np


def active_windows(x: np.ndarray) -> int:
    """Windows where the feature is present (non-zero and known)."""
    return int(np.sum(np.nan_to_num(x, nan=0.0) > 0))


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int, min_pairs: int = 10) -> dict[int, float]:
    """Pearson r of y[t] with x[t - lag] for lag in -max_lag..max_lag (x leads for lag > 0)."""
    n = len(x)
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        a = x[:n - lag] if lag >= 0 else x[-lag:]
        b = y[lag:] if lag >= 0 else y[:n + lag]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() >= min_pairs and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            result[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return result


def shift_test(x: np.ndarray, y: np.ndarray, max_lag: int, permutations: int,
               rng: np.random.Generator) -> dict[str, Any] | None:
    """Strongest forward lag (1..max_lag) and its circular-shift p-value."""
    r = cross_correlation(x, y, max_lag)
    forward = {k: v for k, v in r.items() if k >= 1}
    if not forward:
        return None
    lag, best = max(forward.items(), key=lambda kv: abs(kv[1]))
    n = len(x)
    low, high = max_lag + 1, n - max_lag
    if high <= low:
        return None
    # A short lesson has few distinct rotations (55 windows: 42); then all of
    # them are used (exact test, smallest p = 1 / (rotations + 1)) instead of
    # repeating the same ones at random, which would fake precision.
    shifts = (np.arange(low, high) if high - low <= permutations
              else rng.integers(low, high, size=permutations))
    null = np.empty(len(shifts))
    for i, shift in enumerate(shifts):
        rs = cross_correlation(np.roll(x, int(shift)), y, max_lag)
        null[i] = max((abs(v) for k, v in rs.items() if k >= 1), default=0.0)
    p = (1 + int(np.sum(null >= abs(best) - 1e-12))) / (1 + len(shifts))
    backward = {k: v for k, v in r.items() if k <= -1}
    return {"lag_windows": lag, "r": round(best, 4), "r_lag0": round(r.get(0, float("nan")), 4),
            "r_reverse_max": round(max(backward.values(), key=abs), 4) if backward else None,
            "p_shift": round(p, 4), "null_shifts": len(shifts)}


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """q-values for a list of p-values (same order)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return [round(float(v), 4) for v in out]


def lagged_model(table: Any, outcome: str, predictors: Sequence[str], lag: int,
                 hac_lags: int | None = None) -> dict[str, Any] | None:
    """OLS of outcome_t on predictors_(t-lag) and outcome_(t-1), Newey-West errors."""
    import statsmodels.api as sm
    y = table[outcome].to_numpy(float)
    columns = {f"{name}[t-{lag}]": table[name].to_numpy(float) for name in predictors}
    n = len(y)
    if n <= lag + 5:
        return None
    data = {name: values[:n - lag] for name, values in columns.items()}
    data[f"{outcome}[t-1]"] = y[lag - 1:n - 1] if lag >= 1 else y[:n]
    target = y[lag:]
    X = np.column_stack(list(data.values()))
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(target)
    if ok.sum() < len(data) + 6 or np.std(target[ok]) == 0:
        return None
    X = sm.add_constant(X[ok], has_constant="add")
    maxlags = hac_lags if hac_lags is not None else max(1, int(round(4 * (ok.sum() / 100) ** (2 / 9))))
    fit = sm.OLS(target[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    names = ["const"] + list(data)
    ci = fit.conf_int()
    return {"outcome": outcome, "lag_windows": lag, "windows_used": int(ok.sum()),
            "r_squared": round(float(fit.rsquared), 4), "hac_maxlags": maxlags,
            "terms": {name: {"coef": round(float(fit.params[i]), 4), "se": round(float(fit.bse[i]), 4),
                             "p": round(float(fit.pvalues[i]), 4),
                             "ci95": [round(float(ci[i][0]), 4), round(float(ci[i][1]), 4)]}
                      for i, name in enumerate(names)}}


def peri_event(onset_windows: Sequence[int], y: np.ndarray, before: int, after: int,
               permutations: int, rng: np.random.Generator) -> dict[str, Any] | None:
    """Mean change of y after onsets (windows 1..after) versus before (-before..-1).

    p compares the mean change with the same number of random onsets.
    """
    n = len(y)

    def changes(points: Sequence[int]) -> list[float]:
        out = []
        for w in points:
            if w - before < 0 or w + after >= n:
                continue
            pre, post = y[w - before:w], y[w + 1:w + after + 1]
            if np.isnan(pre).all() or np.isnan(post).all():
                continue
            out.append(float(np.nanmean(post) - np.nanmean(pre)))
        return out
    observed = changes(onset_windows)
    if not observed:
        return None
    mean = float(np.mean(observed))
    valid = np.arange(before, n - after)
    if len(valid) == 0:
        return None
    null = np.empty(permutations)
    for i in range(permutations):
        sample = changes(rng.choice(valid, size=len(observed), replace=len(observed) > len(valid)))
        null[i] = np.mean(sample) if sample else 0.0
    p = (1 + int(np.sum(np.abs(null) >= abs(mean)))) / (1 + permutations)
    curve = []
    for rel in range(-before, after + 1):
        values = [y[w + rel] for w in onset_windows if 0 <= w + rel < n and not np.isnan(y[w + rel])]
        curve.append(round(float(np.mean(values)), 4) if values else None)
    return {"events": len(observed), "mean_change": round(mean, 4), "p_random_onsets": round(p, 4),
            "curve_relative_windows": list(range(-before, after + 1)), "curve": curve}


def split_point(x: np.ndarray, kind: str) -> float | None:
    """Where a teacher feature counts as "present": half the window for shares,
    the median for paths and counts (None when the series is empty)."""
    values = x[~np.isnan(x)]
    if not len(values):
        return None
    return 0.5 if kind in ("share", "episode") else float(np.median(values))


def variation(x: np.ndarray, split: float | None) -> tuple[int, int]:
    """(windows with the feature, windows without it) among known windows."""
    values = x[~np.isnan(x)]
    if split is None or not len(values):
        return 0, 0
    present = int(np.sum(values >= split)) if split > 0 else int(np.sum(values > 0))
    return present, len(values) - present


def comparison(x: np.ndarray, y: np.ndarray, lag: int, split: float,
               min_windows: int = 3, hac_lags: int | None = None,
               groups: np.ndarray | None = None) -> dict[str, Any] | None:
    """Student outcome in the windows lag windows after the teacher feature was
    present (x >= split) versus the other windows, with a Newey-West p-value.

    groups (one lesson label per window, pooled analysis) adds one intercept
    per lesson, so the comparison is made within lessons, and a lag never
    pairs windows of two different lessons.
    """
    import statsmodels.api as sm
    n = len(x)
    if lag < 0 or n <= lag:
        return None
    a, b = x[:n - lag], y[lag:]
    g = groups[lag:] if groups is not None else None
    if groups is not None:
        same = groups[:n - lag] == groups[lag:]
        a, b, g = a[same], b[same], g[same]
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    present = a >= split if split > 0 else a > 0
    if present.sum() < min_windows or (~present).sum() < min_windows:
        return None
    columns = [present.astype(float)]
    if g is not None:
        g = g[ok]
        for label in sorted(set(g))[1:]:
            columns.append((g == label).astype(float))
    X = sm.add_constant(np.column_stack(columns), has_constant="add")
    maxlags = hac_lags if hac_lags is not None else max(1, int(round(4 * (len(b) / 100) ** (2 / 9))))
    try:
        fit = sm.OLS(b, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        p = float(fit.pvalues[1])
    except (ValueError, np.linalg.LinAlgError):
        return None
    return {"lag_windows": lag, "windows_with": int(present.sum()), "windows_without": int((~present).sum()),
            "outcome_with": round(float(b[present].mean()), 4),
            "outcome_without": round(float(b[~present].mean()), 4),
            "difference": round(float(b[present].mean() - b[~present].mean()), 4),
            "difference_within_lessons": round(float(fit.params[1]), 4) if g is not None else None,
            "p_hac": round(p, 4) if np.isfinite(p) else None}


def _tone(desirable: int, effect: float) -> str:
    """strength / weak_point / observation from the outcome's desirable direction."""
    sign = desirable * effect
    return "strength" if sign > 0 else "weak_point" if sign < 0 else "observation"


def analyze(table: Any, metadata: dict[str, Any], settings: dict[str, Any],
            teacher_onsets: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Hypotheses, exploratory links, lagged models, peri-event changes, findings for one lesson.

    Hypotheses (settings["hypotheses"]: teacher feature, student feature,
    expected sign) are fixed in advance and corrected for multiple testing
    among themselves; every other teacher x student pair is exploratory and
    corrected separately. A teacher feature needs at least
    min_active_windows windows with it and as many without it in the lesson.
    """
    rng = np.random.default_rng(int(settings.get("seed", 0)))
    max_lag = int(settings.get("max_lag_windows", 6))
    min_active = int(settings.get("min_active_windows", 5))
    permutations = int(settings.get("permutations", 500))
    alpha = float(settings.get("alpha", 0.05))
    comparison_lag = int(settings.get("comparison_lag_windows", 1))
    window = float(metadata["window_seconds"])
    features = metadata["features"]
    teacher = [name for name, f in features.items() if f["camera"] == "teacher" and name in table]
    students = [name for name, f in features.items() if f["camera"] == "students" and name in table]
    series = {name: table[name].to_numpy(float) for name in teacher + students}
    activity = {name: active_windows(series[name]) for name in teacher + students}
    splits = {name: split_point(series[name], features[name]["kind"]) for name in teacher}
    present = {name: variation(series[name], splits[name]) for name in teacher}
    testable = [name for name in teacher if min(present[name]) >= min_active]
    outcomes = [name for name in students
                if activity[name] >= min_active and np.nanstd(series[name]) > 0]

    def test(x_name: str, y_name: str) -> dict[str, Any] | None:
        result = shift_test(series[x_name], series[y_name], max_lag, permutations, rng)
        if not result:
            return None
        return {**result, "lag_seconds": round(result["lag_windows"] * window, 3),
                "comparison": comparison(series[x_name], series[y_name], comparison_lag, splits[x_name])}

    hypotheses = []
    for spec in settings.get("hypotheses") or []:
        x_name, y_name = spec["teacher"], spec["student"]
        entry = {"teacher_feature": x_name, "student_feature": y_name, "expected_sign": int(spec.get("expect", 0)),
                 "code": spec.get("code")}
        if x_name not in series or y_name not in series:
            entry["not_tested"] = "feature not in this lesson's window table"
        elif x_name not in testable:
            entry["not_tested"] = (f"{x_name} varied too little in this lesson (present in {present[x_name][0]} "
                                   f"and absent in {present[x_name][1]} windows; {min_active} of each needed)")
        elif y_name not in outcomes:
            entry["not_tested"] = f"{y_name} had too little evidence or no variation"
        else:
            result = test(x_name, y_name)
            if result:
                entry.update(result)
                entry["matches_expectation"] = (bool(np.sign(result["r"]) == np.sign(entry["expected_sign"]))
                                                if entry["expected_sign"] else None)
            else:
                entry["not_tested"] = "too few windows for the lags"
        hypotheses.append(entry)
    tested = [h for h in hypotheses if "p_shift" in h]
    for entry, q in zip(tested, benjamini_hochberg([h["p_shift"] for h in tested])):
        entry["q"] = q

    pairs = {(h["teacher_feature"], h["student_feature"]) for h in hypotheses}
    links = []
    for x_name in testable:
        for y_name in outcomes:
            if (x_name, y_name) in pairs:
                continue
            result = test(x_name, y_name)
            if result:
                links.append({"teacher_feature": x_name, "student_feature": y_name, **result})
    for link, q in zip(links, benjamini_hochberg([link["p_shift"] for link in links])):
        link["q"] = q
    links.sort(key=lambda link: (link["q"], link["p_shift"]))

    models = []
    for y_name in [name for name in settings.get("outcomes", outcomes) if name in outcomes]:
        predictors = [name for name in settings.get("model_predictors", testable) if name in testable]
        if not predictors:
            continue
        for lag in settings.get("model_lags", [1, 2, 3]):
            model = lagged_model(table, y_name, predictors, int(lag))
            if model:
                models.append(model)

    events = []
    for name, windows in (teacher_onsets or {}).items():
        if len(windows) < int(settings.get("min_events", 3)):
            events.append({"teacher_event": name, "events": len(windows), "skipped": "too few onsets"})
            continue
        for y_name in outcomes:
            result = peri_event(windows, series[y_name],
                                int(settings.get("peri_event_before_windows", 2)),
                                int(settings.get("peri_event_after_windows", 4)), permutations, rng)
            if result:
                events.append({"teacher_event": name, "student_feature": y_name, **result})
    tested_events = [e for e in events if "p_random_onsets" in e]
    for event, q in zip(tested_events, benjamini_hochberg([e["p_random_onsets"] for e in tested_events])):
        event["q"] = q

    findings = []
    for entry in tested:
        if entry["q"] <= alpha:
            findings.append({"kind": _tone(features[entry["student_feature"]]["desirable"], entry["r"]),
                             "method": "hypothesis_lag_correlation", "teacher_feature": entry["teacher_feature"],
                             "student_feature": entry["student_feature"], "lag_seconds": entry["lag_seconds"],
                             "effect": entry["r"], "q": entry["q"],
                             "matches_expectation": entry.get("matches_expectation")})
    for link in links:
        if link["q"] <= alpha:
            findings.append({"kind": _tone(features[link["student_feature"]]["desirable"], link["r"]),
                             "method": "exploratory_lag_correlation", "teacher_feature": link["teacher_feature"],
                             "student_feature": link["student_feature"], "lag_seconds": link["lag_seconds"],
                             "effect": link["r"], "q": link["q"]})
    for event in tested_events:
        if event["q"] <= alpha:
            findings.append({"kind": _tone(features[event["student_feature"]]["desirable"], event["mean_change"]),
                             "method": "peri_event", "teacher_feature": event["teacher_event"],
                             "student_feature": event["student_feature"],
                             "window_seconds": [-int(settings.get("peri_event_before_windows", 2)) * window,
                                                int(settings.get("peri_event_after_windows", 4)) * window],
                             "effect": event["mean_change"], "q": event["q"]})
    findings.sort(key=lambda f: (f["method"] != "hypothesis_lag_correlation", f["q"], -abs(f["effect"])))
    return {
        "feature_activity_windows": activity,
        "teacher_feature_windows_with_without": {name: list(present[name]) for name in teacher},
        "tested_teacher_features": testable,
        "untested_teacher_features": {name: activity[name] for name in teacher if name not in testable},
        "outcome_features": outcomes,
        "hypotheses": hypotheses,
        "links": links,
        "lagged_models": models,
        "peri_event": events,
        "findings": findings,
        "settings": {"max_lag_windows": max_lag, "min_active_windows": min_active,
                     "permutations": permutations, "alpha": alpha, "seed": int(settings.get("seed", 0)),
                     "comparison_lag_windows": comparison_lag},
    }


def _pooled_r(pieces: list[tuple[np.ndarray, np.ndarray]], lag: int, min_pairs: int = 10) -> float | None:
    """Pearson r of y[t] with x[t - lag] within each lesson (each lesson's means removed), pooled."""
    xs, ys = [], []
    for x, y in pieces:
        n = len(x)
        if n <= lag:
            continue
        a, b = x[:n - lag], y[lag:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 3:
            continue
        xs.append(a[ok] - a[ok].mean())
        ys.append(b[ok] - b[ok].mean())
    if not xs:
        return None
    a, b = np.concatenate(xs), np.concatenate(ys)
    if len(a) < min_pairs or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def between_lessons(tables: Sequence[tuple[str, Any]], x_name: str, y_name: str, kind: str,
                    permutations: int, rng: np.random.Generator) -> dict[str, Any] | None:
    """Lesson-level comparison: each lesson's average teacher feature and student outcome.

    The scenario booklet contrasts lessons (an effective, standing teacher in
    A1/A2; a seated teacher reading notes in A3/A4) more than moments within a
    lesson, so this compares lessons: Spearman's rho over lessons, an exact
    permutation p (every ordering of the lessons, or a random sample when
    there are more than 8), and the outcome in lessons where the teacher
    feature filled at least half the time versus the others. With a handful
    of lessons this is weak evidence: lessons differ in many ways at once.
    """
    from itertools import permutations as orderings
    rows = []
    for name, table in tables:
        if x_name in table and y_name in table:
            x, y = table[x_name].to_numpy(float), table[y_name].to_numpy(float)
            if np.isfinite(x).any() and np.isfinite(y).any():
                rows.append((name, float(np.nanmean(x)), float(np.nanmean(y))))
    if len(rows) < 3:
        return None
    xs, ys = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    if np.std(xs) == 0 or np.std(ys) == 0:
        return {"lessons": len(rows), "not_tested": "the lessons do not differ in one of the two features"}

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        return float(np.corrcoef(ra, rb)[0, 1])
    observed = rho(xs, ys)
    if len(rows) <= 8:
        null = [rho(xs, ys[list(order)]) for order in orderings(range(len(rows)))]
    else:
        null = [rho(xs, rng.permutation(ys)) for _ in range(permutations)]
    p = (1 + sum(abs(v) >= abs(observed) - 1e-12 for v in null)) / (1 + len(null))
    split = 0.5 if kind in ("share", "episode") else float(np.median(xs))
    high = xs >= split
    result = {"lessons": len(rows), "spearman_rho": round(observed, 4), "p_permutation": round(p, 4),
              "per_lesson": [{"lesson": n, "teacher_mean": round(a, 4), "student_mean": round(b, 4)}
                             for n, a, b in rows]}
    if high.any() and (~high).any():
        result.update({"lessons_with": [rows[i][0] for i in np.flatnonzero(high)],
                       "outcome_with": round(float(ys[high].mean()), 4),
                       "outcome_without": round(float(ys[~high].mean()), 4)})
    return result


def pooled_analyze(tables: Sequence[tuple[str, Any]], features: dict[str, Any],
                   settings: dict[str, Any]) -> dict[str, Any]:
    """The hypotheses over several lessons at once, within lessons.

    Each lesson keeps its own mean (so a lesson that is calmer overall does
    not count as an effect), lags never cross from one lesson into the next,
    and the null rotates every lesson's teacher series by its own random
    shift. A lesson where the teacher feature never changes adds nothing to
    that hypothesis (and is listed). Comparisons add one intercept per lesson.
    """
    rng = np.random.default_rng(int(settings.get("seed", 0)))
    max_lag = int(settings.get("max_lag_windows", 6))
    min_active = int(settings.get("min_active_windows", 5))
    permutations = int(settings.get("permutations", 500))
    alpha = float(settings.get("alpha", 0.05))
    comparison_lag = int(settings.get("comparison_lag_windows", 1))
    window = float(settings.get("window_seconds", 5.0))
    results = []
    for spec in settings.get("hypotheses") or []:
        x_name, y_name = spec["teacher"], spec["student"]
        entry = {"teacher_feature": x_name, "student_feature": y_name, "expected_sign": int(spec.get("expect", 0)),
                 "code": spec.get("code"), "lessons_used": [], "lessons_without_variation": []}
        pieces, xs, ys, groups = [], [], [], []
        kind = features.get(x_name, {}).get("kind", "share")
        entry["between_lessons"] = between_lessons(tables, x_name, y_name, kind, permutations, rng)
        for name, table in tables:
            if x_name not in table or y_name not in table:
                continue
            x, y = table[x_name].to_numpy(float), table[y_name].to_numpy(float)
            with_, without = variation(x, split_point(x, kind))
            if min(with_, without) < min_active or len(x) <= 2 * max_lag + 1:
                entry["lessons_without_variation"].append(name)
                continue
            entry["lessons_used"].append(name)
            pieces.append((x, y))
            xs.append(x)
            ys.append(y)
            groups.extend([name] * len(x))
        if len(pieces) < 2:
            entry["not_tested"] = "fewer than two lessons where the teacher feature varied"
            results.append(entry)
            continue
        r = {lag: _pooled_r(pieces, lag) for lag in range(1, max_lag + 1)}
        r = {lag: value for lag, value in r.items() if value is not None}
        if not r:
            entry["not_tested"] = "too few windows"
            results.append(entry)
            continue
        lag, best = max(r.items(), key=lambda kv: abs(kv[1]))
        null = np.empty(permutations)
        for i in range(permutations):
            rotated = [(np.roll(x, int(rng.integers(max_lag + 1, len(x) - max_lag))), y) for x, y in pieces]
            values = [abs(v) for k in range(1, max_lag + 1) if (v := _pooled_r(rotated, k)) is not None]
            null[i] = max(values, default=0.0)
        p = (1 + int(np.sum(null >= abs(best) - 1e-12))) / (1 + permutations)
        per_lesson = {}
        for name, (x, y) in zip(entry["lessons_used"], pieces):
            value = _pooled_r([(x, y)], lag, min_pairs=5)
            per_lesson[name] = round(value, 4) if value is not None else None
        x_all, y_all = np.concatenate(xs), np.concatenate(ys)
        entry.update({"lag_windows": lag, "lag_seconds": round(lag * window, 3), "r": round(best, 4),
                      "p_shift": round(p, 4), "windows": int(len(x_all)), "r_by_lesson": per_lesson,
                      "matches_expectation": (bool(np.sign(best) == np.sign(entry["expected_sign"]))
                                              if entry["expected_sign"] else None),
                      "comparison": comparison(x_all, y_all, comparison_lag,
                                               split_point(x_all, kind) or 0.5, groups=np.array(groups))})
        results.append(entry)
    tested = [h for h in results if "p_shift" in h]
    for entry, q in zip(tested, benjamini_hochberg([h["p_shift"] for h in tested])):
        entry["q"] = q
    findings = [{"kind": _tone(features[h["student_feature"]]["desirable"], h["r"]),
                 "method": "pooled_hypothesis_lag_correlation", "teacher_feature": h["teacher_feature"],
                 "student_feature": h["student_feature"], "lag_seconds": h["lag_seconds"], "effect": h["r"],
                 "q": h["q"], "matches_expectation": h["matches_expectation"]}
                for h in tested if h["q"] <= alpha]
    findings.sort(key=lambda f: (f["q"], -abs(f["effect"])))
    return {"lessons": [name for name, _ in tables], "hypotheses": results, "findings": findings,
            "settings": {"max_lag_windows": max_lag, "min_active_windows": min_active, "permutations": permutations,
                         "alpha": alpha, "seed": int(settings.get("seed", 0)),
                         "comparison_lag_windows": comparison_lag}}
