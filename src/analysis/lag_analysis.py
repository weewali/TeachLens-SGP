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
These are associations in one recording, not causes: a scripted scenario or
a short lesson cannot show that a teacher behaviour works.
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


def analyze(table: Any, metadata: dict[str, Any], settings: dict[str, Any],
            teacher_onsets: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """All teacher x student links, lagged models, peri-event changes, findings."""
    rng = np.random.default_rng(int(settings.get("seed", 0)))
    max_lag = int(settings.get("max_lag_windows", 6))
    min_active = int(settings.get("min_active_windows", 5))
    permutations = int(settings.get("permutations", 500))
    alpha = float(settings.get("alpha", 0.05))
    window = float(metadata["window_seconds"])
    features = metadata["features"]
    teacher = [name for name, f in features.items() if f["camera"] == "teacher"]
    students = [name for name, f in features.items() if f["camera"] == "students"]
    activity = {name: active_windows(table[name].to_numpy(float)) for name in teacher + students}
    testable = [name for name in teacher if activity[name] >= min_active]
    outcomes = [name for name in students if activity[name] >= min_active]

    links = []
    for x_name in testable:
        for y_name in outcomes:
            result = shift_test(table[x_name].to_numpy(float), table[y_name].to_numpy(float),
                                max_lag, permutations, rng)
            if result:
                links.append({"teacher_feature": x_name, "student_feature": y_name,
                              "lag_seconds": round(result["lag_windows"] * window, 3), **result})
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
            result = peri_event(windows, table[y_name].to_numpy(float),
                                int(settings.get("peri_event_before_windows", 2)),
                                int(settings.get("peri_event_after_windows", 4)), permutations, rng)
            if result:
                events.append({"teacher_event": name, "student_feature": y_name, **result})
    tested = [e for e in events if "p_random_onsets" in e]
    for event, q in zip(tested, benjamini_hochberg([e["p_random_onsets"] for e in tested])):
        event["q"] = q

    findings = []
    for link in links:
        if link["q"] > alpha:
            continue
        sign = features[link["student_feature"]]["desirable"]
        tone = ("strength" if sign * link["r"] > 0 else "weak_point" if sign * link["r"] < 0
                else "observation")
        findings.append({"kind": tone, "method": "lag_correlation", "teacher_feature": link["teacher_feature"],
                         "student_feature": link["student_feature"], "lag_seconds": link["lag_seconds"],
                         "effect": link["r"], "q": link["q"]})
    for event in tested:
        if event["q"] > alpha:
            continue
        sign = features[event["student_feature"]]["desirable"]
        tone = ("strength" if sign * event["mean_change"] > 0 else "weak_point" if sign * event["mean_change"] < 0
                else "observation")
        findings.append({"kind": tone, "method": "peri_event", "teacher_feature": event["teacher_event"],
                         "student_feature": event["student_feature"],
                         "window_seconds": [-int(settings.get("peri_event_before_windows", 2)) * window,
                                            int(settings.get("peri_event_after_windows", 4)) * window],
                         "effect": event["mean_change"], "q": event["q"]})
    findings.sort(key=lambda f: (f["q"], -abs(f["effect"])))
    return {
        "feature_activity_windows": activity,
        "tested_teacher_features": testable,
        "untested_teacher_features": {name: activity[name] for name in teacher if name not in testable},
        "outcome_features": outcomes,
        "links": links,
        "lagged_models": models,
        "peri_event": events,
        "findings": findings,
        "settings": {"max_lag_windows": max_lag, "min_active_windows": min_active,
                     "permutations": permutations, "alpha": alpha, "seed": int(settings.get("seed", 0))},
    }
