"""Simulation study of the lag test (report/evaluation_plan.md, level 5).

Instructor and class series are simulated as AR(1) processes with the autocorrelation seen in the
scenario lessons (phi = 0.67 between neighbouring 5-second windows). The class series follows the
instructor series two windows (10 s) later with correlation beta:
    y[t] = beta * x[t-2] + sqrt(1 - beta^2) * e[t]
For each lecture length and beta, the pipeline's own circular-shift test (lag_analysis.shift_test) is
run and three things are recorded: whether p <= 0.05 (false-positive rate when beta = 0, power
otherwise), whether the link survives Benjamini-Hochberg among the 9 pre-registered hypotheses (the
other 8 taken as true nulls with uniform p-values), and whether the reported lag is the true one.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import PIPELINE_ROOT  # noqa: F401

PHI = 0.67
LAG = 2
MAX_LAG = 6
SHIFTS = 200
LENGTHS = [60, 120, 360, 720]          # windows of 5 s: 5, 10, 30, 60 minutes
BETAS = [0.0, 0.2, 0.35, 0.5]
HYPOTHESES = 9
ALPHA = 0.05


def ar1(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(n + 50)
    x[0] = rng.standard_normal()
    s = np.sqrt(1 - PHI ** 2)
    for t in range(1, n + 50):
        x[t] = PHI * x[t - 1] + s * rng.standard_normal()
    return x[50:]


def one_condition(args: tuple[int, float, int, int]) -> dict[str, Any]:
    from src.analysis.lag_analysis import benjamini_hochberg, shift_test
    n, beta, reps, seed = args
    rng = np.random.default_rng(seed)
    hits = bh_hits = lag_ok = valid = 0
    min_p = []
    for _ in range(reps):
        x = ar1(n + LAG, rng)
        e = ar1(n + LAG, rng)
        y = beta * np.roll(x, LAG) + np.sqrt(1 - beta ** 2) * e
        x, y = x[LAG:], y[LAG:]
        result = shift_test(x, y, MAX_LAG, SHIFTS, rng)
        if not result:
            continue
        valid += 1
        p = result["p_shift"]
        min_p.append(1 / (1 + result["null_shifts"]))
        hits += p <= ALPHA
        q = benjamini_hochberg([p] + list(rng.uniform(size=HYPOTHESES - 1)))[0]
        bh_hits += q <= ALPHA
        lag_ok += (p <= ALPHA) and result["lag_windows"] == LAG
    return {"windows": n, "minutes": n * 5 / 60, "beta": beta, "reps": valid,
            "rejection_rate": round(hits / valid, 3), "after_bh_9": round(bh_hits / valid, 3),
            "correct_lag_given_detected": round(lag_ok / hits, 3) if hits else None,
            "smallest_possible_p": round(float(np.mean(min_p)), 4)}


def run_simulation(folder: Path, reps: int = 300) -> dict[str, Any]:
    tasks = [(n, b, reps, 1000 * i + j) for i, n in enumerate(LENGTHS) for j, b in enumerate(BETAS)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one_condition, tasks))
    out = {"phi": PHI, "true_lag_seconds": LAG * 5, "shifts_per_test": SHIFTS, "reps_per_condition": reps,
           "hypotheses_for_bh": HYPOTHESES, "conditions": rows}
    (Path(folder) / "simulation.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    return out
