"""Stitches several analysed lessons onto one real-time line and runs the lag analysis on it.

The scenario lessons were filmed one after another on one evening. Their 5-second window tables
(fusion/windows.csv, with each window's real start time) are placed on a single 5-second grid from
the first lesson's start to the last lesson's end; the minutes between lessons stay empty (NaN), so no
window is moved and no pair of windows from different lessons is ever compared at a lag.

Two versions are analysed with the pipeline's own lag analysis (src/analysis/lag_analysis.analyze):
  real_time        the stitched table as it is
  within_lesson    each feature minus its own lesson's mean, so differences BETWEEN lessons (different
                   scripts) cannot create a link; only changes within lessons remain
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import PIPELINE_ROOT


def stitch_lessons(lessons: list[tuple[str, Path]], folder: Path) -> dict[str, Any]:
    import pandas as pd
    from src.analysis.fusion import FEATURES
    from src.analysis.lag_analysis import analyze
    from src.utils.config import load_config

    settings = dict(load_config(PIPELINE_ROOT / "configs" / "default.yaml").get("fusion") or {})
    window = float(settings.get("window_seconds", 5.0))
    features = {f.name: {"camera": f.camera, "indicator": f.indicator, "labels": list(f.labels), "kind": f.kind,
                         "desirable": f.desirable, "description": f.description} for f in FEATURES}
    tables = []
    for name, lesson in lessons:
        t = pd.read_csv(lesson / "fusion" / "windows.csv")
        t["epoch"] = t["start_real_time"].map(lambda v: datetime.fromisoformat(v).timestamp())
        t["lesson"] = name
        tables.append(t)
    start = min(t["epoch"].min() for t in tables)
    end = max(t["epoch"].max() for t in tables) + window
    n = int(round((end - start) / window))
    columns = [c for c in features if c in tables[0].columns]
    grid = pd.DataFrame(np.nan, index=range(n), columns=columns)
    grid["lesson"] = ""
    for t in tables:
        idx = ((t["epoch"] - start) / window).round().astype(int)
        for c in columns:
            grid.loc[idx.values, c] = t[c].values
        grid.loc[idx.values, "lesson"] = t["lesson"].values
    grid.insert(0, "window", range(n))
    grid.insert(1, "lesson_seconds", [i * window for i in range(n)])
    centred = grid.copy()
    for name in centred["lesson"].unique():
        if not name:
            continue
        rows = centred["lesson"] == name
        for c in columns:
            centred.loc[rows, c] = centred.loc[rows, c] - centred.loc[rows, c].mean()
    occupied = int((grid["lesson"] != "").sum())
    metadata = {"window_seconds": window, "features": features}
    out: dict[str, Any] = {"lessons": [n for n, _ in lessons], "grid_windows": n, "lesson_windows": occupied,
                           "timeline_minutes": round(n * window / 60, 1),
                           "recorded_minutes": round(occupied * window / 60, 1)}
    for label, table in (("real_time", grid), ("within_lesson", centred)):
        result = analyze(table[["window"] + columns], metadata, settings)
        tested = [h for h in result["hypotheses"] if "p_shift" in h]
        out[label] = {
            "tested_teacher_features": result["tested_teacher_features"],
            "hypotheses": [{k: h.get(k) for k in ("code", "teacher_feature", "student_feature", "r", "lag_seconds",
                                                    "p_shift", "q", "null_shifts", "matches_expectation",
                                                    "not_tested")}
                           | ({"comparison": {k: h["comparison"].get(k) for k in
                                              ("outcome_with", "outcome_without", "windows_with", "windows_without",
                                               "p_hac")}} if h.get("comparison") else {})
                           for h in result["hypotheses"]],
            "hypotheses_tested": len(tested),
            "smallest_q": min((h["q"] for h in tested), default=None),
            "links_tested": len(result["links"]),
            "links_q_le_005": sum(1 for link in result["links"] if link["q"] <= 0.05),
            "findings": result["findings"],
        }
        table.to_csv(folder / f"windows_{label}.csv", index=False)
    (folder / "stitched.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf8")
    return out
