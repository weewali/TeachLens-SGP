"""Makes tests/fixtures/fusion_example.json: a fusion layer report on MADE-UP data, for the report step's tests and demo.

The window table is synthetic, with links planted on purpose, but the analysis is the team's real code
(src/analysis/lag_analysis.analyze, settings from configs/default.yaml `fusion:`), and the document is put together the
way scripts/fuse_lesson.py does it, so its shape follows the real output. Not a real lecture.

Only some indicators are used (as when a run switches the others off): with all 24 the fusion tests about 112 pairs,
and in a lesson this short (150 windows) no single link can then reach q <= 0.05, so the example would have no
findings to test with. Planted (window = 5 s): gesturing -> fewer students facing away 10 s later; standing still ->
more phone use 15 s later; facing the class -> more talking 5 s later (neither good nor bad news); after each turn to
the class, more students face the front for 20 s. The board or screen is used in 1 window only (too rare to test).

    python tests/fixtures/make_fusion_example.py        # rewrites the fixture (about 6 s; same seed, same file)
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.fusion import FEATURES  # noqa: E402
from src.analysis.lag_analysis import analyze  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "fusion_example.json"
WINDOW = 5.0
USED = ("T_facing_class", "T_board_or_screen", "T_gesturing", "T_walking", "T_stationary",
        "S_on_target", "S_off_target", "S_phone", "S_talking", "S_hand_raises")
USED_EVENTS = ("board_or_screen_start", "turn_to_class", "walking_start")
NOTE = ("EXAMPLE ONLY: a made-up window table with planted links, analysed by the team's real fusion code "
        "(tests/fixtures/make_fusion_example.py). Not a real lecture.")


def fuse_lesson_module():
    """scripts/fuse_lesson.py (its TEACHER_EVENTS and LIMITATIONS), loaded without running it."""
    spec = importlib.util.spec_from_file_location("fuse_lesson", ROOT / "scripts" / "fuse_lesson.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fusion_settings():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["fusion"]


def feature_metadata(names=None):
    return {f.name: {"camera": f.camera, "indicator": f.indicator, "labels": list(f.labels), "kind": f.kind,
                     "desirable": f.desirable, "description": f.description}
            for f in FEATURES if names is None or f.name in names}


def shifted(x, lag):
    out = np.zeros_like(x)
    out[lag:] = x[:len(x) - lag]
    return out


def example_table(n=150, seed=7):
    """(window table, teacher onsets) with the planted links above."""
    rng = np.random.default_rng(seed)
    noise = lambda sd: rng.normal(0, sd, n)
    clip = lambda x: np.clip(x, 0.0, 1.0)
    bursts = lambda p, low=0.4: (rng.random(n) < p) * rng.uniform(low, 1.0, n)

    facing = clip(0.55 + 0.25 * np.sign(np.sin(np.arange(n) / 6.0)) + noise(0.1))
    gesturing = bursts(0.35)
    stationary = bursts(0.4, 0.6)
    walking = bursts(0.12)
    board = np.zeros(n)
    board[40] = 1.0
    turn_to_class = [8, 22, 37, 55, 71, 88, 104, 121, 137]
    teacher = {
        "T_facing_class": facing, "T_looking_at_students": clip(facing - 0.1 + noise(0.08)),
        "T_board_or_screen": board, "T_writing": np.zeros(n), "T_pointing": bursts(0.1),
        "T_pointing_at_students": bursts(0.06), "T_gesturing": gesturing, "T_walking": walking,
        "T_stationary": stationary, "T_path": walking * rng.uniform(0.2, 0.8, n),
    }
    off_target = clip(0.18 - 0.12 * shifted(gesturing, 2) + noise(0.02))
    looking_down = clip(0.15 + noise(0.04))
    lift = np.zeros(n)
    for w in turn_to_class:
        lift[w + 1:w + 5] = 0.08
    hands = rng.poisson(0.15, n).astype(float)
    students = {
        "S_on_target": clip(1.0 - off_target - looking_down + lift), "S_off_target": off_target,
        "S_looking_down": looking_down, "S_phone": clip(0.05 + 0.12 * shifted(stationary, 3) + noise(0.02)),
        "S_laptop": clip(0.03 + noise(0.01)), "S_writing": clip(0.1 + noise(0.04)), "S_reading": clip(0.05 + noise(0.02)),
        "S_talking": clip(0.08 + 0.2 * shifted(facing, 1) + noise(0.03)), "S_smiling": clip(0.06 + noise(0.03)),
        "S_head_down": clip(0.12 + noise(0.04)), "S_sleep": clip(0.02 + noise(0.01)), "S_moving": clip(0.1 + noise(0.04)),
        "S_hand_raises": hands, "S_answers": np.minimum(hands, (rng.random(n) < 0.5).astype(float)),
    }
    table = pd.DataFrame({"window": range(n), "lesson_seconds": [w * WINDOW for w in range(n)], **teacher, **students})
    onsets = {"board_or_screen_start": [40], "writing_start": [], "pointing_start": [15, 60, 99, 130],
              "pointing_at_students_start": [30, 112], "turn_to_class": turn_to_class,
              "walking_start": [12, 47, 80, 115], "raised_hand_start": []}
    return table, onsets


def fusion_report(table, onsets, settings, features=None, lesson="example"):
    """The document scripts/fuse_lesson.py writes, for a window table made elsewhere."""
    module = fuse_lesson_module()
    n = len(table)
    metadata = {"window_seconds": WINDOW, "windows": n, "start_real_time": "2026-01-01T10:00:00.000+04:00",
                "end_real_time": None, "min_known_fraction": settings.get("min_known_fraction", 0.5),
                "features": features or feature_metadata()}
    onsets = {name: windows for name, windows in onsets.items() if name in module.TEACHER_EVENTS}
    results = analyze(table, metadata, settings, onsets)
    return {"note": NOTE, "lesson": lesson, "student_run": f"{lesson}_student", "instructor_run": f"{lesson}_instructor",
            **metadata, "teacher_onsets": onsets, **results, "limitations": module.LIMITATIONS}


def main():
    table, onsets = example_table()
    table = table[["window", "lesson_seconds", *USED]]
    onsets = {name: onsets[name] for name in USED_EVENTS}
    report = fusion_report(table, onsets, fusion_settings(), feature_metadata(USED))
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(report['links'])} links, {len(report['findings'])} findings")
    for finding in report["findings"]:
        print(f"  {finding['kind']:<11} {finding['teacher_feature']} -> {finding['student_feature']} "
              f"({finding['method']}, effect {finding['effect']:+.3f}, q {finding['q']:.3f})")


if __name__ == "__main__":
    main()
