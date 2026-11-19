"""Fuse a lesson's student-view and instructor-view runs and run the lag analysis.

Reads the two saved runs (outputs/events/<id>.list_a.jsonl; no video or
model), lines them up by real time, cuts the lesson into windows, and tests
which student changes follow which teacher actions (SGP1 proposal FR 6).
Run both cameras with the indicators you want fused first (all switches on
by default, e.g. `run_scenarios.py --scenario A1 --tag full`).

Writes outputs/reports/<name>.fusion.json (links, lagged models, peri-event
changes, findings), <name>.fusion_windows.csv, <name>.fusion_links.csv, and
outputs/visualizations/<name>.fusion.png.

Examples (repository root, TeachLens environment):
    python scripts/fuse_lesson.py --scenario A1 --tag full
    python scripts/fuse_lesson.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1
"""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis.fusion import build_windows, load_events, onsets, overlap_range  # noqa: E402
from src.analysis.lag_analysis import analyze  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import file_in_use, project_path  # noqa: E402

LOGGER = logging.getLogger("fuse_lesson")
# Teacher action onsets for peri-event averages: (indicator, labels).
TEACHER_EVENTS = {
    "board_or_screen_start": ("board_interaction", ("board", "screen")),
    "writing_start": ("writing_on_board", ("writing",)),
    "pointing_start": ("pointing", ("board", "screen", "desk", "students", "side")),
    "pointing_at_students_start": ("pointing", ("students",)),
    "turn_to_class": ("facing_direction", ("toward_class",)),
    "walking_start": ("gesture", ("walking",)),
    "raised_hand_start": ("raised_hand", ("one", "both")),
}
LIMITATIONS = [
    "Associations within one recording, not causes: a teacher behaviour that precedes a student change "
    "may share a cause with it (the lesson plan, the topic, a script).",
    "Cameras are lined up by whole-second recording tags: about +/-1 s, hence 5 s windows.",
    "Student values are class averages of the known student time; unknown time is excluded, not zero.",
    "Teacher features present in fewer than min_active_windows windows are not tested by correlation; "
    "their onsets get peri-event averages when there are enough of them.",
    "Short or scripted lessons have little variation and few windows; findings need full lectures or "
    "several lessons before they say anything about teaching.",
]


def run_ids(scenario: str, tag: str | None, take: str | None = None) -> tuple[str, str]:
    """Student and instructor run IDs of a manifest scenario (video_id + _tag)."""
    from run_scenarios import DEFAULT_MANIFEST, read_manifest
    rows = [row for row in read_manifest(DEFAULT_MANIFEST) if row["scenario"].upper() == scenario.upper()
            and (take is None or row["take"] == take)]
    ids: dict[str, list[str]] = {}
    for row in rows:
        ids.setdefault(row["camera"], []).append(f"{row['video_id']}_{tag}" if tag else row["video_id"])
    if set(ids) != {"student_view", "instructor_view"}:
        raise ValueError(f"Scenario {scenario} needs a student_view and an instructor_view row in the manifest")
    if any(len(v) > 1 for v in ids.values()):
        raise ValueError(f"Scenario {scenario} has several takes; pass --take or --student/--instructor")
    return ids["student_view"][0], ids["instructor_view"][0]


# Categorical hues in fixed order (validated: light surface, CVD-separated;
# below 3:1 contrast for some, so every series is in the legend and the
# window CSV is the table view).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"


def plot(table, name: str, path: Path) -> None:
    """Teacher shares (top) and class shares (bottom) per window; gaps = too little evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    half = table["lesson_seconds"].diff().fillna(0).max() / 2
    t = (table["lesson_seconds"] + half) / 60
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, facecolor=SURFACE)
    panels = ((axes[0], "teacher share of the window",
               (("T_facing_class", "facing class"), ("T_board_or_screen", "board or screen"),
                ("T_gesturing", "gesturing"), ("T_walking", "walking"), ("T_writing", "writing"))),
              (axes[1], "class share of the window",
               (("S_on_target", "on target"), ("S_phone", "phone"), ("S_talking", "talking"),
                ("S_head_down", "head down"), ("S_sleep", "dozing or sleep"))))
    for axis, label, series in panels:
        axis.set_facecolor(SURFACE)
        for color, (column, name_) in zip(SERIES_COLORS, series):
            if column in table:
                axis.plot(t, table[column], color=color, linewidth=1.5, marker="o", markersize=2.5, label=name_)
        axis.set_ylabel(label, color=INK, fontsize=9)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#c9c8c2")
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, labelcolor=INK)
    if "S_hand_raises" in table:
        for minute in (table.loc[table["S_hand_raises"] > 0, "lesson_seconds"] + half) / 60:
            axes[1].axvline(minute, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("lesson minutes (window centres; dotted lines: hand raises; gaps: too little evidence)",
                      color=INK, fontsize=9)
    fig.suptitle(f"{name}: teacher and student timelines", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Manifest scenario (e.g. A1); run IDs come from the manifest")
    parser.add_argument("--tag", help="Tag the runs were made with (run_scenarios.py --tag)")
    parser.add_argument("--take", help="Manifest take when a scenario has several (e.g. 1)")
    parser.add_argument("--student", help="Student-view run ID (instead of --scenario)")
    parser.add_argument("--instructor", help="Instructor-view run ID (instead of --scenario)")
    parser.add_argument("--name", help="Report name (default sgp_<scenario> or the student run ID)")
    parser.add_argument("--window", type=float, help="Window seconds (default fusion.window_seconds)")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.scenario:
            student_id, instructor_id = run_ids(args.scenario, args.tag, args.take)
        elif args.student and args.instructor:
            student_id, instructor_id = args.student, args.instructor
        else:
            parser.error("give --scenario (with --tag) or both --student and --instructor")
        name = args.name or (f"sgp_{args.scenario}" if args.scenario else student_id)
        settings = (load_config(args.config).get("fusion") or {})
        window = float(args.window or settings.get("window_seconds", 5.0))
        outputs = {"json": args.output_dir / "reports" / f"{name}.fusion.json",
                   "windows": args.output_dir / "reports" / f"{name}.fusion_windows.csv",
                   "links": args.output_dir / "reports" / f"{name}.fusion_links.csv",
                   "plot": args.output_dir / "visualizations" / f"{name}.fusion.png"}
        for key, path in outputs.items():
            if key == "plot" and not args.plot:
                continue
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; use --overwrite or another --name")
            if path.exists() and file_in_use(path):
                raise PermissionError(f"{path} is open in another program; close it and run again")
        events = {}
        for view, run in (("student", student_id), ("instructor", instructor_id)):
            path = args.output_dir / "events" / f"{run}.list_a.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing run {path}; run that camera first")
            events[view] = load_events(path)
        table, metadata = build_windows(events["student"], events["instructor"], window_seconds=window,
                                        min_known_fraction=float(settings.get("min_known_fraction", 0.5)))
        start, _ = overlap_range(events["student"], events["instructor"])
        teacher_onsets = {name_: onsets(events["instructor"], indicator, "teacher", labels, start, window,
                                        len(table), float(settings.get("onset_min_gap_seconds", 10)))
                          for name_, (indicator, labels) in TEACHER_EVENTS.items()}
        results = analyze(table, metadata, settings, teacher_onsets)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    report = {"lesson": name, "student_run": student_id, "instructor_run": instructor_id,
              **metadata, "teacher_onsets": teacher_onsets, **results, "limitations": LIMITATIONS}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table.to_csv(outputs["windows"], index=False, float_format="%.4f")
    import pandas as pd
    pd.DataFrame(results["links"]).to_csv(outputs["links"], index=False)
    if args.plot:
        plot(table, name, outputs["plot"])
    print(f"{metadata['windows']} windows of {window:g} s, {metadata['start_real_time'][11:19]}-"
          f"{metadata['end_real_time'][11:19]}")
    untested = ", ".join(f"{k} ({v})" for k, v in results["untested_teacher_features"].items())
    print(f"tested teacher features: {', '.join(results['tested_teacher_features']) or 'none'}")
    print(f"too rare to test (windows present): {untested or 'none'}")
    print("strongest links (teacher -> later students):")
    for link in results["links"][:8]:
        print(f"  {link['teacher_feature']:<24} -> {link['student_feature']:<16} lag {link['lag_seconds']:>4g} s "
              f"r {link['r']:+.2f}  p {link['p_shift']:.3f}  q {link['q']:.3f}")
    print(f"findings (q <= {results['settings']['alpha']}): {len(results['findings'])}")
    for finding in results["findings"][:8]:
        print(f"  {finding['kind']:<11} {finding['teacher_feature']} -> {finding['student_feature']} "
              f"({finding['method']}, effect {finding['effect']:+.3f}, q {finding['q']:.3f})")
    for key, path in outputs.items():
        if key != "plot" or args.plot:
            print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
