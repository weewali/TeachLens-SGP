"""Scores the saved pipeline output against the blind reference labels (report/evaluation_plan.md).

Inputs in one evaluation folder: items.csv (from `make`), labels.txt and labels_repeat.txt.
Label line: frame_id | S<track>:<codes> | ... | I<track>:<codes> | n=<students visible>
  students: F front/focus, A away, D down, U cannot tell; flags P phone, L laptop, W writing, R reading,
            H hand raised, Z dozing, X not a real student (duplicate box, the instructor, a passer-by)
  instructor: st/se/u posture; fc/fa/fu facing; flags g gesturing, b at board/screen, rd reading notes, x not him
Pipeline values come from the same events the fusion layer reads (src/analysis/fusion.py), looked up at the
real time of the analysed frame the picture was taken from, and the on/off-task rule is the pipeline's own
(fusion.student_state), so the two sides are compared like for like.
Output: metrics.json and metrics.md in the folder.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import PIPELINE_ROOT  # noqa: F401  (puts the pipeline on sys.path)
from src.analysis.fusion import STATE_INDICATORS, load_events, student_state  # noqa: E402

RNG = np.random.default_rng(20261120)
BOOT = 2000


# ----------------------------------------------------------------------------- labels

def parse_labels(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.startswith("sgp_"):
            continue
        parts = [p.strip() for p in line.split("|")]
        frame = {"students": {}, "instructor": {}, "n": None}
        for part in parts[1:]:
            if part.startswith("n="):
                frame["n"] = int(part[2:])
                continue
            who, codes = part.split(":", 1)
            codes = codes.split(",")
            if who.startswith("S"):
                frame["students"][int(who[1:])] = codes
            elif who.startswith("I"):
                frame["instructor"][int(who[1:])] = codes
        out[parts[0]] = frame
    return out


def ref_state(codes: list[str]) -> str:
    """Reference on/off task with the pipeline's definition: off wins; looking down alone is unknown."""
    if "X" in codes:
        return "not_student"
    if "P" in codes or "Z" in codes or codes[0] == "A":
        return "off_task"
    if codes[0] == "F" or "W" in codes or "R" in codes:
        return "on_task"
    return "unknown"


# ----------------------------------------------------------------------------- pipeline lookup

class Timeline:
    """Interval events of one camera run: value of (indicator, track) at a real time."""

    def __init__(self, events: list[dict[str, Any]]):
        self.by_key: dict[tuple[str, Any], tuple[list[float], list[tuple[float, Any]]]] = {}
        self.samples: dict[str, dict[float, Any]] = defaultdict(dict)
        for e in sorted(events, key=lambda e: e["t0"]):
            name = e["indicator_name"]
            if name in ("visible_student_count", "teacher_presence_sample"):
                self.samples[name][round(e["t0"], 3)] = e["value"]
                continue
            starts, spans = self.by_key.setdefault((name, e.get("track_id")), ([], []))
            starts.append(e["t0"])
            spans.append((e["t1"], e["value"]))

    def at(self, name: str, track: Any, t: float, tol: float = 0.05) -> Any:
        entry = self.by_key.get((name, track))
        if not entry:
            return None
        starts, spans = entry
        k = bisect_right(starts, t + tol) - 1
        while k >= 0 and starts[k] > t + tol:
            k -= 1
        if k >= 0 and t <= spans[k][0] + tol:
            return spans[k][1]
        return None

    def sample(self, name: str, t: float) -> Any:
        values = self.samples.get(name, {})
        if not values:
            return None
        keys = np.array(list(values))
        k = int(np.argmin(np.abs(keys - t)))
        return values[float(keys[k])] if abs(keys[k] - t) <= 0.6 else None


def pipe_state(tl: Timeline, track: int, t: float) -> str:
    labels = {}
    for name in STATE_INDICATORS:
        v = tl.at(name, track, t)
        if v is not None:
            labels[name] = v
    return student_state(labels)


# ----------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def binary(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs: (reference positive, pipeline positive)."""
    tp = sum(1 for r, p in pairs if r and p)
    fp = sum(1 for r, p in pairs if not r and p)
    fn = sum(1 for r, p in pairs if r and not p)
    tn = sum(1 for r, p in pairs if not r and not p)
    n = len(pairs)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
    spec = tn / (tn + fp) if tn + fp else None
    po = (tp + tn) / n if n else None
    pe = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)) if n else None
    kappa = (po - pe) / (1 - pe) if n and pe is not None and pe < 1 else None
    r3 = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "positives": tp + fn,
            "precision": r3(prec), "precision_ci": wilson(tp, tp + fp), "recall": r3(rec),
            "recall_ci": wilson(tp, tp + fn), "f1": r3(f1), "specificity": r3(spec), "accuracy": r3(po),
            "balanced_accuracy": r3((rec + spec) / 2) if rec is not None and spec is not None else None,
            "kappa": r3(kappa)}


def kappa_multi(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def cluster_boot(rows: list[dict[str, Any]], stat, key: str = "cluster") -> list[float] | None:
    """95% percentile interval of stat(rows) resampling whole clusters (students or lessons)."""
    clusters = defaultdict(list)
    for r in rows:
        clusters[r[key]].append(r)
    names = list(clusters)
    if len(names) < 3:
        return None
    values = []
    for _ in range(BOOT):
        pick = RNG.choice(len(names), size=len(names), replace=True)
        sample = [r for i in pick for r in clusters[names[i]]]
        v = stat(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def ccc(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    d = vx + vy + (mx - my) ** 2
    return round(float(2 * cov / d), 3) if d > 0 else None


# ----------------------------------------------------------------------------- scoring

def score_folder(folder: Path) -> dict[str, Any]:
    folder = Path(folder)
    items = {r["frame_id"]: r for r in csv.DictReader((folder / "items.csv").open(encoding="utf8"))}
    labels = parse_labels(folder / "labels.txt")
    repeat = parse_labels(folder / "labels_repeat.txt") if (folder / "labels_repeat.txt").exists() else {}
    runs: dict[str, tuple[Timeline, Timeline]] = {}
    for item in items.values():
        if item["lesson"] not in runs:
            lesson = Path(item["folder"])
            runs[item["lesson"]] = (Timeline(load_events(lesson / "student" / "events.jsonl")),
                                    Timeline(load_events(lesson / "instructor" / "events.jsonl")))

    student_rows, frame_rows, teacher_rows = [], [], []
    for fid, lab in labels.items():
        item = items[fid]
        lesson = item["lesson"]
        st_tl, te_tl = runs[lesson]
        ts = float(item["student_real_time"]) if item["student_real_time"] else None
        tt = float(item["instructor_real_time"]) if item["instructor_real_time"] else None
        tracked = [int(x) for x in item["student_tracks"].split()] if item["student_tracks"] else []
        real = [k for k, c in lab["students"].items() if "X" not in c]
        count = st_tl.sample("visible_student_count", ts) if ts else None
        frame_rows.append({"frame": fid, "lesson": lesson, "cluster": lesson, "n_ref": lab["n"],
                           "n_pipe": count, "tracked_real": len(real),
                           "false_boxes": sum(1 for c in lab["students"].values() if "X" in c),
                           "missed": max(0, (lab["n"] or 0) - len(real))})
        for track, codes in lab["students"].items():
            if "X" in codes or ts is None:
                continue
            row = {"frame": fid, "lesson": lesson, "cluster": f"{lesson}:S{track}", "track": track,
                   "ref_state": ref_state(codes), "pipe_state": pipe_state(st_tl, track, ts),
                   "ref_facing": codes[0], "pipe_orientation": st_tl.at("orientation_proxy", track, ts),
                   "ref_phone": "P" in codes, "pipe_device": st_tl.at("device_use", track, ts),
                   "ref_doze": "Z" in codes, "pipe_alert": st_tl.at("alertness", track, ts),
                   "ref_hand": "H" in codes, "pipe_hand": st_tl.at("hand_raise_episode", track, ts),
                   "ref_write": "W" in codes, "pipe_note": st_tl.at("note_taking", track, ts)}
            student_rows.append(row)
        for track, codes in lab["instructor"].items():
            if "x" in codes or tt is None:
                continue
            teacher_rows.append({"frame": fid, "lesson": lesson, "cluster": lesson,
                                 "ref_posture": codes[0], "pipe_posture": te_tl.at("sitting_standing", track, tt),
                                 "ref_facing": codes[1], "pipe_facing": te_tl.at("facing_direction", track, tt),
                                 "ref_gesture": "g" in codes, "pipe_gesture": te_tl.at("gesture", track, tt),
                                 "pipe_presence": te_tl.sample("teacher_presence_sample", tt)})

    result: dict[str, Any] = {"frames": len(labels), "lessons": sorted(runs),
                              "student_observations": len(student_rows), "instructor_observations": len(teacher_rows)}

    # L1 people: count and detection
    counted = [r for r in frame_rows if r["n_pipe"] is not None and r["n_ref"] is not None]
    exact = sum(1 for r in counted if r["n_pipe"] == r["n_ref"])
    result["count"] = {"frames": len(counted), "exact": exact, "accuracy": round(exact / len(counted), 3),
                       "accuracy_ci": wilson(exact, len(counted)),
                       "mae": round(float(np.mean([abs(r["n_pipe"] - r["n_ref"]) for r in counted])), 3),
                       "within_one": round(sum(1 for r in counted if abs(r["n_pipe"] - r["n_ref"]) <= 1) / len(counted), 3)}
    boxes = sum(r["tracked_real"] + r["false_boxes"] for r in frame_rows)
    real = sum(r["tracked_real"] for r in frame_rows)
    people = sum(r["n_ref"] or 0 for r in frame_rows)
    result["detection"] = {"boxes": boxes, "real_people_boxed": real, "false_boxes": boxes - real,
                           "people_visible": people, "precision": round(real / boxes, 3), "precision_ci": wilson(real, boxes),
                           "recall": round(real / people, 3), "recall_ci": wilson(real, people)}

    # L2 on/off task
    ref_known = [r for r in student_rows if r["ref_state"] in ("on_task", "off_task")]
    both = [r for r in ref_known if r["pipe_state"] in ("on_task", "off_task")]
    pairs = [(r["ref_state"] == "off_task", r["pipe_state"] == "off_task") for r in both]
    m = binary(pairs)
    m["coverage"] = round(len(both) / len(ref_known), 3) if ref_known else None
    m["coverage_ci"] = wilson(len(both), len(ref_known))
    m["reference_known"] = len(ref_known)
    m["pipeline_unknown_when_reference_known"] = len(ref_known) - len(both)
    m["kappa_ci"] = cluster_boot(both, lambda s: binary([(r["ref_state"] == "off_task", r["pipe_state"] == "off_task")
                                                         for r in s])["kappa"])
    m["f1_ci"] = cluster_boot(both, lambda s: binary([(r["ref_state"] == "off_task", r["pipe_state"] == "off_task")
                                                      for r in s])["f1"])
    m["per_lesson"] = {}
    for lesson in sorted(runs):
        sub = [r for r in both if r["lesson"] == lesson]
        sub_known = [r for r in ref_known if r["lesson"] == lesson]
        if sub:
            b = binary([(r["ref_state"] == "off_task", r["pipe_state"] == "off_task") for r in sub])
            m["per_lesson"][lesson] = {"n": len(sub), "coverage": round(len(sub) / len(sub_known), 3),
                                       "accuracy": b["accuracy"], "kappa": b["kappa"], "f1_off_task": b["f1"]}
    result["on_off_task"] = m

    # L2 single indicators (each scored where the pipeline gave a value)
    def indicator(rows, ref_key, pipe_key, positive):
        covered = [r for r in rows if r[pipe_key] not in (None, "unknown")]
        out = binary([(r[ref_key], r[pipe_key] in positive) for r in covered])
        out["coverage"] = round(len(covered) / len(rows), 3) if rows else None
        out["reference_positives_total"] = sum(1 for r in rows if r[ref_key])
        out["recall_all"] = (round(sum(1 for r in rows if r[ref_key] and r[pipe_key] in positive)
                                   / out["reference_positives_total"], 3)
                             if out["reference_positives_total"] else None)
        return out

    for r in student_rows:
        r["pipe_hand_any"] = "raised" if isinstance(r["pipe_hand"], dict) else "none"
    result["phone"] = indicator(student_rows, "ref_phone", "pipe_device", ("phone", "phone_likely"))
    result["dozing"] = indicator(student_rows, "ref_doze", "pipe_alert", ("dozing", "possible_sleep"))
    result["writing"] = indicator(student_rows, "ref_write", "pipe_note", ("writing",))
    result["hand_raised"] = binary([(r["ref_hand"], r["pipe_hand_any"] == "raised") for r in student_rows])
    for r in student_rows:
        r["ref_front"] = r["ref_facing"] == "F"
    facing_rows = [r for r in student_rows if r["ref_facing"] in ("F", "A")]
    result["student_facing_front"] = indicator(facing_rows, "ref_front", "pipe_orientation", ("on_target", "speaking"))

    # instructor
    t_post = [r for r in teacher_rows if r["ref_posture"] in ("st", "se")]
    for r in t_post:
        r["ref_seated"] = r["ref_posture"] == "se"
    result["instructor_seated"] = indicator(t_post, "ref_seated", "pipe_posture", ("sitting",))
    t_face = [r for r in teacher_rows if r["ref_facing"] in ("fc", "fa")]
    for r in t_face:
        r["ref_fc"] = r["ref_facing"] == "fc"
    result["instructor_facing_class"] = indicator(t_face, "ref_fc", "pipe_facing", ("toward_class",))
    result["instructor_gesturing"] = indicator(teacher_rows, "ref_gesture", "pipe_gesture", ("gesturing",))
    present = [r for r in teacher_rows if r["pipe_presence"] is not None]
    k = sum(1 for r in present if r["pipe_presence"] == "visible")
    result["instructor_presence"] = {"frames": len(present), "visible": k, "recall": round(k / len(present), 3),
                                     "recall_ci": wilson(k, len(present))}

    # L4 class on-task share per frame and per lesson
    shares = []
    for fid in labels:
        rows = [r for r in student_rows if r["frame"] == fid]
        rk = [r for r in rows if r["ref_state"] in ("on_task", "off_task")]
        pk = [r for r in rows if r["pipe_state"] in ("on_task", "off_task")]
        if len(rk) >= 3 and len(pk) >= 3:
            shares.append({"frame": fid, "lesson": items[fid]["lesson"], "cluster": items[fid]["lesson"],
                           "ref": 100 * sum(r["ref_state"] == "on_task" for r in rk) / len(rk),
                           "pipe": 100 * sum(r["pipe_state"] == "on_task" for r in pk) / len(pk)})
    x = np.array([s["pipe"] for s in shares])
    y = np.array([s["ref"] for s in shares])
    diff = x - y
    result["on_task_share"] = {
        "frames": len(shares), "mae_points": round(float(np.abs(diff).mean()), 1),
        "bias_points": round(float(diff.mean()), 1),
        "limits_of_agreement": [round(float(diff.mean() - 1.96 * diff.std(ddof=1)), 1),
                                round(float(diff.mean() + 1.96 * diff.std(ddof=1)), 1)],
        "ccc": ccc(x, y),
        "mae_ci": cluster_boot(shares, lambda s: float(np.mean([abs(r["pipe"] - r["ref"]) for r in s]))),
        "per_lesson": {lesson: {"reference_pct": round(float(np.mean([s["ref"] for s in shares if s["lesson"] == lesson])), 1),
                                "pipeline_pct": round(float(np.mean([s["pipe"] for s in shares if s["lesson"] == lesson])), 1)}
                       for lesson in sorted(runs) if any(s["lesson"] == lesson for s in shares)}}

    # repeat-pass agreement (intra-labeller)
    if repeat:
        a_state, b_state, a_face, b_face, a_tp, b_tp, a_tf, b_tf = [], [], [], [], [], [], [], []
        flags = defaultdict(lambda: [[], []])
        for fid, second in repeat.items():
            first = labels.get(fid)
            if not first:
                continue
            for track, codes in second["students"].items():
                if track in first["students"]:
                    c1 = first["students"][track]
                    a_state.append(ref_state(c1))
                    b_state.append(ref_state(codes))
                    a_face.append(c1[0])
                    b_face.append(codes[0])
                    for f in ("P", "Z", "H", "W", "L"):
                        flags[f][0].append(f in c1)
                        flags[f][1].append(f in codes)
            for track, codes in second["instructor"].items():
                if track in first["instructor"]:
                    c1 = first["instructor"][track]
                    a_tp.append(c1[0])
                    b_tp.append(codes[0])
                    a_tf.append(c1[1])
                    b_tf.append(codes[1])
        result["repeat_agreement"] = {
            "frames": len(repeat), "student_observations": len(a_state),
            "on_off_task_kappa": kappa_multi(a_state, b_state),
            "on_off_task_agreement": round(sum(x == y for x, y in zip(a_state, b_state)) / len(a_state), 3),
            "facing_kappa": kappa_multi(a_face, b_face),
            "flags_kappa": {f: kappa_multi([str(v) for v in a], [str(v) for v in b]) for f, (a, b) in flags.items()},
            "instructor_posture_agreement": round(sum(x == y for x, y in zip(a_tp, b_tp)) / len(a_tp), 3) if a_tp else None,
            "instructor_facing_kappa": kappa_multi(a_tf, b_tf)}

    (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    with (folder / "observations.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in student_rows[0] if not k.startswith("pipe_hand")])
        writer.writeheader()
        for r in student_rows:
            writer.writerow({k: v for k, v in r.items() if k in writer.fieldnames})
    return result
