"""Fuse one lesson's two camera runs into a window table on real time.

Proposal FR 6.1-6.2 (SGP1 report, section 4.1): align instructor and student
outputs on one lecture timeline and divide it into fixed analysis windows.
Input is saved evidence only (the two runs' JSONL), never video. Both runs
need real (wall-clock) times; the cameras then line up to about +/-1 s,
which is why windows are several seconds long (default 5 s).

Each window gets one value per feature:
- share: the share of the window's known time with the given labels. For
  the teacher (one person) that is the time share; for students it is the
  share of known student-seconds (summed over the class), i.e. the class
  average. Unknown time is excluded, and a window with less than
  min_known_fraction of a window of known time is NaN, not 0.
- count: events starting in the window (hand raises, answers).
- path: the teacher's hip-centre path length in the window (frame heights).
Coverage columns say how much evidence each window has.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Feature:
    """One window feature from one camera's saved events."""
    name: str                   # column name, T_ (teacher) or S_ (students) prefix
    camera: str                 # "teacher" or "students"
    indicator: str              # indicator_name in the JSONL
    labels: tuple[str, ...]     # labels counted (share); ignored for path
    kind: str = "share"         # share, count, path
    desirable: int = 0          # students only: +1 more is better, -1 more is worse, 0 neutral
    description: str = ""


# The proposal's teacher features (explanation, board writing, movement,
# interaction) and student features (attention, off-task, hand raises, phone,
# reading, writing), mapped to the indicators that exist (2026-11-18).
FEATURES: tuple[Feature, ...] = (
    Feature("T_facing_class", "teacher", "facing_direction", ("toward_class",),
            description="teacher faces the class (explanation proxy, no audio)"),
    Feature("T_looking_at_students", "teacher", "head_orientation", ("students",),
            description="teacher's head turned to the students"),
    Feature("T_board_or_screen", "teacher", "board_interaction", ("board", "screen"),
            description="teacher works at the board or screen"),
    Feature("T_writing", "teacher", "writing_on_board", ("writing",), description="board writing"),
    Feature("T_pointing", "teacher", "pointing", ("board", "screen", "desk", "students", "side"),
            description="teacher points at something"),
    Feature("T_pointing_at_students", "teacher", "pointing", ("students",),
            description="teacher points toward the class (interaction proxy)"),
    Feature("T_gesturing", "teacher", "gesture", ("gesturing",), description="hand gestures"),
    Feature("T_walking", "teacher", "gesture", ("walking",), description="teacher walks"),
    Feature("T_stationary", "teacher", "stationary_episode", (), kind="episode",
            description="teacher stands still (stationary episode)"),
    Feature("T_path", "teacher", "position", (), kind="path",
            description="teacher hip path length in the window (frame heights)"),
    Feature("S_on_target", "students", "orientation_proxy", ("on_target",), desirable=+1,
            description="students facing the current focus (attention proxy)"),
    Feature("S_off_target", "students", "orientation_proxy", ("off_target",), desirable=-1,
            description="students facing elsewhere"),
    Feature("S_looking_down", "students", "orientation_proxy", ("looking_down",),
            description="students looking down (notes, phone, reading)"),
    Feature("S_phone", "students", "device_use", ("phone", "phone_likely"), desirable=-1,
            description="phone seen in hand or head down for a sustained time"),
    Feature("S_laptop", "students", "device_use", ("laptop",), description="open laptop in use"),
    Feature("S_writing", "students", "note_taking", ("writing",), desirable=+1, description="writing"),
    Feature("S_reading", "students", "reading", ("reading",), description="reading"),
    Feature("S_talking", "students", "talking", ("talking",), description="mouth movement (talking)"),
    Feature("S_smiling", "students", "smile", ("smiling",), description="smiling"),
    Feature("S_head_down", "students", "head_down", ("down",), desirable=-1, description="head down"),
    Feature("S_sleep", "students", "alertness", ("possible_sleep", "dozing"), desirable=-1,
            description="dozing or possible sleep"),
    Feature("S_moving", "students", "motion_level", ("medium", "high"), description="body movement"),
    Feature("S_hand_raises", "students", "hand_raise_episode", (), kind="count", desirable=+1,
            description="hand raises starting in the window"),
    Feature("S_answers", "students", "hand_raise_outcome", ("answered_likely",), kind="count", desirable=+1,
            description="answers after a raise starting in the window"),
)
FEATURES_BY_NAME = {feature.name: feature for feature in FEATURES}
CAMERA_ROLE = {"teacher": "teacher", "students": "student"}


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def load_events(path: Path) -> list[dict[str, Any]]:
    """Events with real times, plus t0/t1 in epoch seconds. Raises without real times."""
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("start_real_time") and event.get("end_real_time"):
                event["t0"], event["t1"] = _epoch(event["start_real_time"]), _epoch(event["end_real_time"])
                events.append(event)
    if not events:
        raise ValueError(f"{path} has no events with real times; fusion needs both cameras' real times")
    return events


def overlap_range(student: Sequence[dict[str, Any]], teacher: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Real-time span both cameras analyzed (from their position samples, else any event)."""
    def span(events):
        times = [e for e in events if e["indicator_name"] == "position"] or list(events)
        return min(e["t0"] for e in times), max(e["t1"] for e in times)
    (s0, s1), (t0, t1) = span(student), span(teacher)
    start, end = max(s0, t0), min(s1, t1)
    if end <= start:
        raise ValueError("The two runs do not overlap in real time; check the runs and their recording starts")
    return start, end


class _Grid:
    def __init__(self, start: float, seconds: float, count: int):
        self.start, self.seconds, self.count = start, seconds, count

    def spread(self, t0: float, t1: float) -> Iterable[tuple[int, float]]:
        """(window, seconds of [t0, t1] inside it); instants give (window, 0)."""
        first = max(0, int(math.floor((t0 - self.start) / self.seconds)))
        last = min(self.count - 1, int(math.floor((t1 - self.start) / self.seconds)))
        for w in range(first, last + 1):
            a = self.start + w * self.seconds
            yield w, max(0.0, min(a + self.seconds, t1) - max(a, t0))

    def index(self, t: float) -> int | None:
        w = int(math.floor((t - self.start) / self.seconds))
        return w if 0 <= w < self.count else None


def _share(events, feature: Feature, grid: _Grid, min_known: float) -> tuple[list[float], list[float]]:
    role = CAMERA_ROLE[feature.camera]
    hit, known = [0.0] * grid.count, [0.0] * grid.count
    for e in events:
        if e["indicator_name"] != feature.indicator or e["role"] != role or e["value"] == "unknown":
            continue
        for w, seconds in grid.spread(e["t0"], e["t1"]):
            known[w] += seconds
            if e["value"] in feature.labels:
                hit[w] += seconds
    values = [h / k if k > 0 and k >= min_known else math.nan for h, k in zip(hit, known)]
    return values, known


def _episode_share(events, feature: Feature, grid: _Grid, presence: list[float]) -> list[float]:
    role = CAMERA_ROLE[feature.camera]
    inside = [0.0] * grid.count
    for e in events:
        if e["indicator_name"] == feature.indicator and e["role"] == role:
            for w, seconds in grid.spread(e["t0"], e["t1"]):
                inside[w] += seconds
    return [min(1.0, i / grid.seconds) if p > 0 else math.nan for i, p in zip(inside, presence)]


def _count(events, feature: Feature, grid: _Grid) -> list[float]:
    role = CAMERA_ROLE[feature.camera]
    counts = [0.0] * grid.count
    for e in events:
        if e["indicator_name"] != feature.indicator or e["role"] != role:
            continue
        if feature.labels and (e["value"].get("outcome") if isinstance(e["value"], dict) else e["value"]) \
                not in feature.labels:
            continue
        w = grid.index(e["t0"])
        if w is not None:
            counts[w] += 1
    return counts


def _path(events, grid: _Grid, max_step_seconds: float) -> list[float]:
    path = [0.0] * grid.count
    hips = sorted((e["t0"], e["value"].get("hip_center_in_shot"), e["value"].get("camera_shot"))
                  for e in events if e["indicator_name"] == "position" and e["role"] == "teacher")
    for (ta, a, shot_a), (tb, b, shot_b) in zip(hips, hips[1:]):
        if a and b and shot_a == shot_b and tb - ta <= max_step_seconds:
            w = grid.index(tb)
            if w is not None:
                path[w] += math.hypot(b[0] - a[0], b[1] - a[1])
    return path


def build_windows(student: Sequence[dict[str, Any]], teacher: Sequence[dict[str, Any]], *,
                  window_seconds: float = 5.0, min_known_fraction: float = 0.5,
                  features: Sequence[Feature] = FEATURES) -> tuple[Any, dict[str, Any]]:
    """Window table (pandas DataFrame, one row per window) and metadata.

    min_known_fraction: a share needs at least this fraction of one window
    of known evidence (teacher: seconds; students: student-seconds).
    """
    import pandas as pd
    start, end = overlap_range(student, teacher)
    count = int(math.floor((end - start) / window_seconds + 1e-9))
    if count < 1:
        raise ValueError("The overlap is shorter than one window")
    grid = _Grid(start, window_seconds, count)
    min_known = min_known_fraction * window_seconds
    columns: dict[str, list[float]] = {}
    coverage: dict[str, list[float]] = {}
    # Seconds the teacher was visible per window: stationary episodes are only
    # judged where the teacher was seen (else NaN, not "moving").
    presence = [0.0] * count
    for e in teacher:
        if e["indicator_name"] == "teacher_presence" and e["value"] == "visible":
            for w, seconds in grid.spread(e["t0"], e["t1"]):
                presence[w] += seconds
    for feature in features:
        events = teacher if feature.camera == "teacher" else student
        if feature.kind == "share":
            columns[feature.name], known = _share(events, feature, grid, min_known)
            if feature.name in ("T_facing_class", "S_on_target"):
                coverage["teacher_known_seconds" if feature.camera == "teacher"
                         else "students_known_seconds"] = known
        elif feature.kind == "episode":
            columns[feature.name] = _episode_share(events, feature, grid, presence)
        elif feature.kind == "count":
            columns[feature.name] = _count(events, feature, grid)
        elif feature.kind == "path":
            columns[feature.name] = _path(events, grid, 2.5)
        else:
            raise ValueError(f"Unknown feature kind {feature.kind}")
    table = pd.DataFrame({
        "window": range(count),
        "lesson_seconds": [round(w * window_seconds, 3) for w in range(count)],
        "start_real_time": [datetime.fromtimestamp(start + w * window_seconds).astimezone()
                            .isoformat(timespec="milliseconds") for w in range(count)],
        **columns,
        **{key: [round(v, 3) for v in values] for key, values in coverage.items()},
    })
    metadata = {
        "window_seconds": window_seconds, "windows": count,
        "start_real_time": table["start_real_time"].iloc[0],
        "end_real_time": datetime.fromtimestamp(start + count * window_seconds).astimezone()
        .isoformat(timespec="milliseconds"),
        "min_known_fraction": min_known_fraction,
        "features": {f.name: {"camera": f.camera, "indicator": f.indicator, "labels": list(f.labels),
                              "kind": f.kind, "desirable": f.desirable, "description": f.description}
                     for f in features},
    }
    return table, metadata


def onsets(events: Sequence[dict[str, Any]], indicator: str, role: str, labels: Sequence[str],
           start: float, window_seconds: float, count: int, min_gap_seconds: float = 0.0) -> list[int]:
    """Windows where an interval with one of the labels starts (for peri-event analysis).

    Interval events are already merged runs, so each event start is an
    onset; starts closer than min_gap_seconds to the previous one are skipped.
    """
    grid = _Grid(start, window_seconds, count)
    times = sorted(e["t0"] for e in events if e["indicator_name"] == indicator and e["role"] == role
                   and e["value"] in labels)
    result, last = [], -math.inf
    for t in times:
        if t - last >= min_gap_seconds:
            w = grid.index(t)
            if w is not None:
                result.append(w)
            last = t
    return result
