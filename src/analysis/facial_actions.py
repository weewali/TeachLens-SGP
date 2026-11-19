"""Student facial actions: rules over face-mesh blendshape scores and head angles.

FoundationAnalyzer calls StudentFaces for confirmed student tracks on a
student camera (never for the teacher: user decision 2026-11-17). Each
sample's FaceMesh (src/inference/face_mesh.py) gives 52 blendshape scores
in [0, 1] and head yaw/pitch/roll; the rules read a few of them:

| Indicator | Labels | Rule |
| --- | --- | --- |
| smile | smiling, not_smiling | mouthSmile (mean of left/right) high and above the person's usual level |
| brows | lowered, raised, neutral | browDown or browUp (inner + outer) above the person's usual level |
| eyes_closed | closed, open | eyeBlink high and above usual for >= eyes_closed_min_seconds |
| yawn | yawning, not_yawning | jawOpen high and above usual for >= yawn_min_seconds |
| head_direction | toward_camera, turned_left, turned_right, down, up | yaw from the camera; pitch against the person's usual pitch |
| talking | talking, silent | inner-lip gap swings >= talk_min_range within a second, in >= 2 of the last 3 seconds |
| head_gesture | nod, shake, none | pitch swings back within nod_window_seconds (nod); yaw reverses twice within shake_window_seconds (shake) |

talking and head_gesture need several faces per second: the pipeline runs the
landmarker on extra frames between samples (faces.frames_per_second, on the
last sample's face crops) and passes them as `series`. Measured on A2
(2026-11-17): students not talking moved the inner-lip gap by at most 0.02
of face height within a second, a talking and laughing student by
0.03-0.12. Chewing and laughing also move the mouth.

Why a per-person baseline: the cameras look down on the class, and on the
scenario videos neutral faces already read eyeSquint 0.4-0.8, eyeLookUp up
to 0.66, and eyeBlink up to 0.46 (glasses, lowered eyes). Each track keeps
its recent scores (baseline_seconds) and a label needs a change from that
track's resting level: the 25th percentile of its own smile scores, the
median of everything else (see REST_QUANTILES). Until baseline_min_samples
faces are seen, labels are unknown. A person who smiles more than about
three quarters of the time raises their own resting level, so their smiles
are under-reported; a brow, eye, or jaw state held for most of the recent
history becomes that person's normal.

These are observed facial actions, not emotions, engagement, or attention.
No face toward the camera (head down on the desk, cap brim, hand over the
face, turned away) gives unknown, never a negative label. Left/right in
head_direction are image directions. At one sample per second, a short
blink is not seen and eyes_closed/yawn episodes are reported from the
sample where their minimum duration is reached.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from src.analysis.events import Event, make_event

# Interval indicators closed with the track (see FoundationAnalyzer._retire).
SEGMENT_NAMES = ("smile", "brows", "eyes_closed", "yawn", "head_direction", "talking", "head_gesture")
INDICATOR_NAMES = SEGMENT_NAMES
SCORE_KEYS = ("smile", "brow_down", "brow_up", "eye_blink", "jaw_open", "yaw", "pitch")
# Usual (resting) level per score, as a quantile of the student's own recent
# scores. Smiles: the low end, so someone who smiles half the time still has
# a neutral resting level (A2: with the median, a 20 s run started during
# smiles missed a 0.65 smile). Everything else: the median, so a lasting
# shift caused by head angle or posture becomes the new normal (A2: a
# student looking down past a classmate read browUp 0.45-0.54 for 40 s
# against an early 0.06-0.24; brows were not raised).
REST_QUANTILES = {"smile": 0.25}


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q))

DEFAULTS: dict[str, float] = {
    "baseline_seconds": 300.0,      # recent history behind each track's usual face
    "baseline_min_samples": 10,     # faces needed before any label is given
    "smile_min": 0.35,              # mean mouthSmile at least this...
    "smile_rise": 0.20,             # ...and this much above the usual level
    "brow_min": 0.30,
    "brow_rise": 0.25,
    "eyes_closed_min": 0.60,        # mean eyeBlink
    "eyes_closed_rise": 0.30,
    "eyes_closed_min_seconds": 2.0,
    "yawn_min": 0.50,               # jawOpen
    "yawn_rise": 0.35,
    "yawn_min_seconds": 2.0,
    "head_turn_degrees": 25.0,      # |yaw| from the camera axis
    "head_pitch_degrees": 15.0,     # pitch change from the usual pitch
    "talk_min_range": 0.03,         # inner-lip gap swing within one second, share of face height
    "talk_min_seconds": 2,          # seconds (of the last 3) with such a swing
    "nod_min_degrees": 5.0,         # pitch swing that counts
    "nod_window_seconds": 1.2,      # down and back up within this time
    "shake_min_degrees": 8.0,       # yaw swing that counts
    "shake_window_seconds": 2.0,    # two yaw reversals within this time
}


def reversals(values: list[float], min_swing: float) -> int:
    """Direction changes of at least min_swing (hysteresis), e.g. down-up = 1."""
    if len(values) < 3:
        return 0
    direction, extreme, count = 0, values[0], 0
    for value in values[1:]:
        if direction == 0:
            if abs(value - extreme) >= min_swing:
                direction, extreme = (1 if value > extreme else -1), value
        elif direction * (value - extreme) > 0:
            extreme = value  # still moving the same way
        elif abs(value - extreme) >= min_swing:
            count, direction, extreme = count + 1, -direction, value
    return count


def face_values(mesh: Any) -> dict[str, float]:
    """The few scores the rules use, from a FaceMesh."""
    s = mesh.scores

    def mean(*names: str) -> float:
        return sum(s.get(name, 0.0) for name in names) / len(names)

    return {"smile": round(mean("mouthSmileLeft", "mouthSmileRight"), 4),
            "brow_down": round(mean("browDownLeft", "browDownRight"), 4),
            "brow_up": round(mean("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), 4),
            "eye_blink": round(mean("eyeBlinkLeft", "eyeBlinkRight"), 4),
            "jaw_open": round(s.get("jawOpen", 0.0), 4),
            "yaw": mesh.yaw, "pitch": mesh.pitch, "roll": mesh.roll}


def head_direction(yaw: float, pitch_change: float, turn: float, tilt: float) -> str:
    """Larger of the turn and the tilt, each measured against its threshold."""
    sideways, vertical = abs(yaw) / turn, abs(pitch_change) / tilt
    if max(sideways, vertical) < 1:
        return "toward_camera"
    if sideways >= vertical:
        return "turned_right" if yaw > 0 else "turned_left"
    return "down" if pitch_change < 0 else "up"


@dataclass
class _FaceTrack:
    history: deque = field(default_factory=deque)
    eyes_since: float | None = None   # start of the current eyes-closed candidate run
    yawn_since: float | None = None
    mouth_ranges: deque = field(default_factory=lambda: deque(maxlen=3))  # per-second lip-gap swings
    last_mouth: float | None = None
    angles: deque = field(default_factory=lambda: deque(maxlen=40))  # (time, pitch, yaw) of recent faces


Segments = Callable[..., list[Event]]  # _Segments.update(name, track_id, time, label, role, confidence, coverage, zone=None)


class StudentFaces:
    """Facial-action rules for student tracks; holds per-track baselines."""

    def __init__(self, video_id: str, enabled: dict[str, bool], sample_period: float,
                 settings: dict[str, Any] | None, update_segment: Segments, always: bool = False):
        """always: compute labels for other rules even when no facial switch is on."""
        self.video_id = video_id
        self.always = always
        self.on = {name: bool(enabled.get(name, False)) for name in INDICATOR_NAMES}
        self.period = sample_period
        self.settings = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if k in DEFAULTS}}
        self.history = max(1, round(float(self.settings["baseline_seconds"]) / sample_period))
        self.segment = update_segment
        self.tracks: dict[int, _FaceTrack] = {}

    @property
    def active(self) -> bool:
        return self.always or any(self.on.values())

    def _run(self, since: float | None, candidate: bool, time: float,
             minimum: float) -> tuple[float | None, bool]:
        """Track a candidate run; True once it has lasted at least minimum seconds."""
        if not candidate:
            return None, False
        since = time if since is None else since
        return since, time - since + self.period >= minimum - 1e-6

    def update(self, time: float, track_id: int, mesh: Any | None, role: str,
               series: Sequence[tuple[float, Any]] = ()) -> tuple[list[Event], dict[str, Any]]:
        """Events for one student sample, plus overlay state (label per indicator).

        series: (time, FaceMesh) from the frames since the previous sample,
        oldest first, for talking and head gestures.
        """
        s = self.settings
        state = self.tracks.setdefault(track_id, _FaceTrack(deque(maxlen=self.history)))
        events: list[Event] = []
        shown: dict[str, Any] = {"face": mesh is not None,
                                 "labels": dict.fromkeys(INDICATOR_NAMES, "unknown")}
        if mesh is None:  # no usable face this sample: unknown, and runs are broken
            state.eyes_since = state.yawn_since = state.last_mouth = None
            state.mouth_ranges.clear()
            for name in INDICATOR_NAMES:
                if self.on[name]:
                    events += self.segment(name, track_id, time, "unknown", role, None, "limited")
                    shown[name] = "unknown"
            return events, shown
        values = face_values(mesh)
        enough = len(state.history) >= int(s["baseline_min_samples"])
        base = ({key: _quantile([h[key] for h in state.history], REST_QUANTILES.get(key, 0.5))
                 for key in SCORE_KEYS} if enough else None)
        state.history.append(values)
        labels: dict[str, str] = {}
        if base is None:  # still learning this person's usual face
            labels = dict.fromkeys(INDICATOR_NAMES, "unknown")
        else:
            labels["smile"] = ("smiling" if values["smile"] >= s["smile_min"]
                               and values["smile"] - base["smile"] >= s["smile_rise"] else "not_smiling")
            down = values["brow_down"] - base["brow_down"]
            up = values["brow_up"] - base["brow_up"]
            lowered = down >= s["brow_rise"] and values["brow_down"] >= s["brow_min"]
            raised = up >= s["brow_rise"] and values["brow_up"] >= s["brow_min"]
            labels["brows"] = ("lowered" if lowered and (not raised or down >= up)
                               else "raised" if raised else "neutral")
            state.eyes_since, closed = self._run(
                state.eyes_since, values["eye_blink"] >= s["eyes_closed_min"]
                and values["eye_blink"] - base["eye_blink"] >= s["eyes_closed_rise"],
                time, float(s["eyes_closed_min_seconds"]))
            labels["eyes_closed"] = "closed" if closed else "open"
            state.yawn_since, yawning = self._run(
                state.yawn_since, values["jaw_open"] >= s["yawn_min"]
                and values["jaw_open"] - base["jaw_open"] >= s["yawn_rise"],
                time, float(s["yawn_min_seconds"]))
            labels["yawn"] = "yawning" if yawning else "not_yawning"
            labels["head_direction"] = head_direction(values["yaw"], values["pitch"] - base["pitch"],
                                                      float(s["head_turn_degrees"]),
                                                      float(s["head_pitch_degrees"]))
        # Talking and head gestures need no baseline: they are changes within seconds.
        faces = [(t, m) for t, m in series if m is not None] + [(time, mesh)]
        mouths = ([state.last_mouth] if state.last_mouth is not None else []) + [m.mouth_open for _, m in faces]
        state.last_mouth = mesh.mouth_open
        state.mouth_ranges.append(max(mouths) - min(mouths) if len(mouths) >= 2 else None)
        known = [r for r in state.mouth_ranges if r is not None]
        labels["talking"] = ("unknown" if not known else
                             "talking" if sum(r >= s["talk_min_range"] for r in known) >= int(s["talk_min_seconds"])
                             else "silent")
        state.angles.extend((t, m.pitch, m.yaw) for t, m in faces)
        nod = [p for t, p, _ in state.angles if time - t <= float(s["nod_window_seconds"]) + 1e-6]
        shake = [y for t, _, y in state.angles if time - t <= float(s["shake_window_seconds"]) + 1e-6]
        labels["head_gesture"] = (
            "unknown" if len(shake) < 5 else
            "shake" if reversals(shake, float(s["shake_min_degrees"])) >= 2 else
            "nod" if reversals(nod, float(s["nod_min_degrees"])) >= 1 else "none")
        values["mouth_range"] = round(state.mouth_ranges[-1], 4) if state.mouth_ranges[-1] is not None else None
        coverage = "adequate" if base is not None else "limited"
        for name in INDICATOR_NAMES:
            if self.on[name]:
                events += self.segment(name, track_id, time, labels[name], role, None, coverage)
                shown[name] = labels[name]
        shown["values"] = values
        shown["labels"] = labels  # all labels, for the student rules (alertness, targets, answers)
        # Unsmoothed scores behind the labels, for threshold tuning.
        events.append(make_event(self.video_id, "facial_scores", time, time,
                                 {**values, "face_width_px": mesh.width_px,
                                  "baseline": ({k: round(v, 4) for k, v in base.items()}
                                               if base is not None else None)},
                                 track_id=track_id, role=role, confidence=None, coverage="limited"))
        return events, shown

    def close(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)
