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
from typing import Any, Callable

import numpy as np

from src.analysis.events import Event, make_event

# Interval indicators closed with the track (see FoundationAnalyzer._retire).
SEGMENT_NAMES = ("smile", "brows", "eyes_closed", "yawn", "head_direction")
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
}


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


Segments = Callable[..., list[Event]]  # _Segments.update(name, track_id, time, label, role, confidence, coverage, zone=None)


class StudentFaces:
    """Facial-action rules for student tracks; holds per-track baselines."""

    def __init__(self, video_id: str, enabled: dict[str, bool], sample_period: float,
                 settings: dict[str, Any] | None, update_segment: Segments):
        self.video_id = video_id
        self.on = {name: bool(enabled.get(name, False)) for name in INDICATOR_NAMES}
        self.period = sample_period
        self.settings = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if k in DEFAULTS}}
        self.history = max(1, round(float(self.settings["baseline_seconds"]) / sample_period))
        self.segment = update_segment
        self.tracks: dict[int, _FaceTrack] = {}

    @property
    def active(self) -> bool:
        return any(self.on.values())

    def _run(self, since: float | None, candidate: bool, time: float,
             minimum: float) -> tuple[float | None, bool]:
        """Track a candidate run; True once it has lasted at least minimum seconds."""
        if not candidate:
            return None, False
        since = time if since is None else since
        return since, time - since + self.period >= minimum - 1e-6

    def update(self, time: float, track_id: int, mesh: Any | None,
               role: str) -> tuple[list[Event], dict[str, Any]]:
        """Events for one student sample, plus overlay state (label per indicator)."""
        s = self.settings
        state = self.tracks.setdefault(track_id, _FaceTrack(deque(maxlen=self.history)))
        events: list[Event] = []
        shown: dict[str, Any] = {"face": mesh is not None}
        if mesh is None:  # no usable face this sample: unknown, and runs are broken
            state.eyes_since = state.yawn_since = None
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
        coverage = "adequate" if base is not None else "limited"
        for name in INDICATOR_NAMES:
            if self.on[name]:
                events += self.segment(name, track_id, time, labels[name], role, None, coverage)
                shown[name] = labels[name]
        shown["values"] = values
        # Unsmoothed scores behind the labels, for threshold tuning.
        events.append(make_event(self.video_id, "facial_scores", time, time,
                                 {**values, "face_width_px": mesh.width_px,
                                  "baseline": ({k: round(v, 4) for k, v in base.items()}
                                               if base is not None else None)},
                                 track_id=track_id, role=role, confidence=None, coverage="limited"))
        return events, shown

    def close(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)
