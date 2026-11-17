"""Student interactions on the student camera: what students face, and answers.

Called by FoundationAnalyzer after StudentBehavior, with every confirmed
student of the sample, so each rule can see the others.

| Indicator | Labels | Rule |
| --- | --- | --- |
| head_target | front, down, up, peer, talking_peer, side, behind | head direction (face mesh yaw/pitch, else which face keypoints show) plus the nearest classmate in that direction |
| orientation_proxy | on_target, off_target, looking_down, speaking | faces the current focus: a student who is answering, else the front of the room |
| hand_raise_outcome | answered_likely, no_answer_seen, unknown (event per raise) | the student talks while facing the front for >= outcome_talk_seconds (3 s), or stands up, from the raise until outcome_window_seconds (12 s) after it |

"front" means toward the student camera, which faces the class from the
front of the room: the teacher, board, and screen cannot be told apart from
this camera alone. A student is "answering" while talking and either
standing or within outcome_window_seconds of their own hand raise; a pair
chatting with each other is not a focus, so facing a chatting classmate is
off target. The orientation proxy is not attention: looking at the front
while thinking about something else, or writing notes with the head down
(looking_down, kept apart from off_target), both happen.

hand_raise_outcome needs talking (face mesh, several frames per second);
an answer given while the face is turned away from the camera is missed,
and a student who talks to a neighbour right after lowering a hand looks
like an answer. Response latency (time from a question to the raise) needs
the teacher's speech, which is out of scope (user decision 2026-11-17).
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from src.analysis.events import Event, make_event
from src.analysis.student_indicators import StudentObservation, head_point, shoulders
from src.analysis.teacher_indicators import facing_direction

SEGMENT_NAMES = ("head_target", "orientation_proxy")
INDICATOR_NAMES = ("head_target", "orientation_proxy", "hand_raise_outcome")
POSE_DIRECTIONS = {"toward_class": "toward_camera", "sideways_left": "turned_left",
                   "sideways_right": "turned_right", "away_from_class": "behind", "unknown": "unknown"}
DEFAULTS: dict[str, float] = {
    "peer_row_widths": 1.5,         # classmates within this height difference count as the same row
    "peer_max_widths": 6.0,         # nobody farther than this is "the peer they face"
    "outcome_window_seconds": 12.0,
    "outcome_talk_seconds": 3.0,     # talking while facing the front (A3: 2 s of any talk was chance)
}


@dataclass
class _Raise:
    start: float
    end: float | None = None
    talk_samples: int = 0
    known_samples: int = 0
    samples: int = 0
    stood: bool = False


Segments = Callable[..., list[Event]]


class StudentInteractions:
    """Head targets, orientation proxy, and hand-raise outcomes for one camera."""

    def __init__(self, video_id: str, enabled: dict[str, bool], min_conf: float,
                 sample_period: float, settings: dict[str, Any] | None, update_segment: Segments):
        self.video_id = video_id
        self.on = {name: bool(enabled.get(name, False)) for name in INDICATOR_NAMES}
        self.min_conf = min_conf
        self.period = sample_period
        self.settings = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if k in DEFAULTS}}
        self.segment = update_segment
        self.raises: dict[int, _Raise] = {}     # current or recent raise per track
        self.last_raise: dict[int, float] = {}  # time a raise was last seen

    @property
    def active(self) -> bool:
        return any(self.on.values())

    def update(self, time: float, students: Sequence[StudentObservation],
               student_labels: dict[int, dict[str, Any]],
               finished_raises: Sequence[dict[str, Any]] = ()) -> tuple[list[Event], dict[int, dict[str, Any]]]:
        events: list[Event] = []
        shown: dict[int, dict[str, Any]] = {obs.track_id: {} for obs in students}
        talking = {obs.track_id: (obs.face_labels or {}).get("talking") for obs in students}
        window = float(self.settings["outcome_window_seconds"])
        for obs in students:
            raising = student_labels.get(obs.track_id, {}).get("hand_raise_now") in ("low", "medium", "high")
            if raising:
                self.last_raise[obs.track_id] = time
        answering = {obs.track_id for obs in students if talking[obs.track_id] == "talking" and (
            obs.posture == "standing" or time - self.last_raise.get(obs.track_id, -math.inf) <= window)}
        positions = {}
        for obs in students:
            head = head_point(obs.person, self.min_conf)
            frame = shoulders(obs.person, self.min_conf)
            if head is not None and frame is not None:
                positions[obs.track_id] = (head, frame[1])
        for obs in students:
            target, peer = self._target(obs, student_labels.get(obs.track_id, {}), positions, talking)
            shown[obs.track_id]["head_target"] = target
            shown[obs.track_id]["head_target_peer"] = peer
            if self.on["head_target"]:
                events += self.segment("head_target", obs.track_id, time, target, obs.role, None,
                                       "limited" if target == "unknown" else "adequate")
            if self.on["orientation_proxy"]:
                proxy = self._proxy(obs.track_id, target, peer, answering)
                events += self.segment("orientation_proxy", obs.track_id, time, proxy, obs.role, None,
                                       "limited" if proxy == "unknown" else "adequate")
                shown[obs.track_id]["orientation_proxy"] = proxy
        events += self._outcomes(time, students, student_labels, finished_raises, talking, shown)
        return events, shown

    # --- head target -------------------------------------------------------------------

    def _target(self, obs: StudentObservation, labels: dict[str, Any],
                positions: dict[int, tuple[tuple[float, float], float]],
                talking: dict[int, str | None]) -> tuple[str, int | None]:
        if labels.get("head_down_state") == "down":
            return "down", None
        face_direction = (obs.face_labels or {}).get("head_direction") if obs.face is not None else None
        direction = (face_direction if face_direction not in (None, "unknown")
                     else POSE_DIRECTIONS[facing_direction(obs.person, self.min_conf)])
        if direction in ("unknown", "down", "up"):
            return direction, None
        if direction == "toward_camera":
            return "front", None
        if obs.track_id not in positions:
            return ("behind" if direction == "behind" else "side"), None
        (hx, hy), width = positions[obs.track_id]
        s = self.settings
        best = None
        for other, ((px, py), _) in positions.items():
            if other == obs.track_id:
                continue
            dx, dy = (px - hx) / width, (py - hy) / width
            if math.hypot(dx, dy) > s["peer_max_widths"]:
                continue
            if direction == "turned_left":
                fits = dx < -0.5 and abs(dy) <= s["peer_row_widths"] + abs(dx) * 0.5
            elif direction == "turned_right":
                fits = dx > 0.5 and abs(dy) <= s["peer_row_widths"] + abs(dx) * 0.5
            else:  # behind: turned around toward the back of the room (farther from the camera)
                fits = dy < -0.5 and abs(dx) <= 2.5
            if fits and (best is None or math.hypot(dx, dy) < best[0]):
                best = (math.hypot(dx, dy), other)
        if best is None:
            return ("behind" if direction == "behind" else "side"), None
        peer = best[1]
        return ("talking_peer" if talking.get(peer) == "talking" else "peer"), peer

    @staticmethod
    def _proxy(track_id: int, target: str, peer: int | None, answering: set[int]) -> str:
        if track_id in answering:
            return "speaking"
        if target == "unknown":
            return "unknown"
        if target == "down":
            return "looking_down"
        focus = answering - {track_id}
        if target == "front":
            return "on_target"
        if focus and peer in focus:
            return "on_target"  # facing the classmate who is answering
        return "off_target"

    # --- hand-raise outcome ---------------------------------------------------------------

    def _outcomes(self, time: float, students: Sequence[StudentObservation],
                  student_labels: dict[int, dict[str, Any]], finished: Sequence[dict[str, Any]],
                  talking: dict[int, str | None], shown: dict[int, dict[str, Any]]) -> list[Event]:
        events: list[Event] = []
        for obs in students:
            now = student_labels.get(obs.track_id, {}).get("hand_raise_now")
            if now in ("low", "medium", "high") and obs.track_id not in self.raises:
                self.raises[obs.track_id] = _Raise(time)
        for record in finished:
            entry = self.raises.setdefault(record["track_id"], _Raise(record["start"]))
            entry.end = record["end"]
        by_id = {obs.track_id: obs for obs in students}
        for track_id, entry in list(self.raises.items()):
            obs = by_id.get(track_id)
            if obs is not None:
                entry.samples += 1
                said = talking.get(track_id)
                entry.known_samples += said in ("talking", "silent")
                # An answer is spoken toward the front; talking to a neighbour is a chat.
                entry.talk_samples += said == "talking" and shown.get(track_id, {}).get("head_target") == "front"
                entry.stood |= obs.posture == "standing"
            answered = (entry.talk_samples * self.period >= self.settings["outcome_talk_seconds"]
                        or entry.stood)
            due = entry.end is not None and time >= entry.end + self.settings["outcome_window_seconds"]
            if entry.end is not None and (answered or due):
                events += self._emit(track_id, obs.role if obs else "student", entry, time, answered)
                del self.raises[track_id]
        return events

    def _emit(self, track_id: int, role: str, entry: _Raise, time: float, answered: bool) -> list[Event]:
        if not self.on["hand_raise_outcome"]:
            return []
        outcome = ("answered_likely" if answered else
                   "no_answer_seen" if entry.known_samples * 2 >= max(entry.samples, 1) else "unknown")
        return [make_event(self.video_id, "hand_raise_outcome", entry.start, max(time, entry.start),
                           {"outcome": outcome, "raise_end": round(entry.end, 3),
                            "talk_seconds": round(entry.talk_samples * self.period, 1),
                            "stood_up": entry.stood},
                           track_id=track_id, role=role,
                           coverage="limited" if outcome == "unknown" else "adequate")]

    def close(self, track_id: int, role: str, end: float) -> list[Event]:
        """A track ended: decide its pending raise with what was seen."""
        entry = self.raises.pop(track_id, None)
        self.last_raise.pop(track_id, None)
        if entry is None:
            return []
        if entry.end is None:
            entry.end = end
        answered = entry.talk_samples * self.period >= self.settings["outcome_talk_seconds"] or entry.stood
        return self._emit(track_id, role, entry, end, answered)
