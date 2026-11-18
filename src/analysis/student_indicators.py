"""List C: student behaviour on the student camera.

FoundationAnalyzer calls StudentBehavior once per analyzed sample with every
confirmed student track (StudentObservation): pose keypoints, the List A
posture/motion state, the face mesh and facial-action labels
(src/analysis/facial_actions.py), hand gestures (src/inference/hands.py),
and the COCO objects detected in the frame (src/inference/objects.py).

| Indicator | Labels | Basis |
| --- | --- | --- |
| hand_raise | high, medium, low, none | wrist well above the head; above the nose; forearm raised near vertical below that (elbow on the desk), held 2 s or with an open palm; hands at the face never count |
| seat_occupancy | seat ID, away | a "home" seat learnt from the track's seated shoulder position |
| out_of_seat | in_seat, out_of_seat | shoulder midpoint far from the home seat (standing up in place stays in the seat) |
| head_down | down, up | head height above the shoulders drops or nearly reaches them; face pitch drops; nose seen but both eyes hidden (head bowed, seen from above); no face and no head above the shoulders, or shoulders dropped (head on the desk) |
| alertness | awake, tired, dozing, possible_sleep | yawn; eyes closed >= 2 s; head down and still for 20 s with the face hidden (head on the arms), or eyes closed for 10 s; head down with the face visible is awake (phone, notes) |
| device_use | phone, phone_likely, laptop, none | phone: a detected phone at a hand while looking down at it or holding it up, or lying in front while looked at; phone_likely: the head pointed down (not turned to a neighbour, not resting on the arms) for 5 s while not writing, not reading something detected, and not facing an open laptop (phone under the desk or covered by the hands, not seen); laptop: hands on, or the head down at, an open laptop in front |
| note_taking | writing, not_writing | looking down with a hand on the desk moving a little, no device |
| reading | reading, not_reading | eyes looking down with hands still, no device (a book nearby makes it adequate) |
| response_gesture | thumbs_up, thumbs_down, open_palm, pointing_up, victory, i_love_you, none | Gesture Recognizer on the wrist crops |

Every label is `unknown` when the needed body parts are not visible. These
are observed proxies, not attention, engagement, or intent: a phone in hand
may be course work, and a head down may be reading, writing, or resting.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Callable, Sequence

from src.analysis.events import Event, make_event
from src.analysis.teacher_indicators import facing_direction
from src.inference.pose import (
    LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST, PosePerson,
)

Point = tuple[float, float]
SEGMENT_NAMES = ("hand_raise", "seat_occupancy", "out_of_seat", "head_down", "alertness",
                 "device_use", "note_taking", "reading", "response_gesture")
INDICATOR_NAMES = SEGMENT_NAMES
ARMS = {"left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)}
RAISE_RANK = {"unknown": -1, "none": 0, "low": 1, "medium": 2, "high": 3}
RAISE_GESTURES = ("open_palm", "pointing_up", "victory")  # hand shapes of a raised hand
RESPONSE_GESTURES = ("thumbs_up", "thumbs_down", "open_palm", "pointing_up", "victory", "i_love_you")

DEFAULTS: dict[str, float] = {
    # Hand raise (distances in shoulder widths)
    "raise_face_dx": 0.55,            # face box: wrist this close sideways to the nose...
    "raise_face_above": 0.8,          # ...from this far above the nose...
    "raise_face_below": 0.9,          # ...to this far below it (chin, cheek, hair, phone call)
    "raise_high_above_nose": 0.5,     # high: wrist this far above the nose; medium: above the nose
    "raise_max_tilt_degrees": 35.0,   # forearm within this of vertical (medium, low)
    "raise_low_min_lift": 0.6,        # low: wrist above the elbow by this share of the forearm
    "raise_low_min_seconds": 2.0,     # low raises without a raised-hand gesture must last this long
    # Seats
    "seat_min_samples": 5,            # seated samples before the home seat is set
    "seat_match_widths": 0.8,         # a home this close to a known seat takes that seat
    "out_of_seat_widths": 1.5,        # sideways (or beyond standing height) from the seat
    "out_of_seat_min_seconds": 2.0,
    # Head down and alertness
    "head_down_drop": 0.35,           # head height below the usual height
    "head_down_min_height": 0.1,      # nose this close to the shoulder line
    "head_hidden_shoulder_drop": 0.3, # no head points and shoulders this far below seated level
    "head_on_desk_room": 0.9,         # no face and the box top this close above the shoulders = head down on the desk
    "bowed_ear_rise": 0.45,           # ears this many eye distances above the eyes = head bowed (B1 upright: up to -0.52 once)
    "bowed_nose_drop": 0.75,          # nose this many eye distances below the eyes = looking down (B1 upright: up to 0.69)
    "head_rest_height": 0.0,          # head point at or below the shoulder line = head resting (B1 sleeper: -0.19)
    "laptop_open_ratio": 0.7,         # laptop box height/width at least this = open (A2: closed 0.34-0.54, open ~1.07)
    "head_down_pitch_degrees": 15.0,  # face pitch below the usual pitch
    "eyes_down_min": 0.45,            # mean eyeLookDown (face mesh)...
    "eyes_down_rise": 0.15,           # ...and this much above the student's usual level
    "sleep_min_seconds": 20.0,        # head down and still this long
    "long_eyes_closed_seconds": 10.0, # eyes closed this long
    # Desk activity
    "object_near_wrist": 0.5,         # object centre within this of a wrist = in hand (A3: 1.0 took phones lying by a writing hand)
    "held_phone_max_gap": 0.6,        # wrists this close together in front of the body = holding something
    "writing_min_speed": 0.03,        # wrist speed relative to the shoulders, widths per second
    "writing_max_speed": 0.6,
    "phone_head_down_seconds": 5.0,   # head pointed down in all but one of the last 5 s, phone not seen = phone_likely
    "phone_memory_seconds": 10.0,     # a phone seen at the hands stays in use this long while the head stays down
    "phone_eyes_down": 0.55,          # mean eyeLookDown this high counts as pointed down (A4 phone users 0.44-0.63)
    "activity_smoothing_samples": 3,
    "gesture_min_score": 0.6,
    "history_seconds": 300.0,
}


@dataclass(frozen=True)
class StudentObservation:
    """What the analyzer knows about one confirmed student at one sample."""
    track_id: int
    person: PosePerson
    role: str
    posture: str | None = None
    shoulder_rise: float | None = None
    motion: str | None = None
    face: Any | None = None                 # FaceMesh
    face_labels: dict[str, str] | None = None  # StudentFaces labels (all indicators)
    gestures: dict[str, Any] | None = None  # side -> HandGesture


def _px(person: PosePerson, index: int, min_conf: float) -> Point | None:
    point = person.point(index, min_conf)
    return None if point is None else (point.x * person.width, point.y * person.height)


def _mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def shoulders(person: PosePerson, min_conf: float) -> tuple[Point, float] | None:
    """Shoulder midpoint and shoulder width in pixels (both shoulders needed)."""
    left, right = _px(person, LEFT_SHOULDER, min_conf), _px(person, RIGHT_SHOULDER, min_conf)
    if left is None or right is None:
        return None
    width = math.dist(left, right)
    return (_mid(left, right), width) if width >= 8 else None


def head_point(person: PosePerson, min_conf: float) -> Point | None:
    """Nose, else the mean of visible eyes, else of visible ears."""
    nose = _px(person, NOSE, min_conf)
    if nose is not None:
        return nose
    for group in ((LEFT_EYE, RIGHT_EYE), (LEFT_EAR, RIGHT_EAR)):
        points = [p for i in group if (p := _px(person, i, min_conf)) is not None]
        if points:
            return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    return None


def pose_looks_down(person: PosePerson, min_conf: float, settings: dict[str, float]) -> bool | None:
    """Head pitched down from face keypoints alone (works without a face mesh).

    Measured on the student camera (2026-11-18), in eye distances: upright
    faces have the ears near the eyes (A1: -0.17..+0.08; B1 after the camera
    was moved: -0.52..+0.44) and the nose 0.47-0.69 below them. A bowed head lifts the ears above the eyes (A2
    student in a cap on his phone: -0.55..-0.73) or drops the nose further
    (A2 student bent over his phone: 0.77-0.92). None without both eyes.
    """
    left, right = _px(person, LEFT_EYE, min_conf), _px(person, RIGHT_EYE, min_conf)
    if left is None or right is None:
        return None
    gap = abs(left[0] - right[0])
    if gap < 4:
        return None
    eye_y = (left[1] + right[1]) / 2
    nose = _px(person, NOSE, min_conf)
    ears = [p for i in (LEFT_EAR, RIGHT_EAR) if (p := _px(person, i, min_conf)) is not None]
    ears_up = bool(ears) and (sum(p[1] for p in ears) / len(ears) - eye_y) / gap <= -settings["bowed_ear_rise"]
    nose_low = nose is not None and (nose[1] - eye_y) / gap >= settings["bowed_nose_drop"]
    return ears_up or nose_low


def raise_level(person: PosePerson, min_conf: float, settings: dict[str, float],
                gestures: dict[str, Any] | None = None) -> tuple[str, str | None, bool]:
    """(level, side, palm) for the higher arm; level unknown without arms.

    Heights are measured against the head (nose), with the forearm pointing up:
    high = wrist well above the head, medium = wrist above the nose, low =
    forearm raised near vertical below that (elbow on the desk, hand up).
    A wrist in the box around the face (chin, cheek, hair, phone call) only
    counts with a raised-hand shape from the gesture model (open palm,
    pointing finger, victory): on A1 a raise beside the face and a cheek rest
    had the same wrist position. A detected fist, thumbs up/down, etc. held
    up is not a raise. palm says whether a raised-hand shape was seen.
    """
    frame = shoulders(person, min_conf)
    if frame is None:
        return "unknown", None, False
    (_, mid_y), width = frame
    nose = _px(person, NOSE, min_conf)
    best, best_side, palm, evaluated = "none", None, False, False
    for side, (shoulder_i, elbow_i, wrist_i) in ARMS.items():
        shoulder, elbow, wrist = (_px(person, i, min_conf) for i in (shoulder_i, elbow_i, wrist_i))
        if shoulder is None or elbow is None or wrist is None:
            continue
        evaluated = True
        at_face = (nose is not None and abs(wrist[0] - nose[0]) < settings["raise_face_dx"] * width
                   and nose[1] - settings["raise_face_above"] * width < wrist[1]
                   < nose[1] + settings["raise_face_below"] * width)
        forearm = math.dist(elbow, wrist)
        if forearm < 0.2 * width:
            continue
        tilt = (math.degrees(math.atan2(abs(wrist[0] - elbow[0]), elbow[1] - wrist[1]))
                if wrist[1] < elbow[1] else 180.0)
        # Heights against the head, not the shoulders: the camera looks down
        # on the class, so a hand raised to head height sits only a little
        # above the shoulders in the image (A1 0:23, five raises at head height).
        head = head_point(person, min_conf)
        head_y = head[1] if head is not None else mid_y - 0.7 * width
        lifted = wrist[1] < elbow[1]
        level = "none"
        if lifted and wrist[1] < head_y - settings["raise_high_above_nose"] * width:
            level = "high"
        elif lifted and wrist[1] < head_y:
            level = "medium"
        elif (wrist[1] < shoulder[1] + 0.5 * width and tilt <= settings["raise_max_tilt_degrees"]
              and elbow[1] - wrist[1] >= settings["raise_low_min_lift"] * forearm):
            level = "low"
        hand = (gestures or {}).get(side)
        shown_palm = bool(hand and hand.found and hand.gesture in RAISE_GESTURES)
        if level != "none" and at_face and not shown_palm:
            level = "none"  # hand at the face without a raised-hand shape: chin/cheek rest
        if level == "low" and hand is not None and hand.found and hand.gesture not in RAISE_GESTURES + (None,):
            level = "none"  # e.g. a fist or thumbs up held up: not a raised hand
        if RAISE_RANK[level] > RAISE_RANK[best]:
            best, best_side, palm = level, side, shown_palm
    return (best if evaluated else "unknown"), best_side, palm


@dataclass
class Seat:
    seat_id: str
    x: float          # pixels, shoulder midpoint while seated
    y: float
    width: float      # shoulder width, pixels
    tracks: list[int] = field(default_factory=list)


class SeatRegistry:
    """Seats learnt from where students sit; a returning student keeps the seat ID."""

    def __init__(self, match_widths: float):
        self.match = match_widths
        self.seats: list[Seat] = []

    def assign(self, track_id: int, home: Point, width: float, taken: set[str]) -> Seat:
        free = [seat for seat in self.seats if seat.seat_id not in taken
                and math.dist((seat.x, seat.y), home) <= self.match * max(width, seat.width)]
        if free:
            seat = min(free, key=lambda s: math.dist((s.x, s.y), home))
        else:
            seat = Seat(f"seat_{len(self.seats) + 1:02d}", home[0], home[1], width)
            self.seats.append(seat)
        seat.tracks.append(track_id)
        return seat


@dataclass
class _Student:
    anchors: deque                       # seated shoulder midpoints and widths
    heights: deque                       # head height above the shoulders, widths
    pitches: deque                       # face pitch, degrees
    eyes: deque = field(default_factory=lambda: deque(maxlen=300))  # eyeLookDown, for the usual level
    seat: Seat | None = None
    previous_mid: Point | None = None
    away_since: float | None = None
    down_since: float | None = None
    eyes_closed_since: float | None = None
    phone_down: deque = field(default_factory=lambda: deque(maxlen=30))  # head pointed down, per sample
    phone_seen_at: float | None = None
    moved: deque = field(default_factory=lambda: deque(maxlen=3))
    low_raise_since: float | None = None
    raise_episode: dict[str, Any] | None = None
    previous_wrists: dict[str, Point] | None = None
    previous_time: float | None = None
    activities: deque = field(default_factory=lambda: deque(maxlen=3))


Segments = Callable[..., list[Event]]


class StudentBehavior:
    """List C rules for student tracks; holds per-track state between samples."""

    def __init__(self, video_id: str, enabled: dict[str, bool], min_conf: float,
                 sample_period: float, settings: dict[str, Any] | None, update_segment: Segments):
        self.video_id = video_id
        self.on = {name: bool(enabled.get(name, False)) for name in INDICATOR_NAMES}
        self.min_conf = min_conf
        self.period = sample_period
        self.settings = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if k in DEFAULTS}}
        history = max(1, round(float(self.settings["history_seconds"]) / sample_period))
        self._history = history
        self.segment = update_segment
        self.seats = SeatRegistry(float(self.settings["seat_match_widths"]))
        self.tracks: dict[int, _Student] = {}
        self.finished_raises: list[dict[str, Any]] = []  # episodes closed this sample (for interactions)
        self.frame_size: tuple[int, int] = (1, 1)

    @property
    def active(self) -> bool:
        return any(self.on.values())

    def _state(self, track_id: int) -> _Student:
        if track_id not in self.tracks:
            n = self._history
            self.tracks[track_id] = _Student(deque(maxlen=60), deque(maxlen=n), deque(maxlen=n))
        return self.tracks[track_id]

    def _held(self, since: float | None, condition: bool, time: float,
              seconds: float) -> tuple[float | None, bool]:
        if not condition:
            return None, False
        since = time if since is None else since
        return since, time - since + self.period >= seconds - 1e-6

    # --- per-sample entry point -------------------------------------------

    def update(self, time: float, students: Sequence[StudentObservation],
               objects: Sequence[Any] = ()) -> tuple[list[Event], dict[int, dict[str, Any]]]:
        """Events for one sample and overlay state per track ID."""
        events: list[Event] = []
        shown: dict[int, dict[str, Any]] = {}
        self.finished_raises = []
        if students:
            self.frame_size = (students[0].person.width, students[0].person.height)
        owners = self._assign_objects(students, objects)
        taken = {self.tracks[o.track_id].seat.seat_id for o in students
                 if o.track_id in self.tracks and self.tracks[o.track_id].seat is not None}
        for obs in students:
            state = self._state(obs.track_id)
            labels: dict[str, Any] = {}
            events += self._hand_raise(time, obs, state, labels)
            events += self._seat(time, obs, state, labels, taken)
            events += self._head_and_alertness(time, obs, state, labels)
            events += self._desk_activity(time, obs, state, labels, owners.get(obs.track_id, []))
            events += self._response_gesture(time, obs, labels)
            shown[obs.track_id] = labels
        return events, shown

    # --- objects ------------------------------------------------------------

    def _assign_objects(self, students: Sequence[StudentObservation],
                        objects: Sequence[Any]) -> dict[int, list[tuple[Any, float]]]:
        """Each phone/laptop/book goes to the nearest student in front of whom it lies.

        Returns per track: (object, distance to the nearest wrist in shoulder widths, or inf).
        """
        owners: dict[int, list[tuple[Any, float]]] = {}
        for item in objects:
            if item.label not in ("cell phone", "laptop", "book"):
                continue
            best = None
            for obs in students:
                frame = shoulders(obs.person, self.min_conf)
                if frame is None:
                    continue
                (mx, my), width = frame
                w, h = obs.person.width, obs.person.height
                cx, cy = (item.box[0] + item.box[2]) / 2 * w, (item.box[1] + item.box[3]) / 2 * h
                x1, _, x2, y2 = obs.person.box
                inside_x = x1 * w - 0.3 * width <= cx <= x2 * w + 0.3 * width
                if not inside_x or cy < my - 0.5 * width or cy > max(y2 * h, my + 4 * width):
                    continue
                wrists = [p for i in (LEFT_WRIST, RIGHT_WRIST) if (p := _px(obs.person, i, self.min_conf))]
                near = min((math.dist((cx, cy), p) / width for p in wrists), default=math.inf)
                score = min(near, math.dist((cx, cy), (mx, my)) / width)
                if best is None or score < best[0]:
                    best = (score, obs.track_id, near)
            if best is not None:
                owners.setdefault(best[1], []).append((item, best[2]))
        return owners

    # --- hand raise -----------------------------------------------------

    def _hand_raise(self, time: float, obs: StudentObservation, state: _Student,
                    labels: dict[str, Any]) -> list[Event]:
        s = self.settings
        level, side, palm = raise_level(obs.person, self.min_conf, s, obs.gestures)
        # A low raise (elbow on the desk) must be held; on A1/A3 one-second "raises"
        # with an open palm were hands moving while talking.
        state.low_raise_since, held = self._held(state.low_raise_since, level == "low", time,
                                                 float(s["raise_low_min_seconds"]))
        if level == "low" and not held:
            level = "none"
        events: list[Event] = []
        episode = state.raise_episode
        if level in ("low", "medium", "high"):
            if episode is None:
                state.raise_episode = episode = {"start": time, "max": level, "side": side,
                                                 "palm": palm, "samples": 0}
            episode["end"] = time
            episode["samples"] += 1
            episode["palm"] = episode["palm"] or palm
            if RAISE_RANK[level] > RAISE_RANK[episode["max"]]:
                episode["max"], episode["side"] = level, side
        elif episode is not None and level != "unknown":
            events.append(self._close_raise(obs.track_id, obs.role, episode))
            state.raise_episode = None
        labels["hand_raise_now"] = level  # for hand-raise outcomes, even with hand_raise off
        if self.on["hand_raise"]:
            events += self.segment("hand_raise", obs.track_id, time, level, obs.role, None,
                                   "limited" if level in ("unknown", "low") else "adequate")
            labels["hand_raise"] = level
            labels["hand_raise_side"] = side
        return events

    def _close_raise(self, track_id: int, role: str, episode: dict[str, Any]) -> Event:
        end = episode["end"] + self.period
        record = {**episode, "track_id": track_id, "end": end}
        self.finished_raises.append(record)
        return make_event(self.video_id, "hand_raise_episode", episode["start"], end,
                          {"max_level": episode["max"], "side": episode["side"],
                           "open_palm_seen": episode["palm"], "samples": episode["samples"]},
                          track_id=track_id, role=role,
                          coverage="limited" if episode["max"] == "low" else "adequate")

    # --- seats --------------------------------------------------------------

    def _seat(self, time: float, obs: StudentObservation, state: _Student,
              labels: dict[str, Any], taken: set[str]) -> list[Event]:
        if not (self.on["seat_occupancy"] or self.on["out_of_seat"]):
            return []
        s = self.settings
        frame = shoulders(obs.person, self.min_conf)
        distance = None
        if frame is not None:
            (x, y), width = frame
            steady = (state.previous_mid is not None
                      and math.dist(state.previous_mid, (x, y)) <= 0.3 * width)
            state.previous_mid = (x, y)
            seated = obs.posture == "sitting" or (obs.posture in (None, "unknown") and steady)
            if state.seat is not None:
                dx, dy = x - state.seat.x, y - state.seat.y
                # Standing up in place raises the shoulders ~1.3 widths: not leaving the seat.
                distance = math.hypot(dx, max(0.0, abs(dy) - 1.5 * width)) / width
            if seated and (distance is None or distance < 0.5):
                state.anchors.append((x, y, width))
            if state.seat is None and len(state.anchors) >= int(s["seat_min_samples"]):
                home = (median(a[0] for a in state.anchors), median(a[1] for a in state.anchors))
                state.seat = self.seats.assign(obs.track_id, home, median(a[2] for a in state.anchors), taken)
                taken.add(state.seat.seat_id)
                distance = 0.0
        state.away_since, away = self._held(state.away_since,
                                            distance is not None and distance > s["out_of_seat_widths"],
                                            time, float(s["out_of_seat_min_seconds"]))
        out = ("unknown" if state.seat is None or distance is None else "out_of_seat" if away else "in_seat")
        events: list[Event] = []
        if self.on["seat_occupancy"]:
            seat_label = (state.seat.seat_id if out == "in_seat" else "away" if out == "out_of_seat" else "unknown")
            events += self.segment("seat_occupancy", obs.track_id, time, seat_label, obs.role, None,
                                   "limited" if seat_label == "unknown" else "adequate",
                                   zone=state.seat.seat_id if state.seat else None)
            labels["seat"] = seat_label
        if self.on["out_of_seat"]:
            events += self.segment("out_of_seat", obs.track_id, time, out, obs.role, None,
                                   "limited" if out == "unknown" else "adequate",
                                   zone=state.seat.seat_id if state.seat else None)
            labels["out_of_seat"] = out
            labels["seat_distance"] = round(distance, 2) if distance is not None else None
        return events

    def seat_map_event(self, end: float) -> Event | None:
        """Learnt seats (normalized positions) for the summary's seat and row tables."""
        if not self.seats.seats:
            return None
        width, height = self.frame_size
        return make_event(self.video_id, "seat_map", end, end,
                          {"seats": [{"seat_id": seat.seat_id, "x": round(seat.x / width, 4),
                                      "y": round(seat.y / height, 4),
                                      "shoulder_width": round(seat.width / width, 4),
                                      "tracks": seat.tracks} for seat in self.seats.seats]},
                          role="classroom", coverage="limited")

    # --- head down and alertness -----------------------------------------------

    def _head_and_alertness(self, time: float, obs: StudentObservation, state: _Student,
                            labels: dict[str, Any]) -> list[Event]:
        s = self.settings
        frame = shoulders(obs.person, self.min_conf)
        head = head_point(obs.person, self.min_conf)
        down = None
        if obs.face is not None:
            pitch = obs.face.pitch
            base = median(state.pitches) if len(state.pitches) >= 5 else None
            state.pitches.append(pitch)
            eyes_down = (obs.face.scores.get("eyeLookDownLeft", 0) + obs.face.scores.get("eyeLookDownRight", 0)) / 2
            # Eyes down against the student's usual level: the camera is above
            # the teacher, so watching the front already reads as looking down
            # (A3: two students labelled PHONE with their heads up).
            usual_eyes = median(state.eyes) if len(state.eyes) >= 5 else None
            state.eyes.append(eyes_down)
            eyes_low = (usual_eyes is not None and eyes_down >= s["eyes_down_min"]
                        and eyes_down - usual_eyes >= s["eyes_down_rise"])
            labels["eyes_down"] = round(eyes_down, 2)
            labels["eyes_looking_down"] = eyes_low
            if base is not None:
                down = pitch - base <= -s["head_down_pitch_degrees"] or (eyes_low and pitch - base <= -8)
        eyes_seen = any(obs.person.point(i, self.min_conf) is not None for i in (LEFT_EYE, RIGHT_EYE))
        nose_seen = obs.person.point(NOSE, self.min_conf) is not None
        if frame is not None and head is not None:
            (_, my), width = frame
            height = (my - head[1]) / width
            usual = median(state.heights) if len(state.heights) >= 5 else None
            state.heights.append(height)
            pose_down = height <= s["head_down_min_height"] or (usual is not None and height <= usual - s["head_down_drop"])
            # Absolute cues (no usual level needed; a student down for the whole
            # run has no "up" to compare with). From the camera above, a head
            # bowed steeply shows the nose but hides both eyes (A2: a student
            # in a cap on his phone, user correction 2026-11-18).
            bowed = (nose_seen and not eyes_seen) or bool(pose_looks_down(obs.person, self.min_conf, s))
            down = bool(down) or pose_down or bowed
        if frame is not None and not nose_seen and not eyes_seen:
            # Face hidden: head resting on the arms or the desk (B1). No head
            # sticks up above the shoulders, or the shoulders sank below the
            # seated level.
            (_, my), width = frame
            head_room = (my - obs.person.box[1] * obs.person.height) / width
            if head_room <= s["head_on_desk_room"] or (
                    obs.shoulder_rise is not None and obs.shoulder_rise <= -s["head_hidden_shoulder_drop"]):
                down = True
        head_label = "unknown" if down is None else "down" if down else "up"
        labels["head_down_state"] = head_label
        events: list[Event] = []
        if self.on["head_down"]:
            events += self.segment("head_down", obs.track_id, time, head_label, obs.role, None,
                                   "limited" if head_label == "unknown" or head is None else "adequate")
            labels["head_down"] = head_label
        # Alertness: possible sleep, dozing, tired (yawning), awake.
        face = obs.face_labels or {}
        still = obs.motion in ("still", "low", None, "unknown")
        # Sleep needs the face hidden (head resting on the arms or the desk) or
        # the eyes closed. A student bent over a phone also keeps the head down
        # and still, but the face stays visible with the eyes open (A2, user
        # correction 2026-11-18: that was labelled possible_sleep).
        face_hidden = all(obs.person.point(i, self.min_conf) is None for i in (NOSE, LEFT_EYE, RIGHT_EYE))
        # The pose model still guesses face points on a head lying on the arms
        # (B1: nose 0.5, an eye 0.41), so a head at or below the shoulder line
        # counts as resting too; a student bent over a phone keeps the head above it.
        resting = face_hidden or (frame is not None and head is not None
                                  and (frame[0][1] - head[1]) / frame[1] <= s["head_rest_height"])
        labels["head_resting"] = resting and head_label == "down"
        state.down_since, long_down = self._held(state.down_since,
                                                 head_label == "down" and still and resting, time,
                                                 float(s["sleep_min_seconds"]))
        closed = face.get("eyes_closed") == "closed"
        state.eyes_closed_since, long_closed = self._held(state.eyes_closed_since, closed, time,
                                                          float(s["long_eyes_closed_seconds"]))
        if long_down or long_closed:
            alert = "possible_sleep"
        elif closed:
            alert = "dozing"
        elif face.get("yawn") == "yawning":
            alert = "tired"
        elif obs.face is not None and face.get("eyes_closed") == "open":
            alert = "awake"
        elif head_label == "up" and obs.motion in ("medium", "high"):
            alert = "awake"
        elif head_label == "down" and not resting:
            alert = "awake"  # looking down with the head up off the desk: reading, writing, a phone
        else:
            alert = "unknown"
        labels["alertness_state"] = alert
        if self.on["alertness"]:
            events += self.segment("alertness", obs.track_id, time, alert, obs.role, None,
                                   "limited" if alert in ("unknown", "possible_sleep") else "adequate")
            labels["alertness"] = alert
        return events

    # --- desk activity: device use, note-taking, reading -----------------------

    def _desk_activity(self, time: float, obs: StudentObservation, state: _Student,
                       labels: dict[str, Any], owned: list[tuple[Any, float]]) -> list[Event]:
        if not (self.on["device_use"] or self.on["note_taking"] or self.on["reading"]):
            return []
        s = self.settings
        frame = shoulders(obs.person, self.min_conf)
        wrists = {side: _px(obs.person, ARMS[side][2], self.min_conf) for side in ARMS}
        speeds = []
        if frame is not None and state.previous_wrists and state.previous_time is not None:
            dt = time - state.previous_time
            (mx, my), width = frame
            if 0 < dt <= 2.5 * self.period:
                for side, point in wrists.items():
                    before = state.previous_wrists.get(side)
                    if point is not None and before is not None:
                        rel_now = (point[0] - mx, point[1] - my)
                        rel_before = (before[0] - state.previous_wrists["_mid"][0],
                                      before[1] - state.previous_wrists["_mid"][1])
                        speeds.append(math.dist(rel_now, rel_before) / width / dt)
        if frame is not None:
            state.previous_wrists = {**wrists, "_mid": frame[0]}
            state.previous_time = time
        head = labels.get("head_down_state", "unknown")
        eyes_low = bool(labels.get("eyes_looking_down"))
        looking_down = head == "down" or eyes_low
        phone = [d for item, d in owned if item.label == "cell phone"]
        # Only an open laptop is a device in use: a closed one is a flat slab
        # (A2: a closed laptop in front of a student on his phone read as laptop).
        laptops = [(item, d) for item, d in owned if item.label == "laptop"
                   and (item.box[3] - item.box[1]) * obs.person.height
                   >= s["laptop_open_ratio"] * (item.box[2] - item.box[0]) * obs.person.width]
        books = [item for item, _ in owned if item.label == "book"]
        if frame is None and head == "unknown":
            activity = "unknown"
        else:
            width = frame[1] if frame else 1.0
            size = (obs.person.width, obs.person.height)
            on_laptop = any(self._hand_on(tuple(v * size[i % 2] for i, v in enumerate(item.box)), wrists,
                                          0.3 * width) for item, _ in laptops)
            visible = [p for p in wrists.values() if p is not None]
            # Both hands together in front of the body: holding something (a
            # phone), not writing (A2 back row read as writing before 2026-11-18).
            both_held = (frame is not None and len(visible) == 2
                         and math.dist(*visible) <= s["held_phone_max_gap"] * width
                         and all(frame[0][1] - 0.5 * width <= p[1] <= frame[0][1] + 3.0 * width for p in visible)
                         and abs((visible[0][0] + visible[1][0]) / 2 - frame[0][0]) <= 1.2 * width)
            moving = [v for v in speeds if s["writing_min_speed"] <= v <= s["writing_max_speed"]]
            # Writing is stop-and-go: a hand that moved in any of the last 3 samples.
            state.moved.append(bool(moving))
            moved_recently = any(state.moved)
            on_desk = frame is not None and any(
                frame[0][1] + 0.3 * width <= p[1] <= frame[0][1] + 2.5 * width for p in visible)
            # Looking down at the own hands or desk, not turned to a neighbour.
            direction = ((obs.face_labels or {}).get("head_direction") if obs.face is not None
                         else None)
            sideways = (direction in ("turned_left", "turned_right") if direction not in (None, "unknown")
                        else facing_direction(obs.person, self.min_conf) in ("sideways_left", "sideways_right"))
            eyes_far_down = (labels.get("eyes_down") or 0) >= s["phone_eyes_down"]
            pointed_down = (looking_down or eyes_far_down) and not sideways
            # Phone use without seeing the phone (user, 2026-11-18): a phone under
            # the desk or covered by the hands is not detected, but the head stays
            # pointed down at it. A head resting on the arms is sleep, not a phone.
            # Most of the last phone_head_down_seconds, so one frame where the
            # head angle flickers does not restart the count (A4 front right).
            window = max(1, round(float(s["phone_head_down_seconds"]) / self.period))
            state.phone_down.append(pointed_down and not labels.get("head_resting"))
            recent_down = list(state.phone_down)[-window:]
            sustained = len(recent_down) >= window and sum(recent_down) >= max(1, window - 1)
            # A detected phone at a hand counts while the student looks down at it
            # or holds it up (A1: a phone lying by a resting hand with the head up
            # is not use); a phone lying in front counts while looked at.
            in_hand = any(d <= s["object_near_wrist"] for d in phone)
            held_up = frame is not None and any(
                d <= s["object_near_wrist"] and (item.box[1] + item.box[3]) / 2 * obs.person.height < frame[0][1]
                for item, d in owned if item.label == "cell phone")
            seen_now = (in_hand and (pointed_down or held_up)) or (phone and pointed_down and (eyes_low or eyes_far_down))
            if seen_now:
                state.phone_seen_at = time
            # The detector finds a held phone in only some frames (A4: about half);
            # while the head stays down, a phone seen moments ago is still in use
            # and outranks a paper on the desk (read as a book) or thumbs moving
            # (read as writing).
            recent_phone = (state.phone_seen_at is not None and pointed_down
                            and time - state.phone_seen_at <= s["phone_memory_seconds"])
            if seen_now or recent_phone:
                activity = "phone"
            elif laptops and not sideways and (on_laptop or pointed_down):
                activity = "laptop"  # hands on, or the head down at, an open laptop in front
            elif looking_down and on_desk and moved_recently and not both_held:
                activity = "writing"
            elif books and pointed_down and not moved_recently \
                    and (obs.face_labels or {}).get("eyes_closed") != "closed":
                activity = "reading"  # something to read in front, hands still
            elif sustained:
                activity = "phone_likely"
            elif looking_down:
                activity = "looking_down"
            else:
                activity = "none"
        # Short look-ups while writing or reading: majority of the last samples.
        state.activities.append(activity)
        if activity in ("writing", "reading", "none", "looking_down"):
            recent = [a for a in state.activities if a != "unknown"]
            for candidate in ("writing", "reading"):
                if recent.count(candidate) * 2 > len(recent):
                    activity = candidate
        labels["activity"] = activity
        events: list[Event] = []
        known = activity != "unknown"
        if self.on["device_use"]:
            device = (activity if activity in ("phone", "phone_likely", "laptop")
                      else ("none" if known else "unknown"))
            events += self.segment("device_use", obs.track_id, time, device, obs.role, None,
                                   "limited" if device in ("unknown", "phone_likely") else "adequate")
            labels["device_use"] = device
        if self.on["note_taking"]:
            note = "writing" if activity == "writing" else ("not_writing" if known else "unknown")
            events += self.segment("note_taking", obs.track_id, time, note, obs.role, None,
                                   "limited" if note == "unknown" else "adequate")
            labels["note_taking"] = note
        if self.on["reading"]:
            reading = "reading" if activity == "reading" else ("not_reading" if known else "unknown")
            events += self.segment("reading", obs.track_id, time, reading, obs.role, None,
                                   "adequate" if reading == "reading" and books else
                                   "limited" if reading in ("unknown", "reading") else "adequate")
            labels["reading"] = reading
        return events

    @staticmethod
    def _hand_on(box: tuple[float, float, float, float], wrists: dict[str, Point | None],
                 margin: float) -> bool:
        """A wrist inside a pixel box, with a margin."""
        x1, y1, x2, y2 = box
        return any(point is not None and x1 - margin <= point[0] <= x2 + margin
                   and y1 - margin <= point[1] <= y2 + margin for point in wrists.values())

    # --- response gestures ---------------------------------------------------------------

    def _response_gesture(self, time: float, obs: StudentObservation,
                          labels: dict[str, Any]) -> list[Event]:
        if not self.on["response_gesture"]:
            return []
        hands = obs.gestures or {}
        found = [h for h in hands.values() if h.found and h.gesture in RESPONSE_GESTURES
                 and h.score >= self.settings["gesture_min_score"]]
        if found:
            label = max(found, key=lambda h: h.score).gesture
        elif any(h.found for h in hands.values()):
            label = "none"
        else:
            label = "unknown"  # no hand seen in the wrist crops
        labels["response_gesture"] = label
        return self.segment("response_gesture", obs.track_id, time, label, obs.role, None,
                            "limited" if label == "unknown" else "adequate")

    # --- track end ------------------------------------------------------------------------

    def close(self, track_id: int, role: str) -> list[Event]:
        state = self.tracks.pop(track_id, None)
        if state is None or state.raise_episode is None:
            return []
        return [self._close_raise(track_id, role, state.raise_episode)]
