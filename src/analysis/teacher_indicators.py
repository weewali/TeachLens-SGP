"""List B: teacher visual behaviour from pose keypoints (instructor camera).

Every rule reads one tracked person's COCO-17 keypoints at one analyzed
sample, plus optional calibrated *targets* (board, screen, desk polygons from
the camera's zone file). The rules reuse the pose model's output, so they add
no model and almost no run time. FoundationAnalyzer calls TeacherBehavior for
tracks whose camera-assumed role is teacher.

Assumptions and limits (2D, one camera):
- The instructor camera faces the front wall from the students' side, so
  "toward the class" means toward the camera.
- Orientation is where the head and body point, never eye gaze.
- Leaning toward or away from the camera cannot be measured; only sideways
  tilt in the image can.
- At one analyzed frame per second, quick gestures and brief pointing are
  sampled coarsely; episodes shorter than a second can be missed.

Facing uses which face keypoints the model sees (checked on the scenario
videos 2026-11-17): facing the class shows the nose, both eyes and both ears;
a three-quarter turn hides one ear; a profile along the wall shows one eye
and one ear; a back view shows neither nose nor eyes.
"""

from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Callable

from src.analysis.events import Event, make_event
from src.inference.pose import (
    LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST, PosePerson,
)

Point = tuple[float, float]
Polygon = tuple[tuple[float, float], ...]
WALL_TARGETS = ("board", "screen")  # what a teacher can write on or present from
ARMS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))

# Interval indicators closed with the track (see FoundationAnalyzer._retire).
SEGMENT_NAMES = ("facing_direction", "head_orientation", "board_interaction",
                 "writing_on_board", "pointing", "gesture", "raised_hand", "torso_lean")

DEFAULTS: dict[str, float] = {
    "head_down_ratio": 1.0,            # nose this many eye-gaps below the eyes -> looking down
    "pointing_min_elbow_degrees": 150.0,
    "pointing_min_reach": 0.6,         # shoulder-to-wrist / torso length
    "pointing_min_raise_degrees": 40.0,  # arm at least this far from hanging straight down
    "raised_max_from_vertical_degrees": 35.0,  # a straight arm this close to vertical is a raised hand
    "target_margin": 0.03,             # normalized x margin around a target
    "gesture_min_speed": 0.15,         # wrist speed relative to the torso, torso lengths / s
    "gesture_raised_min_speed": 0.08,  # enough for a hand raised to chest height or above
    "gesture_raised_max_drop": 0.6,    # "raised": wrist at most 0.6 torso lengths below the shoulders
    "gesture_wrist_confidence": 0.5,   # wrists below this confidence are ignored for gestures
    "gesture_face_distance": 0.6,      # wrist within this many shoulder widths of the nose = hand at the face
    "locomotion_min_speed": 0.25,      # hip speed (body heights / s) treated as walking
    "stationary_max_speed": 0.08,      # hip speed (body heights / s) treated as standing still
    "stationary_min_seconds": 5.0,
    "lean_upright_max_degrees": 10.0,
    "lean_strong_min_degrees": 25.0,
    "writing_smoothing_samples": 3,
}


# --- geometry -------------------------------------------------------------

def _px(person: PosePerson, index: int, min_conf: float) -> Point | None:
    """Keypoint in pixels (aspect ratio kept), or None below min_conf."""
    point = person.point(index, min_conf)
    return None if point is None else (point.x * person.width, point.y * person.height)


def _mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _angle(a: Point, b: Point, c: Point) -> float:
    """Angle ABC in degrees."""
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*v1) * math.hypot(*v2)
    if norm == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / norm))))


def inside(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test (normalized coordinates)."""
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _x_range(polygon: Polygon) -> tuple[float, float]:
    xs = [x for x, _ in polygon]
    return min(xs), max(xs)


def ray_target(person: PosePerson, start: Point, direction: Point,
               targets: dict[str, Polygon], max_length: float = 1.5) -> str | None:
    """First target polygon a ray from start (pixels) along direction meets."""
    length = math.hypot(*direction)
    if length == 0:
        return None
    step = max(person.width, person.height) * max_length / 120
    dx, dy = direction[0] / length * step, direction[1] / length * step
    x, y = start
    for _ in range(120):
        x, y = x + dx, y + dy
        point = (x / person.width, y / person.height)
        if not (-0.1 <= point[0] <= 1.1 and -0.1 <= point[1] <= 1.1):
            return None
        hit = next((name for name, polygon in targets.items() if inside(point, polygon)), None)
        if hit is not None:
            return hit
    return None


@dataclass(frozen=True)
class Body:
    """Shoulder/hip frame of one person in pixels."""
    shoulder_mid: Point
    shoulder_width: float
    torso: float  # shoulder midpoint to hip midpoint (or an estimate)
    hip_mid: Point | None
    left_shoulder: Point
    right_shoulder: Point


def body(person: PosePerson, min_conf: float) -> Body | None:
    left, right = _px(person, LEFT_SHOULDER, min_conf), _px(person, RIGHT_SHOULDER, min_conf)
    if left is None or right is None:
        return None
    shoulder_mid = _mid(left, right)
    hip_left, hip_right = _px(person, LEFT_HIP, min_conf), _px(person, RIGHT_HIP, min_conf)
    hip_mid = _mid(hip_left, hip_right) if hip_left and hip_right else None
    box_height = (person.box[3] - person.box[1]) * person.height
    torso = math.dist(shoulder_mid, hip_mid) if hip_mid else 0.3 * box_height
    if torso < 8:
        return None
    return Body(shoulder_mid, math.dist(left, right), torso, hip_mid, left, right)


# --- per-sample rules -------------------------------------------------------

def facing_direction(person: PosePerson, min_conf: float) -> str:
    """toward_class, sideways_left/right (image direction), away_from_class, or unknown.

    toward_class covers frontal and three-quarter views (both eyes seen);
    sideways is a profile along the wall (one eye, or the nose alone); away
    is a back view: shoulders seen but no nose or eyes. Pose models do not
    reliably swap left and right for people seen from behind, so shoulder
    order is not used.
    """
    nose = person.point(NOSE, min_conf)
    left_side = [person.point(i, min_conf) is not None for i in (LEFT_EYE, LEFT_EAR)]
    right_side = [person.point(i, min_conf) is not None for i in (RIGHT_EYE, RIGHT_EAR)]
    eyes = left_side[0] + right_side[0]
    if eyes == 2:
        return "toward_class"
    if nose is not None or eyes == 1:
        # The visible side of the face faces the camera: seeing the person's
        # left eye/ear means they look toward the image left.
        if any(left_side) and not any(right_side):
            return "sideways_left"
        if any(right_side) and not any(left_side):
            return "sideways_right"
        frame = body(person, min_conf)
        if nose is not None and frame is not None and frame.shoulder_width > 0:
            offset = (nose.x * person.width - frame.shoulder_mid[0]) / frame.shoulder_width
            if abs(offset) >= 0.15:
                return "sideways_right" if offset > 0 else "sideways_left"
        return "unknown"
    # No nose or eyes but shoulders visible: the back of the head faces the camera.
    return "unknown" if body(person, min_conf) is None else "away_from_class"


def head_pitch_down(person: PosePerson, min_conf: float, down_ratio: float) -> bool | None:
    """True when the nose sits far below the eyes (looking down); None if not measurable."""
    nose = _px(person, NOSE, min_conf)
    left, right = _px(person, LEFT_EYE, min_conf), _px(person, RIGHT_EYE, min_conf)
    if nose is None or left is None or right is None:
        return None
    gap = math.dist(left, right)
    if gap < 2:
        return None
    return (nose[1] - _mid(left, right)[1]) / gap >= down_ratio


def head_orientation(person: PosePerson, facing: str, min_conf: float,
                     targets: dict[str, Polygon], down_ratio: float, margin: float) -> str:
    """Approximate target of the head: students, notes, board, screen, desk, front_wall,
    side_left, side_right, or unknown. Orientation, not gaze."""
    if facing == "unknown":
        return "unknown"
    head = _head_point(person, min_conf)
    if head is None:
        return "unknown"
    if facing == "toward_class":
        down = head_pitch_down(person, min_conf, down_ratio)
        if down:
            return "notes"  # looking down: notes, laptop, desk, or phone
        return "students"
    if facing == "away_from_class":
        walls = {name: polygon for name, polygon in targets.items() if name in WALL_TARGETS}
        for name, polygon in walls.items():
            low, high = _x_range(polygon)
            if low - margin <= head[0] <= high + margin:
                return name
        return "front_wall"
    # Profile: the first target at head height on that side, else the side wall.
    side = 1 if facing == "sideways_right" else -1
    pixel = (head[0] * person.width, head[1] * person.height)
    hit = ray_target(person, pixel, (side, 0.0), targets)
    return hit or ("side_right" if side > 0 else "side_left")


def _head_point(person: PosePerson, min_conf: float) -> Point | None:
    """Normalized head position: nose, else eye/ear average, else shoulder midpoint lifted."""
    points = [person.point(i, min_conf) for i in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)]
    points = [p for p in points if p is not None]
    if points:
        return (sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))
    return None


def raised_hands(person: PosePerson, min_conf: float) -> str:
    """none, one, both, or unknown: wrist above the head (or well above the shoulder)."""
    count, seen = 0, 0
    nose = _px(person, NOSE, min_conf)
    for shoulder_i, elbow_i, wrist_i in ARMS:
        shoulder, wrist = _px(person, shoulder_i, min_conf), _px(person, wrist_i, min_conf)
        if shoulder is None or wrist is None:
            continue
        seen += 1
        elbow = _px(person, elbow_i, min_conf)
        frame = body(person, min_conf)
        width = frame.shoulder_width if frame else 0.0
        top = nose[1] if nose is not None else shoulder[1] - 0.6 * width
        if wrist[1] < top and (elbow is None or elbow[1] < shoulder[1] + 0.1 * width):
            count += 1
    if seen == 0:
        return "unknown"
    return ("none", "one", "both")[count]


def torso_tilt(person: PosePerson, min_conf: float) -> float | None:
    """Sideways tilt of the torso in degrees from vertical (+ = toward the image right)."""
    frame = body(person, min_conf)
    if frame is None or frame.hip_mid is None:
        return None
    dx = frame.shoulder_mid[0] - frame.hip_mid[0]
    dy = frame.hip_mid[1] - frame.shoulder_mid[1]
    if dy <= 0:
        return None
    return round(math.degrees(math.atan2(dx, dy)), 1)


def lean_label(tilt: float | None, upright_max: float, strong_min: float) -> str:
    if tilt is None:
        return "unknown"
    if abs(tilt) <= upright_max:
        return "upright"
    side = "right" if tilt > 0 else "left"
    return f"strong_lean_{side}" if abs(tilt) >= strong_min else f"lean_{side}"


@dataclass(frozen=True)
class Pointing:
    target: str  # board, screen, desk, students, side, or unknown
    start: Point  # normalized wrist
    end: Point  # normalized point along the arm direction, for drawing


def pointing(person: PosePerson, min_conf: float, facing: str, targets: dict[str, Polygon],
             settings: dict[str, float]) -> Pointing | None:
    """The most extended pointing arm and its approximate target, or None.

    An arm points when it is nearly straight, reaches away from the body, and
    is lifted from hanging down, but is not held straight up (a raised hand).
    The shoulder-to-wrist ray picks a target; with none hit, a teacher facing
    the class is pointing at students and one turned sideways at the side of
    the room (a door, a window, or a student group at that side).
    """
    frame = body(person, min_conf)
    if frame is None:
        return None
    best: tuple[float, Point, Point] | None = None
    for shoulder_i, elbow_i, wrist_i in ARMS:
        shoulder, elbow, wrist = (_px(person, i, min_conf) for i in (shoulder_i, elbow_i, wrist_i))
        if shoulder is None or elbow is None or wrist is None:
            continue
        reach = math.dist(shoulder, wrist) / frame.torso
        if (_angle(shoulder, elbow, wrist) < settings["pointing_min_elbow_degrees"]
                or reach < settings["pointing_min_reach"]):
            continue
        direction = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
        from_down = math.degrees(math.atan2(abs(direction[0]), direction[1]))  # 0 = hanging
        if from_down < settings["pointing_min_raise_degrees"]:
            continue
        if 180 - from_down <= settings["raised_max_from_vertical_degrees"]:
            continue  # straight up: a raised hand, not pointing
        if best is None or reach > best[0]:
            best = (reach, wrist, direction)
    if best is None:
        return None
    _, wrist, direction = best
    target = ray_target(person, wrist, direction, targets)
    if target is None:
        target = ("students" if facing == "toward_class" else
                  "side" if facing.startswith("sideways") else "unknown")
    length = math.hypot(*direction) or 1.0
    reach_px = 0.25 * max(person.width, person.height)
    end = (wrist[0] + direction[0] / length * reach_px, wrist[1] + direction[1] / length * reach_px)
    return Pointing(target, (wrist[0] / person.width, wrist[1] / person.height),
                    (end[0] / person.width, end[1] / person.height))


def near_target(person: PosePerson, frame: Body, polygon: Polygon, margin: float) -> bool:
    """Body in front of a wall target in the image: shoulders within its x range and the
    box reaching up to it."""
    low, high = _x_range(polygon)
    x = frame.shoulder_mid[0] / person.width
    top = min(y for _, y in polygon)
    bottom = max(y for _, y in polygon)
    return low - margin <= x <= high + margin and person.box[1] <= bottom and person.box[3] >= top


def forearm_lifted(person: PosePerson, elbow_i: int, wrist_i: int, min_conf: float) -> bool:
    """Forearm level or raised: the wrist no more than ~30 degrees below the elbow.

    Writing low on a board keeps the forearm about level (HowToSpeak 47:49),
    so "wrist above elbow" alone misses it.
    """
    elbow, wrist = _px(person, elbow_i, min_conf), _px(person, wrist_i, min_conf)
    if elbow is None or wrist is None:
        return False
    return wrist[1] - elbow[1] <= 0.5 * math.dist(elbow, wrist)


def wrist_in_target(person: PosePerson, min_conf: float, polygon: Polygon) -> Point | None:
    """A wrist inside the target polygon with the forearm level or raised."""
    for _, elbow_i, wrist_i in ARMS:
        wrist = person.point(wrist_i, min_conf)
        if wrist is None or not forearm_lifted(person, elbow_i, wrist_i, min_conf):
            continue
        if inside((wrist.x, wrist.y), polygon):
            return (wrist.x, wrist.y)
    return None


def board_part(x: float, polygon: Polygon) -> str:
    low, high = _x_range(polygon)
    third = (x - low) / max(1e-6, high - low)
    return "board_left" if third < 1 / 3 else "board_center" if third < 2 / 3 else "board_right"


# --- per-track state ----------------------------------------------------------

@dataclass
class _TeacherTrack:
    hips: Point | None = None  # previous hip midpoint (pixels)
    hands: dict[int, Point] = field(default_factory=dict)  # wrist minus shoulder midpoint, in torsos
    time: float | None = None
    box_height: float = 0.0
    writing: list[bool] = field(default_factory=list)
    hip_speeds: list[float] = field(default_factory=list)  # recent, for smoothing
    still_start: float | None = None
    still_last: float | None = None
    still_points: list[Point] = field(default_factory=list)
    still_zones: list[str] = field(default_factory=list)


Segments = Callable[..., list[Event]]  # _Segments.update(name, track_id, time, label, role, confidence, coverage, zone=None)


class TeacherBehavior:
    """List B rules for teacher tracks; holds per-track state between samples."""

    def __init__(self, video_id: str, enabled: dict[str, bool], targets: dict[str, Polygon],
                 zones: dict[str, Polygon], min_conf: float, sample_period: float,
                 settings: dict[str, Any] | None, update_segment: Segments):
        self.video_id = video_id
        self.on = {name: bool(enabled.get(name, False)) for name in (
            "stationary_episodes", "facing_direction", "head_orientation", "board_interaction",
            "writing_on_board", "pointing", "gestures", "raised_hand", "torso_lean")}
        self.targets = targets
        self.zones = zones
        self.min_conf = min_conf
        self.period = sample_period
        self.settings = {**DEFAULTS, **(settings or {})}
        self.segment = update_segment
        self.tracks: dict[int, _TeacherTrack] = {}

    @property
    def active(self) -> bool:
        return any(self.on.values())

    def update(self, time: float, track_id: int, person: PosePerson,
               role: str) -> tuple[list[Event], dict[str, Any]]:
        """Events for one teacher sample, plus overlay state (label per indicator)."""
        s, c = self.settings, self.min_conf
        state = self.tracks.setdefault(track_id, _TeacherTrack())
        events: list[Event] = []
        shown: dict[str, Any] = {}
        frame = body(person, c)
        box_height = (person.box[3] - person.box[1]) * person.height
        dt = time - state.time if state.time is not None else None

        # Hip speed: locomotion vs standing still (body heights / second).
        hip_speed = None
        hips = frame.hip_mid if frame else None
        if hips is not None and state.hips is not None and dt and dt <= 2.5 * self.period:
            hip_speed = math.dist(hips, state.hips) / max(1.0, (box_height + state.box_height) / 2) / dt

        facing = facing_direction(person, c)
        head = head_orientation(person, facing, c, self.targets, s["head_down_ratio"], s["target_margin"])

        if self.on["facing_direction"]:
            events += self.segment("facing_direction", track_id, time, facing, role, None,
                                   "limited" if facing == "unknown" else "adequate")
            shown["facing"] = facing
        if self.on["head_orientation"]:
            events += self.segment("head_orientation", track_id, time, head, role, None,
                                   "limited" if head in ("unknown", "front_wall") else "adequate")
            shown["head"] = head

        point = pointing(person, c, facing, self.targets, s) if frame else None
        interaction = "unknown" if frame is None else "none"
        if frame is not None:
            for name in WALL_TARGETS:
                polygon = self.targets.get(name)
                if polygon is None or not near_target(person, frame, polygon, s["target_margin"]):
                    continue
                # Attending to it: looking at it, pointing at it, or back to the
                # class with a hand on it. A hand merely in front of it while
                # facing the class is a gesture, not interaction.
                if (head == name or (point is not None and point.target == name)
                        or (facing == "away_from_class" and wrist_in_target(person, c, polygon) is not None)):
                    interaction = name
                    break
        if self.on["board_interaction"]:
            events += self.segment("board_interaction", track_id, time, interaction, role, None,
                                   "limited" if interaction == "unknown" or not self.targets else "adequate")
            shown["board"] = interaction

        if self.on["writing_on_board"]:
            events += self._writing(time, track_id, person, role, state, facing, interaction,
                                    hip_speed, shown)

        if self.on["pointing"]:
            label = "unknown" if frame is None else point.target if point else "none"
            if shown.get("writing") == "writing":
                label, point = "none", None  # an arm on the board while writing is not pointing
            events += self.segment("pointing", track_id, time, label, role, None,
                                   "limited" if label in ("unknown", "students", "side") else "adequate")
            shown["pointing"] = label
            shown["pointing_ray"] = (point.start, point.end) if point else None

        if self.on["gestures"]:
            events += self._gestures(time, track_id, person, role, state, frame, hip_speed, dt, shown)

        if self.on["raised_hand"]:
            raised = raised_hands(person, c)
            events += self.segment("raised_hand", track_id, time, raised, role, None,
                                   "limited" if raised == "unknown" else "adequate")
            shown["raised"] = raised

        if self.on["torso_lean"]:
            tilt = torso_tilt(person, c)
            lean = lean_label(tilt, s["lean_upright_max_degrees"], s["lean_strong_min_degrees"])
            events += self.segment("torso_lean", track_id, time, lean, role, None,
                                   "limited" if lean == "unknown" else "adequate")
            if tilt is not None:
                events.append(make_event(self.video_id, "torso_tilt", time, time,
                                         {"degrees_from_vertical": tilt}, track_id=track_id,
                                         role=role, confidence=None, coverage="limited"))
            shown["lean"] = lean

        if self.on["stationary_episodes"]:
            events += self._stationary(time, track_id, person, role, state, hips, hip_speed, shown)

        state.hips, state.time, state.box_height = hips, time, box_height
        return events, shown

    def _writing(self, time, track_id, person, role, state, facing, interaction, hip_speed, shown):
        s = self.settings
        board = self.targets.get("board")
        wrist = None
        if board is not None:
            wrist = wrist_in_target(person, self.min_conf, board) if interaction == "board" else None
            candidate = (wrist is not None and facing in ("away_from_class", "sideways_left", "sideways_right")
                         and (hip_speed is None or hip_speed < s["locomotion_min_speed"]))
            coverage = "adequate"
        else:
            # No calibrated board: back to the class with a forearm lifted to
            # shoulder height and the feet still (weaker evidence).
            frame = body(person, self.min_conf)
            lifted = False
            if frame is not None:
                for _, elbow_i, wrist_i in ARMS:
                    w = _px(person, wrist_i, self.min_conf)
                    if (w is not None and forearm_lifted(person, elbow_i, wrist_i, self.min_conf)
                            and w[1] < frame.shoulder_mid[1] + 0.5 * frame.torso):
                        lifted = True
            candidate = (facing == "away_from_class" and lifted
                         and (hip_speed is None or hip_speed < s["locomotion_min_speed"]))
            coverage = "limited"
        window = int(s["writing_smoothing_samples"])
        state.writing = (state.writing + [candidate])[-window:]
        writing = sum(state.writing) * 2 > len(state.writing)  # majority of recent samples
        label = "writing" if writing else "not_writing"
        zone = board_part(wrist[0], board) if writing and wrist is not None and board is not None else None
        shown["writing"] = label
        return self.segment("writing_on_board", track_id, time, label, role, None, coverage, zone=zone)

    def _gestures(self, time, track_id, person, role, state, frame, hip_speed, dt, shown):
        s = self.settings
        events: list[Event] = []
        hands: dict[int, Point] = {}
        busy: set[int] = set()  # hands at the face or on the desk: not gestures
        if frame is not None:
            nose = _px(person, NOSE, self.min_conf)
            desk = self.targets.get("desk")
            # Only wrists the model is fairly sure of: hidden hands (behind the
            # back) get guessed positions that jump between samples.
            for _, _, wrist_i in ARMS:
                wrist = _px(person, wrist_i, max(self.min_conf, s["gesture_wrist_confidence"]))
                if wrist is None:
                    continue
                hands[wrist_i] = ((wrist[0] - frame.shoulder_mid[0]) / frame.torso,
                                  (wrist[1] - frame.shoulder_mid[1]) / frame.torso)
                # Chin or cheek resting on a hand, or hands typing / turning
                # pages on the desk, move without communicating anything.
                if ((nose is not None and math.dist(wrist, nose) < s["gesture_face_distance"] * frame.shoulder_width)
                        or (desk is not None and inside((wrist[0] / person.width, wrist[1] / person.height), desk))):
                    busy.add(wrist_i)
        moves = {i: math.dist(hands[i], state.hands[i]) / dt for i in hands
                 if i in state.hands and i not in busy and dt and dt <= 2.5 * self.period}
        known = any(i in state.hands for i in hands) and dt and dt <= 2.5 * self.period
        if not known:
            label = "unknown"
        elif hip_speed is not None and hip_speed >= s["locomotion_min_speed"]:
            label = "walking"  # hands swing while walking; not counted as gestures
        elif shown.get("writing") == "writing":
            label = "writing"
        else:
            # A clear hand movement, or a smaller one with the hand raised to
            # chest height or above (talking with the hands, or holding a hand
            # up toward the screen, which moves little between samples).
            gesturing = any(
                speed >= s["gesture_min_speed"]
                or (speed >= s["gesture_raised_min_speed"]
                    and hands[i][1] <= s["gesture_raised_max_drop"])
                for i, speed in moves.items())
            label = "gesturing" if gesturing else "hands_still"
        events += self.segment("gesture", track_id, time, label, role, None,
                               "limited" if label == "unknown" else "adequate")
        if label == "gesturing" and frame is not None and frame.shoulder_width > 0:
            spread = max(abs(hands[i][0]) * frame.torso / frame.shoulder_width for i in moves)
            events.append(make_event(self.video_id, "gesture_expansiveness", time, time,
                                     {"wrist_reach_shoulder_widths": round(spread, 2)},
                                     track_id=track_id, role=role, confidence=None, coverage="limited"))
        state.hands = hands
        shown["gesture"] = label
        return events

    def _stationary(self, time, track_id, person, role, state, hips, hip_speed, shown):
        # Median of the last three hip speeds, so one jittery sample does not split an episode.
        if hip_speed is None:
            state.hip_speeds = []
        else:
            state.hip_speeds = (state.hip_speeds + [hip_speed])[-3:]
        smoothed = median(state.hip_speeds) if state.hip_speeds else None
        still = smoothed is not None and smoothed <= self.settings["stationary_max_speed"]
        events: list[Event] = []
        if still:
            if state.still_start is None:
                state.still_start = state.time if state.time is not None else time
                state.still_points, state.still_zones = [], []
            state.still_last = time
            state.still_points.append((hips[0] / person.width, hips[1] / person.height))
            foot = ((person.box[0] + person.box[2]) / 2, person.box[3])
            state.still_zones.append(next((name for name, polygon in self.zones.items()
                                           if inside(foot, polygon)), "outside_calibrated_zones")
                                     if self.zones else None)
        else:
            events += self._end_stationary(track_id, role, state)
        shown["stationary"] = "stationary" if still else "moving" if smoothed is not None else "unknown"
        return events

    def _end_stationary(self, track_id: int, role: str, state: _TeacherTrack,
                        end: float | None = None) -> list[Event]:
        if state.still_start is None or state.still_last is None:
            return []
        start, last = state.still_start, end if end is not None else state.still_last
        points, zones = state.still_points, [z for z in state.still_zones if z]
        state.still_start = state.still_last = None
        state.still_points, state.still_zones = [], []
        if last - start < self.settings["stationary_min_seconds"] or not points:
            return []
        where = [round(median(p[0] for p in points), 4), round(median(p[1] for p in points), 4)]
        zone = max(set(zones), key=zones.count) if zones else None
        return [make_event(self.video_id, "stationary_episode", start, last,
                           {"hip_center": where, "samples": len(points)}, track_id=track_id,
                           role=role, confidence=None, coverage="adequate", zone=zone)]

    def close(self, track_id: int, role: str, end: float) -> list[Event]:
        """Finish a retired or final track: emit an open stationary episode."""
        state = self.tracks.pop(track_id, None)
        if state is None or not self.on["stationary_episodes"]:
            return []
        return self._end_stationary(track_id, role, state,
                                    end=min(end, (state.still_last or end) + self.period))
