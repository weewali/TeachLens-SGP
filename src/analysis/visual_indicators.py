"""List A: conservative, observable indicators from anonymous pose tracks.

This module is deliberately independent of Ultralytics. Thresholds are
configured in YAML, and uncertain role/posture/motion is reported as unknown.
"""

from collections import deque
from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Any

from src.analysis.events import Event, make_event
from src.inference.pose import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
    LEFT_WRIST, NOSE, RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP,
    RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, Keypoint, PosePerson,
)
from src.preprocessing.camera_motion import Affine, CameraMotion, apply_affine


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Ray-casting test in normalized frame coordinates."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def zone_for(point: tuple[float, float], zones: dict[str, tuple[tuple[float, float], ...]]) -> str:
    """First matching zone wins; define overlapping polygons in priority order."""
    return next((name for name, polygon in zones.items() if point_in_polygon(point, polygon)),
                "outside_calibrated_zones")


def restrict_joints(person: PosePerson, allowed: frozenset[int]) -> PosePerson:
    """Zero the confidence of joints that rules must not use.

    Those are joints outside the camera's pose_joints (e.g. legs hidden by
    desks, whose positions the model still guesses) and joints placed outside
    the image (e.g. feet below the frame when someone is near the camera).
    """
    return replace(person, keypoints=tuple(
        point if index in allowed and 0 <= point.x <= 1 and 0 <= point.y <= 1
        else replace(point, confidence=0.0)
        for index, point in enumerate(person.keypoints)))


def _angle(person: PosePerson, a: int, b: int, c: int, min_conf: float) -> float | None:
    points = [person.point(i, min_conf) for i in (a, b, c)]
    if any(point is None for point in points):
        return None
    first, middle, last = points
    v1 = ((first.x - middle.x) * person.width, (first.y - middle.y) * person.height)
    v2 = ((last.x - middle.x) * person.width, (last.y - middle.y) * person.height)
    lengths = math.hypot(*v1) * math.hypot(*v2)
    if lengths < 1:
        return None
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / lengths))
    return math.degrees(math.acos(cosine))


LEG_SIDES = ((LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
             (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))


def sitting_standing(person: PosePerson, min_conf: float,
                     sitting_max: float, standing_min: float) -> str:
    """Classify clear 2D leg geometry only; desks/occlusion become unknown."""
    return posture_with_basis(person, min_conf, sitting_max, standing_min)[0]


def posture_with_basis(person: PosePerson, min_conf: float, sitting_max: float,
                       standing_min: float, *, upper_legs: bool = False,
                       thigh_tolerance: float = 30.0,
                       thigh_torso_ratio: float = 0.5) -> tuple[str, str | None]:
    """Return (posture, basis); basis is "full_legs", "upper_legs", or None.

    The full shoulder-hip-knee-ankle chain is always preferred. Only when no
    side has it (ankles below the frame or hidden) and upper_legs is enabled
    does the shoulder-hip-knee fallback run; ambiguous full-leg geometry is
    never overridden by it.
    """
    knees, hips = [], []
    for shoulder, hip, knee, ankle in LEG_SIDES:
        knee_angle = _angle(person, hip, knee, ankle, min_conf)
        hip_angle = _angle(person, shoulder, hip, knee, min_conf)
        if knee_angle is not None and hip_angle is not None:
            knees.append(knee_angle)
            hips.append(hip_angle)
    if knees:
        knee, hip = median(knees), median(hips)
        if knee >= standing_min and hip >= standing_min - 10:
            return "standing", "full_legs"
        if knee <= sitting_max and hip <= standing_min - 5:
            return "sitting", "full_legs"
        return "unknown", None
    if upper_legs:
        label = _upper_leg_posture(person, min_conf, sitting_max, standing_min,
                                   thigh_tolerance, thigh_torso_ratio)
        if label != "unknown":
            return label, "upper_legs"
    return "unknown", None


def _upper_leg_posture(person: PosePerson, min_conf: float, sitting_max: float,
                       standing_min: float, thigh_tolerance: float,
                       thigh_torso_ratio: float) -> str:
    """Standing: straight hip and near-vertical thigh; sitting: bent hip and
    near-horizontal thigh. A thigh shorter than thigh_torso_ratio x torso is
    foreshortened (e.g. sitting facing the camera), so its direction is not
    trusted. Every usable side must agree; waist-up views stay unknown.
    """
    labels = set()
    for shoulder, hip, knee, _ in LEG_SIDES:
        hip_angle = _angle(person, shoulder, hip, knee, min_conf)
        points = [person.point(i, min_conf) for i in (shoulder, hip, knee)]
        if hip_angle is None or any(point is None for point in points):
            continue
        top, middle, bottom = points
        torso = math.hypot((top.x - middle.x) * person.width, (top.y - middle.y) * person.height)
        dx = (bottom.x - middle.x) * person.width
        dy = (bottom.y - middle.y) * person.height  # image y grows downward
        thigh = math.hypot(dx, dy)
        if torso < 1 or thigh < thigh_torso_ratio * torso:
            labels.add("unknown")
            continue
        from_vertical = math.degrees(math.atan2(abs(dx), dy))  # > 90 when knee is above hip
        if hip_angle >= standing_min - 10 and from_vertical <= thigh_tolerance:
            labels.add("standing")
        elif hip_angle <= sitting_max and 90 - thigh_tolerance <= from_vertical <= 90 + thigh_tolerance:
            labels.add("sitting")
        else:
            labels.add("unknown")
    return labels.pop() if len(labels) == 1 else "unknown"


SEATED_LEVEL_QUANTILE = 0.8  # of image y (grows downward): the low end of shoulder heights
SEAT_HISTORY_SECONDS = 300.0

ShoulderSample = tuple[float, float, float]  # midpoint x, y and shoulder width, in pixels


def shoulder_sample(person: PosePerson, min_conf: float) -> ShoulderSample | None:
    """Shoulder midpoint and width in pixels; None unless both shoulders are visible."""
    left, right = person.point(LEFT_SHOULDER, min_conf), person.point(RIGHT_SHOULDER, min_conf)
    if left is None or right is None:
        return None
    return ((left.x + right.x) / 2 * person.width, (left.y + right.y) / 2 * person.height,
            math.hypot((left.x - right.x) * person.width, (left.y - right.y) * person.height))


def shoulder_rise_posture(current: ShoulderSample | None, history: Any,
                          standing_min: float, sitting_max: float,
                          max_seat_shift: float, min_history: int) -> tuple[str, float | None]:
    """Posture without legs: shoulder rise above the track's own seated level.

    For seated people whose legs are hidden (students behind desks) on a
    fixed camera. history holds the track's earlier shoulder samples. The
    seated level is the low end of their heights (SEATED_LEVEL_QUANTILE), so
    it assumes the person sits most of the time and brief bending does not
    move it; the seat is their median x. Rise is in median shoulder widths
    (standing raises the shoulders about one width): standing >= standing_min,
    sitting <= sitting_max, including below the level (leaning on the desk).
    Returns (label, rise); too little history, narrow shoulders, a rise between
    the thresholds, or a person away from their seat give unknown.
    """
    if current is None or len(history) < min_history:
        return "unknown", None
    width = median(sample[2] for sample in history)
    if width < 8:
        return "unknown", None
    heights = sorted(sample[1] for sample in history)
    seated_y = heights[round(SEATED_LEVEL_QUANTILE * (len(heights) - 1))]
    if abs(current[0] - median(sample[0] for sample in history)) > max_seat_shift * width:
        return "unknown", None
    rise = round((seated_y - current[1]) / width, 3)
    return ("standing" if rise >= standing_min else "sitting" if rise <= sitting_max
            else "unknown"), rise


def person_in_shot(person: PosePerson, matrix: Affine) -> PosePerson:
    """Map a person's box and joints into camera-shot reference coordinates."""
    corners = [apply_affine(matrix, (x, y)) for x in (person.box[0], person.box[2])
               for y in (person.box[1], person.box[3])]
    xs, ys = [corner[0] for corner in corners], [corner[1] for corner in corners]
    keypoints = tuple(Keypoint(*apply_affine(matrix, (point.x, point.y)), point.confidence)
                      for point in person.keypoints)
    return replace(person, box=(min(xs), min(ys), max(xs), max(ys)), keypoints=keypoints)


def motion_level(current: PosePerson, previous: PosePerson | None, seconds: float,
                 min_conf: float, low_max: float, medium_max: float,
                 high_min: float,
                 camera_fixed: bool) -> tuple[str, float | None]:
    """Median joint displacement in body-heights/second, for a fixed camera.

    This is apparent image motion, not measured physical speed. For a moving
    camera, pass both poses in the same camera-shot coordinates (see
    person_in_shot) with camera_fixed=True, or camera_fixed=False for unknown.
    """
    if not camera_fixed or previous is None or seconds <= 0 or seconds > 2:
        return "unknown", None
    scale = ((current.box[3] - current.box[1]) +
             (previous.box[3] - previous.box[1])) * current.height / 2
    if scale < 12:
        return "unknown", None
    distances = []
    for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                  LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP):
        now, before = current.point(index, min_conf), previous.point(index, min_conf)
        if now is not None and before is not None:
            distances.append(math.hypot((now.x - before.x) * current.width,
                                        (now.y - before.y) * current.height))
    if len(distances) < 3:
        return "unknown", None
    speed = median(distances) / scale / seconds
    label = ("still" if speed < low_max else "low" if speed < medium_max
             else "medium" if speed < high_min else "high")
    return label, round(speed, 4)


def visibility(person: PosePerson, others: tuple[PosePerson, ...],
               min_conf: float, joints: frozenset[int] | None = None) -> dict[str, Any]:
    """Keypoint coverage and overlap proxies, never a definitive occlusion claim.

    A body part with joints outside the camera's pose_joints is None (not
    evaluated) rather than False, e.g. students' lower body behind desks.
    """
    def all_visible(indices: tuple[int, ...]) -> bool | None:
        if joints is not None and not joints.issuperset(indices):
            return None
        return all(person.point(i, min_conf) is not None for i in indices)

    x1, y1, x2, y2 = person.box
    area = max((x2 - x1) * (y2 - y1), 1e-9)
    overlap = 0.0
    for other in others:
        if other.track_id == person.track_id:
            continue
        a, b, c, d = other.box
        intersection = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
        overlap = max(overlap, intersection / area)
    upper = all_visible((LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                         LEFT_WRIST, RIGHT_WRIST))
    lower = all_visible((LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                         LEFT_ANKLE, RIGHT_ANKLE))
    truncated = min(x1, y1, 1 - x2, 1 - y2) < 0.01
    return {
        "face_landmarks_visible": all_visible((NOSE, LEFT_EYE, RIGHT_EYE)),
        "upper_body_visible": upper,
        "lower_body_visible": lower,
        "full_body_visible": None if None in (upper, lower) else upper and lower and not truncated,
        "box_truncated": truncated,
        "maximum_box_overlap_fraction": round(overlap, 3),
        "possible_occlusion": overlap >= 0.25 and False in (upper, lower),
        "limitation": "Low keypoint confidence may reflect occlusion, distance, blur, or pose error.",
    }


def _near_edge(box: tuple[float, float, float, float], margin: float) -> bool:
    """Box touching the left, top, or right edge, or its centre near any edge.

    Touching only the bottom edge does not count: people near the camera and
    students behind front desks are cut off there without leaving the view.
    """
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (min(box[0], box[1], 1 - box[2]) <= 0.01
            or min(center_x, center_y, 1 - center_x, 1 - center_y) <= 2 * margin)


@dataclass(frozen=True)
class TrackSnapshot:
    """Rule state of one tracked person at one sample, for review overlays.

    None means the indicator was not evaluated (switched off, no calibration,
    or the track is still provisional); "unknown" means it was evaluated but
    the evidence was insufficient. Mirrors the events emitted for the sample.
    """
    track_id: int
    confirmed: bool
    samples: int
    role: str
    posture: str | None = None
    posture_basis: str | None = None  # "full_legs", "upper_legs", or "shoulder_rise" for a known posture
    posture_rise: float | None = None  # shoulder_rise: shoulder widths above the seated level
    motion: str | None = None
    motion_score: float | None = None  # smoothed body-heights/second behind the label
    zone: str | None = None
    foot_point: tuple[float, float] | None = None
    foot_point_source: str | None = None
    foot_point_in_shot: tuple[float, float] | None = None  # None for an uncompensated moving camera
    hip_center_in_shot: tuple[float, float] | None = None  # path anchor; needs both hips
    possible_occlusion: bool | None = None
    box_truncated: bool | None = None


@dataclass(frozen=True)
class SampleSnapshot:
    """Frame-level state after one analyzed sample; None means not evaluated."""
    time: float
    tracks: tuple[TrackSnapshot, ...]
    visible_student_count: int | None = None
    student_count_coverage: str | None = None
    teacher_presence: str | None = None
    shot: int | None = None  # 0 for a fixed camera; None if a moving camera is uncompensated
    camera: CameraMotion | None = None


@dataclass
class _Track:
    first: float
    last: float
    first_box: tuple[float, float, float, float]
    last_box: tuple[float, float, float, float]
    samples: int = 0
    role: str = "unknown"
    previous: PosePerson | None = None  # in camera-shot coordinates
    previous_time: float | None = None
    previous_shot: int | None = None
    motion_speeds: list[float] | None = None
    shoulders: deque | None = None  # recent ShoulderSamples for shoulder_rise posture


@dataclass
class _Segment:
    start: float
    last: float
    label: str
    role: str
    confidence_sum: float
    confidence_count: int
    coverage: str
    zone: str | None


class _Segments:
    def __init__(self, video_id: str, sample_period: float, max_gap: float):
        self.video_id = video_id
        self.sample_period = sample_period
        self.max_gap = max_gap
        self.open: dict[tuple[str, int | None], _Segment] = {}

    def _close(self, key: tuple[str, int | None], max_end: float | None = None) -> Event | None:
        segment = self.open.pop(key, None)
        if segment is None:
            return None
        name, track_id = key
        end = segment.last + self.sample_period
        if max_end is not None:
            end = min(end, max_end)
        return make_event(self.video_id, name, segment.start, end, segment.label,
                          track_id=track_id, role=segment.role,
                          confidence=(segment.confidence_sum / segment.confidence_count
                                      if segment.confidence_count else None),
                          coverage=segment.coverage, zone=segment.zone)

    def update(self, name: str, track_id: int | None, time: float, label: str,
               role: str, confidence: float | None, coverage: str,
               zone: str | None = None) -> list[Event]:
        key = (name, track_id)
        old = self.open.get(key)
        events = []
        if old is not None and (old.label != label or old.role != role or old.zone != zone
                                or time - old.last > self.max_gap):
            closed = self._close(key)
            if closed is not None:
                events.append(closed)
            old = None
        if old is None:
            self.open[key] = _Segment(time, time, label, role, confidence or 0.0,
                                      int(confidence is not None), coverage, zone)
        else:
            if time - old.last > self.sample_period * 1.5:
                old.coverage = "limited"
            old.last = time
            old.confidence_sum += confidence or 0.0
            old.confidence_count += int(confidence is not None)
            if coverage != "adequate":
                old.coverage = "limited"
        return events

    def expire(self, time: float) -> list[Event]:
        events = []
        for key, segment in list(self.open.items()):
            if time - segment.last > self.max_gap:
                closed = self._close(key)
                if closed is not None:
                    events.append(closed)
        return events

    def close(self, key: tuple[str, int | None], max_end: float | None = None) -> Event | None:
        return self._close(key, max_end)

    def close_all(self, max_end: float | None = None) -> list[Event]:
        return [event for key in list(self.open)
                if (event := self._close(key, max_end)) is not None]


class FoundationAnalyzer:
    """Streaming List A analysis for one camera/video; instantiate per recording."""

    def __init__(self, video_id: str, expected_role: str, role_policy: str,
                 camera_fixed: bool, zones: dict[str, tuple[tuple[float, float], ...]],
                 sample_period: float, settings: dict[str, Any],
                 enabled: dict[str, bool] | None = None, start_time: float = 0.0,
                 posture_method: str = "legs", joints: frozenset[int] | None = None):
        """start_time is where analysis begins (a --start segment); people
        already visible then are not reported as entries.

        posture_method is "legs" or "shoulder_rise" (fixed camera, legs not
        used). joints is the camera's pose_joints; people passed to process()
        must already be limited to them (restrict_joints), and body parts
        outside them are reported as not evaluated.
        """
        self.video_id = video_id
        self.start_time = start_time
        self.expected_role = expected_role
        self.role_policy = role_policy
        self.camera_fixed = camera_fixed
        if posture_method not in {"legs", "shoulder_rise"}:
            raise ValueError(f"Unknown posture_method: {posture_method}")
        if posture_method == "shoulder_rise" and not camera_fixed:
            raise ValueError("shoulder_rise posture needs a fixed camera")
        self.posture_method = posture_method
        self.joints = joints
        self.zones = zones
        self.period = sample_period
        self.settings = settings
        self.enabled = enabled or {name: True for name in (
            "teacher_presence", "visible_student_count", "anonymous_tracking",
            "role_classification", "position", "classroom_zone", "entry_exit",
            "visibility_occlusion", "sitting_standing", "motion_level")}
        self.min_samples = int(settings["min_track_samples"])
        self.max_gap = float(settings["max_track_gap_seconds"])
        self.min_conf = float(settings["keypoint_confidence"])
        self.seat_history = max(1, round(SEAT_HISTORY_SECONDS / sample_period))
        self.rise_history = max(1, round(float(settings.get("shoulder_rise_history_seconds", 3.0))
                                         / sample_period))
        self.tracks: dict[int, _Track] = {}
        self.segments = _Segments(video_id, sample_period, self.max_gap)
        # State behind the most recent process() call; read-only for overlays.
        self.last_sample: SampleSnapshot | None = None

    def _role(self, visible_confirmed: int, visible_total: int) -> str:
        if self.role_policy == "all_tracks":
            return self.expected_role
        if self.role_policy == "single_visible" and visible_confirmed == 1 and visible_total == 1:
            return self.expected_role
        return "unknown"

    def _retire(self, track_id: int, recording_end: float, *, lost_before_end: bool) -> list[Event]:
        state = self.tracks.pop(track_id)
        events = []
        for name in ("classroom_zone", "sitting_standing", "motion_level"):
            event = self.segments.close((name, track_id), recording_end)
            if event is not None:
                events.append(event)
        if state.samples < self.min_samples:
            return events
        end = min(state.last + self.period, recording_end)
        if self.enabled["anonymous_tracking"]:
            events.append(make_event(self.video_id, "persistent_anonymous_identity",
                                     state.first, end,
                                     {"samples": state.samples, "temporary_id": True},
                                     track_id=track_id, role=state.role,
                                     coverage="limited" if state.role == "unknown" else "adequate"))
        margin = float(self.settings["edge_margin"])
        # An appearance in the middle of the image is NOT proof of entry/exit.
        if (self.enabled["entry_exit"] and state.first - self.start_time > self.period + self.max_gap
                and _near_edge(state.first_box, margin)):
            events.append(make_event(self.video_id, "entry_exit", state.first, state.first,
                                     "apparent_entry_at_frame_edge", track_id=track_id,
                                     role=state.role, coverage="limited"))
        if (self.enabled["entry_exit"] and lost_before_end
                and _near_edge(state.last_box, margin)):
            events.append(make_event(self.video_id, "entry_exit", state.last, state.last,
                                     "apparent_exit_at_frame_edge", track_id=track_id,
                                     role=state.role, coverage="limited"))
        return events

    def process(self, time: float, people: tuple[PosePerson, ...],
                camera: CameraMotion | None = None) -> list[Event]:
        """Analyze one sample; camera is the estimated motion of a moving camera.

        With camera=None, a fixed camera's image is its own shot (shot 0) and a
        moving camera leaves motion and in-shot positions unknown.
        """
        events = []
        shot = camera.shot if camera is not None else 0 if self.camera_fixed else None
        if camera is not None:
            events.append(make_event(
                self.video_id, "camera_motion", time, time,
                {"shot": camera.shot, "step_reliable": camera.step_reliable,
                 "step_shift_fraction_of_width": camera.step_shift,
                 "step_scale": camera.step_scale, "background_inliers": camera.inliers,
                 "to_shot_reference": [round(v, 6) for v in camera.to_shot_reference]},
                role="recording",
                coverage={True: "adequate", False: "limited", None: "unknown"}[camera.step_reliable]))
        for track_id, state in list(self.tracks.items()):
            if time - state.last > self.max_gap:
                events.extend(self._retire(track_id, recording_end=time, lost_before_end=True))
        events.extend(self.segments.expire(time))
        unique = {person.track_id: person for person in people}
        for track_id, person in unique.items():
            state = self.tracks.get(track_id)
            if state is None:
                state = _Track(time, time, person.box, person.box,
                               shoulders=deque(maxlen=self.seat_history))
                self.tracks[track_id] = state
            state.samples += 1
            state.last = time
            state.last_box = person.box
        confirmed = [person for person in unique.values()
                     if self.tracks[person.track_id].samples >= self.min_samples]
        role = self._role(len(confirmed), len(unique))
        teacher_seen = False
        student_count = 0
        snapshots: dict[int, TrackSnapshot] = {}
        for person in confirmed:
            track_id = person.track_id
            state = self.tracks[track_id]
            posture = basis = rise = motion = zone = smoothed = visible = None
            # Same person in the current shot's coordinates, so samples are
            # comparable while the camera pans or zooms; None if not measurable.
            scene = (person_in_shot(person, camera.to_shot_reference) if camera is not None
                     else person if self.camera_fixed else None)
            if self.enabled["role_classification"] and (state.role != role or state.samples == self.min_samples):
                events.append(make_event(self.video_id, "role_classification", time, time,
                                         {"label": role, "source": "camera_context",
                                          "verified_by_model": False}, track_id=track_id,
                                         role=role, confidence=None, coverage="limited"))
            state.role = role
            teacher_seen |= role == "teacher"
            student_count += int(role == "student")
            center = ((person.box[0] + person.box[2]) / 2,
                      (person.box[1] + person.box[3]) / 2)
            foot, foot_source = person.foot_point(self.min_conf)
            foot_in_shot = (apply_affine(camera.to_shot_reference, foot) if camera is not None
                            else foot if self.camera_fixed else None)
            # Hips are visible far more often than ankles, so they anchor paths.
            hips = [scene.point(i, self.min_conf) for i in (LEFT_HIP, RIGHT_HIP)] if scene else []
            hip_in_shot = (((hips[0].x + hips[1].x) / 2, (hips[0].y + hips[1].y) / 2)
                           if len(hips) == 2 and None not in hips else None)
            keypoints = [{"index": i, "x": round(p.x, 4), "y": round(p.y, 4),
                          "confidence": round(p.confidence, 3)}
                         for i, p in enumerate(person.keypoints)
                         if p.confidence >= self.min_conf]
            if self.enabled["position"]:
                events.append(make_event(self.video_id, "position", time, time,
                                         {"box_xyxy": [round(x, 4) for x in person.box],
                                          "box_center": [round(x, 4) for x in center],
                                          "foot_point": [round(x, 4) for x in foot],
                                          "foot_point_source": foot_source,
                                          # Path data: comparable across samples of one shot.
                                          "foot_point_in_shot": ([round(x, 4) for x in foot_in_shot]
                                                                 if foot_in_shot is not None else None),
                                          "hip_center_in_shot": ([round(x, 4) for x in hip_in_shot]
                                                                 if hip_in_shot is not None else None),
                                          "camera_shot": shot,
                                          "keypoints": keypoints},
                                         track_id=track_id, role=role,
                                         confidence=person.confidence,
                                         coverage="adequate" if keypoints else "limited"))
            if self.enabled["visibility_occlusion"]:
                visible = visibility(person, people, self.min_conf, self.joints)
                events.append(make_event(self.video_id, "visibility_occlusion", time, time,
                                         visible, track_id=track_id, role=role,
                                         confidence=None,
                                         coverage="adequate" if visible["upper_body_visible"] else "limited"))
            if self.enabled["classroom_zone"] and self.zones:
                # A box cut off by the image bottom does not show where the person stands.
                zone = (zone_for(foot, self.zones) if foot_source != "frame_bottom_cut_off"
                        else "unknown")
                events.extend(self.segments.update("classroom_zone", track_id, time, zone,
                                                   role, None,
                                                   "adequate" if foot_source == "ankles" else "limited",
                                                   zone=zone))
            if self.enabled["sitting_standing"]:
                if self.posture_method == "shoulder_rise":
                    shoulders = shoulder_sample(person, self.min_conf)
                    posture, rise = shoulder_rise_posture(
                        shoulders, state.shoulders,
                        float(self.settings.get("shoulder_rise_standing_min", 0.7)),
                        float(self.settings.get("shoulder_rise_sitting_max", 0.35)),
                        float(self.settings.get("shoulder_rise_max_seat_shift", 1.0)),
                        self.rise_history)
                    basis = "shoulder_rise" if posture != "unknown" else None
                    if shoulders is not None:
                        state.shoulders.append(shoulders)
                else:
                    posture, basis = posture_with_basis(
                        person, self.min_conf, float(self.settings["sitting_angle_degrees"]),
                        float(self.settings["standing_angle_degrees"]),
                        upper_legs=bool(self.settings.get("upper_leg_posture", False)),
                        thigh_tolerance=float(self.settings.get("thigh_axis_tolerance_degrees", 30)),
                        thigh_torso_ratio=float(self.settings.get("thigh_torso_min_ratio", 0.5)))
                # Only full legs are adequate; upper legs and shoulder rise are weaker bases.
                events.extend(self.segments.update("sitting_standing", track_id, time, posture,
                                                   role, None,
                                                   "adequate" if basis == "full_legs" else "limited"))
                if rise is not None:
                    # Unsmoothed sample behind the label, for threshold tuning.
                    events.append(make_event(self.video_id, "shoulder_rise", time, time,
                                             {"shoulder_widths_above_seated_level": rise},
                                             track_id=track_id, role=role, confidence=None,
                                             coverage="limited"))
            if self.enabled["motion_level"]:
                delta = time - state.previous_time if state.previous_time is not None else 0.0
                previous = state.previous if state.previous_shot == shot else None
                motion, speed = motion_level(scene or person, previous, delta, self.min_conf,
                                             float(self.settings["motion_low_max"]),
                                             float(self.settings["motion_medium_max"]),
                                             float(self.settings["motion_high_min"]),
                                             scene is not None)
                if speed is None:
                    state.motion_speeds = []
                else:
                    state.motion_speeds = (state.motion_speeds or []) + [speed]
                    state.motion_speeds = state.motion_speeds[-int(self.settings.get("motion_smoothing_samples", 3)):]
                    smoothed = median(state.motion_speeds)
                    motion = ("still" if smoothed < float(self.settings["motion_low_max"])
                              else "low" if smoothed < float(self.settings["motion_medium_max"])
                              else "medium" if smoothed < float(self.settings["motion_high_min"])
                              else "high")
                events.extend(self.segments.update("motion_level", track_id, time, motion,
                                                   role, None,
                                                   "adequate" if speed is not None else "limited"))
                if speed is not None:
                    events.append(make_event(self.video_id, "motion_speed", time, time,
                                             {"body_heights_per_second": speed,
                                              "assumes_fixed_camera": camera is None,
                                              "camera_motion_compensated": camera is not None},
                                             track_id=track_id, role=role, confidence=None))
            state.previous, state.previous_time, state.previous_shot = scene, time, shot
            snapshots[track_id] = TrackSnapshot(
                track_id, True, state.samples, role, posture=posture, posture_basis=basis,
                posture_rise=rise, motion=motion, motion_score=round(smoothed, 4) if smoothed is not None else None,
                zone=zone, foot_point=foot, foot_point_source=foot_source,
                foot_point_in_shot=foot_in_shot, hip_center_in_shot=hip_in_shot,
                possible_occlusion=visible["possible_occlusion"] if visible is not None else None,
                box_truncated=visible["box_truncated"] if visible is not None else None)
        provisional = any(self.tracks[person.track_id].samples < self.min_samples
                          for person in unique.values())
        count = count_coverage = presence = None
        if self.enabled["visible_student_count"] and self.expected_role == "student":
            count = student_count
            count_coverage = "limited" if role == "unknown" or provisional else "adequate"
            events.append(make_event(self.video_id, "visible_student_count", time, time,
                                     student_count, coverage=count_coverage))
        if self.enabled["teacher_presence"] and self.expected_role == "teacher":
            presence = ("visible" if teacher_seen else "unknown" if confirmed or provisional
                        else "not_detected")
            events.append(make_event(self.video_id, "teacher_presence_sample", time, time,
                                     presence, coverage="limited" if presence == "unknown" else "adequate"))
        self.last_sample = SampleSnapshot(
            time,
            tuple(snapshots.get(track_id) or TrackSnapshot(track_id, False,
                                                           self.tracks[track_id].samples,
                                                           self.tracks[track_id].role)
                  for track_id in unique),
            count, count_coverage, presence, shot, camera)
        if self.enabled["teacher_presence"] and teacher_seen and self.expected_role == "teacher":
            events.extend(self.segments.update("teacher_presence", None, time, "visible",
                                               "teacher", None, "limited"))
        elif confirmed and role == "unknown":
            closed = self.segments.close(("teacher_presence", None))
            if closed is not None:
                events.append(closed)
        return events

    def finish(self, recording_end: float) -> list[Event]:
        events = []
        for track_id in list(self.tracks):
            events.extend(self._retire(track_id, recording_end=recording_end,
                                       lost_before_end=False))
        events.extend(self.segments.close_all(recording_end))
        return events
