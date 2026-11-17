"""Small List A/B summary computed from saved evidence, without rerunning YOLO."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from src.analysis.facial_actions import SEGMENT_NAMES as FACE_SEGMENTS
from src.analysis.interaction_indicators import SEGMENT_NAMES as INTERACTION_SEGMENTS
from src.analysis.student_indicators import SEGMENT_NAMES as STUDENT_SEGMENTS
from src.analysis.teacher_indicators import SEGMENT_NAMES as TEACHER_SEGMENTS


def summarize_list_a(event_path: Path, camera_id: str, video_id: str,
                     video_duration: float, sample_period: float,
                     analyzed_start: float = 0.0,
                     analyzed_end: float | None = None,
                     recording_start: str | None = None,
                     recording_start_source: str | None = None,
                     frame_aspect: float | None = None) -> dict[str, Any]:
    """Report coverage explicitly; unknown/uncalibrated time is not a zero.

    analyzed_start/analyzed_end bound a --start/--end segment; every duration
    in the summary belongs to that range, not to the whole recording.
    recording_start (ISO 8601) and its metadata tag are what the events'
    real times were computed from; both are None without a tag.
    frame_aspect (width / height) turns the teacher's hip path into frame
    heights; the path is image-plane distance, not metres on the floor.
    """
    if analyzed_end is None:
        analyzed_end = video_duration
    analyzed_seconds = max(0.0, analyzed_end - analyzed_start)
    counts: list[int] = []
    count_valid = 0
    teacher_seconds = 0.0
    teacher_sample_values = Counter()
    role_tracks: dict[str, set[int]] = defaultdict(set)
    track_episodes = Counter()
    zone_seconds: dict[str, float] = defaultdict(float)
    posture_seconds: dict[str, float] = defaultdict(float)
    motion_seconds: dict[str, float] = defaultdict(float)
    entries = Counter()
    camera_steps = Counter()
    shots: set[int] = set()
    teacher_seconds_by = defaultdict(lambda: defaultdict(float))
    teacher_episodes = defaultdict(Counter)
    stationary = {"episodes": 0, "seconds": 0.0, "by_zone": defaultdict(float)}
    path_points: dict[int, list[tuple[float, int | None, tuple[float, float]]]] = defaultdict(list)
    face_seconds = defaultdict(lambda: defaultdict(float))
    face_episodes = defaultdict(Counter)
    face_samples = 0
    student_seconds = defaultdict(lambda: defaultdict(float))
    track_seconds = defaultdict(lambda: defaultdict(float))  # student track -> (indicator, label) -> s
    raises = Counter()
    raises_by_track = Counter()
    outcomes = Counter()
    answered_by_track = Counter()
    seat_map = None
    with event_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            name, value = event["indicator_name"], event["value"]
            if name == "visible_student_count":
                if event["coverage"] == "adequate":
                    counts.append(value)
                    count_valid += 1
            elif name == "teacher_presence":
                teacher_seconds += event["duration"]
            elif name == "teacher_presence_sample":
                teacher_sample_values[value] += 1
            elif name == "persistent_anonymous_identity":
                role_tracks[event["role"]].add(event["track_id"])
                track_episodes[event["role"]] += 1
            elif name == "classroom_zone":
                zone_seconds[value] += event["duration"]
            elif name == "sitting_standing":
                posture_seconds[value] += event["duration"]
            elif name == "motion_level":
                motion_seconds[value] += event["duration"]
                if event["role"] == "student":
                    track_seconds[event["track_id"]][("motion_level", value)] += event["duration"]
            elif name == "entry_exit":
                entries[value] += 1
            elif name in TEACHER_SEGMENTS and event["role"] == "teacher":
                teacher_seconds_by[name][value] += event["duration"]
                teacher_episodes[name][value] += 1
            elif name in FACE_SEGMENTS and event["role"] == "student":
                face_seconds[name][value] += event["duration"]
                face_episodes[name][value] += 1
                track_seconds[event["track_id"]][(name, value)] += event["duration"]
            elif name in STUDENT_SEGMENTS + INTERACTION_SEGMENTS and event["role"] == "student":
                student_seconds[name][value] += event["duration"]
                track_seconds[event["track_id"]][(name, value)] += event["duration"]
            elif name == "hand_raise_episode" and event["role"] == "student":
                raises[value["max_level"]] += 1
                raises_by_track[event["track_id"]] += 1
            elif name == "hand_raise_outcome":
                outcomes[value["outcome"]] += 1
                answered_by_track[event["track_id"]] += value["outcome"] == "answered_likely"
            elif name == "seat_map":
                seat_map = value
            elif name == "facial_scores" and event["role"] == "student":
                face_samples += 1
            elif name == "stationary_episode" and event["role"] == "teacher":
                stationary["episodes"] += 1
                stationary["seconds"] += event["duration"]
                stationary["by_zone"][event["zone"] or "uncalibrated"] += event["duration"]
            elif (name == "position" and event["role"] == "teacher"
                  and value.get("hip_center_in_shot") is not None):
                path_points[event["track_id"]].append(
                    (event["start_time"], value.get("camera_shot"), tuple(value["hip_center_in_shot"])))
            elif name == "camera_motion":
                camera_steps[{True: "measured", False: "unmeasured", None: "first"}[value["step_reliable"]]] += 1
                shots.add(value["shot"])
    # Teacher path: hip centre steps within one track and shot, skipping gaps.
    path_length, path_steps = 0.0, 0
    for points in path_points.values():
        points.sort()
        for (t0, shot0, a), (t1, shot1, b) in zip(points, points[1:]):
            if shot0 == shot1 and t1 - t0 <= 2.5 * sample_period:
                path_length += math.hypot((b[0] - a[0]) * (frame_aspect or 1.0), b[1] - a[1])
                path_steps += 1
    gesture_known = sum(v for k, v in teacher_seconds_by["gesture"].items() if k != "unknown")
    return {
        "video_id": video_id,
        "camera_id": camera_id,
        "video_duration_seconds": round(video_duration, 3),
        "analyzed_start_seconds": round(analyzed_start, 3),
        "analyzed_end_seconds": round(analyzed_end, 3),
        "analyzed_duration_seconds": round(analyzed_seconds, 3),
        "recording_start_real_time": recording_start,
        "recording_start_source": recording_start_source,
        "teacher_visible_seconds": round(teacher_seconds, 3) if teacher_sample_values else None,
        "teacher_visible_fraction_of_valid_samples": (
            round(teacher_sample_values["visible"] /
                  (teacher_sample_values["visible"] + teacher_sample_values["not_detected"]), 4)
            if teacher_sample_values["visible"] + teacher_sample_values["not_detected"] else None),
        "teacher_presence_sample_coverage": dict(teacher_sample_values),
        "visible_student_count": ({"min": min(counts), "max": max(counts),
                                   "median": median(counts)} if counts else None),
        "student_count_valid_sampled_seconds": round(min(count_valid * sample_period, analyzed_seconds), 3),
        "distinct_tracker_id_values_by_role": {role: len(ids) for role, ids in role_tracks.items()},
        "track_episodes_by_role": dict(track_episodes),
        "zone_track_seconds": {key: round(value, 3) for key, value in zone_seconds.items()},
        "posture_track_seconds": {key: round(value, 3) for key, value in posture_seconds.items()},
        "motion_track_seconds": {key: round(value, 3) for key, value in motion_seconds.items()},
        "apparent_frame_edge_transitions": dict(entries),
        # List B (teacher behaviour): seconds and episode counts per label.
        "teacher_behavior_seconds": {name: {k: round(v, 3) for k, v in values.items()}
                                     for name, values in teacher_seconds_by.items()},
        "teacher_behavior_episodes": {name: dict(values) for name, values in teacher_episodes.items()},
        "teacher_gestures_per_minute": (round(teacher_episodes["gesture"]["gesturing"] * 60 / gesture_known, 2)
                                        if gesture_known else None),
        "teacher_stationary": ({"episodes": stationary["episodes"],
                                "seconds": round(stationary["seconds"], 3),
                                "seconds_by_zone": {k: round(v, 3) for k, v in stationary["by_zone"].items()}}
                               if stationary["episodes"] else None),
        "teacher_path": ({"length_frame_heights": round(path_length, 3), "steps": path_steps,
                          "basis": "hip centre, image plane" + ("" if frame_aspect else ", aspect unknown")}
                         if path_steps else None),
        # Student facial actions (student camera): per-track seconds and episodes
        # per label, summed over students; unknown = no usable face or still learning.
        "student_face_seconds": {name: {k: round(v, 3) for k, v in values.items()}
                                 for name, values in face_seconds.items()},
        "student_face_episodes": {name: dict(values) for name, values in face_episodes.items()},
        "student_face_samples_with_mesh": face_samples if face_seconds else None,
        # List C (students): per-track seconds summed over students, per label.
        "student_behavior_seconds": {name: {k: round(v, 3) for k, v in values.items()}
                                     for name, values in student_seconds.items()},
        "student_hand_raises": ({"episodes_by_max_level": dict(raises), "outcomes": dict(outcomes)}
                                if raises or outcomes else None),
        "student_orientation_proxy": _orientation(student_seconds.get("orientation_proxy")),
        "seats": _seats(seat_map, track_seconds, raises_by_track, answered_by_track, frame_aspect),
        # Present only for compensated moving cameras; unmeasured steps start new shots.
        "camera_motion": ({"sample_steps": dict(camera_steps), "shots": len(shots)}
                          if camera_steps else None),
        "limitations": [
            "Roles are camera-context assumptions, not model-verified identities.",
            "Counts include only confirmed visible tracks, not official attendance.",
            "Track IDs can fragment or switch; unique track count is not a unique-person count.",
            "Posture and motion unknown time is excluded from known-state claims.",
            "Zone durations are per-track seconds and may exceed recording length.",
        ] + (["List C student labels are pose/face/object proxies (head down may be reading or resting; "
              "a phone may be course work); hand-raise outcomes come from talking or standing after a raise."]
             if student_seconds else [])
        + (["Facial actions are face-landmark score changes against each student's own usual "
              "face, not emotions or attention; faces not turned toward the camera are unknown."]
             if face_seconds else [])
        + (["Only the analyzed range is summarized; tracks are cut at its boundaries."]
             if analyzed_start > 0 or analyzed_end < video_duration else [])
        + (["Real times are the camera's whole-second recording-start tag plus video time; "
            "accurate to about 1 s."] if recording_start else
           ["The video has no recording-time tag, so real times are null."]),
    }


def _orientation(seconds: dict[str, float] | None) -> dict[str, Any] | None:
    """Share of known student time facing the current focus (a proxy, not attention)."""
    if not seconds:
        return None
    on, off, down = (seconds.get(k, 0.0) for k in ("on_target", "off_target", "looking_down"))
    return {"on_target_fraction_of_known": round(on / (on + off + down), 4) if on + off + down else None,
            "on_target_fraction_excluding_looking_down": round(on / (on + off), 4) if on + off else None}


# Per-seat activity: (indicator, labels) whose seconds are summed per seat.
SEAT_ACTIVITY = {
    "talking_seconds": ("talking", ("talking",)),
    "smiling_seconds": ("smile", ("smiling",)),
    "phone_seconds": ("device_use", ("phone", "phone_hidden_possible")),
    "laptop_seconds": ("device_use", ("laptop",)),
    "writing_seconds": ("note_taking", ("writing",)),
    "reading_seconds": ("reading", ("reading",)),
    "head_down_seconds": ("head_down", ("down",)),
    "sleep_or_dozing_seconds": ("alertness", ("possible_sleep", "dozing")),
    "on_target_seconds": ("orientation_proxy", ("on_target",)),
    "off_target_seconds": ("orientation_proxy", ("off_target",)),
    "out_of_seat_seconds": ("out_of_seat", ("out_of_seat",)),
    "moving_seconds": ("motion_level", ("medium", "high")),
}


def _seats(seat_map: dict[str, Any] | None, track_seconds: dict[int, dict[tuple[str, str], float]],
           raises: Counter, answered: Counter, frame_aspect: float | None) -> dict[str, Any] | None:
    """Seats (learnt from where students sat) and rows, with activity per seat.

    Rows: seats sorted from the camera (front, larger image y) to the back; a
    new row starts where the gap in y exceeds one shoulder width. Heat-map
    input for later reports; image-plane, not a floor plan.
    """
    if not seat_map or not seat_map.get("seats"):
        return None
    seats = sorted(seat_map["seats"], key=lambda seat: -seat["y"])
    gap = median(seat["shoulder_width"] for seat in seats) * (frame_aspect or 1.0)
    rows: list[list[dict[str, Any]]] = []
    for seat in seats:
        if rows and rows[-1][-1]["y"] - seat["y"] <= gap:
            rows[-1].append(seat)
        else:
            rows.append([seat])
    table = []
    for row_number, row in enumerate(rows, start=1):
        for place, seat in enumerate(sorted(row, key=lambda s: s["x"]), start=1):
            entry = {"seat_id": seat["seat_id"], "row": row_number, "place_in_row": place,
                     "x": seat["x"], "y": seat["y"], "tracks": seat["tracks"],
                     "hand_raises": sum(raises[t] for t in seat["tracks"]),
                     "answers_likely": sum(answered[t] for t in seat["tracks"]),
                     "occupied_seconds": round(sum(track_seconds[t][("seat_occupancy", seat["seat_id"])]
                                                   for t in seat["tracks"]), 3)}
            for key, (indicator, labels) in SEAT_ACTIVITY.items():
                entry[key] = round(sum(track_seconds[t][(indicator, label)]
                                       for t in seat["tracks"] for label in labels), 3)
            table.append(entry)
    row_totals = []
    for row_number in range(1, len(rows) + 1):
        members = [e for e in table if e["row"] == row_number]
        totals = {"row": row_number, "seats": len(members),
                  "hand_raises": sum(e["hand_raises"] for e in members)}
        for key in ("occupied_seconds",) + tuple(SEAT_ACTIVITY):
            totals[key] = round(sum(e[key] for e in members), 3)
        row_totals.append(totals)
    return {"seats": table, "rows": row_totals,
            "basis": "seats learnt from seated shoulder positions; row 1 = nearest the camera"}


def summarize_objects(event_path: Path, camera_id: str, video_id: str, video_duration: float,
                      sample_period: float, labels: list[str], analyzed_start: float = 0.0,
                      analyzed_end: float | None = None, recording_start: str | None = None,
                      recording_start_source: str | None = None) -> dict[str, Any]:
    """Per-class presence from saved object detections (scripts/detect_objects.py).

    Every analyzed sample has an object_count event, so a class that is never
    seen has an explicit zero, not a missing value.
    """
    if analyzed_end is None:
        analyzed_end = video_duration
    samples = 0
    seen = Counter()
    most = Counter()
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    detections = Counter()
    confidence = defaultdict(float)
    with event_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["indicator_name"] == "object_count":
                samples += 1
                for label, count in event["value"].items():
                    if count:
                        seen[label] += 1
                        most[label] = max(most[label], count)
                        first.setdefault(label, event["start_time"])
                        last[label] = event["start_time"]
            elif event["indicator_name"] == "object":
                detections[event["value"]["label"]] += 1
                confidence[event["value"]["label"]] += event["confidence"] or 0.0
    return {
        "video_id": video_id,
        "camera_id": camera_id,
        "video_duration_seconds": round(video_duration, 3),
        "analyzed_start_seconds": round(analyzed_start, 3),
        "analyzed_end_seconds": round(analyzed_end, 3),
        "recording_start_real_time": recording_start,
        "recording_start_source": recording_start_source,
        "sample_period_seconds": round(sample_period, 3),
        "analyzed_samples": samples,
        "objects": {label: {
            "samples_seen": seen[label],
            "fraction_of_samples": round(seen[label] / samples, 4) if samples else None,
            "max_count_in_one_sample": most[label],
            "first_seen_seconds": first.get(label),
            "last_seen_seconds": last.get(label),
            "mean_confidence": round(confidence[label] / detections[label], 3) if detections[label] else None,
        } for label in labels},
        "limitations": [
            "COCO detector on sampled frames: small, partly hidden, or unusual items are missed.",
            "A detected object is not use: a phone on a desk does not show anyone is using it.",
            "Counts are per frame; the same item is counted again in every sample (no tracking).",
        ],
    }
