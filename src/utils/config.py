"""Load and validate List A configuration and optional camera calibration."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from src.inference.pose import JOINT_GROUPS
from src.utils.paths import project_path


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(default: Path, local: Path | None = None) -> dict[str, Any]:
    """Apply local settings over defaults; CLI overrides are applied by the caller."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required in the TeachLens environment") from exc

    def read(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        if not isinstance(document, dict):
            raise ValueError(f"Expected YAML mapping: {path}")
        return document

    config = read(default)
    if local is not None:
        config = _merge(config, read(local))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    video = config.get("video", {})
    analysis = config.get("analysis", {})
    inference = config.get("inference", {})
    cameras = config.get("cameras", {})
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("At least one camera context is required")
    for name, camera in cameras.items():
        if camera.get("expected_role") not in {"teacher", "student", "unknown"}:
            raise ValueError(f"Invalid expected_role for camera {name}")
        if camera.get("role_policy") not in {"all_tracks", "single_visible", "unknown"}:
            raise ValueError(f"Invalid role_policy for camera {name}")
        joints = camera.get("pose_joints", list(JOINT_GROUPS))
        if (not isinstance(joints, list) or not set(joints) <= set(JOINT_GROUPS)
                or "arms" not in joints):
            raise ValueError(f"cameras.{name}.pose_joints must list groups from "
                             f"{list(JOINT_GROUPS)} and include arms (shoulders)")
        method = camera.get("posture_method", "legs")
        if method not in {"legs", "shoulder_rise"}:
            raise ValueError(f"cameras.{name}.posture_method must be legs or shoulder_rise")
        if method == "legs" and "knees" not in joints:
            raise ValueError(f"cameras.{name}: posture_method legs needs knees in pose_joints")
        if method == "shoulder_rise" and not camera.get("fixed_camera", True):
            raise ValueError(f"cameras.{name}: posture_method shoulder_rise needs fixed_camera: true")
        if not isinstance(camera.get("keep_teacher_track", False), bool):
            raise ValueError(f"cameras.{name}.keep_teacher_track must be true or false")
        inside = camera.get("ignore_people_inside_targets")
        if inside is not None and (not isinstance(inside, list)
                                   or not all(isinstance(name, str) for name in inside)):
            raise ValueError(f"cameras.{name}.ignore_people_inside_targets must be a list of target names")
        every = camera.get("find_screens_every_seconds")
        if every is not None and (isinstance(every, bool) or not isinstance(every, (int, float)) or every <= 0):
            raise ValueError(f"cameras.{name}.find_screens_every_seconds must be null or a positive number")
        below = camera.get("ignore_people_starting_below")
        if below is not None and (isinstance(below, bool) or not isinstance(below, (int, float))
                                  or not 0 < below <= 1):
            raise ValueError(f"cameras.{name}.ignore_people_starting_below must be null or in (0, 1]")
    if float(video.get("sample_every_seconds", 0)) <= 0:
        raise ValueError("video.sample_every_seconds must be positive")
    if not isinstance(video.get("hardware_decoding", True), bool):
        raise ValueError("video.hardware_decoding must be true or false")
    faces = config.get("faces") or {}
    for key in ("smile_min", "smile_rise", "brow_min", "brow_rise", "eyes_closed_min",
                "eyes_closed_rise", "yawn_min", "yawn_rise", "min_detection_confidence",
                "max_nose_offset", "min_mesh_width"):
        if key in faces and not 0 <= float(faces[key]) <= 1:
            raise ValueError(f"faces.{key} must be in [0, 1]")
    for key in ("baseline_seconds", "eyes_closed_min_seconds", "yawn_min_seconds",
                "head_turn_degrees", "head_pitch_degrees", "crop_size", "talk_min_range",
                "nod_min_degrees", "nod_window_seconds", "shake_min_degrees", "shake_window_seconds"):
        if key in faces and float(faces[key]) <= 0:
            raise ValueError(f"faces.{key} must be positive")
    if int(faces.get("baseline_min_samples", 1)) < 1:
        raise ValueError("faces.baseline_min_samples must be at least one")
    if not 0 <= int(faces.get("frames_per_second", 1)) <= 30:
        raise ValueError("faces.frames_per_second must be from 0 to 30")
    tracking = config.get("tracking") or {}
    if tracking.get("stitch") not in (None, True, False):
        raise ValueError("tracking.stitch must be true or false")
    for key in ("stitch_max_speed", "stitch_margin"):
        if key in tracking and float(tracking[key]) <= 0:
            raise ValueError(f"tracking.{key} must be positive")
    students = config.get("students") or {}
    for key in ("seat_min_samples", "out_of_seat_widths", "sleep_min_seconds", "outcome_window_seconds",
                "outcome_talk_seconds", "raise_low_min_seconds"):
        if key in students and float(students[key]) <= 0:
            raise ValueError(f"students.{key} must be positive")
    fusion = config.get("fusion") or {}
    for key in ("window_seconds", "min_active_windows", "permutations", "max_lag_windows", "min_events"):
        if key in fusion and float(fusion[key]) <= 0:
            raise ValueError(f"fusion.{key} must be positive")
    for key in ("min_known_fraction", "alpha"):
        if key in fusion and not 0 < float(fusion[key]) <= 1:
            raise ValueError(f"fusion.{key} must be in (0, 1]")
    for spec in fusion.get("hypotheses") or []:
        if not isinstance(spec, dict) or not spec.get("teacher") or not spec.get("student")                 or int(spec.get("expect", 0)) not in (-1, 0, 1):
            raise ValueError("fusion.hypotheses entries need teacher, student, and expect -1, 0, or 1")
    if int(fusion.get("comparison_lag_windows", 1)) < 0:
        raise ValueError("fusion.comparison_lag_windows must be 0 or more")
    lags = fusion.get("model_lags", [1])
    if not isinstance(lags, list) or not all(isinstance(k, int) and k >= 1 for k in lags):
        raise ValueError("fusion.model_lags must be a list of positive whole numbers of windows")
    objects = config.get("objects") or {}
    classes = objects.get("classes", [])
    if not isinstance(classes, list) or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("objects.classes must be a list of detector class names")
    for key in ("confidence_threshold", "iou_threshold"):
        if not 0 <= float(objects.get(key, 0.25)) <= 1:
            raise ValueError(f"objects.{key} must be in [0, 1]")
    size = objects.get("image_size")
    if size is not None and (not isinstance(size, int) or size % 32 or not 320 <= size <= 3840):
        raise ValueError("objects.image_size must be null or a multiple of 32 from 320 to 3840")
    for key in ("confidence_threshold", "iou_threshold", "keypoint_confidence"):
        value = float(inference.get(key, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"inference.{key} must be in [0, 1]")
    size = inference.get("image_size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int)
                             or not 320 <= size <= 3840 or size % 32):
        raise ValueError("inference.image_size must be null or a multiple of 32 from 320 to 3840")
    for key in ("min_track_samples", "max_track_gap_seconds", "edge_margin",
                "standing_angle_degrees", "sitting_angle_degrees",
                "motion_low_max", "motion_medium_max", "motion_high_min"):
        if key not in analysis:
            raise ValueError(f"Missing analysis.{key}")
    if int(analysis["min_track_samples"]) < 1:
        raise ValueError("min_track_samples must be at least one")
    if float(analysis["max_track_gap_seconds"]) <= 0:
        raise ValueError("max_track_gap_seconds must be positive")
    if float(video["sample_every_seconds"]) > float(analysis["max_track_gap_seconds"]):
        raise ValueError("sample_every_seconds must not exceed max_track_gap_seconds")
    if int(analysis.get("motion_smoothing_samples", 3)) < 1:
        raise ValueError("motion_smoothing_samples must be at least one")
    if not 0 < float(analysis["edge_margin"]) < 0.5:
        raise ValueError("edge_margin must be in (0, 0.5)")
    if not 0 < float(analysis["sitting_angle_degrees"]) < float(analysis["standing_angle_degrees"]) < 180:
        raise ValueError("Expected sitting_angle < standing_angle < 180")
    if not (0 < float(analysis["motion_low_max"]) <
            float(analysis["motion_medium_max"]) < float(analysis["motion_high_min"])):
        raise ValueError("Expected 0 < motion_low_max < motion_medium_max < motion_high_min")
    if not 0 < float(analysis.get("thigh_axis_tolerance_degrees", 30)) < 45:
        raise ValueError("thigh_axis_tolerance_degrees must be in (0, 45)")
    if float(analysis.get("thigh_torso_min_ratio", 0.5)) <= 0:
        raise ValueError("thigh_torso_min_ratio must be positive")
    distance = analysis.get("duplicate_max_keypoint_distance")
    if distance is not None and not 0 < float(distance) <= 1:
        raise ValueError("duplicate_max_keypoint_distance must be null or in (0, 1] shoulder widths")
    if not 0 < float(analysis.get("duplicate_min_box_inside", 0.9)) <= 1:
        raise ValueError("duplicate_min_box_inside must be in (0, 1]")
    if not (0 < float(analysis.get("shoulder_rise_sitting_max", 0.35))
            < float(analysis.get("shoulder_rise_standing_min", 0.7))):
        raise ValueError("Expected 0 < shoulder_rise_sitting_max < shoulder_rise_standing_min")
    if not (0 < float(analysis.get("desk_sitting_max_widths", 1.1))
            < float(analysis.get("desk_standing_min_widths", 1.4))):
        raise ValueError("Expected 0 < desk_sitting_max_widths < desk_standing_min_widths")
    for key, default in (("shoulder_rise_history_seconds", 3.0), ("shoulder_rise_max_seat_shift", 1.0)):
        if float(analysis.get(key, default)) <= 0:
            raise ValueError(f"analysis.{key} must be positive")
    camera_motion = config.get("camera_motion", {})
    if int(camera_motion.get("min_inliers", 40)) < 4:
        raise ValueError("camera_motion.min_inliers must be at least 4")
    if not 0 < float(camera_motion.get("max_scale_step", 0.1)) < 1:
        raise ValueError("camera_motion.max_scale_step must be in (0, 1)")
    review = config.get("review", {})
    codec = review.get("video_codec", "avc1")
    if not isinstance(codec, str) or len(codec) != 4:
        raise ValueError("review.video_codec must be a four-character code such as avc1 or mp4v")
    if (review.get("path") or "off") not in {"off", "teacher", "all"}:
        raise ValueError("review.path must be off, teacher, or all")
    if review.get("path_seconds") is not None and float(review["path_seconds"]) <= 0:
        raise ValueError("review.path_seconds must be positive or null")
    if float(review.get("path_max_gap_seconds", 1.0)) <= 0:
        raise ValueError("review.path_max_gap_seconds must be positive")


def load_zones(path: str | Path | None, camera_id: str) -> dict[str, tuple[tuple[float, float], ...]]:
    """Return validated normalized polygons. No file means no zone claims."""
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for camera zones") from exc
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if document.get("camera_id") != camera_id:
        raise ValueError(f"Zone file camera_id must be {camera_id}: {resolved}")
    if not isinstance(document.get("zones"), dict):
        raise ValueError("Zone file must contain a zones mapping")
    zones = {}
    for name, item in document["zones"].items():
        points = item.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Zone {name} needs at least three points")
        polygon = tuple((float(x), float(y)) for x, y in points)
        if any(not 0 <= coord <= 1 for point in polygon for coord in point):
            raise ValueError(f"Zone {name} coordinates must be in [0, 1]")
        zones[name] = polygon
    return zones


def load_targets(path: str | Path | None, camera_id: str) -> dict[str, tuple[tuple[float, float], ...]]:
    """Optional `targets:` polygons (board, screen, desk) from a camera's zone file.

    They are what List B rules test head direction, pointing rays, and wrists
    against, unlike `zones:`, which locate a person's feet. No file or no
    section means no target claims.
    """
    if path is None:
        return {}
    import yaml
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if document.get("camera_id") != camera_id:
        raise ValueError(f"Zone file camera_id must be {camera_id}: {resolved}")
    targets = {}
    for name, item in (document.get("targets") or {}).items():
        polygon = tuple((float(x), float(y)) for x, y in item.get("points", []))
        if len(polygon) < 3 or any(not 0 <= v <= 1 for point in polygon for v in point):
            raise ValueError(f"Target {name} needs at least three points in [0, 1]")
        targets[name] = polygon
    return targets


def load_indicator_flags(path: str | Path) -> dict[str, bool]:
    """Read List A (visual:) and List B (teacher:) switches; absent ones are off."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for indicator configuration") from exc
    with project_path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    visual, teacher = document.get("visual", {}), document.get("teacher", {})
    faces = document.get("faces", {})
    students = document.get("students", {})
    flags = {name: bool(visual.get(name, {}).get("enabled", False)) for name in LIST_A_INDICATORS}
    flags.update({name: bool(teacher.get(name, {}).get("enabled", False))
                  for name in LIST_B_INDICATORS})
    flags.update({name: bool(faces.get(name, {}).get("enabled", False))
                  for name in FACE_INDICATORS})
    flags.update({name: bool(students.get(name, {}).get("enabled", False))
                  for name in STUDENT_INDICATORS})
    return flags


LIST_A_INDICATORS = (
    "teacher_presence", "visible_student_count", "anonymous_tracking",
    "role_classification", "position", "classroom_zone", "entry_exit",
    "visibility_occlusion", "sitting_standing", "motion_level",
)


# List B: teacher visual behaviour (src/analysis/teacher_indicators.py).
LIST_B_INDICATORS = (
    "stationary_episodes", "facing_direction", "head_orientation", "board_interaction",
    "writing_on_board", "pointing", "gestures", "raised_hand", "torso_lean",
)
# Student facial actions from face landmarks, student cameras only
# (src/analysis/facial_actions.py; same names as its INDICATOR_NAMES).
FACE_INDICATORS = ("smile", "brows", "eyes_closed", "yawn", "head_direction", "talking", "head_gesture")
# List C: student behaviour, student cameras only (src/analysis/student_indicators.py
# and interaction_indicators.py; same names as their INDICATOR_NAMES).
STUDENT_INDICATORS = ("seat_occupancy", "hand_raise", "out_of_seat", "head_down", "alertness",
                      "device_use", "note_taking", "reading", "response_gesture",
                      "head_target", "orientation_proxy", "hand_raise_outcome")
INDICATORS = LIST_A_INDICATORS + LIST_B_INDICATORS + FACE_INDICATORS + STUDENT_INDICATORS


def select_indicators(enabled: dict[str, bool],
                      chosen: Sequence[str] | None) -> dict[str, bool]:
    """Keep only the chosen List A/B switches on (None keeps the file's switches).

    Used to test one indicator at a time: the JSONL and the review video then
    show only what was chosen. A chosen switch is turned on even if the file
    has it off.
    """
    if chosen is None:
        return enabled
    unknown = sorted(set(chosen) - set(INDICATORS))
    if unknown or not chosen:
        raise ValueError(f"Unknown or no indicators {unknown}; choose from {list(INDICATORS)}")
    return {name: name in chosen for name in enabled}
