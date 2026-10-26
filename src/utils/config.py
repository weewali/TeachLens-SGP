"""Load and validate List A configuration and optional camera calibration."""

from copy import deepcopy
from pathlib import Path
from typing import Any

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
    if float(video.get("sample_every_seconds", 0)) <= 0:
        raise ValueError("video.sample_every_seconds must be positive")
    for key in ("confidence_threshold", "iou_threshold", "keypoint_confidence"):
        value = float(inference.get(key, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"inference.{key} must be in [0, 1]")
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
    if not (0 < float(analysis.get("shoulder_rise_sitting_max", 0.35))
            < float(analysis.get("shoulder_rise_standing_min", 0.7))):
        raise ValueError("Expected 0 < shoulder_rise_sitting_max < shoulder_rise_standing_min")
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


def load_indicator_flags(path: str | Path) -> dict[str, bool]:
    """Read List A switches; absent switches are disabled, not assumed ready."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for indicator configuration") from exc
    with project_path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    visual = document.get("visual", {})
    names = (
        "teacher_presence", "visible_student_count", "anonymous_tracking",
        "role_classification", "position", "classroom_zone", "entry_exit",
        "visibility_occlusion", "sitting_standing", "motion_level",
    )
    return {name: bool(visual.get(name, {}).get("enabled", False)) for name in names}
