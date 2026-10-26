"""Small List A summary computed from saved evidence, without rerunning YOLO."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any


def summarize_list_a(event_path: Path, camera_id: str, video_id: str,
                     video_duration: float, sample_period: float,
                     analyzed_start: float = 0.0,
                     analyzed_end: float | None = None) -> dict[str, Any]:
    """Report coverage explicitly; unknown/uncalibrated time is not a zero.

    analyzed_start/analyzed_end bound a --start/--end segment; every duration
    in the summary belongs to that range, not to the whole recording.
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
            elif name == "entry_exit":
                entries[value] += 1
            elif name == "camera_motion":
                camera_steps[{True: "measured", False: "unmeasured", None: "first"}[value["step_reliable"]]] += 1
                shots.add(value["shot"])
    return {
        "video_id": video_id,
        "camera_id": camera_id,
        "video_duration_seconds": round(video_duration, 3),
        "analyzed_start_seconds": round(analyzed_start, 3),
        "analyzed_end_seconds": round(analyzed_end, 3),
        "analyzed_duration_seconds": round(analyzed_seconds, 3),
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
        # Present only for compensated moving cameras; unmeasured steps start new shots.
        "camera_motion": ({"sample_steps": dict(camera_steps), "shots": len(shots)}
                          if camera_steps else None),
        "limitations": [
            "Roles are camera-context assumptions, not model-verified identities.",
            "Counts include only confirmed visible tracks, not official attendance.",
            "Track IDs can fragment or switch; unique track count is not a unique-person count.",
            "Posture and motion unknown time is excluded from known-state claims.",
            "Zone durations are per-track seconds and may exceed recording length.",
        ] + (["Only the analyzed range is summarized; tracks are cut at its boundaries."]
             if analyzed_start > 0 or analyzed_end < video_duration else []),
    }
