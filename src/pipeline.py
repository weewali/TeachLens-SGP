"""Orchestrate one physical List A analysis run; no rules are defined here."""

from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from src.analysis.aggregates import summarize_list_a
from src.analysis.events import with_real_time, write_jsonl
from src.analysis.visual_indicators import FoundationAnalyzer, restrict_joints
from src.inference.pose import YoloPoseTracker, joint_indices
from src.preprocessing.camera_motion import CameraMotionEstimator
from src.preprocessing.recording_time import read_recording_time
from src.preprocessing.video import VideoInfo, frames
from src.reporting.annotated_video import (
    AnnotatedVideoWriter, LivePreview, ReviewOverlay, camera_notes,
)
from src.utils.config import load_indicator_flags, load_zones
from src.utils.paths import project_path


LOGGER = logging.getLogger(__name__)


class AnalysisStopped(RuntimeError):
    """The user stopped the run from the live preview; nothing was published."""


def recording_start(video: Path) -> tuple[datetime | None, str | None]:
    """Recording start time and the metadata tag it came from, or (None, None)."""
    try:
        recording = read_recording_time(video)
    except RuntimeError as exc:
        LOGGER.warning("Real times will be null: %s", exc)
        return None, None
    if recording.local_start is None:
        LOGGER.warning("%s has no recording-time tag; real times will be null", video.name)
    return recording.local_start, recording.start_source


def analyze_video(video: Path, video_id: str, camera_id: str,
                  config: dict[str, Any], *, weight: Path | None = None,
                  output_dir: Path | None = None,
                  device: str | None = None,
                  overwrite: bool = False,
                  save_video: bool | None = None,
                  live_preview: bool | None = None,
                  start: float = 0.0, end: float | None = None,
                  path: str | None = None) -> tuple[Path, Path, Path | None]:
    """Write JSONL evidence, a JSON summary, and optionally an annotated MP4.

    save_video, live_preview, and path (overlay trail: off/teacher/all)
    default to the config's review section. The MP4 path is None when no
    video is saved. start/end (seconds from the source start, end exclusive)
    limit analysis to one segment; event times stay source-relative so they
    can be matched with the original video.
    """
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not video_id or not all(c.isalnum() or c in "-_" for c in video_id):
        raise ValueError("video_id may contain only letters, digits, '-' and '_'")
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Expected 0 <= start < end for the analyzed range")
    if camera_id not in config["cameras"]:
        raise ValueError(f"Unknown camera_id {camera_id}; choose from {list(config['cameras'])}")
    camera = config["cameras"][camera_id]
    review = config.get("review", {})
    if save_video is None:
        save_video = bool(review.get("save_video", False))
    if live_preview is None:
        live_preview = bool(review.get("live_preview", False))
    path_mode = path or review.get("path") or "off"  # YAML may read a bare off as false
    model_path = weight or project_path(config["models"]["pose_estimator"])
    root = output_dir or project_path(config["paths"]["outputs"])
    events_path = root / "events" / f"{video_id}.list_a.jsonl"
    summary_path = root / "reports" / f"{video_id}.list_a.json"
    video_path = root / "debug" / f"{video_id}.list_a.mp4" if save_video else None
    if not overwrite:
        for path in (events_path, summary_path, video_path):
            if path is not None and path.exists():
                raise FileExistsError(f"Output already exists: {path}; choose a new video ID or --overwrite")
    zones = load_zones(camera.get("zones"), camera_id)
    joints = joint_indices(camera.get("pose_joints"))
    started, started_source = recording_start(video)
    enabled = load_indicator_flags(config["paths"]["indicators"])
    if enabled["classroom_zone"] and not zones:
        LOGGER.warning("camera=%s has no calibrated zones; classroom_zone will be unknown", camera_id)
    tracker = YoloPoseTracker(model_path, float(config["inference"]["confidence_threshold"]),
                              float(config["inference"]["iou_threshold"]),
                              config["models"].get("tracker", "bytetrack.yaml"), device)
    info: VideoInfo | None = None
    last_time = 0.0
    analyzer: FoundationAnalyzer | None = None
    annotated = (AnnotatedVideoWriter(video_path, str(review.get("video_codec", "avc1")),
                                      overwrite=overwrite)
                 if video_path is not None else None)
    preview = LivePreview() if live_preview else None
    overlay: ReviewOverlay | None = None
    segment = start > 0 or end is not None
    motion_settings = config.get("camera_motion", {})
    estimator = (CameraMotionEstimator(int(motion_settings.get("min_inliers", 40)),
                                       float(motion_settings.get("max_scale_step", 0.1)))
                 if not camera.get("fixed_camera", True) and camera.get("compensate_camera_motion", False)
                 else None)

    def analyzed_end() -> float:
        """End of the analyzed range: --end, else the recording's end."""
        assert info is not None
        source_end = info.duration if info.duration > 0 else last_time + info.sample_period
        return min(end, source_end) if end is not None else source_end

    def evidence():
        nonlocal info, last_time, analyzer, overlay
        for info, timestamp, frame in frames(video, float(config["video"]["sample_every_seconds"]),
                                             start, end):
            if analyzer is None:
                analyzer = FoundationAnalyzer(video_id, camera["expected_role"],
                                              camera["role_policy"],
                                              bool(camera.get("fixed_camera", True)), zones,
                                              info.sample_period,
                                              {**config["analysis"],
                                               "keypoint_confidence": config["inference"]["keypoint_confidence"]},
                                              enabled, start_time=start,
                                              posture_method=camera.get("posture_method", "legs"),
                                              joints=joints)
                LOGGER.info("video=%s camera=%s fps=%.2f sample_period=%.3fs range=%.3fs-%s",
                            video_id, camera_id, info.fps, info.sample_period, start,
                            f"{end:.3f}s" if end is not None else "end")
                if annotated is not None or preview is not None:
                    overlay = ReviewOverlay(
                        f"{video_id} | {camera_id}", float(config["inference"]["keypoint_confidence"]),
                        duration=info.duration or None, zones=zones,
                        notes=camera_notes(camera, enabled, bool(zones)),
                        show_rejected_keypoints=bool(review.get("show_rejected_keypoints", False)),
                        time_range=(start, end) if segment else None, path=path_mode,
                        path_seconds=(float(review["path_seconds"])
                                      if review.get("path_seconds") is not None else None),
                        path_max_gap_seconds=float(review.get("path_max_gap_seconds", 1.0)))
                if annotated is not None:
                    # One video frame per analyzed sample keeps source timing.
                    annotated.open(1 / info.sample_period, info.width, info.height)
            # Rules and overlays see only the joints this camera may use.
            people = tuple(restrict_joints(person, joints) for person in tracker.infer(frame))
            camera_motion = (estimator.update(frame, [person.box for person in people])
                             if estimator is not None else None)
            LOGGER.debug("video=%s t=%.3f tracked_people=%d", video_id, timestamp, len(people))
            for event in analyzer.process(timestamp, people, camera_motion):
                yield with_real_time(event, started)
            if overlay is not None:
                image = overlay.draw(frame, people, analyzer.last_sample)
                if annotated is not None:
                    annotated.write(image)
                if preview is not None and not preview.show(image):
                    raise AnalysisStopped("Stopped from the live preview; no outputs were written")
            last_time = timestamp
        if analyzer is None or info is None:
            raise ValueError(f"No decodable frames found: {video}"
                             + (f" in the requested range; is --start ({start:g}s) before the video ends?"
                                if segment else ""))
        for event in analyzer.finish(analyzed_end()):
            yield with_real_time(event, started)

    # The MP4 is published right after the JSONL; any failure or stop discards both.
    with ExitStack() as stack:
        for resource in (annotated, preview):
            if resource is not None:
                stack.enter_context(resource)
        write_jsonl(events_path, evidence(), overwrite=overwrite)
    assert info is not None
    duration = info.duration if info.duration > 0 else last_time + info.sample_period
    summary = summarize_list_a(events_path, camera_id, video_id, duration, info.sample_period,
                               start, analyzed_end(),
                               started.isoformat(timespec="milliseconds") if started else None,
                               started_source)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=summary_path.parent,
                                         prefix=f".{summary_path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, summary_path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in (events_path, summary_path, video_path)
                                        if path is not None))
    return events_path, summary_path, video_path
