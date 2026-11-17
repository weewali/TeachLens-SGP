"""Orchestrate one physical List A analysis run; no rules are defined here."""

from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from src.analysis.aggregates import summarize_list_a, summarize_objects
from src.analysis.events import make_event, with_real_time, write_jsonl
from src.analysis.visual_indicators import (
    DuplicateFilter, FoundationAnalyzer, inside_any_target, restrict_joints, starts_below,
)
from src.inference.objects import ObjectDetector
from src.inference.pose import YoloPoseTracker, joint_indices
from src.preprocessing.camera_motion import CameraMotionEstimator
from src.preprocessing.recording_time import read_recording_time
from src.preprocessing.video import VideoInfo, frames
from src.reporting.annotated_video import (
    AnnotatedVideoWriter, LivePreview, ObjectOverlay, ReviewOverlay, camera_notes,
)
from src.utils.config import load_indicator_flags, load_targets, load_zones, select_indicators
from src.utils.paths import file_in_use, project_path


LOGGER = logging.getLogger(__name__)


class AnalysisStopped(RuntimeError):
    """The user stopped the run from the live preview; nothing was published."""


def tracker_settings(name: str) -> str:
    """A repository tracker YAML as an absolute path; a bare built-in name as is.

    Ultralytics resolves relative paths from the working directory, so a
    repository file like configs/bytetrack_1fps.yaml must be made absolute.
    """
    path = project_path(name)
    return str(path) if path.is_file() else name


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
                  path: str | None = None,
                  indicators: Sequence[str] | None = None,
                  real_start: datetime | None = None) -> tuple[Path, Path, Path | None]:
    """Write JSONL evidence, a JSON summary, and optionally an annotated MP4.

    save_video, live_preview, and path (overlay trail: off/teacher/all)
    default to the config's review section. The MP4 path is None when no
    video is saved. start/end (seconds from the source start, end exclusive)
    limit analysis to one segment; event times stay source-relative so they
    can be matched with the original video. indicators (List A switch
    names) turns on only those indicators, so the JSONL and the review video
    show just them. real_start replaces the file's recording-start tag, for
    clips whose tag is known to be wrong (e.g. trimmed on an iPhone, which
    keeps the uncut start).
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
    else:
        # Check now rather than fail after the whole run when publishing.
        busy = [path for path in (events_path, summary_path, video_path)
                if path is not None and file_in_use(path)]
        if busy:
            raise PermissionError(f"Cannot replace {', '.join(str(path) for path in busy)}: it is open "
                                  "in another program (for example a video player); close it and run again")
    zones = load_zones(camera.get("zones"), camera_id)
    targets = load_targets(camera.get("zones"), camera_id)  # List B board/screen/desk
    ignore_inside = tuple(camera.get("ignore_people_inside_targets") or ())
    joints = joint_indices(camera.get("pose_joints"))
    ignore_below = camera.get("ignore_people_starting_below")
    analysis = config["analysis"]
    duplicates = (DuplicateFilter(float(config["inference"]["keypoint_confidence"]),
                                  float(analysis["duplicate_max_keypoint_distance"]),
                                  float(analysis.get("duplicate_min_box_inside", 0.9)))
                  if analysis.get("duplicate_max_keypoint_distance") is not None else None)
    started, started_source = ((real_start, "given") if real_start is not None
                               else recording_start(video))
    enabled = select_indicators(load_indicator_flags(config["paths"]["indicators"]), indicators)
    if enabled["classroom_zone"] and not zones:
        LOGGER.warning("camera=%s has no calibrated zones; classroom_zone will be unknown", camera_id)
    tracker = YoloPoseTracker(model_path, float(config["inference"]["confidence_threshold"]),
                              float(config["inference"]["iou_threshold"]),
                              tracker_settings(config["models"].get("tracker", "bytetrack.yaml")), device,
                              config["inference"].get("image_size"))
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
                                              joints=joints,
                                              keep_teacher_track=bool(camera.get("keep_teacher_track", False)),
                                              targets=targets, teacher_settings=config.get("teacher"))
                LOGGER.info("video=%s camera=%s fps=%.2f sample_period=%.3fs range=%.3fs-%s",
                            video_id, camera_id, info.fps, info.sample_period, start,
                            f"{end:.3f}s" if end is not None else "end")
                if annotated is not None or preview is not None:
                    overlay = ReviewOverlay(
                        f"{video_id} | {camera_id}", float(config["inference"]["keypoint_confidence"]),
                        duration=info.duration or None, zones=zones, targets=targets,
                        notes=camera_notes(camera, enabled, bool(zones)),
                        show_rejected_keypoints=bool(review.get("show_rejected_keypoints", False)),
                        time_range=(start, end) if segment else None, path=path_mode,
                        path_seconds=(float(review["path_seconds"])
                                      if review.get("path_seconds") is not None else None),
                        path_max_gap_seconds=float(review.get("path_max_gap_seconds", 1.0)))
                if annotated is not None:
                    # One video frame per analyzed sample keeps source timing.
                    annotated.open(1 / info.sample_period, info.width, info.height)
            # Rules and overlays see only the joints this camera may use, and
            # not people cut off at the bottom (e.g. student heads in front
            # of the instructor camera).
            people = tuple(restrict_joints(person, joints) for person in tracker.infer(frame)
                           if not starts_below(person, ignore_below)
                           and not inside_any_target(person, targets, ignore_inside))
            if duplicates is not None:  # one detection per body (desks split seated people)
                people = duplicates(people)
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
                               started_source,
                               frame_aspect=info.width / info.height if info.height else None)
    _publish_json(summary_path, summary)
    if duplicates is not None:
        LOGGER.info("video=%s removed %d duplicate detections of the same person",
                    video_id, duplicates.dropped)
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in (events_path, summary_path, video_path)
                                        if path is not None))
    return events_path, summary_path, video_path


def _publish_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to a same-directory temporary file, then publish it with os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def detect_objects(video: Path, video_id: str, camera_id: str, config: dict[str, Any], *,
                   weight: Path | None = None, output_dir: Path | None = None,
                   device: str | None = None, overwrite: bool = False,
                   save_video: bool | None = None, start: float = 0.0, end: float | None = None,
                   labels: Sequence[str] | None = None,
                   real_start: datetime | None = None) -> tuple[Path, Path, Path | None]:
    """Detect classroom objects (no poses) on the sampled frames of one video.

    Writes <id>.objects.jsonl (one `object` event per detection and one
    `object_count` event per sample, zeros included), <id>.objects.json (per
    class summary), and optionally <id>.objects.mp4 (boxes and counts).
    Sampling, start/end, and real times work as in analyze_video().
    """
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not video_id or not all(c.isalnum() or c in "-_" for c in video_id):
        raise ValueError("video_id may contain only letters, digits, '-' and '_'")
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Expected 0 <= start < end for the analyzed range")
    settings = config.get("objects", {})
    labels = list(dict.fromkeys(labels or settings.get("classes") or []))
    if not labels:
        raise ValueError("No object classes chosen (objects.classes in the config or --classes)")
    review = config.get("review", {})
    if save_video is None:
        save_video = bool(review.get("save_video", False))
    root = output_dir or project_path(config["paths"]["outputs"])
    events_path = root / "events" / f"{video_id}.objects.jsonl"
    summary_path = root / "reports" / f"{video_id}.objects.json"
    video_path = root / "debug" / f"{video_id}.objects.mp4" if save_video else None
    outputs = [path for path in (events_path, summary_path, video_path) if path is not None]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}; choose a new video ID or --overwrite")
    else:
        busy = [path for path in outputs if file_in_use(path)]
        if busy:
            raise PermissionError(f"Cannot replace {', '.join(str(path) for path in busy)}: it is open "
                                  "in another program (for example a video player); close it and run again")
    started, started_source = ((real_start, "given") if real_start is not None
                               else recording_start(video))
    detector = ObjectDetector(weight or project_path(settings.get("detector", "models/detection/yolo26n.pt")),
                              labels, float(settings.get("confidence_threshold", 0.25)),
                              float(settings.get("iou_threshold", 0.45)),
                              settings.get("image_size"), device)
    annotated = (AnnotatedVideoWriter(video_path, str(review.get("video_codec", "avc1")),
                                      overwrite=overwrite) if video_path is not None else None)
    state: dict[str, Any] = {"info": None, "last_time": 0.0, "overlay": None}

    def evidence():
        for info, timestamp, frame in frames(video, float(config["video"]["sample_every_seconds"]),
                                             start, end):
            if state["overlay"] is None:
                LOGGER.info("video=%s camera=%s objects=%s sample_period=%.3fs range=%.3fs-%s",
                            video_id, camera_id, ", ".join(labels), info.sample_period, start,
                            f"{end:.3f}s" if end is not None else "end")
                state["overlay"] = ObjectOverlay(
                    f"{video_id} | {camera_id} | objects", labels, duration=info.duration or None,
                    notes=("COCO detector, no poses: an object seen is not an object in use",))
                if annotated is not None:
                    annotated.open(1 / info.sample_period, info.width, info.height)
            state["info"] = info
            found = detector.detect(frame)
            counts = {label: 0 for label in labels}
            for item in found:
                counts[item.label] += 1
                yield with_real_time(make_event(
                    video_id, "object", timestamp, timestamp,
                    {"label": item.label, "box_xyxy": list(item.box)},
                    role="classroom", confidence=item.confidence), started)
            yield with_real_time(make_event(video_id, "object_count", timestamp, timestamp, counts,
                                            role="classroom"), started)
            if annotated is not None:
                annotated.write(state["overlay"].draw(frame, found, timestamp))
            state["last_time"] = timestamp
        if state["info"] is None:
            raise ValueError(f"No decodable frames found: {video}")

    with ExitStack() as stack:
        if annotated is not None:
            stack.enter_context(annotated)
        write_jsonl(events_path, evidence(), overwrite=overwrite)
    info: VideoInfo = state["info"]
    duration = info.duration if info.duration > 0 else state["last_time"] + info.sample_period
    analyzed_end = min(end, duration) if end is not None else duration
    _publish_json(summary_path, summarize_objects(
        events_path, camera_id, video_id, duration, info.sample_period, labels, start, analyzed_end,
        started.isoformat(timespec="milliseconds") if started else None, started_source))
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in outputs))
    return events_path, summary_path, video_path
