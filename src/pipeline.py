"""The TeachLens pipeline: one camera run, or a whole lesson, from video to outputs.

analyze_video(): one camera recording -> pose, faces, hands, objects ->
rules -> events.jsonl, summary.json, review.mp4, run.json in its run folder.
detect_objects(): the isolated COCO object detector on one recording.
analyze_lesson(): a lesson's two recordings (planned and lined up by
src/preprocessing/alignment.py) -> both camera runs -> camera links ->
fusion and lag analysis -> facts.json for the LLM report stage ->
side_by_side.mp4 -> lesson.json, all in one lesson folder.
pool_lessons(): the lag hypotheses over several analyzed lessons.
Folders and file names: src/utils/output_layout.py. No rules are defined here.
"""

from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from src.analysis.aggregates import summarize_list_a, summarize_objects
from src.analysis.events import make_event, with_real_time, write_jsonl
from collections import defaultdict

from src.analysis.visual_indicators import (
    FACE_DEPENDENT_STUDENT_INDICATORS, DuplicateFilter, FoundationAnalyzer, inside_any_target,
    restrict_joints, starts_below,
)
from src.inference.face_mesh import FaceLandmarker
from src.inference.hands import HandGestures
from src.inference.objects import ObjectDetector
from src.inference.tracking import TrackStitcher
from src.inference.pose import YoloPoseTracker, joint_indices
from src.preprocessing.camera_motion import CameraMotionEstimator
from src.preprocessing.recording_time import read_recording_time
from src.preprocessing.video import VideoInfo, frames, frames_with_subsamples
from src.reporting.annotated_video import (
    AnnotatedVideoWriter, LivePreview, ObjectOverlay, ReviewOverlay, camera_notes,
)
from src.utils.config import (
    FACE_INDICATORS, STUDENT_INDICATORS, load_indicator_flags, load_targets, load_zones,
    select_indicators,
)
from src.utils.output_layout import (CAMERA_FOLDERS, find_lesson, find_run, lesson_paths, new_folder,
                                     read_info, run_folder, run_paths)
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
                  real_start: datetime | None = None,
                  folder: Path | None = None) -> tuple[Path, Path, Path | None]:
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

    Outputs go to folder (a lesson's student/ or instructor/ folder), else to
    outputs/runs/<YYYY-MM-DD_HHMM>_<video_id>/ (the existing one with
    overwrite): events.jsonl, summary.json, review.mp4, and run.json (what was
    analyzed and the review frames' timing). Returns (events, summary, mp4).
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
    folder = folder or run_folder(root, video_id, overwrite)
    files = run_paths(folder)
    events_path, summary_path, info_path = files["events"], files["summary"], files["info"]
    video_path = files["review"] if save_video else None
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
    # Face landmarks: student camera only (user decision 2026-11-17), and only
    # when a facial switch is on; points and crops never leave memory.
    face_settings = config.get("faces") or {}
    student_settings = config.get("students") or {}
    keypoint_conf = float(config["inference"]["keypoint_confidence"])
    is_student_camera = camera["expected_role"] == "student"
    on = {name for name, value in enabled.items() if value}
    landmarker = gestures_model = object_model = None
    if on & set(FACE_INDICATORS + STUDENT_INDICATORS) and not is_student_camera:
        LOGGER.warning("camera=%s: facial actions and List C run only on student cameras; skipped", camera_id)
    if is_student_camera and on & set(FACE_INDICATORS + FACE_DEPENDENT_STUDENT_INDICATORS):
        landmarker = FaceLandmarker(
            project_path(face_settings.get("model", "models/face/face_landmarker.task")),
            face_settings, keypoint_conf)
    # Talking and nods need several faces per second: extra frames between
    # samples, face model only (no pose), on the last sample's face crops.
    extra_faces = (max(0, int(face_settings.get("frames_per_second", 5)) - 1)
                   if landmarker is not None and on & {"talking", "head_gesture", "hand_raise_outcome",
                                                       "head_target", "orientation_proxy"} else 0)
    if is_student_camera and on & {"device_use", "note_taking", "reading"}:
        objects_settings = config.get("objects") or {}
        object_model = ObjectDetector(
            project_path(objects_settings.get("detector", "models/detection/yolo26n.pt")),
            list(student_settings.get("object_classes") or ["cell phone", "laptop", "book"]),
            float(student_settings.get("object_confidence", objects_settings.get("confidence_threshold", 0.25))),
            float(objects_settings.get("iou_threshold", 0.45)),
            student_settings.get("object_image_size", objects_settings.get("image_size")), device)
    if is_student_camera and on & {"response_gesture", "hand_raise", "hand_raise_outcome"}:
        gestures_model = HandGestures(
            project_path(student_settings.get("gesture_model", "models/hands/gesture_recognizer.task")),
            None, keypoint_conf)
    tracking = config.get("tracking") or {}
    stitcher = (TrackStitcher(float(config["analysis"]["max_track_gap_seconds"]),
                              float(tracking.get("stitch_max_speed", 1.5)),
                              float(tracking.get("stitch_margin", 0.5)))
                if tracking.get("stitch", True) else None)
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

    first_sample: float | None = None
    samples = 0

    def analyzed_end() -> float:
        """End of the analyzed range: --end, else the recording's end."""
        assert info is not None
        source_end = info.duration if info.duration > 0 else last_time + info.sample_period
        return min(end, source_end) if end is not None else source_end

    def evidence():
        nonlocal info, last_time, analyzer, overlay, first_sample, samples
        series: dict[int, list] = defaultdict(list)  # faces on frames between samples
        for info, timestamp, frame, is_sample in frames_with_subsamples(
                video, float(config["video"]["sample_every_seconds"]), start, end, extra_faces,
                bool(config["video"].get("hardware_decoding", True))):
            if not is_sample:
                for track_id, mesh in landmarker.estimate_crops(frame).items():
                    series[track_id].append((timestamp, mesh))
                continue
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
                                              targets=targets, teacher_settings=config.get("teacher"),
                                              face_settings=face_settings,
                                              student_settings=student_settings)
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
            if stitcher is not None:  # one ID for a person walking between samples
                people = stitcher(timestamp, people)
            camera_motion = (estimator.update(frame, [person.box for person in people])
                             if estimator is not None else None)
            faces = landmarker.estimate(frame, people) if landmarker is not None else None
            found = object_model.detect(frame) if object_model is not None else ()
            hands = gestures_model.estimate(frame, people) if gestures_model is not None else None
            LOGGER.debug("video=%s t=%.3f tracked_people=%d", video_id, timestamp, len(people))
            for event in analyzer.process(timestamp, people, camera_motion, faces,
                                          face_series=series, objects=found, gestures=hands):
                yield with_real_time(event, started)
            series = defaultdict(list)
            if overlay is not None:
                image = overlay.draw(frame, people, analyzer.last_sample, faces, objects=found)
                if annotated is not None:
                    annotated.write(image)
                if preview is not None and not preview.show(image):
                    raise AnalysisStopped("Stopped from the live preview; no outputs were written")
            last_time = timestamp
            first_sample = timestamp if first_sample is None else first_sample
            samples += 1
        if analyzer is None or info is None:
            raise ValueError(f"No decodable frames found: {video}"
                             + (f" in the requested range; is --start ({start:g}s) before the video ends?"
                                if segment else ""))
        for event in analyzer.finish(analyzed_end()):
            yield with_real_time(event, started)

    # The MP4 is published right after the JSONL; any failure or stop discards both.
    with ExitStack() as stack:
        for model in (landmarker, gestures_model):
            if model is not None:
                stack.callback(model.close)
        for resource in (annotated, preview):
            if resource is not None:
                stack.enter_context(resource)
        write_jsonl(events_path, evidence(), overwrite=overwrite)
    assert info is not None
    for name, model in (("faces", landmarker), ("hands", gestures_model)):
        if model is not None:
            LOGGER.info("video=%s %s: %s", video_id, name,
                        ", ".join(f"{key} {value}" for key, value in model.counts.items()))
    if stitcher is not None and stitcher.stitched:
        LOGGER.info("video=%s joined %d tracker IDs of people moving between samples",
                    video_id, stitcher.stitched)
    duration = info.duration if info.duration > 0 else last_time + info.sample_period
    summary = summarize_list_a(events_path, camera_id, video_id, duration, info.sample_period,
                               start, analyzed_end(),
                               started.isoformat(timespec="milliseconds") if started else None,
                               started_source,
                               frame_aspect=info.width / info.height if info.height else None)
    _publish_json(summary_path, summary)
    _publish_json(info_path, {
        "run_id": video_id, "analysis": "pose", "camera": camera_id, "video": str(video),
        "analyzed_start_seconds": start, "analyzed_end_seconds": round(analyzed_end(), 3),
        "recording_start_real_time": started.isoformat(timespec="milliseconds") if started else None,
        "recording_start_source": started_source,
        "indicators": sorted(name for name, value in enabled.items() if value),
        "created_real_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        # Review frame k shows source second first_sample_seconds + k * sample_period.
        "review_video": video_path.name if video_path is not None else None,
        "first_sample_seconds": first_sample, "sample_period": info.sample_period, "samples": samples,
        "files": {key: path.name for key, path in (("events", events_path), ("summary", summary_path),
                                                   ("review", video_path)) if path is not None}},
        sort_keys=False)
    if duplicates is not None:
        LOGGER.info("video=%s removed %d duplicate detections of the same person",
                    video_id, duplicates.dropped)
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in (events_path, summary_path, video_path)
                                        if path is not None))
    return events_path, summary_path, video_path


def _publish(path: Path, write) -> None:
    """Call write(temporary path) for a same-directory temporary file, then publish it with os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_json(path: Path, data: dict[str, Any], sort_keys: bool = True) -> None:
    """Write JSON atomically (temporary file, then os.replace)."""
    _publish(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n",
                                              encoding="utf-8"))


def detect_objects(video: Path, video_id: str, camera_id: str, config: dict[str, Any], *,
                   weight: Path | None = None, output_dir: Path | None = None,
                   device: str | None = None, overwrite: bool = False,
                   save_video: bool | None = None, start: float = 0.0, end: float | None = None,
                   labels: Sequence[str] | None = None,
                   real_start: datetime | None = None,
                   folder: Path | None = None) -> tuple[Path, Path, Path | None]:
    """Detect classroom objects (no poses) on the sampled frames of one video.

    Writes objects.jsonl (one `object` event per detection and one
    `object_count` event per sample, zeros included), objects_summary.json
    (per class), objects_run.json, and optionally objects_review.mp4 (boxes
    and counts) to folder, else to outputs/runs/<stamp>_<video_id>/.
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
    folder = folder or run_folder(root, video_id, overwrite)
    files = run_paths(folder)
    events_path, summary_path = files["objects_events"], files["objects_summary"]
    video_path = files["objects_review"] if save_video else None
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
                                             start, end,
                                             bool(config["video"].get("hardware_decoding", True))):
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
    _publish_json(files["objects_info"], {
        "run_id": video_id, "analysis": "objects", "camera": camera_id, "video": str(video),
        "analyzed_start_seconds": start, "analyzed_end_seconds": round(analyzed_end, 3), "classes": labels,
        "recording_start_real_time": started.isoformat(timespec="milliseconds") if started else None,
        "created_real_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": {key: path.name for key, path in (("events", events_path), ("summary", summary_path),
                                                   ("review", video_path)) if path is not None}},
        sort_keys=False)
    LOGGER.info("Wrote %s", ", ".join(str(path) for path in outputs))
    return events_path, summary_path, video_path


# ---------------------------------------------------------------------------
# Whole lessons: two camera runs -> links -> fusion -> facts -> side-by-side video.


def review_timing(folder: Path) -> "ReviewTiming | None":
    """When each frame of a run's review.mp4 was taken (run.json; older runs: from their events)."""
    from src.reporting.side_by_side import ReviewTiming
    files = run_paths(folder)
    if not files["review"].is_file():
        return None
    info = read_info(files["info"])
    start = info.get("recording_start_real_time") or read_info(files["summary"]).get("recording_start_real_time")
    if not start:
        return None
    first, period = info.get("first_sample_seconds"), info.get("sample_period")
    if first is None or not period:  # runs made before run.json: one frame per sample event
        times = sorted({round(float(json.loads(line)["start_time"]), 3)
                        for line in files["events"].open(encoding="utf-8")
                        if '"teacher_presence_sample"' in line or '"visible_student_count"' in line})
        if len(times) < 2:
            return None
        first, period = times[0], (times[-1] - times[0]) / (len(times) - 1)
    return ReviewTiming(files["review"], datetime.fromisoformat(start), float(first), float(period))


def link_runs(student_events: Path, instructor_events: Path, config: dict[str, Any],
              window_seconds: float = 20.0) -> dict[str, Any]:
    """Cross-camera timing links of two saved runs (board target from the instructor camera's zones)."""
    from src.analysis.cross_camera import link_cameras, read_events
    board = load_targets(config["cameras"]["instructor_view"].get("zones"), "instructor_view").get("board")
    return link_cameras(read_events(student_events), read_events(instructor_events),
                        window_seconds=window_seconds, board=board,
                        sample_period=float(config["video"]["sample_every_seconds"]))


def fuse_and_write(student_folder: Path, instructor_folder: Path, folder: Path, config: dict[str, Any], *,
                   title: str, header: dict[str, Any] | None = None, window_seconds: float | None = None,
                   plot: bool = True) -> tuple[Any, dict[str, Any]]:
    """Fuse two saved runs, run the lag analysis, and write fusion/ and facts.json in a lesson folder."""
    import pandas as pd
    from src.analysis.fusion import fuse_runs, load_events
    from src.reporting.facts import lesson_facts
    student, instructor = run_paths(student_folder), run_paths(instructor_folder)
    table, report = fuse_runs(load_events(student["events"]), load_events(instructor["events"]),
                              config.get("fusion") or {}, window_seconds)
    report = {**(header or {}), **report}
    paths = lesson_paths(folder)
    _publish_json(paths["fusion"], report, sort_keys=False)
    _publish(paths["fusion_windows"], lambda tmp: table.to_csv(tmp, index=False, float_format="%.4f"))
    rows = [{"type": "hypothesis", **{k: v for k, v in h.items() if k != "comparison"}}
            for h in report["hypotheses"]] + [{"type": "exploratory", **{k: v for k, v in link.items()
                                                                         if k != "comparison"}}
                                              for link in report["links"]]
    _publish(paths["fusion_links"], lambda tmp: pd.DataFrame(rows).to_csv(tmp, index=False))
    if plot:
        from src.reporting.charts import plot_fusion_timelines
        _publish(paths["fusion_plot"], lambda tmp: plot_fusion_timelines(table, title, tmp))
    facts = lesson_facts(table, report, lesson=title,
                         videos=[read_info(student["info"]).get("video", ""), read_info(instructor["info"]).get("video", "")],
                         student_summary=read_info(student["summary"]),
                         instructor_summary=read_info(instructor["summary"]))
    facts["lecture"]["videos"] = [Path(v).name for v in facts["lecture"]["videos"] if v]
    _publish_json(paths["facts"], facts, sort_keys=False)
    return table, report


def lesson_folder(root: Path, name: str, *, overwrite: bool, reuse: bool) -> Path:
    """The lesson's folder: the existing one (with overwrite or reuse) or a new lessons/<stamp>_<name>."""
    existing = find_lesson(root, name)
    if existing is not None and not (overwrite or reuse):
        raise FileExistsError(f"Lesson {name} already exists in {existing}; use --reuse, --overwrite, "
                              "or another name")
    return existing or new_folder(root, "lessons", name)


def analyze_lesson(runs: Sequence[Any], name: str, config: dict[str, Any], *,
                   output_dir: Path | None = None, weight: Path | None = None, device: str | None = None,
                   overwrite: bool = False, reuse: bool = False, save_video: bool | None = None,
                   objects: bool = False, plot: bool = True, side_by_side: bool = True,
                   redo: Sequence[str] = ()) -> dict[str, Any]:
    """Run every step for one lesson; returns what lesson.json records.

    runs: the two CameraRun plans (src/preprocessing/alignment.py). Everything
    goes to outputs/lessons/<stamp>_<name>/: student/ and instructor/ (camera
    runs), cross_camera.json, fusion/, facts.json, side_by_side.mp4,
    lesson.json. reuse keeps a camera run that already exists (in this
    lesson's folder or anywhere else, by run ID) and redoes only the later
    steps; overwrite replaces outputs. redo ("student_view", "instructor_view")
    re-analyzes those cameras even with reuse, e.g. after changing a teacher rule.
    """
    from src.preprocessing.alignment import CAMERAS, valid_name
    if not valid_name(name):
        raise ValueError("The lesson name may contain only letters, digits, '-' and '_'")
    cameras = {run.camera: run for run in runs}
    if set(cameras) != set(CAMERAS) or len(runs) != 2:
        raise ValueError("A lesson needs exactly one student_view and one instructor_view run")
    for run in runs:
        if not run.video.is_file() and not reuse:
            raise FileNotFoundError(f"Video not found: {run.video}")
    root = output_dir or project_path(config["paths"]["outputs"])
    folder = lesson_folder(root, name, overwrite=overwrite, reuse=reuse)
    paths = lesson_paths(folder)
    if not overwrite:  # fail before hours of analysis, not after
        for key in ("lesson", "fusion", "facts", "cross_camera"):
            if paths[key].exists() and not reuse:
                raise FileExistsError(f"{paths[key]} exists; use --overwrite or another name")
    busy = [path for path in paths.values() if path.exists() and file_in_use(path)]
    if busy:
        raise PermissionError(f"Close {', '.join(map(str, busy))} (open in another program) and run again")
    began = time.perf_counter()
    recorded, used = [], {}
    review = config.get("review", {})
    for number, camera in enumerate(CAMERAS, start=1):
        run = cameras[camera]
        here = folder / CAMERA_FOLDERS[camera]
        files = run_paths(here)
        found = (here if files["events"].is_file() and read_info(files["info"]).get("run_id") in (None, run.video_id)
                 else find_run(root, run.video_id)) if reuse and camera not in redo else None
        started = time.perf_counter()
        if found is not None and run_paths(found)["events"].is_file() and run_paths(found)["summary"].is_file():
            status, used[camera] = "reused", found
            LOGGER.info("[%d/2] %s: saved run %s reused from %s", number, camera, run.video_id, found)
        else:
            LOGGER.info("[%d/2] %s: analyzing %s, %g-%s s -> %s", number, camera, run.video.name, run.start,
                        "end" if run.end is None else f"{run.end:g}", here)
            analyze_video(run.video, run.video_id, camera, config, weight=weight, output_dir=root,
                          device=device, overwrite=overwrite or reuse, save_video=save_video,
                          live_preview=False, start=run.start, end=run.end, path=run.path,
                          real_start=run.real_start, folder=here)
            status, used[camera] = "analyzed", here
        files = run_paths(used[camera])
        entry = {"camera": camera, "video": str(run.video), "video_id": run.video_id, "folder": str(used[camera]),
                 "analyzed_start_seconds": run.start, "analyzed_end_seconds": run.end,
                 "real_start_given": run.real_start.isoformat(timespec="milliseconds") if run.real_start else None,
                 "status": status, "seconds": round(time.perf_counter() - started, 1),
                 "files": {key: str(files[key]) for key in ("events", "summary", "review", "info")
                           if files[key].is_file()}}
        if objects:
            if not (reuse and files["objects_events"].is_file()):
                LOGGER.info("[%d/2] %s: detecting objects", number, camera)
                detect_objects(run.video, f"{run.video_id}_objects", camera, config, output_dir=root,
                               device=device, overwrite=True, save_video=save_video, start=run.start,
                               end=run.end, real_start=run.real_start, folder=used[camera])
            entry["files"].update({key: str(files[key]) for key in ("objects_events", "objects_summary",
                                                                    "objects_review") if files[key].is_file()})
        recorded.append(entry)
    header = {"lesson": name, "student_run": cameras["student_view"].video_id,
              "instructor_run": cameras["instructor_view"].video_id}
    student, instructor = run_paths(used["student_view"]), run_paths(used["instructor_view"])
    LOGGER.info("Linking the cameras by real time")
    _publish_json(paths["cross_camera"], {**header, **link_runs(student["events"], instructor["events"], config)},
                  sort_keys=False)
    LOGGER.info("Fusing the runs and running the lag analysis")
    table, fusion = fuse_and_write(used["student_view"], used["instructor_view"], folder, config, title=name,
                                   header=header, plot=plot)
    side = None
    if side_by_side:
        timings = (review_timing(used["student_view"]), review_timing(used["instructor_view"]))
        if None in timings:
            LOGGER.warning("No side-by-side video: a camera run has no review video or no timing")
        else:
            from src.reporting.side_by_side import write_side_by_side
            LOGGER.info("Writing the side-by-side lesson video")
            side = write_side_by_side(timings[0], timings[1], table, datetime.fromisoformat(fusion["start_real_time"]),
                                      fusion["windows"] * fusion["window_seconds"], fusion["window_seconds"],
                                      paths["side_by_side"], name, codec=str(review.get("video_codec", "avc1")),
                                      overwrite=True)
    links = read_info(paths["cross_camera"]).get("links", [])
    lesson = {
        **header,
        "folder": str(folder),
        "finished_real_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - began, 1),
        "runs": recorded,
        "cross_camera_links": len(links),
        "fusion": {key: fusion[key] for key in ("window_seconds", "windows", "start_real_time", "end_real_time",
                                                "tested_teacher_features", "untested_teacher_features",
                                                "findings")},
        "hypotheses": [{k: h.get(k) for k in ("code", "teacher_feature", "student_feature", "expected_sign",
                                              "lag_seconds", "r", "q", "matches_expectation", "not_tested")}
                       for h in fusion["hypotheses"]],
        "outputs": {key: str(path) for key, path in paths.items()
                    if path.exists() or key == "lesson"},
        "next_step": f"LLM report: python \"Downloaded from prototype/llm_report.py\" --facts \"{paths['facts']}\"",
        "limitations": fusion["limitations"],
    }
    if side is None:
        lesson["outputs"].pop("side_by_side", None)
    _publish_json(paths["lesson"], lesson, sort_keys=False)
    return lesson


def pool_lessons(folders: Sequence[Path], name: str, config: dict[str, Any], *,
                 output_dir: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Test the fusion hypotheses over several analyzed lessons (their fusion/windows.csv), within lessons.

    Writes outputs/pooled/<stamp>_<name>/pooled.json and facts.json.
    """
    import pandas as pd
    from src.analysis.fusion import FEATURES
    from src.analysis.lag_analysis import pooled_analyze
    from src.reporting.facts import pooled_facts
    settings = dict(config.get("fusion") or {})
    tables = []
    for lesson in folders:
        windows = lesson_paths(lesson)["fusion_windows"]
        if not windows.is_file():
            raise FileNotFoundError(f"{windows} is missing; fuse that lesson first")
        tables.append((read_info(lesson_paths(lesson)["lesson"]).get("lesson") or lesson.name, pd.read_csv(windows)))
    if len(tables) < 2:
        raise ValueError("Pooling needs at least two lessons")
    features = {f.name: {"camera": f.camera, "indicator": f.indicator, "labels": list(f.labels), "kind": f.kind,
                         "desirable": f.desirable, "description": f.description} for f in FEATURES}
    pooled = pooled_analyze(tables, features, settings)
    root = output_dir or project_path(config["paths"]["outputs"])
    existing = find_lesson(root, name, kind="pooled")
    if existing is not None and not overwrite:
        raise FileExistsError(f"Pooled analysis {name} exists in {existing}; use --overwrite or another name")
    folder = existing or new_folder(root, "pooled", name)
    report = {"name": name, "lesson_folders": [str(f) for f in folders], **pooled, "features": features}
    _publish_json(folder / "pooled.json", report, sort_keys=False)
    _publish_json(folder / "facts.json",
                  pooled_facts(pooled, features, float(settings.get("window_seconds", 5.0)), name), sort_keys=False)
    report["folder"] = str(folder)
    return report
