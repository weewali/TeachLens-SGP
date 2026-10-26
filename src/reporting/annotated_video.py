"""List A review video: frame overlays, atomic MP4 output, and a live window.

Overlays are drawn from the same pose observations and rule state that produced
this run's JSONL evidence; nothing here runs a model or adds a rule. Only the
analyzed (sampled) frames are drawn, so every video frame is one the rules saw.
The MP4 contains the original classroom footage: keep it local (outputs/ is
ignored by Git) and do not share it without the recording's permission.
"""

from bisect import bisect_left
from collections.abc import Sequence
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from src.analysis.visual_indicators import SampleSnapshot, TrackSnapshot
from src.inference.pose import (
    JOINT_GROUPS, LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE,
    LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW,
    RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, PosePerson,
)
from src.preprocessing.camera_motion import IDENTITY, invert


LOGGER = logging.getLogger(__name__)

# COCO-17 limbs, drawn only when both joints pass keypoint_confidence.
SKELETON = (
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE), (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
)

# BGR colours by camera-assumed role. Provisional tracks are not yet counted
# by any rule; magenta marks confirmed people whose role stayed unknown.
ROLE_COLORS = {"teacher": (0, 140, 255), "student": (255, 160, 0), "unknown": (255, 0, 255)}
PROVISIONAL_COLOR = (170, 170, 170)
ZONE_COLOR = (80, 220, 80)
PATH_COLOR = (0, 255, 255)  # teacher path; "all" paths use role colours
PATH_MODES = ("off", "teacher", "all")


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required in the TeachLens environment") from exc
    return cv2


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{rest:04.1f}" if hours else f"{minutes:02d}:{rest:04.1f}"


def describe_track(person: PosePerson, track: TrackSnapshot | None) -> list[str]:
    """Label lines for one person; '?' marks evaluated-but-unknown states."""
    if track is None or not track.confirmed:
        return [f"#{person.track_id} provisional {person.confidence:.2f}"]
    lines = [f"#{person.track_id} {track.role} {person.confidence:.2f}"]
    states = []
    if track.posture is not None:
        posture = ("posture ?" if track.posture == "unknown" else
                   f"{track.posture} (partial)" if track.posture_basis == "upper_legs"
                   else track.posture)
        if track.posture_rise is not None:
            posture += f" rise {track.posture_rise:.2f}"
        states.append(posture)
    if track.motion is not None:
        motion = "motion ?" if track.motion == "unknown" else track.motion
        if track.motion_score is not None:
            motion += f" {track.motion_score:.2f} bh/s"
        states.append(motion)
    if states:
        lines.append(" | ".join(states))
    flags = []
    if track.zone is not None:
        flags.append(f"zone {track.zone}")
    if track.possible_occlusion:
        flags.append("occluded?")
    if track.box_truncated:
        flags.append("cut off")
    if flags:
        lines.append(" | ".join(flags))
    return lines


def camera_notes(camera: dict[str, Any], enabled: dict[str, bool], has_zones: bool) -> tuple[str, ...]:
    """Fixed reminders about why some states are unknown for this camera."""
    notes = [f"roles: camera context ({camera['role_policy']}), not verified"]
    unused = [group for group in JOINT_GROUPS if group not in camera.get("pose_joints", JOINT_GROUPS)]
    if unused:
        notes.append(f"joints not used: {', '.join(unused)}")
    if (enabled.get("motion_level") and not camera.get("fixed_camera", True)
            and not camera.get("compensate_camera_motion", False)):
        notes.append("motion: unknown, camera not fixed")
    if enabled.get("classroom_zone") and not has_zones:
        notes.append("zones: not calibrated")
    return tuple(notes)


Rect = tuple[int, int, int, int]  # pixel x1, y1, x2, y2


def _text_size(cv2: Any, lines: Sequence[str], scale: float) -> tuple[int, int]:
    thickness, pad = max(1, round(scale)), max(2, round(4 * scale))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, thickness)[0]
             for line in lines]
    return max(w for w, _ in sizes) + pad * 2, (max(h for _, h in sizes) + pad * 2) * len(lines)


def _clamp(image: Any, x: int, y: int, size: tuple[int, int]) -> Rect:
    """Keep a text block inside the frame."""
    height, width = image.shape[:2]
    x = max(0, min(x, width - size[0]))
    y = max(0, min(y, height - size[1]))
    return x, y, x + size[0], y + size[1]


def _overlaps(first: Rect, second: Rect) -> bool:
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _text_block(cv2: Any, image: Any, lines: Sequence[str], rect: Rect, scale: float,
                background: tuple[int, int, int] | None, text_color: tuple[int, int, int]) -> None:
    """Draw stacked lines on a filled (or translucent dark) background."""
    thickness, pad = max(1, round(scale)), max(2, round(4 * scale))
    x1, y1, x2, y2 = rect
    if background is None:  # darken in place so the scene stays faintly visible
        image[y1:y2, x1:x2] = cv2.convertScaleAbs(image[y1:y2, x1:x2], alpha=0.35)
    else:
        image[y1:y2, x1:x2] = background
    line_height = (y2 - y1) // len(lines)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (x1 + pad, y1 + line_height * (index + 1) - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, text_color, thickness, cv2.LINE_AA)


class ReviewOverlay:
    """Draw one analyzed sample: boxes, joints, assumed roles, rule states, HUD."""

    def __init__(self, title: str, keypoint_confidence: float, *,
                 duration: float | None = None,
                 zones: dict[str, tuple[tuple[float, float], ...]] | None = None,
                 notes: Sequence[str] = (), show_rejected_keypoints: bool = False,
                 time_range: tuple[float, float | None] | None = None,
                 path: str = "off", path_seconds: float | None = None,
                 path_max_gap_seconds: float = 1.0):
        if path not in PATH_MODES:
            raise ValueError(f"path must be one of {PATH_MODES}")
        self.title = title
        self.min_conf = keypoint_confidence
        self.duration = duration
        self.zones = zones or {}
        self.notes = tuple(notes)
        self.show_rejected = show_rejected_keypoints
        self.time_range = time_range  # (start, end or None) of a --start/--end segment
        # Path trails: hip-centre points in the current camera shot's
        # coordinates, keyed by "teacher" or track ID; cleared at a new shot.
        self.path = path
        self.path_seconds = path_seconds
        self.path_max_gap = path_max_gap_seconds
        self._paths: dict[Any, list[tuple[float, float, float, str]]] = {}
        self._path_shot: int | None = None

    def hud_lines(self, sample: SampleSnapshot | None) -> list[str]:
        lines = [self.title]
        if sample is None:
            return lines + list(self.notes)
        clock = _clock(sample.time)
        if self.duration:
            clock += f" / {_clock(self.duration)}"
        if self.time_range is not None:
            start, end = self.time_range
            clock += f" | range {_clock(start)}-{_clock(end) if end is not None else 'end'}"
        confirmed = sum(track.confirmed for track in sample.tracks)
        lines += [f"t {clock}",
                  f"tracked {len(sample.tracks)}: {confirmed} confirmed, "
                  f"{len(sample.tracks) - confirmed} provisional"]
        if sample.visible_student_count is not None:
            lines.append(f"visible students {sample.visible_student_count} "
                         f"({sample.student_count_coverage})")
        if sample.teacher_presence is not None:
            lines.append(f"teacher presence {sample.teacher_presence}")
        camera = sample.camera
        if camera is not None:
            if camera.step_reliable is False:
                lines.append(f"camera: cut or unmeasured -> shot {camera.shot}, motion unknown")
            elif camera.step_reliable:
                lines.append(f"camera: shot {camera.shot}, pan {100 * camera.step_shift:.1f}% "
                             f"zoom x{camera.step_scale:.3f} (compensated)")
            else:
                lines.append(f"camera: shot {camera.shot} start (compensated)")
        if self.path != "off":
            lines.append(f"path: {self.path}, hip centre"
                         + (f", last {self.path_seconds:g}s" if self.path_seconds else "")
                         + (", resets at camera cuts" if camera is not None else "")
                         if sample.shot is not None else
                         f"path: {self.path} unavailable (moving camera, not compensated)")
        return lines + list(self.notes)

    def _update_paths(self, sample: SampleSnapshot) -> None:
        if self.path == "off" or sample.shot is None:
            return
        if sample.shot != self._path_shot:  # another shot's coordinates are unrelated
            self._paths.clear()
            self._path_shot = sample.shot
        for track in sample.tracks:
            if (not track.confirmed or track.hip_center_in_shot is None
                    or (self.path == "teacher" and track.role != "teacher")):
                continue
            key = "teacher" if self.path == "teacher" else track.track_id
            self._paths.setdefault(key, []).append((sample.time, *track.hip_center_in_shot, track.role))
        if self.path_seconds:
            cutoff = sample.time - self.path_seconds
            for key, points in list(self._paths.items()):
                del points[:bisect_left(points, (cutoff,))]
                if not points:
                    del self._paths[key]

    def _draw_paths(self, cv2: Any, image: Any, sample: SampleSnapshot, scale: float) -> None:
        if not self._paths:
            return
        import numpy as np
        height, width = image.shape[:2]
        to_image = np.array(invert(sample.camera.to_shot_reference) if sample.camera is not None
                            else IDENTITY).reshape(2, 3)
        thickness = max(1, round(2 * scale))
        for key, points in self._paths.items():
            color = PATH_COLOR if key == "teacher" else ROLE_COLORS.get(points[-1][3], ROLE_COLORS["unknown"])
            data = np.array([point[:3] for point in points])
            pixels = data[:, 1:3] @ to_image[:, :2].T + to_image[:, 2]
            pixels = np.rint(pixels * (width, height)).astype(np.int32)
            breaks = np.flatnonzero(np.diff(data[:, 0]) > self.path_max_gap) + 1
            for run in np.split(pixels, breaks):
                if len(run) > 1:
                    cv2.polylines(image, [run], False, color, thickness, cv2.LINE_AA)
                else:
                    cv2.circle(image, tuple(int(v) for v in run[0]), thickness, color, -1, cv2.LINE_AA)
            if points[-1][0] == sample.time:  # current position
                cv2.circle(image, tuple(int(v) for v in pixels[-1]), max(3, round(5 * scale)),
                           color, -1, cv2.LINE_AA)

    def draw(self, frame: Any, people: Sequence[PosePerson],
             sample: SampleSnapshot | None) -> Any:
        """Return an annotated copy; the decoded frame itself is not changed."""
        cv2 = _cv2()
        image = frame.copy()
        height, width = image.shape[:2]
        scale = max(0.6, min(width, height) / 720)
        self._draw_zones(cv2, image, scale)
        if sample is not None:
            self._update_paths(sample)
            self._draw_paths(cv2, image, sample, scale)
        states = {track.track_id: track for track in sample.tracks} if sample else {}
        # Confirmed and larger (usually nearer) people first, so their labels
        # get the preferred spot and their skeletons are drawn on top.
        ordered = sorted(people, key=lambda p: (
            not (states.get(p.track_id) is not None and states[p.track_id].confirmed),
            -(p.box[2] - p.box[0]) * (p.box[3] - p.box[1])))
        for person in reversed(ordered):
            self._draw_body(cv2, image, person, states.get(person.track_id), scale)
        hud = self.hud_lines(sample)
        hud_rect = _clamp(image, 0, 0, _text_size(cv2, hud, scale))
        placed = [hud_rect]
        for person in ordered:
            placed.append(self._draw_label(cv2, image, person, states.get(person.track_id),
                                           0.8 * scale, placed))
        _text_block(cv2, image, hud, hud_rect, scale, None, (255, 255, 255))
        return image

    def _draw_zones(self, cv2: Any, image: Any, scale: float) -> None:
        if not self.zones:
            return
        import numpy as np
        height, width = image.shape[:2]
        for name, polygon in self.zones.items():
            points = np.array([(round(x * width), round(y * height)) for x, y in polygon], np.int32)
            cv2.polylines(image, [points], True, ZONE_COLOR, max(1, round(scale)), cv2.LINE_AA)
            cv2.putText(image, name, tuple(int(v) for v in points[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, ZONE_COLOR, max(1, round(scale)), cv2.LINE_AA)

    @staticmethod
    def _color(track: TrackSnapshot | None) -> tuple[int, int, int]:
        if track is None or not track.confirmed:
            return PROVISIONAL_COLOR
        return ROLE_COLORS.get(track.role, ROLE_COLORS["unknown"])

    @staticmethod
    def _pixel_box(image: Any, person: PosePerson) -> Rect:
        height, width = image.shape[:2]
        return (round(person.box[0] * width), round(person.box[1] * height),
                round(person.box[2] * width), round(person.box[3] * height))

    def _draw_body(self, cv2: Any, image: Any, person: PosePerson,
                   track: TrackSnapshot | None, scale: float) -> None:
        height, width = image.shape[:2]
        confirmed = track is not None and track.confirmed
        color = self._color(track)
        line = max(1, round(2 * scale))
        radius = max(2, round(3 * scale))
        x1, y1, x2, y2 = self._pixel_box(image, person)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line if confirmed else max(1, line // 2))
        joints = {}
        for index, point in enumerate(person.keypoints):
            xy = (round(point.x * width), round(point.y * height))
            if point.confidence >= self.min_conf:
                joints[index] = xy
            elif self.show_rejected and point.confidence > 0:
                cv2.circle(image, xy, radius, PROVISIONAL_COLOR, 1, cv2.LINE_AA)
        for first, second in SKELETON:
            if first in joints and second in joints:
                cv2.line(image, joints[first], joints[second], color, line, cv2.LINE_AA)
        for xy in joints.values():
            cv2.circle(image, xy, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, xy, radius, (0, 0, 0), 1, cv2.LINE_AA)
        if confirmed and track.zone is not None and track.foot_point is not None:
            foot = (round(track.foot_point[0] * width), round(track.foot_point[1] * height))
            marker = (cv2.MARKER_CROSS if track.foot_point_source == "ankles"
                      else cv2.MARKER_TILTED_CROSS)
            cv2.drawMarker(image, foot, ZONE_COLOR, marker, round(14 * scale), line)

    def _draw_label(self, cv2: Any, image: Any, person: PosePerson,
                    track: TrackSnapshot | None, scale: float, placed: list[Rect]) -> Rect:
        """Place the label above, inside, or below the box, avoiding earlier labels."""
        lines = describe_track(person, track)
        size = _text_size(cv2, lines, scale)
        x1, y1, _, y2 = self._pixel_box(image, person)
        candidates = [_clamp(image, x1, y, size)
                      for y in (y1 - size[1], y1, y2, y2 - size[1])]
        rect = next((c for c in candidates if not any(_overlaps(c, other) for other in placed)),
                    candidates[0])
        color = self._color(track)
        text = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        _text_block(cv2, image, lines, rect, scale, color, text)
        return rect


class AnnotatedVideoWriter:
    """Stream review frames to a hidden temporary MP4; publish only a full run.

    Use as a context manager and call open() once the frame size is known. Any
    exception, including Ctrl+C or a stop from the live preview, discards the
    temporary file rather than publishing a partial video.
    """

    def __init__(self, path: Path, codec: str = "avc1", *, overwrite: bool = False):
        self.path = path
        self.codec = codec
        self.overwrite = overwrite
        self.frames_written = 0
        self._writer: Any = None
        self._temporary: Path | None = None

    def __enter__(self) -> "AnnotatedVideoWriter":
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.path}; choose a new video ID or --overwrite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def open(self, fps: float, width: int, height: int) -> None:
        """Start writing at the analysis rate so playback keeps source timing."""
        cv2 = _cv2()
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                            suffix=f".tmp{self.path.suffix}")
        os.close(descriptor)
        self._temporary = Path(name)
        attempts = [(self.codec, cv2.CAP_ANY), ("mp4v", cv2.CAP_ANY)]
        if os.name == "nt" and self.codec.lower() in {"avc1", "h264"}:
            # Windows pip OpenCV encodes H.264 via Media Foundation; its FFmpeg
            # lacks OpenH264 and logs errors before falling back on its own.
            attempts.insert(0, (self.codec, cv2.CAP_MSMF))
        # mp4v is the portable fallback when a machine's OpenCV lacks H.264.
        for codec, backend in dict.fromkeys(attempts):
            writer = cv2.VideoWriter(str(self._temporary), backend, cv2.VideoWriter_fourcc(*codec),
                                     fps, (width, height))
            if writer.isOpened():
                if codec != self.codec:
                    LOGGER.warning("Video codec %s is unavailable; writing %s instead", self.codec, codec)
                    self.codec = codec
                LOGGER.info("Annotated video codec=%s backend=%s fps=%.3f",
                            codec, writer.getBackendName(), fps)
                self._writer = writer
                return
            writer.release()
        raise RuntimeError(f"OpenCV could not open an MP4 writer for {self.path}")

    def write(self, frame: Any) -> None:
        if self._writer is None:
            raise RuntimeError("AnnotatedVideoWriter.open() must be called before write()")
        self._writer.write(frame)
        self.frames_written += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._writer is not None:
            self._writer.release()
        if self._temporary is None:
            return
        try:
            if exc_type is None and self.frames_written:
                os.replace(self._temporary, self.path)
        finally:
            self._temporary.unlink(missing_ok=True)


class LivePreview:
    """Show annotated samples in an OpenCV window while the analysis runs.

    Keys: space pauses or resumes, n steps one sample while paused, q or Esc
    asks the caller to stop. Closing the window hides the preview and the
    analysis continues. Frames appear as fast as they are analyzed, so the
    pace depends on hardware rather than matching real time.
    """

    WINDOW = "TeachLens live review"
    MAX_SIZE = (1280, 720)

    def __init__(self) -> None:
        self._opened = False
        self._closed = False
        self._paused = False

    def __enter__(self) -> "LivePreview":
        return self

    def show(self, image: Any) -> bool:
        """Display one frame; return False when the user asks to stop."""
        if self._closed:
            return True
        cv2 = _cv2()
        if not self._opened:
            try:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            except cv2.error as exc:
                raise RuntimeError("Live preview needs a desktop session and opencv-python with "
                                   "GUI support; run again with --no-live") from exc
            height, width = image.shape[:2]
            fit = min(1.0, self.MAX_SIZE[0] / width, self.MAX_SIZE[1] / height)
            cv2.resizeWindow(self.WINDOW, round(width * fit), round(height * fit))
            self._opened = True
        cv2.imshow(self.WINDOW, self._with_status(cv2, image))
        while True:
            key = cv2.waitKey(50 if self._paused else 1) & 0xFF
            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                LOGGER.info("Live preview closed; the analysis continues without it")
                self._closed = True
                return True
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key == ord(" "):
                self._paused = not self._paused
                cv2.imshow(self.WINDOW, self._with_status(cv2, image))
            elif key in (ord("n"), ord("N")) and self._paused:
                return True
            if not self._paused:
                return True

    def _with_status(self, cv2: Any, image: Any) -> Any:
        status = ("PAUSED | space resume | n next sample | q stop (no outputs saved)"
                  if self._paused else "LIVE | space pause | q stop (no outputs saved)")
        shown = image.copy()
        scale = max(0.6, min(image.shape[:2]) / 720)
        size = _text_size(cv2, [status], scale)
        _text_block(cv2, shown, [status], _clamp(shown, 0, shown.shape[0] - size[1], size),
                    scale, None, (0, 255, 255) if self._paused else (255, 255, 255))
        return shown

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._opened and not self._closed:
            cv2 = _cv2()
            try:
                cv2.destroyWindow(self.WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass  # the user closed the window during the last frame
