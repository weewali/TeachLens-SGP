"""Read sampled video frames while retaining source-video timestamps."""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    sample_stride: int

    @property
    def sample_period(self) -> float:
        return self.sample_stride / self.fps


def open_capture(path: Path, hardware_decoding: bool = True):
    """OpenCV capture, with GPU (e.g. D3D11) decoding when available.

    Measured 2026-11-18 on the 4K HEVC scenario files: 10 s of video took
    37 s to decode in software and 1.7 s with hardware decoding; frames
    differ by about 1 grey level on average (colour conversion), and seeking
    stays frame-exact. Without a hardware decoder OpenCV decodes in software.
    """
    import cv2
    if hardware_decoding:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if capture.isOpened():
            LOGGER.info("%s: hardware decoding %s", path.name,
                        "on" if capture.get(cv2.CAP_PROP_HW_ACCELERATION) > 0 else "unavailable, software")
            return capture
        capture.release()
    return cv2.VideoCapture(str(path))


def frames(path: Path, sample_every_seconds: float, start_seconds: float = 0.0,
           end_seconds: float | None = None,
           hardware_decoding: bool = True) -> Iterator[tuple[VideoInfo, float, object]]:
    """Yield (metadata, time in seconds, BGR frame); release capture on exit.

    Only frames with start_seconds <= time < end_seconds are read. Times stay
    relative to the start of the source video, and samples keep the same
    frame grid as a full run (index divisible by the stride), so a segment
    analyzes exactly the frames a full run would analyze in that range.
    """
    for info, time, frame, is_sample in frames_with_subsamples(
            path, sample_every_seconds, start_seconds, end_seconds, 0, hardware_decoding):
        if is_sample:
            yield info, time, frame


def frames_with_subsamples(path: Path, sample_every_seconds: float, start_seconds: float = 0.0,
                           end_seconds: float | None = None, extra_per_sample: int = 0,
                           hardware_decoding: bool = True,
                           ) -> Iterator[tuple[VideoInfo, float, object, bool]]:
    """Like frames(), plus extra_per_sample evenly spaced frames between samples.

    Yields (metadata, time, frame, is_sample). Every frame is decoded anyway,
    so the extra frames cost no decoding; they are for cheap per-face work
    (talking, nods) that one frame per second cannot see. The extra frames
    after a sample come before the next sample. Frames that are not yielded
    are only grabbed (decoded, not converted to BGR).
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required in the TeachLens environment") from exc
    capture = open_capture(path, hardware_decoding)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid video metadata (fps or dimensions): {path}")
        stride = max(1, round(sample_every_seconds * fps))
        info = VideoInfo(fps, width, height, count, count / fps if count > 0 else 0.0, stride)
        # Offsets of the extra frames after each sample (frame indexes).
        extras = {round(stride * k / (extra_per_sample + 1)) for k in range(1, extra_per_sample + 1)}
        extras.discard(0)
        extras.discard(stride)
        # First sampled frame at or after the requested start, on the full-run grid.
        index = math.ceil(math.ceil(start_seconds * fps - 1e-6) / stride) * stride
        # Frame-accurate for the project's H.264/AV1 files; if a backend
        # refuses to seek, decode forward instead of guessing timestamps.
        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            for _ in range(index):
                if not capture.grab():
                    return
        while end_seconds is None or index / fps < end_seconds:
            if not capture.grab():
                break
            sample, extra = index % stride == 0, index % stride in extras
            if sample or extra:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                # Decoding timestamps may be missing/inaccurate; frame index is
                # deterministic for ordinary constant-frame-rate recordings.
                yield info, index / fps, frame, sample
            index += 1
    finally:
        capture.release()
