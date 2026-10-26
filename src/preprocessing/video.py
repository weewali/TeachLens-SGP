"""Read sampled video frames while retaining source-video timestamps."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator


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


def frames(path: Path, sample_every_seconds: float, start_seconds: float = 0.0,
           end_seconds: float | None = None) -> Iterator[tuple[VideoInfo, float, object]]:
    """Yield (metadata, time in seconds, BGR frame); release capture on exit.

    Only frames with start_seconds <= time < end_seconds are read. Times stay
    relative to the start of the source video, and samples keep the same
    frame grid as a full run (index divisible by the stride), so a segment
    analyzes exactly the frames a full run would analyze in that range.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required in the TeachLens environment") from exc
    capture = cv2.VideoCapture(str(path))
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
        # First sampled frame at or after the requested start, on the full-run grid.
        index = math.ceil(math.ceil(start_seconds * fps - 1e-6) / stride) * stride
        # Frame-accurate for the project's H.264/AV1 files; if a backend
        # refuses to seek, decode forward instead of guessing timestamps.
        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            for _ in range(index):
                if not capture.grab():
                    return
        while end_seconds is None or index / fps < end_seconds:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                # Decoding timestamps may be missing/inaccurate; frame index is
                # deterministic for ordinary constant-frame-rate recordings.
                yield info, index / fps, frame
            index += 1
    finally:
        capture.release()
