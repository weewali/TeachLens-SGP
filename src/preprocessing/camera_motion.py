"""Estimate camera pans and zooms between sampled frames from the background.

An operated lecture camera that follows the instructor makes image-space
motion and paths describe the camera rather than the person. This module
matches background features (people masked out) between consecutive sampled
frames, fits a similarity transform (shift, zoom, small rotation), and chains
it into the coordinates of the current shot's first frame. A cut, a sudden
zoom, heavy blur, or too few background features starts a new shot instead
of guessing; drift accumulates slowly within long shots.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence


Affine = tuple[float, float, float, float, float, float]  # row-major 2x3, normalized [0, 1] coords
IDENTITY: Affine = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class CameraMotion:
    """Camera state at one sample.

    to_shot_reference maps this frame's normalized (x, y) into the first frame
    of the current shot, so positions from different samples in one shot are
    comparable. step_reliable is None at the first sample of a run, False when
    the change since the previous sample could not be measured (a new shot
    starts), and True otherwise.
    """
    shot: int
    to_shot_reference: Affine
    step_reliable: bool | None
    step_shift: float | None = None  # translation since the previous sample, fraction of frame width
    step_scale: float | None = None  # zoom factor since the previous sample
    inliers: int | None = None


def apply_affine(matrix: Affine, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + b * y + c, d * x + e * y + f


def compose(outer: Affine, inner: Affine) -> Affine:
    """Matrix for applying inner first, then outer."""
    a, b, c, d, e, f = outer
    g, h, i, j, k, l = inner
    return (a * g + b * j, a * h + b * k, a * i + b * l + c,
            d * g + e * j, d * h + e * k, d * i + e * l + f)


def invert(matrix: Affine) -> Affine:
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Camera transform is not invertible")
    return (e / det, -b / det, (b * f - c * e) / det,
            -d / det, a / det, (c * d - a * f) / det)


class CameraMotionEstimator:
    """Track one camera's motion across a recording; create one per run."""

    def __init__(self, min_inliers: int = 40, max_scale_step: float = 0.1,
                 working_width: int = 960, max_features: int = 1500):
        self.min_inliers = min_inliers
        self.max_scale_step = max_scale_step
        self.working_width = working_width
        self.max_features = max_features
        self.shot = 0
        self.to_reference = IDENTITY
        self._previous: tuple[Any, Any] | None = None
        self._orb: Any = None

    def update(self, frame: Any,
               people_boxes: Sequence[tuple[float, float, float, float]]) -> CameraMotion:
        """Measure the change since the previous call; boxes are normalized xyxy."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("opencv-python is required in the TeachLens environment") from exc
        if self._orb is None:
            self._orb = cv2.ORB_create(self.max_features)
        height, width = frame.shape[:2]
        size = (self.working_width, max(1, round(height * self.working_width / width)))
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in people_boxes:  # people move on their own; ignore them
            pad_x, pad_y = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            mask[max(0, int((y1 - pad_y) * size[1])):int(math.ceil((y2 + pad_y) * size[1])),
                 max(0, int((x1 - pad_x) * size[0])):int(math.ceil((x2 + pad_x) * size[0]))] = 0
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        previous, self._previous = self._previous, (keypoints, descriptors)
        if previous is None:
            return CameraMotion(self.shot, self.to_reference, None)
        step, inliers = self._estimate(cv2, np, previous, (keypoints, descriptors))
        scale = math.hypot(step[0], step[3]) if step is not None else None
        if step is None or inliers < self.min_inliers or abs(scale - 1) > self.max_scale_step:
            self.shot += 1
            self.to_reference = IDENTITY
            return CameraMotion(self.shot, IDENTITY, False, None, scale, inliers)
        a, b, c, d, e, f = step
        shift = math.hypot(c, f)
        if shift < 0.5 and abs(scale - 1) < 0.002:
            step = IDENTITY  # sub-pixel jitter of a still camera: avoid random-walk drift
        else:
            # Pixel transform of the working image -> normalized coordinates.
            w, h = size
            step = (a, b * h / w, c / w, d * w / h, e, f / h)
        self.to_reference = compose(self.to_reference, invert(step))
        return CameraMotion(self.shot, self.to_reference, True, round(shift / size[0], 5),
                            round(scale, 5), inliers)

    @staticmethod
    def _estimate(cv2: Any, np: Any, previous: tuple[Any, Any],
                  current: tuple[Any, Any]) -> tuple[Affine | None, int]:
        """Similarity transform previous -> current in working-image pixels."""
        (old_points, old_descriptors), (new_points, new_descriptors) = previous, current
        if old_descriptors is None or new_descriptors is None:
            return None, 0
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(old_descriptors,
                                                                         new_descriptors)
        if len(matches) < 4:
            return None, len(matches)
        source = np.float32([old_points[m.queryIdx].pt for m in matches])
        target = np.float32([new_points[m.trainIdx].pt for m in matches])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                          ransacReprojThreshold=2.0)
        if matrix is None:
            return None, 0
        return tuple(float(v) for v in matrix.reshape(-1)), int(inlier_mask.sum())
