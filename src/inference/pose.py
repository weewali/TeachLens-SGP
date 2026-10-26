"""YOLO26 pose + ByteTrack adapter; emits model-independent observations."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger(__name__)

# COCO-17 keypoint indices used by the pretrained YOLO26 pose checkpoint.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = range(5)
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW = range(5, 9)
LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP = range(9, 13)
LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = range(13, 17)

# Joint groups a camera may let rules use (cameras.<id>.pose_joints).
JOINT_GROUPS = {
    "head": (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR),
    "arms": (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST),
    "hips": (LEFT_HIP, RIGHT_HIP),
    "knees": (LEFT_KNEE, RIGHT_KNEE),
    "ankles": (LEFT_ANKLE, RIGHT_ANKLE),
}


def joint_indices(groups: Iterable[str] | None) -> frozenset[int]:
    """COCO-17 indices of the named joint groups; None means every joint."""
    return frozenset(index for name in (JOINT_GROUPS if groups is None else groups)
                     for index in JOINT_GROUPS[name])


@dataclass(frozen=True)
class Keypoint:
    x: float  # normalized by image width
    y: float  # normalized by image height
    confidence: float


@dataclass(frozen=True)
class PosePerson:
    track_id: int
    box: tuple[float, float, float, float]  # normalized xyxy
    confidence: float
    keypoints: tuple[Keypoint, ...]
    width: int
    height: int

    def point(self, index: int, minimum_confidence: float) -> Keypoint | None:
        if index >= len(self.keypoints):
            return None
        point = self.keypoints[index]
        return point if point.confidence >= minimum_confidence else None

    def foot_point(self, minimum_confidence: float) -> tuple[tuple[float, float], str]:
        """Mean visible ankle, else the box-bottom midpoint as a floor proxy.

        A box cut off by the bottom of the image (someone near the camera)
        does not reach the floor; its source is frame_bottom_cut_off.
        """
        ankles = [self.point(i, minimum_confidence) for i in (LEFT_ANKLE, RIGHT_ANKLE)]
        visible = [point for point in ankles if point is not None]
        if visible:
            return ((sum(p.x for p in visible) / len(visible),
                     sum(p.y for p in visible) / len(visible)), "ankles")
        x1, _, x2, y2 = self.box
        return (((x1 + x2) / 2, y2), "frame_bottom_cut_off" if y2 >= 0.99 else "box_bottom_proxy")


def _normalized_box(raw: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    return (max(0.0, min(1.0, raw[0] / width)),
            max(0.0, min(1.0, raw[1] / height)),
            max(0.0, min(1.0, raw[2] / width)),
            max(0.0, min(1.0, raw[3] / height)))


def adapt_result(result: Any, width: int, height: int) -> tuple[PosePerson, ...]:
    """Convert one Ultralytics result, skipping untracked detections."""
    if result.boxes is None or result.keypoints is None or result.boxes.id is None:
        return ()
    ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    points = result.keypoints.xy.cpu().tolist()
    point_scores = (result.keypoints.conf.cpu().tolist()
                    if result.keypoints.conf is not None else None)
    if not (len(ids) == len(boxes) == len(scores) == len(points)):
        raise ValueError("YOLO returned misaligned boxes, IDs, and keypoints")
    people = []
    for index, (track_id, box, score, raw_points) in enumerate(zip(ids, boxes, scores, points)):
        if len(raw_points) < 17:
            LOGGER.warning("Skipping track %s: expected 17 COCO keypoints", track_id)
            continue
        confidences = point_scores[index] if point_scores is not None else [0.0] * len(raw_points)
        keypoints = tuple(Keypoint(float(x) / width, float(y) / height, float(c))
                          for (x, y), c in zip(raw_points, confidences))
        people.append(PosePerson(int(track_id), _normalized_box(box, width, height),
                                 float(score), keypoints, width, height))
    return tuple(people)


class YoloPoseTracker:
    """Own one model per video so ByteTrack IDs never leak across recordings."""

    def __init__(self, weight_path: Path, confidence: float, iou: float,
                 tracker: str = "bytetrack.yaml", device: str | None = None):
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"YOLO26 pose weights are missing: {weight_path}. "
                "Place yolo26n-pose.pt there before analysis; no automatic download is allowed."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics in the TeachLens Conda environment") from exc
        self.model = YOLO(str(weight_path))
        names = self.model.names
        if isinstance(names, dict) and names.get(0) != "person":
            raise ValueError("The pose checkpoint must use class 0 = person")
        self.confidence = confidence
        self.iou = iou
        self.tracker = tracker
        self.device = device

    def infer(self, frame: Any) -> tuple[PosePerson, ...]:
        height, width = frame.shape[:2]
        kwargs: dict[str, Any] = dict(persist=True, tracker=self.tracker,
                                      conf=self.confidence, iou=self.iou,
                                      classes=[0], verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        return adapt_result(results[0], width, height) if results else ()
