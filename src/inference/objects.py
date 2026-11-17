"""Classroom object detection with a local COCO-trained YOLO26 detector (no poses).

The COCO detector (`models/detection/yolo26n.pt`) already knows the common
classroom items: cell phone, laptop, book, bottle, cup, keyboard, mouse,
remote, scissors, backpack, handbag. Items COCO lacks (pens, notebooks,
calculators, markers) need a fine-tuned model later. Detections are image
evidence only: a phone on a desk does not show that anyone is using it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class DetectedObject:
    """One detected item in one frame; box is normalized xyxy."""
    label: str
    confidence: float
    box: tuple[float, float, float, float]


def adapt_detections(result: Any, names: dict[int, str]) -> tuple[DetectedObject, ...]:
    """Convert one Ultralytics detection result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    coordinates = boxes.xyxyn.tolist()
    scores = boxes.conf.tolist()
    classes = boxes.cls.int().tolist()
    return tuple(DetectedObject(names[int(c)], round(float(s), 4),
                                tuple(round(float(v), 4) for v in box))
                 for box, s, c in zip(coordinates, scores, classes))


class ObjectDetector:
    """Own one detector per video; only the requested COCO classes are kept."""

    def __init__(self, weight_path: Path, labels: Sequence[str], confidence: float,
                 iou: float = 0.45, image_size: int | None = None, device: str | None = None):
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Object detector weights are missing: {weight_path}. Place yolo26n.pt there "
                "before running; no automatic download is allowed.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics in the TeachLens Conda environment") from exc
        self.model = YOLO(str(weight_path))
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError(f"{weight_path.name} is a {self.model.task} model, not a detector")
        names = dict(self.model.names)
        unknown = sorted(set(labels) - set(names.values()))
        if unknown:
            raise ValueError(f"The detector does not know {unknown}; it knows {sorted(names.values())}")
        self.names = names
        self.class_ids = sorted(i for i, name in names.items() if name in set(labels))
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

    def detect(self, frame: Any) -> tuple[DetectedObject, ...]:
        kwargs: dict[str, Any] = dict(conf=self.confidence, iou=self.iou,
                                      classes=self.class_ids, verbose=False)
        if self.image_size:
            kwargs["imgsz"] = self.image_size
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(frame, **kwargs)
        return adapt_detections(results[0], self.names) if results else ()
