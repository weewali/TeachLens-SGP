"""Face landmarks for students: MediaPipe Face Landmarker on pose-based face crops.

The Face Landmarker gives 478 face points, 52 blendshape scores (ARKit-style
facial movements such as mouthSmileLeft, browDownLeft, eyeBlinkLeft, jawOpen),
and a head-pose matrix per face. Its own face detector is made for faces close
to a phone camera, and classroom faces in 4K frames are only 100-170 px wide,
so each face is cut out around the pose model's head keypoints, enlarged to
crop_size, and only that crop is passed in.

Used on student cameras only (user decision 2026-11-17). Points and crops
stay in memory: the JSONL stores blendshape scores and head angles, never
face points or images. No identity matching.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.inference.pose import LEFT_EAR, LEFT_EYE, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_EYE, \
    RIGHT_SHOULDER, PosePerson

NOSE_TIP = 1  # MediaPipe face-mesh index of the nose tip
DEFAULTS: dict[str, float] = {
    "crop_size": 256,             # pixels fed to the landmarker
    "min_detection_confidence": 0.3,
    "min_face_width_px": 60,      # narrower faces (in the source frame) are not used
    # Sanity check, measured on A2 (13 good meshes): nose offset 0.03-0.07,
    # mesh width 0.42-0.49 of the crop side.
    "max_nose_offset": 0.15,      # mesh nose tip vs pose nose, in crop sides
    "min_mesh_width": 0.30,       # mesh width at least this share of the crop side
}


@dataclass(frozen=True)
class FaceMesh:
    """One face: points in normalized frame coordinates, scores, head angles.

    yaw/pitch/roll are degrees from the head-pose matrix (relative to the
    camera): yaw > 0 turns the face toward the image right, pitch < 0 tilts it
    down. width_px is the mesh width in source-frame pixels.
    """
    track_id: int
    points: tuple[tuple[float, float], ...]
    scores: dict[str, float]
    yaw: float
    pitch: float
    roll: float
    width_px: float
    mouth_open: float = 0.0  # inner-lip gap / face height (forehead to chin)


def face_crop(person: PosePerson, min_conf: float) -> tuple[float, float, float] | None:
    """Square crop around the face in source pixels: (left, top, side), or None.

    Needs the nose and at least one eye, i.e. a face turned toward the camera.
    The side comes from the widest of ear distance, eye distance, and shoulder
    width (each scaled to roughly a face-and-hair width), plus a margin.
    """
    def px(index: int) -> tuple[float, float] | None:
        point = person.point(index, min_conf)
        return None if point is None else (point.x * person.width, point.y * person.height)

    nose = px(NOSE)
    eyes = [p for p in (px(LEFT_EYE), px(RIGHT_EYE)) if p is not None]
    if nose is None or not eyes:
        return None
    ears = [p for p in (px(LEFT_EAR), px(RIGHT_EAR)) if p is not None]
    shoulders = [p for p in (px(LEFT_SHOULDER), px(RIGHT_SHOULDER)) if p is not None]
    spans = []
    if len(ears) == 2:
        spans.append(math.dist(*ears) * 1.6)
    if len(eyes) == 2:
        spans.append(math.dist(*eyes) * 3.2)
    if len(shoulders) == 2:
        spans.append(math.dist(*shoulders) * 0.75)
    if not spans:
        return None
    head = [nose] + eyes + ears
    center_x = sum(p[0] for p in head) / len(head)
    center_y = sum(p[1] for p in head) / len(head)
    side = max(spans) * 1.25
    return center_x - side / 2, center_y - side * 0.55, side


def head_angles(matrix: Any) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 4x4 face-pose matrix."""
    import cv2
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    rotation = rotation / np.linalg.norm(rotation, axis=0)  # remove scale
    (pitch, yaw, roll), *_ = cv2.RQDecomp3x3(rotation)
    return float(yaw), float(pitch), float(roll)


# Face-mesh indices: inner lips (13 upper, 14 lower), forehead top 10, chin 152.
UPPER_LIP, LOWER_LIP, FOREHEAD, CHIN = 13, 14, 10, 152


def mouth_opening(points: Sequence[tuple[float, float]]) -> float:
    """Inner-lip gap as a share of face height; about 0 with the mouth closed."""
    face = math.dist(points[FOREHEAD], points[CHIN])
    return math.dist(points[UPPER_LIP], points[LOWER_LIP]) / face if face > 0 else 0.0


def plausible(points: Sequence[tuple[float, float]], crop: tuple[float, float, float],
              nose: tuple[float, float], max_nose_offset: float, min_width: float) -> bool:
    """Reject meshes that do not sit on the pose model's face.

    points are crop-relative [0, 1]; nose is the pose nose in source pixels.
    A three-quarter face resting on a hand once produced a mesh squeezed onto
    the ear (A2, 2026-11-17): far from the nose and much too narrow.
    """
    left, top, side = crop
    xs = [x for x, _ in points]
    tip = points[NOSE_TIP]
    offset = math.dist((left + tip[0] * side, top + tip[1] * side), nose) / side
    return offset <= max_nose_offset and max(xs) - min(xs) >= min_width


class FaceLandmarker:
    """Run MediaPipe Face Landmarker on face crops of the given people."""

    def __init__(self, model_path: Path, settings: dict[str, Any] | None = None,
                 min_conf: float = 0.35):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face landmark model is missing: {model_path}. Place face_landmarker.task there "
                "before running; no automatic download is allowed.")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError("Install mediapipe in the TeachLens environment with --no-deps "
                               "(see AGENTS.md section 5)") from exc
        self.settings = {**DEFAULTS, **(settings or {})}
        self.min_conf = min_conf
        self._mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=float(self.settings["min_detection_confidence"]),
            min_face_presence_confidence=float(self.settings["min_detection_confidence"]),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        # Why faces were not used, for the run log: no face toward the camera,
        # no mesh found, mesh rejected by the sanity check, face too small.
        self.counts = {"crops": 0, "meshes": 0, "no_face": 0, "no_mesh": 0, "rejected": 0, "too_small": 0}

    def close(self) -> None:
        self.landmarker.close()

    def estimate(self, frame: Any, people: Sequence[PosePerson]) -> dict[int, FaceMesh]:
        """Face meshes by track ID for people whose face is turned toward the camera.

        The crops used are kept in last_crops, so estimate_crops() can reuse
        them on the frames between samples.
        """
        height, width = frame.shape[:2]
        self.last_crops = {}
        meshes = {}
        for person in people:
            crop = face_crop(person, self.min_conf)
            if crop is None:
                self.counts["no_face"] += 1
                continue
            self.counts["crops"] += 1
            nose = person.point(NOSE, self.min_conf)
            box = (round(crop[0]), round(crop[1]), max(1, round(crop[2])))
            self.last_crops[person.track_id] = (box, (nose.x * width, nose.y * height))
            mesh = self._mesh(frame, person.track_id, *self.last_crops[person.track_id])
            if mesh is not None:
                meshes[person.track_id] = mesh
                self.counts["meshes"] += 1
        return meshes

    def estimate_crops(self, frame: Any) -> dict[int, FaceMesh]:
        """Meshes on the last sample's crops, for a frame between samples."""
        meshes = {}
        for track_id, (box, nose) in getattr(self, "last_crops", {}).items():
            mesh = self._mesh(frame, track_id, box, nose, count=False)
            if mesh is not None:
                meshes[track_id] = mesh
                self.counts["between_meshes"] = self.counts.get("between_meshes", 0) + 1
        return meshes

    def _mesh(self, frame: Any, track_id: int, box: tuple[int, int, int],
              nose: tuple[float, float], count: bool = True) -> FaceMesh | None:
        import cv2
        height, width = frame.shape[:2]
        size = int(self.settings["crop_size"])
        x0, y0, s = box

        def miss(reason: str) -> None:
            if count:
                self.counts[reason] += 1
        patch = np.zeros((s, s, 3), np.uint8)  # black where the crop leaves the frame
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x0 + s), min(height, y0 + s)
        if sx1 <= sx0 or sy1 <= sy0:
            return miss("no_mesh")
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        patch = cv2.resize(patch, (size, size),
                           interpolation=cv2.INTER_CUBIC if s < size else cv2.INTER_AREA)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        found = self.landmarker.detect(image)
        if not found.face_landmarks:
            return miss("no_mesh")
        points = [(p.x, p.y) for p in found.face_landmarks[0]]
        if not plausible(points, (x0, y0, s), nose, float(self.settings["max_nose_offset"]),
                         float(self.settings["min_mesh_width"])):
            return miss("rejected")
        xs = [x for x, _ in points]
        width_px = (max(xs) - min(xs)) * s
        if width_px < float(self.settings["min_face_width_px"]):
            return miss("too_small")
        yaw, pitch, roll = head_angles(found.facial_transformation_matrixes[0])
        return FaceMesh(
            track_id, tuple(((x0 + x * s) / width, (y0 + y * s) / height) for x, y in points),
            {b.category_name: round(float(b.score), 4) for b in found.face_blendshapes[0]},
            round(yaw, 1), round(pitch, 1), round(roll, 1), round(width_px, 1),
            round(mouth_opening(points), 4))
