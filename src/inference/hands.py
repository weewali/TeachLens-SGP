"""Student hand gestures: MediaPipe Gesture Recognizer on wrist crops.

The pose model has no finger points, so thumbs up, an open palm, or a
victory sign cannot be told apart from a fist with it. The Gesture
Recognizer (models/hands/gesture_recognizer.task, Google, Apache 2.0,
downloaded 2026-11-17 at the user's request) finds one hand in a crop and
names one of seven gestures. Each crop is a square around the wrist,
shifted along the forearm toward the fingers, enlarged to crop_size.

Student camera only; crops stay in memory and nothing about the hand is
saved except the gesture name and score.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.inference.pose import (
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST, PosePerson,
)

# MediaPipe category -> event label; "None" means a hand without a known gesture.
GESTURES = {"Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down", "Open_Palm": "open_palm",
            "Pointing_Up": "pointing_up", "Victory": "victory", "Closed_Fist": "fist",
            "ILoveYou": "i_love_you"}
ARMS = {"left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)}
DEFAULTS: dict[str, float] = {
    "crop_size": 256,
    "min_detection_confidence": 0.3,
    "crop_forearms": 1.4,   # crop side in forearm lengths
    "min_crop_px": 48,
}


@dataclass(frozen=True)
class HandGesture:
    """One hand at one sample: found = a hand was detected in the crop."""
    side: str            # the person's left or right arm (pose keypoints)
    found: bool
    gesture: str | None  # GESTURES value, or None for no known gesture
    score: float


def hand_crop(person: PosePerson, side: str, min_conf: float,
              forearms: float = 1.4, min_px: float = 48) -> tuple[int, int, int] | None:
    """Square crop (left, top, side) in pixels around one hand, or None."""
    _, elbow_i, wrist_i = ARMS[side]
    elbow, wrist = person.point(elbow_i, min_conf), person.point(wrist_i, min_conf)
    if elbow is None or wrist is None:
        return None
    ex, ey = elbow.x * person.width, elbow.y * person.height
    wx, wy = wrist.x * person.width, wrist.y * person.height
    forearm = math.hypot(wx - ex, wy - ey)
    side_px = max(forearms * forearm, min_px)
    # The hand continues past the wrist along the forearm.
    cx, cy = wx + 0.35 * (wx - ex), wy + 0.35 * (wy - ey)
    return round(cx - side_px / 2), round(cy - side_px / 2), max(1, round(side_px))


class HandGestures:
    """Gesture per arm for each person; one recognizer per video."""

    def __init__(self, model_path: Path, settings: dict[str, Any] | None = None, min_conf: float = 0.35):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Gesture model is missing: {model_path}. Place gesture_recognizer.task there "
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
        confidence = float(self.settings["min_detection_confidence"])
        self.recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence))
        self.counts = {"crops": 0, "hands": 0, "gestures": 0}

    def close(self) -> None:
        self.recognizer.close()

    def estimate(self, frame: Any, people: Sequence[PosePerson]) -> dict[int, dict[str, HandGesture]]:
        import cv2
        height, width = frame.shape[:2]
        size = int(self.settings["crop_size"])
        found_all: dict[int, dict[str, HandGesture]] = {}
        for person in people:
            hands = {}
            for side in ARMS:
                crop = hand_crop(person, side, self.min_conf, float(self.settings["crop_forearms"]),
                                 float(self.settings["min_crop_px"]))
                if crop is None:
                    continue
                x0, y0, s = crop
                sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x0 + s), min(height, y0 + s)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                self.counts["crops"] += 1
                patch = np.zeros((s, s, 3), np.uint8)
                patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
                patch = cv2.resize(patch, (size, size),
                                   interpolation=cv2.INTER_CUBIC if s < size else cv2.INTER_AREA)
                result = self.recognizer.recognize(self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
                if not result.hand_landmarks:
                    hands[side] = HandGesture(side, False, None, 0.0)
                    continue
                self.counts["hands"] += 1
                top = result.gestures[0][0] if result.gestures and result.gestures[0] else None
                name = GESTURES.get(top.category_name) if top is not None else None
                if name is not None:
                    self.counts["gestures"] += 1
                hands[side] = HandGesture(side, True, name, round(float(top.score), 3) if top else 0.0)
            if hands:
                found_all[person.track_id] = hands
        return found_all
