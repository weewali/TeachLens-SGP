"""Takes frames from a lesson's review videos and pixelates every head before they go into the report.

Heads are found with the local YOLO26n-pose model on the frame itself (low confidence, so nobody is missed):
the box around the nose, eyes and ears is enlarged and pixelated; for a person whose face points are not
found, the top quarter of the person box is pixelated instead. Extra boxes (x1, y1, x2, y2 in 0-1 units)
can be given for anyone the model misses. Every output must still be checked by eye.

    python -m evaluation.blur_frames <review.mp4> <frame index> <out.png> [x1,y1,x2,y2 ...]
"""
from pathlib import Path
import sys

import cv2
import numpy as np

from evaluation import PIPELINE_ROOT

MODEL = PIPELINE_ROOT / "models" / "pose" / "yolo26n-pose.pt"


def pixelate(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int = 7) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    part = image[y1:y2, x1:x2]
    small = cv2.resize(part, (blocks, max(1, round(blocks * (y2 - y1) / (x2 - x1)))), interpolation=cv2.INTER_AREA)
    image[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def blur_heads(image: np.ndarray, extra: list[tuple[float, float, float, float]] = ()) -> int:
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    result = model.predict(image, imgsz=1280, conf=0.15, verbose=False)[0]
    h, w = image.shape[:2]
    count = 0
    for box, kp in zip(result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()):
        bx1, by1, bx2, by2 = box
        face = kp[:5]
        seen = face[face[:, 2] > 0.2]
        if len(seen) >= 2:
            fx1, fy1 = seen[:, 0].min(), seen[:, 1].min()
            fx2, fy2 = seen[:, 0].max(), seen[:, 1].max()
            size = max(fx2 - fx1, fy2 - fy1, 0.18 * (bx2 - bx1))
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            r = 1.1 * size
            pixelate(image, int(cx - r), int(cy - 1.3 * r), int(cx + r), int(cy + r))
        else:
            pixelate(image, int(bx1), int(by1), int(bx2), int(by1 + 0.28 * (by2 - by1)))
        count += 1
    for x1, y1, x2, y2 in extra:
        pixelate(image, int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    return count


def main() -> int:
    video, index, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    extra = [tuple(float(v) for v in arg.split(",")) for arg in sys.argv[4:]]
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        print("could not read frame", index)
        return 1
    n = blur_heads(frame, extra)
    frame = cv2.resize(frame, (1920, round(1920 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"{n} heads pixelated -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
