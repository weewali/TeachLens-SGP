"""Blind labelling frames from saved lesson folders.

For every sampled second of a lesson, one picture holds: the student view and
the instructor view (downscaled, with the pipeline's boxes and anonymous track
numbers), and an enlarged crop of every tracked person. It shows only where the
pipeline found people, never what it decided about them, so the labeller is not
pulled toward the pipeline's answers. The pictures show real faces: they stay
under outputs/evaluation/ (ignored by Git) and never go into the report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OVERVIEW = (960, 540)
CELL = (274, 330)
STUDENT_COLOUR = (40, 160, 255)
TEACHER_COLOUR = (60, 200, 60)


def epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


@dataclass
class Frame:
    """The analysed frame of one camera closest to a sampled real time."""
    video_time: float
    real_time: float
    boxes: list[tuple[int, str, list[float]]]  # (track_id, role, box_xyxy normalised)


def load_positions(events_path: Path) -> dict[float, Frame]:
    """Analysed frames of one camera run: real time -> boxes (position events only)."""
    frames: dict[float, Frame] = {}
    with events_path.open(encoding="utf8") as handle:
        for line in handle:
            e = json.loads(line)
            if e["indicator_name"] != "position" or e.get("track_id") is None:
                continue
            t = round(epoch(e["start_real_time"]), 3)
            frame = frames.setdefault(t, Frame(float(e["start_time"]), t, []))
            frame.boxes.append((int(e["track_id"]), e["role"], list(e["value"]["box_xyxy"])))
    return frames


def nearest(frames: dict[float, Frame], times: np.ndarray, t: float, tolerance: float) -> Frame | None:
    if not len(times):
        return None
    k = int(np.argmin(np.abs(times - t)))
    return frames[float(times[k])] if abs(times[k] - t) <= tolerance else None


def read_frame(capture: cv2.VideoCapture, video_time: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000.0)
    ok, image = capture.read()
    return image if ok else None


def _overview(image: np.ndarray | None, boxes, colour_for) -> np.ndarray:
    canvas = np.full((OVERVIEW[1], OVERVIEW[0], 3), 40, np.uint8)
    if image is None:
        cv2.putText(canvas, "no frame", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return canvas
    canvas = cv2.resize(image, OVERVIEW, interpolation=cv2.INTER_AREA)
    for track, role, (x1, y1, x2, y2) in boxes:
        p1 = (int(x1 * OVERVIEW[0]), int(y1 * OVERVIEW[1]))
        p2 = (int(x2 * OVERVIEW[0]), int(y2 * OVERVIEW[1]))
        cv2.rectangle(canvas, p1, p2, colour_for(role), 2)
        cv2.putText(canvas, str(track), (p1[0] + 3, p1[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(canvas, str(track), (p1[0] + 3, p1[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_for(role), 2)
    return canvas


def _crop(image: np.ndarray | None, box: list[float], title: str, colour) -> np.ndarray:
    cell = np.full((CELL[1], CELL[0], 3), 40, np.uint8)
    if image is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = 0.15 * (x2 - x1), 0.10 * (y2 - y1)
        a, b = max(0, int((x1 - mx) * w)), max(0, int((y1 - my) * h))
        c, d = min(w, int((x2 + mx) * w)), min(h, int((y2 + my) * h))
        if c > a and d > b:
            part = image[b:d, a:c]
            scale = min(CELL[0] / part.shape[1], (CELL[1] - 30) / part.shape[0])
            part = cv2.resize(part, (max(1, int(part.shape[1] * scale)), max(1, int(part.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            y0, x0 = 30, (CELL[0] - part.shape[1]) // 2
            cell[y0:y0 + part.shape[0], x0:x0 + part.shape[1]] = part
    cv2.putText(cell, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return cell


def composite(student_image, student: Frame | None, teacher_image, teacher: Frame | None, header: str) -> np.ndarray:
    def colour(role: str):
        return TEACHER_COLOUR if role == "teacher" else STUDENT_COLOUR

    top = np.hstack([_overview(student_image, student.boxes if student else [], colour),
                     _overview(teacher_image, teacher.boxes if teacher else [], colour)])
    cells = []
    for frame, image, prefix in ((student, student_image, "S"), (teacher, teacher_image, "I")):
        for track, role, box in sorted(frame.boxes if frame else []):
            cells.append(_crop(image, box, f"{prefix}{track}", colour(role)))
    per_row = 7
    rows = []
    for k in range(0, max(1, len(cells)), per_row):
        row = cells[k:k + per_row]
        row += [np.full((CELL[1], CELL[0], 3), 40, np.uint8)] * (per_row - len(row))
        rows.append(np.hstack(row))
    band = np.full((36, top.shape[1], 3), 255, np.uint8)
    cv2.putText(band, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    rows = [r[:, :top.shape[1]] if r.shape[1] > top.shape[1] else
            np.hstack([r, np.full((r.shape[0], top.shape[1] - r.shape[1], 3), 40, np.uint8)]) for r in rows]
    return np.vstack([band, top] + rows)


def make_frames(lessons: list[tuple[str, Path]], out_dir: Path, every_seconds: float,
                tolerance: float = 0.6) -> list[dict[str, Any]]:
    """Writes frames/<frame_id>.jpg and returns one item per sampled second."""
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for name, folder in lessons:
        lesson = json.loads((folder / "lesson.json").read_text(encoding="utf8"))
        runs = {r["camera"]: r for r in lesson["runs"]}
        student = load_positions(folder / "student" / "events.jsonl")
        teacher = load_positions(folder / "instructor" / "events.jsonl")
        s_times, t_times = np.array(sorted(student)), np.array(sorted(teacher))
        start, end = epoch(lesson["fusion"]["start_real_time"]), epoch(lesson["fusion"]["end_real_time"])
        s_cap = cv2.VideoCapture(runs["student_view"]["video"])
        t_cap = cv2.VideoCapture(runs["instructor_view"]["video"])
        k = 0
        t = start + every_seconds / 2
        while t < end:
            s, i = nearest(student, s_times, t, tolerance), nearest(teacher, t_times, t, tolerance)
            frame_id = f"{name}_{k:03d}"
            clock = int(round(t - start))
            s_img = read_frame(s_cap, s.video_time) if s else None
            t_img = read_frame(t_cap, i.video_time) if i else None
            image = composite(s_img, s, t_img, i, f"{frame_id}   lesson time {clock // 60}:{clock % 60:02d}")
            cv2.imwrite(str(out_dir / "frames" / f"{frame_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            items.append({"frame_id": frame_id, "lesson": name, "folder": str(folder), "lesson_seconds": clock,
                          "real_time": round(t, 3),
                          "student_real_time": s.real_time if s else "", "student_video_time": s.video_time if s else "",
                          "student_tracks": " ".join(str(b[0]) for b in sorted(s.boxes)) if s else "",
                          "instructor_real_time": i.real_time if i else "",
                          "instructor_video_time": i.video_time if i else "",
                          "instructor_tracks": " ".join(str(b[0]) for b in sorted(i.boxes)) if i else ""})
            k += 1
            t += every_seconds
        s_cap.release()
        t_cap.release()
    return items


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
