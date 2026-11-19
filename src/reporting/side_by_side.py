"""One lesson video: the student and instructor review videos side by side, lined up by real time.

User request 2026-11-19. Each camera's review MP4 holds one annotated frame
per analyzed sample; its timing (first sample, sample period, recording
start) comes from the run's run.json. For every second of the stretch both
cameras recorded, the frame nearest that real time is taken from each video,
both are scaled to the same height and placed side by side under a header:
real clock time, lesson time, what the instructor was doing, and the class's
on-task / off-task / phone shares in that 5 s window (from the fusion table).
The video holds original classroom footage: it stays local (outputs/ is
ignored by Git).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.annotated_video import AnnotatedVideoWriter

PANEL_HEIGHT = 1080
HEADER_HEIGHT = 96
WHITE, GREY, DARK = (255, 255, 255), (190, 190, 190), (40, 40, 40)
TEACHER_STATES = (("T_seated", "seated"), ("T_at_desk", "at desk"), ("T_facing_class", "facing class"),
                  ("T_back_to_class", "back to class"), ("T_reading_notes", "reading notes"),
                  ("T_gesturing", "gesturing"), ("T_walking", "walking"), ("T_board_or_screen", "board/screen"))


@dataclass(frozen=True)
class ReviewTiming:
    """Real time of a review video's frames: frame k shows real_start + first + k * period."""
    video: Path
    real_start: datetime
    first_sample_seconds: float
    sample_period: float

    def frame_at(self, moment: datetime) -> int:
        offset = (moment - self.real_start).total_seconds() - self.first_sample_seconds
        return int(round(offset / self.sample_period))


class _Reader:
    """Sequential frame access by index (indexes only grow, as in a timeline)."""

    def __init__(self, path: Path):
        import cv2
        self.capture = cv2.VideoCapture(str(path))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index, self.frame = -1, None

    def get(self, index: int) -> Any:
        if index < 0 or index >= self.count:
            return None
        while self.index < index:
            ok, frame = self.capture.read()
            if not ok:
                self.count = self.index + 1
                return None
            self.index, self.frame = self.index + 1, frame
        return self.frame

    def close(self) -> None:
        self.capture.release()


def _panel(cv2: Any, frame: Any, width: int, label: str) -> Any:
    if frame is None:
        panel = np.zeros((PANEL_HEIGHT, width, 3), np.uint8)
        cv2.putText(panel, f"{label}: no frame at this time", (40, PANEL_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, GREY, 2, cv2.LINE_AA)
        return panel
    return cv2.resize(frame, (width, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)


def _pct(value: Any) -> str:
    return "?" if value is None or not np.isfinite(value) else f"{100 * value:.0f}%"


def header_lines(row: Any, moment: datetime, lesson_seconds: float, title: str) -> tuple[str, str, str]:
    """(left, middle, right) header texts for one second of the lesson."""
    minutes, seconds = divmod(int(lesson_seconds), 60)
    left = f"{title}  {moment.strftime('%H:%M:%S')}  lesson {minutes:02d}:{seconds:02d}"
    states = []
    if row is not None:
        for column, word in TEACHER_STATES:
            value = row.get(column)
            if value is not None and np.isfinite(value) and value >= 0.5:
                states.append(word)
    middle = "instructor: " + (", ".join(states) if states else "-")
    right = ("class: on task " + _pct(row.get("S_on_task")) + "  off task " + _pct(row.get("S_off_task"))
             + "  phone " + _pct(row.get("S_phone"))) if row is not None else "class: -"
    return left, middle, right


def write_side_by_side(student: ReviewTiming, instructor: ReviewTiming, table: Any, start: datetime,
                       seconds: float, window_seconds: float, output: Path, title: str, *,
                       codec: str = "avc1", overwrite: bool = False) -> Path:
    """Write output (1 frame per lesson second): students left, instructor right."""
    import cv2
    readers = {"students": _Reader(student.video), "instructor": _Reader(instructor.video)}
    sizes = {}
    for name, reader in readers.items():
        width = int(reader.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(reader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        sizes[name] = int(round(width * PANEL_HEIGHT / height / 2)) * 2
    total_width = sizes["students"] + sizes["instructor"]
    rows = table.to_dict("records") if table is not None else []
    try:
        with AnnotatedVideoWriter(output, codec, overwrite=overwrite) as writer:
            writer.open(1.0, total_width, PANEL_HEIGHT + HEADER_HEIGHT)
            for tick in range(int(seconds)):
                moment = start + timedelta(seconds=tick)
                frames = {name: readers[name].get(timing.frame_at(moment))
                          for name, timing in (("students", student), ("instructor", instructor))}
                canvas = np.full((PANEL_HEIGHT + HEADER_HEIGHT, total_width, 3), DARK, np.uint8)
                canvas[HEADER_HEIGHT:, :sizes["students"]] = _panel(cv2, frames["students"], sizes["students"],
                                                                     "student camera")
                canvas[HEADER_HEIGHT:, sizes["students"]:] = _panel(cv2, frames["instructor"], sizes["instructor"],
                                                                     "instructor camera")
                window = int(tick // window_seconds)
                row = rows[window] if 0 <= window < len(rows) else None
                left, middle, right = header_lines(row, moment, tick, title)
                for text, x in ((left, 24), (middle, int(total_width * 0.36)), (right, int(total_width * 0.66))):
                    cv2.putText(canvas, text, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)
                writer.write(canvas)
    finally:
        for reader in readers.values():
            reader.close()
    return output
