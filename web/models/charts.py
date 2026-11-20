"""Chart model: the two heatmap images on the report page, drawn from a lesson folder's saved events.

- instructor_heatmap    where the instructor was in the instructor camera's view (hip centre per sample), with their
                        path, over a frame of the instructor video
- seat_heatmap          one box per anonymous seat (student/summary.json "seats"), coloured by the share of the time
                        its student faced the front (an observed posture, not attention), over a frame of the student
                        video

Privacy (user requests 2026-11-19 and 2026-11-20): in both background frames every head the pipeline located is
pixelated: everyone detected in the first ANON_FIRST_SAMPLES samples (any role), plus where each seat's student
usually was. If anything goes wrong, the chart is drawn without a background, never over the unblurred frame. Only
these pictures are changed; the videos and the analysis are not.

The lecture timeline and the activity comparison are drawn in the page itself (web/models/stat_charts.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

POSE_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose" / "yolo26n-pose.pt"   # local, never downloaded
CHART_STYLE = "3"         # saved charts drawn in an older style are redrawn (3: lesson pipeline data, both blurred)
MIN_KNOWN_SECONDS = 3      # a seat needs this much facing-front / facing-away time before it gets a percentage
ANON_FIRST_SAMPLES = 5     # heads of everyone found in the first 5 samples (5 s at 1 sample/s) are blurred
HEAD_SHARE = 0.35          # head area: the top 35% of a person box (at most one box width tall), full box width
HEAD_PAD = 0.2             # each head area grows by this share on every side
PIXEL_BLOCKS = 6           # a head area is shown as about 6 coloured blocks across

# file -> (title shown above the chart, description used only as the image's text alternative)
CHARTS = {
    "instructor_heatmap.png": ("Instructor Movement", "Brighter = more time spent there, in the instructor camera's "
                               "view. The line is your path, sampled every second. Faces are blurred for privacy."),
    "seat_heatmap.png": ("Student Seats", "Each box is where one student usually sat. The percentage is the share of "
                         "the time that student faced the front. A posture, not a measure of attention; writing and "
                         "reading look like facing down. Faces are blurred for privacy."),
}


def chart_path(lecture, name):
    return lecture.charts_dir / name


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _events(lesson, camera, indicator=None):
    path = lesson / camera / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if indicator is None or r.get("indicator_name") == indicator]


def head_areas(boxes):
    """[(x1, y1, x2, y2)] as fractions of the frame: the head part of each person box (fractions too), padded."""
    areas = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1):
            continue
        height = min(HEAD_SHARE * (y2 - y1), x2 - x1)
        if y2 >= 0.97:   # cut off by the bottom edge (someone close to the camera): the box is mostly head
            height = y2 - y1
        pad_x, pad_y = HEAD_PAD * (x2 - x1), HEAD_PAD * height
        areas.append((max(0.0, x1 - pad_x), max(0.0, y1 - pad_y), min(1.0, x2 + pad_x), min(1.0, y1 + height + pad_y)))
    return areas


def pixelate(image, areas, blocks=PIXEL_BLOCKS):
    """A copy of the image with each area (fractions of the frame) replaced by about `blocks` flat blocks across."""
    out = np.array(image, copy=True)
    height, width = out.shape[:2]
    for fx1, fy1, fx2, fy2 in areas:
        x1, x2 = int(np.floor(fx1 * width)), int(np.ceil(fx2 * width))
        y1, y2 = int(np.floor(fy1 * height)), int(np.ceil(fy2 * height))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        size = max(4, int(np.ceil(max(x2 - x1, y2 - y1) / blocks)))
        region = out[y1:y2, x1:x2]
        for by in range(0, region.shape[0], size):
            for bx in range(0, region.shape[1], size):
                block = region[by:by + size, bx:bx + size]
                block[...] = block.mean(axis=(0, 1))
    return out


def grab_frame(lesson, camera):
    """The frame at the first analysed sample of a camera's source video (RGB), or None."""
    run = _json(lesson / camera / "run.json")
    source = run.get("video")
    if not source or not Path(source).exists():
        return None, None
    import cv2
    first = float(run.get("first_sample_seconds") or run.get("analyzed_start_seconds") or 0.0)
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, first * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    return (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), first) if ok else (None, None)


def people_in(image):
    """Person boxes (fractions of the frame) the local pose model finds in this one frame: people the analysis did not
    track (e.g. students in front of the instructor camera) get blurred too. Empty when the model is not available."""
    if not POSE_MODEL.exists():
        return []
    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never download or install anything from the web app
    os.environ.setdefault("YOLO_OFFLINE", "True")
    from ultralytics import YOLO
    result = YOLO(str(POSE_MODEL)).predict(image, imgsz=1280, conf=0.15, verbose=False)[0]
    return [tuple(float(v) for v in box) for box in result.boxes.xyxyn.tolist()]


def blurred_background(lesson, camera, extra_boxes=()):
    """The camera's first analysed frame with every located head pixelated, or None (never the raw frame)."""
    try:
        image, first = grab_frame(lesson, camera)
        if image is None:
            return None
        positions = _events(lesson, camera, "position")
        times = sorted({float(p["start_time"]) for p in positions})
        early = set(times[:ANON_FIRST_SAMPLES])
        boxes = [p["value"]["box_xyxy"] for p in positions
                 if float(p["start_time"]) in early and isinstance(p.get("value"), dict) and "box_xyxy" in p["value"]]
        return pixelate(image, head_areas(list(boxes) + list(extra_boxes) + people_in(image)))
    except Exception:
        return None


def _figure(background, alpha=0.55):
    shape = background.shape if background is not None else (9, 16)
    fig, ax = plt.subplots(figsize=(8, 8 * shape[0] / shape[1]))
    if background is not None:
        ax.imshow(background, extent=[0, 1, 1, 0], alpha=alpha, aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def _point(value):
    for key in ("hip_center", "box_center"):
        point = value.get(key) if isinstance(value, dict) else None
        if point and len(point) == 2 and all(v is not None for v in point):
            return float(point[0]), float(point[1])
    return None


def draw_instructor_heatmap(lecture, out_path):
    lesson = lecture.lesson_dir
    teacher = [e for e in _events(lesson, "instructor", "position") if e.get("role") == "teacher"]
    points = [(float(e["start_time"]), _point(e["value"])) for e in teacher]
    points = sorted((t, p) for t, p in points if p)
    if not points:
        return False
    xs, ys = np.array([p[0] for _, p in points]), np.array([p[1] for _, p in points])
    fig, ax = _figure(blurred_background(lesson, "instructor"))
    counts, _, _ = np.histogram2d(ys, xs, bins=(12, 20), range=[[0, 1], [0, 1]])
    shown = np.where(counts > 0, counts, np.nan)   # empty cells stay see-through
    image = ax.imshow(shown, extent=[0, 1, 1, 0], cmap="inferno", alpha=0.8, interpolation="nearest", aspect="auto",
                      vmin=0)
    ax.plot(xs, ys, color="#38bdf8", linewidth=1.2, alpha=0.9)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("samples (one every second)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def seat_boxes(lesson):
    """{seat_id: (seat entry, median person box of its tracks)} for the seats in student/summary.json."""
    seats = (_json(lesson / "student" / "summary.json").get("seats") or {}).get("seats", [])
    boxes = {}
    for event in _events(lesson, "student", "position"):
        value = event.get("value")
        if isinstance(value, dict) and "box_xyxy" in value:
            boxes.setdefault(event.get("track_id"), []).append(value["box_xyxy"])
    result = {}
    for seat in seats:
        found = [b for track in seat.get("tracks", []) for b in boxes.get(track, [])]
        if found:
            result[seat["seat_id"]] = (seat, tuple(float(v) for v in np.median(np.array(found, float), axis=0)))
    return result


def draw_seat_heatmap(lecture, out_path):
    lesson = lecture.lesson_dir
    seats = seat_boxes(lesson)
    if not seats:
        return False
    background = blurred_background(lesson, "student", extra_boxes=[box for _, box in seats.values()])
    fig, ax = _figure(background, alpha=0.8)
    cmap = plt.get_cmap("viridis")
    for seat_id, (seat, (x1, y1, x2, y2)) in sorted(seats.items(), key=lambda item: item[1][1][1]):
        number = str(seat_id).split("_")[-1].lstrip("0") or "0"
        front, away = float(seat.get("on_target_seconds") or 0), float(seat.get("off_target_seconds") or 0)
        if front + away >= MIN_KNOWN_SECONDS:
            share = front / (front + away)
            colour, label = cmap(share), f"Seat {number}\n{100 * share:.0f}%"
        else:
            colour, label = (0.55, 0.57, 0.6, 1.0), f"Seat {number}\nnot readable"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=colour, alpha=0.22, edgecolor="none"))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", edgecolor=colour, linewidth=2.5))
        light = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2] > 0.55   # dark text on light labels
        ax.text((x1 + x2) / 2, y1, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold", linespacing=1.0,
                multialignment="center", color="#111111" if light else "white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": colour, "edgecolor": "none"})
    image = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("% of the time facing the front")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def ensure_charts(lecture):
    """Draws the heatmaps if they are missing, older than the lesson's events or drawn in an older style; returns the
    charts that exist, in order."""
    lesson = lecture.lesson_dir
    if lesson is None:
        return []
    events = lesson / "student" / "events.jsonl"
    lecture.charts_dir.mkdir(parents=True, exist_ok=True)
    style = lecture.charts_dir / "style.txt"
    old_style = not style.exists() or style.read_text(encoding="utf-8").strip() != CHART_STYLE
    for name, draw in (("instructor_heatmap.png", draw_instructor_heatmap), ("seat_heatmap.png", draw_seat_heatmap)):
        path = chart_path(lecture, name)
        stale = old_style or not path.exists() or (events.exists() and path.stat().st_mtime < events.stat().st_mtime)
        if stale and not draw(lecture, path):
            path.unlink(missing_ok=True)   # a chart that cannot be drawn now is not shown at all
    if old_style:   # written only when it changes: its time tells the PDF whether the charts are new
        style.write_text(CHART_STYLE, encoding="utf-8")
    return [{"file": name, "title": title, "description": description} for name, (title, description) in CHARTS.items()
            if chart_path(lecture, name).exists()]
