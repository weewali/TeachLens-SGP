"""Chart model: the heatmap images the report page shows, drawn from a lecture's saved tables (no video is reprocessed).

- instructor_heatmap    where the instructor was in the instructor camera's view, with their path
- seat_heatmap          for each seat, the share of the readable time its student had their head up (a posture proxy,
                        not attention), drawn as a box where that student usually sat. The camera frame behind it has
                        every head it can locate pixelated (user request 2026-11-19); only this picture is changed,
                        never the frames the analysis reads.

The lecture timeline and the activity comparison are drawn in the page itself (web/models/stat_charts.py), so they
follow the page's light or dark theme; the pipeline's own timeline.png is no longer shown.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEAT_MIN_SHARE = 0.10      # as in Main Code.py: a seat must be occupied in at least this share of the samples
MIN_READABLE = 3           # samples with a readable head needed before a seat gets a percentage
CHART_STYLE = "2"          # saved charts drawn in an older style are redrawn (2: faces pixelated in the seat chart)

# Faces in the seat chart's background frame (the first sampled frame of the student camera). A person the detector
# missed in that one frame still gets blurred if they were found in one of the next samples or usually sat in a seat.
ANON_FIRST_FRAMES = 5      # heads of everyone found in the first 5 samples (10 s), whatever their role
HEAD_WIDTH = 1.3           # head area: this many shoulder widths wide, from the top of the person box to the shoulders
HEAD_TOP_SHARE = 0.45      # without readable shoulders: the top 45% of the person box, full width
HEAD_PAD = 0.2             # each head area grows by this share on every side
PIXEL_BLOCKS = 6           # a head area is shown as about 6 coloured blocks across

# file -> (title shown above the chart, description used only as the image's text alternative; the page shows no
# caption under the heatmaps, user request 2026-11-19)
CHARTS = {
    "instructor_heatmap.png": ("Instructor Movement", "Brighter = more time spent there, in the instructor camera's "
                               "view. The line is your path, sampled every 2 seconds."),
    "seat_heatmap.png": ("Student Seats", "Each box is where one student usually sat. The percentage is the share of "
                         "the time their head could be read that it was up. A posture, not a measure of attention; "
                         "writing and reading look like head down. Faces are blurred for privacy."),
}


def chart_path(lecture, name):
    return lecture.charts_dir / name


def _detections(lecture):
    path = lecture.tables_dir / "frame_detections.csv"
    return pd.read_csv(path) if path.exists() else None


def _background(lecture, camera):
    frame = lecture.session_dir / "frames" / camera / "frame_0000.jpg"
    return plt.imread(str(frame)) if frame.exists() else None


def head_areas(people):
    """[(x1, y1, x2, y2)] as fractions of the frame: the head of each detection (pixel boxes with shoulders sx, sy, sw
    and the frame size img_w, img_h, as in frame_detections.csv), from the top of the person box down to the shoulder
    line, or the top of the box when the shoulders were not readable; padded on every side."""
    areas = []
    for p in people.itertuples(index=False):
        w, h = float(p.img_w), float(p.img_h)
        if not (w > 0 and h > 0 and np.isfinite([p.x1, p.y1, p.x2, p.y2]).all()):
            continue
        if np.isfinite([p.sx, p.sy, p.sw]).all() and p.sw > 0 and p.sy > p.y1:
            x1, x2 = p.sx - HEAD_WIDTH * p.sw / 2, p.sx + HEAD_WIDTH * p.sw / 2
            y1, y2 = p.y1, p.sy
        else:
            x1, x2, y1, y2 = p.x1, p.x2, p.y1, p.y1 + HEAD_TOP_SHARE * (p.y2 - p.y1)
        pad_x, pad_y = HEAD_PAD * (x2 - x1), HEAD_PAD * (y2 - y1)
        areas.append((max(0.0, (x1 - pad_x) / w), max(0.0, (y1 - pad_y) / h),
                      min(1.0, (x2 + pad_x) / w), min(1.0, (y2 + pad_y) / h)))
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


def _student_background(lecture, students_camera):
    """The student camera's first frame with every head found around it pixelated, or None. A failure gives no
    background at all rather than the unblurred frame."""
    image = _background(lecture, "students")
    if image is None:
        return None
    try:
        early = students_camera[students_camera["frame"] < ANON_FIRST_FRAMES]
        usual = students_camera[students_camera["role"] == "student"].groupby("seat")[
            ["x1", "y1", "x2", "y2", "sx", "sy", "sw", "img_w", "img_h"]].median()   # where each seat's head usually was
        return pixelate(image, head_areas(early) + head_areas(usual))
    except Exception:
        return None


def _figure(background, alpha=0.55):
    fig, ax = plt.subplots(figsize=(8, 4.5) if background is None else (8, 8 * background.shape[0] / background.shape[1]))
    if background is not None:
        ax.imshow(background, extent=[0, 1, 1, 0], alpha=alpha, aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def draw_instructor_heatmap(lecture, out_path):
    people = _detections(lecture)
    if people is None:
        return False
    instructor = people[(people["camera"] == "instructor") & (people["role"] == "instructor")].sort_values("frame")
    if instructor.empty:
        return False
    fig, ax = _figure(_background(lecture, "instructor"))
    counts, _, _ = np.histogram2d(instructor["cy"], instructor["cx"], bins=(12, 20), range=[[0, 1], [0, 1]])
    shown = np.where(counts > 0, counts, np.nan)   # empty cells stay see-through (no smoothing: it would hide them)
    image = ax.imshow(shown, extent=[0, 1, 1, 0], cmap="inferno", alpha=0.8, interpolation="nearest", aspect="auto",
                      vmin=0)
    ax.plot(instructor["cx"], instructor["cy"], color="#38bdf8", linewidth=1.2, alpha=0.9)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("samples (one every 2 s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def draw_seat_heatmap(lecture, out_path):
    people = _detections(lecture)
    if people is None:
        return False
    students = people[(people["camera"] == "students") & (people["role"] == "student")].copy()
    if students.empty:
        return False
    background = _student_background(lecture, people[people["camera"] == "students"])
    frames_path = lecture.tables_dir / "frames.csv"
    n_frames = len(pd.read_csv(frames_path)) if frames_path.exists() else int(people["frame"].max()) + 1
    for side, size in (("x1", "img_w"), ("x2", "img_w"), ("y1", "img_h"), ("y2", "img_h")):
        students[side] = students[side] / students[size]
    head = students["head_up"].astype(bool) | students["head_down"].astype(bool)
    students["readable"], students["up"] = head, students["head_up"].astype(bool)
    seats = students.groupby("seat").agg(frames=("frame", "nunique"), x1=("x1", "median"), y1=("y1", "median"),
                                         x2=("x2", "median"), y2=("y2", "median"), readable=("readable", "sum"),
                                         up=("up", "sum"))
    # older results used grid cells, where a chair or bag seen twice became a seat: keep the places really sat in
    seats = seats[seats["frames"] >= max(2, SEAT_MIN_SHARE * n_frames)]
    fig, ax = _figure(background, alpha=0.8)
    cmap = plt.get_cmap("viridis")
    for seat, s in seats.sort_values("y1").iterrows():   # back rows first, so front-row labels stay on top
        number = str(seat).split("_")[-1].lstrip("0") or "0"
        if s["readable"] >= MIN_READABLE:
            share = s["up"] / s["readable"]
            colour, label = cmap(share), f"Seat {number}\n{100 * share:.0f}%"
        else:
            colour, label = (0.55, 0.57, 0.6, 1.0), f"Seat {number}\nnot readable"
        ax.add_patch(plt.Rectangle((s["x1"], s["y1"]), s["x2"] - s["x1"], s["y2"] - s["y1"], facecolor=colour,
                                   alpha=0.22, edgecolor="none"))
        ax.add_patch(plt.Rectangle((s["x1"], s["y1"]), s["x2"] - s["x1"], s["y2"] - s["y1"], facecolor="none",
                                   edgecolor=colour, linewidth=2.5))
        light = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2] > 0.55   # dark text on light labels
        ax.text((s["x1"] + s["x2"]) / 2, s["y1"], label, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                linespacing=1.0, multialignment="center", color="#111111" if light else "white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": colour, "edgecolor": "none"})
    image = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("% of readable time with head up")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def ensure_charts(lecture):
    """Draws the heatmaps if they are missing, older than the tables or drawn in an older style (e.g. before faces were
    blurred); returns the charts that exist, in order."""
    tables = lecture.tables_dir / "frame_detections.csv"
    lecture.charts_dir.mkdir(parents=True, exist_ok=True)
    style = lecture.charts_dir / "style.txt"
    old_style = not style.exists() or style.read_text(encoding="utf-8").strip() != CHART_STYLE
    for name, draw in (("instructor_heatmap.png", draw_instructor_heatmap), ("seat_heatmap.png", draw_seat_heatmap)):
        path = chart_path(lecture, name)
        stale = old_style or not path.exists() or (tables.exists() and path.stat().st_mtime < tables.stat().st_mtime)
        if stale and tables.exists() and not draw(lecture, path) and old_style:
            path.unlink(missing_ok=True)   # an old chart that cannot be redrawn (e.g. unblurred faces) is not shown
    if tables.exists():
        style.write_text(CHART_STYLE, encoding="utf-8")
    return [{"file": name, "title": title, "description": description} for name, (title, description) in CHARTS.items()
            if chart_path(lecture, name).exists()]
