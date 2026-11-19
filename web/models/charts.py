"""Chart model: the heatmap images the report page shows, drawn from a lecture's saved tables (no video is reprocessed).

- instructor_heatmap    where the instructor was in the instructor camera's view, with their path
- seat_heatmap          for each seat, the share of the readable time its student had their head up (a posture proxy,
                        not attention), drawn as a box where that student usually sat

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

# file -> (title shown above the chart, description used only as the image's text alternative; the page shows no
# caption under the heatmaps, user request 2026-11-19)
CHARTS = {
    "instructor_heatmap.png": ("Instructor Movement", "Brighter = more time spent there, in the instructor camera's "
                               "view. The line is your path, sampled every 2 seconds."),
    "seat_heatmap.png": ("Student Seats","Each box is where one student usually sat. The percentage is the share of "
                         "the time their head could be read that it was up. A posture, not a measure of attention; "
                         "writing and reading look like head down."),
}


def chart_path(lecture, name):
    return lecture.charts_dir / name


def _detections(lecture):
    path = lecture.tables_dir / "frame_detections.csv"
    return pd.read_csv(path) if path.exists() else None


def _background(lecture, camera):
    frame = lecture.session_dir / "frames" / camera / "frame_0000.jpg"
    return plt.imread(str(frame)) if frame.exists() else None


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
    fig, ax = _figure(_background(lecture, "students"), alpha=0.8)
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
    """Draws the heatmaps if they are missing or older than the tables; returns the charts that exist, in order."""
    tables = lecture.tables_dir / "frame_detections.csv"
    lecture.charts_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in (("instructor_heatmap.png", draw_instructor_heatmap), ("seat_heatmap.png", draw_seat_heatmap)):
        path = chart_path(lecture, name)
        stale = not path.exists() or (tables.exists() and path.stat().st_mtime < tables.stat().st_mtime)
        if stale and tables.exists():
            draw(lecture, path)
    return [{"file": name, "title": title, "description": description} for name, (title, description) in CHARTS.items()
            if chart_path(lecture, name).exists()]
