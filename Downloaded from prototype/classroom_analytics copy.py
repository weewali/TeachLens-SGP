import os
import re
import cv2
import json
import glob
import shutil
import argparse
from collections import Counter
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template
from ultralytics import YOLO

# scipy / statsmodels are only needed for the lag analysis (report FR6.3). If they are not installed
# the pipeline still runs and the lag stage falls back to plain pandas correlations without p-values.
try:
    from scipy import stats
except ImportError:
    stats = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

# 1. Path Configuration
BASE_DIR = os.environ.get("TEACHLENS_BASE", "classroom_project")   # tests point this somewhere else
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
STD_DIR = os.path.join(BASE_DIR, "standardized")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
TABLE_DIR = os.path.join(BASE_DIR, "tables")
YOLO_OUT_DIR = os.path.join(BASE_DIR, "yolo_outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# 2. Pipeline Parameters (report chapter 4)
STANDARD_FPS = 25                  # video standardization: one frame rate ...
STANDARD_MAX_HEIGHT = 720          # ... and one maximum resolution (never upscaled)
FRAME_EVERY_SECONDS = 2            # frame sampling interval
WINDOW_SECONDS = 10                # fusion window length (a multiple of FRAME_EVERY_SECONDS)
MAX_LAG_WINDOWS = 3                # lags 1..3 windows for the lag analysis
MIN_WINDOWS_FOR_STATS = 8          # below this many valid windows, no lag statistics are reported

DETECT_WEIGHTS = "yolov8n.pt"      # COCO detector, used here for phone / book cues
POSE_WEIGHTS = "yolo11n-pose.pt"   # person boxes + body keypoints (pose-based cues)
CONF = 0.25
IMGSZ = 640
DEVICE = "cpu"                     # Change to "cuda" or 0 if GPU is available
COCO_OBJECTS = {67: "phone", 73: "book"}
PREDICT_CHUNK = 20                 # frames per predict() call; one call over all frames grew past 3 GB and was killed

# Behaviour heuristics. These are rule-based stand-ins for the trained instructor / student
# behaviour models the report plans for SGP2, so treat their outputs as a baseline.
KP_CONF = 0.5                      # minimum keypoint confidence to use a keypoint
HAND_RAISE_MARGIN = 0.05           # wrist must be this fraction of the box height above the shoulder
HAND_RAISE_MIN_FRAMES = 2          # ... in this many consecutive sampled frames
HEAD_DOWN_RATIO = 0.25             # nose within this many shoulder-widths of the shoulder line = head down
OFFTASK_MIN_FRAMES = 3             # head down this many consecutive sampled frames (no book) = off-task
FACING_MARGIN = 0.15               # nose offset (in shoulder-widths) toward the students = facing them
LONG_STRETCH_WINDOWS = 6           # explanation with no interaction for this many windows = flagged
SEATED_THIGH_RATIO = 0.5           # thigh / torso vertical extent below this = seated (needs hips and knees in view)
STATIONARY_MOVE = 0.01             # instructor moving less than this fraction of the frame between samples = stationary

# Report points only use windows where student posture could be read for enough seats.
MIN_POSE_COVERAGE = 0.4            # share of seat-frames with a readable posture
MIN_SEATS_OBSERVED = 3             # seats seen per frame, on average
MIN_GROUP_WINDOWS = 5              # a behaviour needs this many windows (and as many others) to be compared
MIN_ATTENTION_GAP = 0.05           # attention must differ by at least 5 percentage points

# 3. Classroom calibration (report FR2.3): fractions of the frame as [x1, y1, x2, y2].
# These defaults were set by eye for the 10-minute test video; edit classroom_project/config/calibration.json
# for other camera positions.
DEFAULT_CALIBRATION = {
    "teacher_zone": [0.76, 0.00, 1.00, 0.85],
    "board_region": [0.80, 0.00, 1.00, 0.45],
    "student_region": [0.00, 0.15, 0.76, 1.00],
    "seat_grid": [3, 4],            # rows, columns of anonymous seats inside the student region
}

TEACHER_FEATURES = ["explanation_ratio", "board_writing_ratio", "movement_amount", "interaction_count",
                    "teacher_seated_ratio", "teacher_gesture", "teacher_student_distance", "teacher_stationary_ratio"]
STUDENT_OUTCOMES = ["attention_level", "off_task_rate", "hand_raise_count"]
MODEL_PREDICTORS = ["explanation_ratio", "board_writing_ratio", "interaction_count"]
# Hypotheses fixed before these indicators had ever been computed. Only these pairs are judged against the small
# correction; every other pair is exploratory and gets the larger one. Do not add a pair after looking at results.
PRESPECIFIED = [("teacher_seated_ratio", "attention_level"), ("teacher_student_distance", "hand_raise_count")]

# The instructor-facing report: positive and negative points, each tied to what the instructor was doing.
FEEDBACK_TEMPLATE = """\
positive points:

{% for line in positive %}
- {{ line }}
{% endfor %}

negative points:

{% for line in negative %}
- {{ line }}
{% endfor %}

{{ note }}
"""

FEEDBACK_NOTE = ("Note: \"focused\" means head up rather than looking down; it is a posture proxy, not a measure of attention or "
                 "learning. Times are mm:ss from the start of the lecture. Where the camera moves, roles and posture can be wrong; "
                 "see analysis_details.txt for the indicators, statistics and limitations.")

# What the language model is told the words in facts.json mean, and the alternative explanations it may offer (as
# possibilities only). Keeping them here means the model cannot invent its own definitions or science.
GLOSSARY = {
    "observable students": "Seats where a student's head and shoulders were readable in the video. Not every student in the room.",
    "focused": "Head up rather than looking down. A posture proxy; it is not a measure of attention, understanding or learning.",
    "window": f"A {WINDOW_SECONDS}-second slice of the lecture.",
    "explaining to the class": "The instructor was visible, facing the students, and not writing on the board.",
    "not facing the class": "The instructor was visible but not turned toward the students or the board.",
    "moving around": "The most mobile quarter of windows, by how far the instructor travelled.",
    "close to the students": "The quarter of windows where the instructor was closest to the nearest student.",
    "seated": "The instructor's thighs looked short beside the torso (needs hips and knees in view).",
    "evidence.strength": "weak = the gap could easily be chance; suggestive = unlikely to be pure chance; descriptive = a single observation or a count, not a test.",
}
ALTERNATIVE_EXPLANATIONS = {
    "E1": "Students may have looked down to write notes, which the head-up rule counts as not focused.",
    "E2": "Students may have been reading a slide or the board rather than watching the instructor.",
    "E3": "The camera moves in this recording, so some people may have been mislabelled or out of view.",
    "E4": "The comparison rests on a small number of 10-second windows.",
    "E5": "The type of activity (lecture, seat-work, discussion) was not recorded, and it changes posture on its own.",
}

# The supporting detail behind the report above.
DETAILS_TEMPLATE = """\
Classroom Analytics: analysis details
{{ "=" * 50 }}

Video: {{ video }}
Lecture duration: {{ duration }}  |  Frames analysed: {{ n_frames }} (one every {{ step }} s)  |  Windows: {{ n_windows }} x {{ window_s }} s
Mode: {{ mode }}

1. Instructor indicators
{% for line in instructor_lines %}
- {{ line }}
{% endfor %}

2. Student indicators
{% for line in student_lines %}
- {{ line }}
{% endfor %}

3. Teacher-student linkage (lag analysis)
{% for line in linkage_lines %}
- {{ line }}
{% endfor %}

4. Suggestions
{% for line in suggestions %}
- {{ line }}
{% endfor %}

5. Supporting evidence (timestamps are mm:ss from the start of the lecture)
{% for line in evidence %}
- {{ line }}
{% endfor %}

6. Limitations
{% for line in limitations %}
- {{ line }}
{% endfor %}
"""


def setup_directories():
    """Creates the folders shared by every lecture: inputs and configuration."""
    for folder in [BASE_DIR, VIDEO_DIR, CONFIG_DIR]:
        os.makedirs(folder, exist_ok=True)
    print("Directories initialized.")


def set_session_dirs(session_id):
    """Points the output folders at classroom_project/sessions/<session_id>/ so one lecture never overwrites another."""
    global STD_DIR, FRAME_DIR, TABLE_DIR, YOLO_OUT_DIR, REPORT_DIR
    root = os.path.join(BASE_DIR, "sessions", session_id)
    STD_DIR, FRAME_DIR, TABLE_DIR, YOLO_OUT_DIR, REPORT_DIR = (
        os.path.join(root, name) for name in ("standardized", "frames", "tables", "yolo_outputs", "reports"))
    for folder in (STD_DIR, FRAME_DIR, TABLE_DIR, YOLO_OUT_DIR, REPORT_DIR):
        os.makedirs(folder, exist_ok=True)


def load_calibration(session_id=None):
    """Loads the classroom calibration (teacher zone, board region, student seats); writes the defaults if missing.

    A camera with its own layout can have config/calibration_<session_id>.json, which wins over the shared file."""
    path = os.path.join(CONFIG_DIR, "calibration.json")
    specific = os.path.join(CONFIG_DIR, f"calibration_{session_id}.json") if session_id else None
    if specific and os.path.exists(specific):
        path = specific
        print(f"Using this session's own calibration: {specific}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION, f, indent=2)
        print(f"Default classroom calibration written to {path} (edit it to match your camera).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- stage 1: video input (FR1)
def validate_video(video_path):
    """Checks that the video can be opened and read before any analysis starts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if fps <= 0 or frame_count <= 0 or not ok:
        raise ValueError(f"Could not read FPS / frames from video at {video_path}")
    print(f"Video OK: {width}x{height}, {fps:.2f} fps, {frame_count / fps:.0f} s")
    return {"fps": fps, "width": width, "height": height, "duration_s": frame_count / fps}


# ---------------------------------------------------------------- stage 2: preprocessing (FR2.1, FR2.2)
def standardize_video(video_path, info, output_dir):
    """Uses FFmpeg to normalize the video to one frame rate, codec and (maximum) resolution."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for video standardization but was not found on PATH.")
    out_path = os.path.join(output_dir, "standardized.mp4")
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(video_path):
        print(f"Using existing standardized video: {out_path}")
        return out_path

    height = min(STANDARD_MAX_HEIGHT, info["height"])
    height -= height % 2
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-an",
        "-vf", f"fps={STANDARD_FPS},scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", out_path,
    ]
    print("Standardizing video with FFmpeg...")
    subprocess.run(command, check=True)
    return out_path


def extract_frames(video_path, output_dir, every_n_seconds=FRAME_EVERY_SECONDS):
    """Extracts one frame every N seconds and records each frame's time in the lecture."""
    for old in glob.glob(os.path.join(output_dir, "frame_*.jpg")):
        os.remove(old)  # frames from a previous run would otherwise be mixed into this one

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(fps * every_n_seconds))
    rows, frame_idx = [], 0

    while cap.grab():
        if frame_idx % frame_interval == 0:
            ok, frame = cap.retrieve()
            if ok:
                path = os.path.join(output_dir, f"frame_{len(rows):04d}.jpg")
                cv2.imwrite(path, frame)
                rows.append({"frame": len(rows), "time_s": frame_idx / fps, "path": path})
        frame_idx += 1

    cap.release()
    frames_df = pd.DataFrame(rows)
    frames_df.to_csv(os.path.join(TABLE_DIR, "frames.csv"), index=False)
    print(f"Extracted {len(frames_df)} frames to {output_dir}")
    return frames_df


# ---------------------------------------------------------------- stage 3: detection + pose
def load_model(weights, required=True):
    """Loads local YOLO weights. Never downloads: ultralytics would fetch missing weights automatically."""
    if os.path.exists(weights):
        return YOLO(weights)
    if required:
        raise FileNotFoundError(f"Model weights '{weights}' not found in {os.getcwd()}. Place the file there and run again.")
    print(f"WARNING: '{weights}' not found - running without pose cues (hand raise, head posture, facing).")
    return None


def inside(cx, cy, region):
    """True if the point (fractions of the frame) lies inside the [x1, y1, x2, y2] region."""
    x1, y1, x2, y2 = region
    return x1 <= cx <= x2 and y1 <= cy <= y2


def seat_for(cx, cy, calib):
    """Maps a position inside the student region to an anonymous seat ID (seat-based tracking)."""
    x1, y1, x2, y2 = calib["student_region"]
    rows, cols = calib["seat_grid"]
    col = min(cols - 1, max(0, int((cx - x1) / (x2 - x1) * cols)))
    row = min(rows - 1, max(0, int((cy - y1) / (y2 - y1) * rows)))
    return f"seat_{row * cols + col + 1:02d}"


def pose_cues(kxy, kconf, box_h):
    """Reads posture and gesture cues from one person's 17 COCO keypoints (NaN where a point is not visible)."""
    def ok(i):
        return kconf[i] >= KP_CONF

    hand_up = any(
        ok(w) and ok(s) and kxy[w][1] < kxy[s][1] - HAND_RAISE_MARGIN * box_h
        for w, s in ((9, 5), (10, 6))  # (wrist, shoulder) pairs: left, right
    )
    head_down = head_up = False
    face_dx = np.nan
    if ok(0) and ok(5) and ok(6):
        shoulder_w = max(abs(kxy[5][0] - kxy[6][0]), 1e-6)
        head_down = (((kxy[5][1] + kxy[6][1]) / 2) - kxy[0][1]) < HEAD_DOWN_RATIO * shoulder_w
        head_up = not head_down
        face_dx = (kxy[0][0] - (kxy[5][0] + kxy[6][0]) / 2) / shoulder_w  # nose offset in shoulder-widths

    # seated (1.0) or standing (0.0) from how long the thigh looks beside the torso; NaN when hips or knees are hidden
    seated = np.nan
    if all(ok(i) for i in (5, 6, 11, 12, 13, 14)):
        torso = abs((kxy[11][1] + kxy[12][1]) / 2 - (kxy[5][1] + kxy[6][1]) / 2)
        thigh = abs((kxy[13][1] + kxy[14][1]) / 2 - (kxy[11][1] + kxy[12][1]) / 2)
        if torso > 1e-6:
            seated = float(thigh / torso < SEATED_THIGH_RATIO)

    # gesture reach: how far the wrists get from the shoulders, in shoulder-widths
    gesture = np.nan
    if ok(5) and ok(6):
        shoulder_w = max(abs(kxy[5][0] - kxy[6][0]), 1e-6)
        reach = [np.hypot(*(kxy[w] - kxy[s])) / shoulder_w for w, s in ((9, 5), (10, 6)) if ok(w) and ok(s)]
        if reach:
            gesture = float(max(reach))
    return {"hand_up": bool(hand_up), "head_down": bool(head_down), "head_up": bool(head_up), "face_dx": face_dx,
            "seated": seated, "gesture": gesture}


def predict_in_chunks(model, paths, **kwargs):
    """Yields YOLO results frame by frame, calling predict() on small chunks so memory stays flat."""
    for start in range(0, len(paths), PREDICT_CHUNK):
        yield from model.predict(source=paths[start:start + PREDICT_CHUNK], stream=True, conf=CONF, imgsz=IMGSZ,
                                 device=DEVICE, verbose=False, **kwargs)


def detect_people(frames_df, pose_model, det_model):
    """Detects every person (with keypoints when a pose model is available) in each sampled frame."""
    paths = frames_df["path"].tolist()
    if pose_model is not None:
        stream = predict_in_chunks(pose_model, paths)
    else:
        stream = predict_in_chunks(det_model, paths, classes=[0])

    rows = []
    for i, r in enumerate(stream):
        h, w = r.orig_shape
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kxy = kconf = None
        if pose_model is not None and r.keypoints is not None and r.keypoints.conf is not None:
            kxy, kconf = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            cues = {"hand_up": False, "head_down": False, "head_up": False, "face_dx": np.nan,
                    "seated": np.nan, "gesture": np.nan}
            if kxy is not None:
                cues = pose_cues(kxy[j], kconf[j], y2 - y1)
            rows.append({
                "frame": i, "time_s": frames_df.loc[i, "time_s"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h, "conf": float(confs[j]), **cues,
                "phone": False, "book": False, "role": "other", "seat": "",
            })
    return pd.DataFrame(rows)


def detect_objects(frames_df, det_model):
    """Detects phones and books (COCO classes) used as student-behaviour cues."""
    stream = predict_in_chunks(det_model, frames_df["path"].tolist(), classes=list(COCO_OBJECTS))
    rows = []
    for i, r in enumerate(stream):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            rows.append({"frame": i, "label": COCO_OBJECTS[int(c)], "px": (x1 + x2) / 2, "py": (y1 + y2) / 2})
    return pd.DataFrame(rows, columns=["frame", "label", "px", "py"])


def assign_roles(people, objects, calib):
    """Assigns instructor / student roles and seat IDs from the calibration zones, then attaches object cues."""
    for _, grp in people.groupby("frame"):
        in_teacher = [inside(r.cx, r.cy, calib["teacher_zone"]) for r in grp.itertuples()]
        candidates = grp[in_teacher]
        if len(candidates) > 0:
            people.loc[candidates["conf"].idxmax(), "role"] = "instructor"  # one instructor timeline (FR3.1)
        for r in grp.itertuples():
            if people.loc[r.Index, "role"] != "instructor" and inside(r.cx, r.cy, calib["student_region"]):
                people.loc[r.Index, "role"] = "student"
                people.loc[r.Index, "seat"] = seat_for(r.cx, r.cy, calib)

    # a phone / book belongs to the smallest student box that contains its centre
    students = people[people["role"] == "student"]
    for o in objects.itertuples():
        grp = students[students["frame"] == o.frame]
        hits = grp[(grp["x1"] <= o.px) & (o.px <= grp["x2"]) & (grp["y1"] <= o.py) & (o.py <= grp["y2"])]
        if len(hits) > 0:
            area = (hits["x2"] - hits["x1"]) * (hits["y2"] - hits["y1"])
            people.loc[area.idxmin(), o.label] = True
    return people


# ---------------------------------------------------------------- stage 4: behaviour recognition -> timeline
def run_segments(mask, min_len=1):
    """(start, end) index pairs, inclusive, for each run of True values that is at least min_len long."""
    segments, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segments.append((start, i - 1))
            start = None
    return segments


def sustained(mask, min_len):
    """Keeps only the runs of True that last at least min_len sampled frames."""
    out = np.zeros(len(mask), dtype=bool)
    for s, e in run_segments(mask, min_len):
        out[s:e + 1] = True
    return out


def build_timeline(frames_df, people, calib, use_pose):
    """Turns per-frame detections into one instructor state timeline and per-seat student state timelines."""
    n = len(frames_df)
    tl = {"times": frames_df["time_s"].to_numpy(), "n": n}

    # instructor with pose: absent / board_writing / interacting / explaining / other (present, not facing students)
    # instructor without pose (only the position is known): absent / at_board / present
    cx, cy, conf = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    state = np.array(["absent"] * n, dtype=object)
    zone = np.array([""] * n, dtype=object)
    seated_i, gesture_i, dist_i = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    hand_i = np.zeros(n, dtype=bool)
    students = people[people["role"] == "student"]
    student_centres = {f: g[["cx", "cy"]].to_numpy() for f, g in students.groupby("frame")}
    students_x = (calib["student_region"][0] + calib["student_region"][2]) / 2
    for r in people[people["role"] == "instructor"].itertuples():
        i = r.frame
        cx[i], cy[i], conf[i] = r.cx, r.cy, r.conf
        seated_i[i], gesture_i[i], hand_i[i] = r.seated, r.gesture, r.hand_up
        if i in student_centres:   # distance to the nearest student, in frame-widths
            dist_i[i] = float(np.hypot(student_centres[i][:, 0] - r.cx, student_centres[i][:, 1] - r.cy).min())
        at_board = inside(r.cx, r.cy, calib["board_region"])
        toward = 1 if students_x > r.cx else -1
        facing = use_pose and not np.isnan(r.face_dx) and r.face_dx * toward > FACING_MARGIN
        if not use_pose:
            state[i] = "at_board" if at_board else "present"
        elif at_board and r.hand_up:
            state[i] = "board_writing"
        elif facing and r.hand_up:
            state[i] = "interacting"          # facing the students and gesturing
        elif facing:
            state[i] = "explaining"           # facing the students, not writing on the board
        else:
            state[i] = "other"
        zone[i] = "board" if at_board else "teacher_zone"
    tl.update(ins_cx=cx, ins_cy=cy, ins_conf=conf, ins_state=state, ins_zone=zone,
              ins_seated=seated_i, ins_gesture=gesture_i, ins_dist=dist_i, ins_hand=hand_i)

    # students: one timeline per anonymous seat
    seats = sorted(students["seat"].unique())
    shape = (len(seats), n)
    present, hand, down, up = (np.zeros(shape, dtype=bool) for _ in range(4))
    phone, book, standing = (np.zeros(shape, dtype=bool) for _ in range(3))
    sconf = np.zeros(shape)
    seat_index = {s: k for k, s in enumerate(seats)}
    for r in students.itertuples():
        k, i = seat_index[r.seat], r.frame
        present[k, i] = True
        hand[k, i] |= r.hand_up
        down[k, i] |= r.head_down
        up[k, i] |= r.head_up
        phone[k, i] |= r.phone
        book[k, i] |= r.book
        standing[k, i] |= (r.seated == 0.0)   # NaN (hidden legs) is not standing
        sconf[k, i] = max(sconf[k, i], r.conf)

    hand_s, off_task = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for k in range(len(seats)):
        hand_s[k] = sustained(hand[k], HAND_RAISE_MIN_FRAMES)
        off_task[k] = sustained(down[k] & ~book[k], OFFTASK_MIN_FRAMES) | phone[k]
    focused = up & ~off_task
    tl.update(seats=seats, present=present, hand=hand_s, phone=phone, book=book, off_task=off_task,
              focused=focused, standing=standing, sconf=sconf)
    return tl


def make_events(tl):
    """Builds the timestamped event table: label, start, end, confidence, entity ID (FR3.3, FR4.3)."""
    times, step = tl["times"], FRAME_EVERY_SECONDS
    rows = []

    def add(entity, entity_id, label, mask, conf):
        for s, e in run_segments(mask):
            rows.append({"entity": entity, "entity_id": entity_id, "label": label, "start_s": times[s],
                         "end_s": times[e] + step, "duration_s": times[e] + step - times[s],
                         "confidence": float(np.nanmean(conf[s:e + 1]))})

    for label in ("explaining", "board_writing", "interacting", "at_board", "present"):
        add("instructor", "instructor", label, tl["ins_state"] == label, tl["ins_conf"])
    for k, seat in enumerate(tl["seats"]):
        for label, mask in (("hand_raise", tl["hand"][k]), ("phone_use", tl["phone"][k]), ("reading", tl["book"][k]),
                            ("off_task", tl["off_task"][k]), ("focused", tl["focused"][k])):
            add("student", seat, label, mask, tl["sconf"][k])

    columns = ["entity", "entity_id", "label", "start_s", "end_s", "duration_s", "confidence"]
    events = pd.DataFrame(rows, columns=columns).sort_values("start_s").reset_index(drop=True)
    events["start_window"] = (events["start_s"] // WINDOW_SECONDS).astype(int)
    return events


def nanmean_or_nan(values):
    """Mean of the values that exist, or NaN when none do (no empty-slice warning)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


# ---------------------------------------------------------------- stage 5: windows + indicators (FR5, FR6.1-6.2)
def build_window_features(tl, events, use_pose):
    """Aligns instructor and student outputs on one timeline and computes features per fixed window."""
    win = (tl["times"] // WINDOW_SECONDS).astype(int)
    lecture_end = float(tl["times"][-1]) + FRAME_EVERY_SECONDS   # the last window is shorter than WINDOW_SECONDS
    counts = events.groupby(["label", "start_window"]).size()
    cx, cy = tl["ins_cx"], tl["ins_cy"]
    rows = []
    for w in range(int(win.max()) + 1):
        idx = np.where(win == w)[0]
        if len(idx) == 0:
            continue
        steps = [float(np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]))
                 for i in idx if i > 0 and not np.isnan(cx[i]) and not np.isnan(cx[i - 1])]
        move = sum(steps)
        visible = tl["ins_state"][idx] != "absent"
        observed = tl["present"][:, idx].sum()
        focused, off = tl["focused"][:, idx].sum(), tl["off_task"][:, idx].sum()
        row = {
            "window_id": w, "start_s": w * WINDOW_SECONDS, "end_s": min((w + 1) * WINDOW_SECONDS, lecture_end), "n_frames": len(idx),
            "instructor_present": float(np.mean(tl["ins_state"][idx] != "absent")),
            "explanation_ratio": float(np.mean(tl["ins_state"][idx] == "explaining")) if use_pose else np.nan,
            "board_writing_ratio": float(np.mean(np.isin(tl["ins_state"][idx], ["board_writing", "at_board"]))),
            "movement_amount": move,
            "teacher_stationary_ratio": float(np.mean(np.array(steps) < STATIONARY_MOVE)) if steps else np.nan,
            "teacher_seated_ratio": nanmean_or_nan(tl["ins_seated"][idx]),
            "teacher_posture_readable": float(np.mean(~np.isnan(tl["ins_seated"][idx][visible]))) if visible.any() else np.nan,
            "teacher_gesture": nanmean_or_nan(tl["ins_gesture"][idx]),
            "teacher_hand_up_ratio": float(np.mean(tl["ins_hand"][idx])) if use_pose else np.nan,
            "teacher_student_distance": nanmean_or_nan(tl["ins_dist"][idx]),
            "students_standing": tl["standing"][:, idx].sum() / len(idx),
            "minutes": (w * WINDOW_SECONDS + WINDOW_SECONDS / 2) / 60,
            "interaction_count": int(counts.get(("interacting", w), 0)) if use_pose else np.nan,
            "seats_observed": observed / len(idx),
            "attention_level": focused / observed if observed and use_pose else np.nan,
            "off_task_rate": off / observed if observed and use_pose else np.nan,
            "pose_coverage": (focused + off) / observed if observed else np.nan,
            "hand_raise_count": int(counts.get(("hand_raise", w), 0)) if use_pose else np.nan,
            "phone_use_count": int(counts.get(("phone_use", w), 0)),
            "reading_count": int(counts.get(("reading", w), 0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 6: fusion / lag analysis (FR6.3, FR6.4)
def shifted(windows, columns, lag):
    """Shifts window features back in time, never across the boundary between two lectures."""
    if "session_id" in windows.columns:
        return windows.groupby("session_id")[columns].shift(lag)
    return windows[columns].shift(lag)


def analyze_lags(windows):
    """Checks whether student indicators change after teacher actions (cross-correlation + lagged OLS)."""
    lag_rows, model_rows = [], []
    # when several lectures are pooled, each lecture gets its own baseline (a dummy column per extra lecture)
    lecture_dummies = (pd.get_dummies(windows["session_id"], prefix="lecture", drop_first=True, dtype=float)
                       if "session_id" in windows.columns else pd.DataFrame(index=windows.index))
    for feature in TEACHER_FEATURES:
        for outcome in STUDENT_OUTCOMES:
            for lag in range(1, MAX_LAG_WINDOWS + 1):
                pair = pd.concat([shifted(windows, feature, lag), windows[outcome]], axis=1).dropna()
                if len(pair) < MIN_WINDOWS_FOR_STATS or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
                    continue
                if stats is not None:
                    r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                else:
                    r, p = pair.iloc[:, 0].corr(pair.iloc[:, 1]), np.nan
                # neighbouring windows are alike, so also test the feature after controlling for the outcome's previous value
                p_controlled, coef_controlled = np.nan, np.nan
                if sm is not None:
                    ctrl = pd.concat([shifted(windows, feature, lag).rename("x"), shifted(windows, outcome, 1).rename("prev"),
                                      windows[outcome].rename("y"), lecture_dummies], axis=1).dropna()
                    if len(ctrl) >= MIN_WINDOWS_FOR_STATS and ctrl["x"].std() > 0 and ctrl["prev"].std() > 0:
                        fit = sm.OLS(ctrl["y"], sm.add_constant(ctrl[["x", "prev", *lecture_dummies.columns]])).fit()
                        p_controlled, coef_controlled = float(fit.pvalues["x"]), float(fit.params["x"])
                lag_rows.append({"teacher_feature": feature, "student_outcome": outcome, "lag_windows": lag,
                                 "lag_seconds": lag * WINDOW_SECONDS, "correlation": float(r),
                                 "p_value": float(p), "p_controlled": p_controlled, "coef_controlled": coef_controlled,
                                 "n_windows": len(pair), "prespecified": (feature, outcome) in PRESPECIFIED})

    if sm is not None:  # Y_t = a + b1*E_{t-1} + b2*W_{t-1} + b3*I_{t-1} + b4*Y_{t-1} + b5*minutes (time into the lecture)
        for outcome in STUDENT_OUTCOMES:
            data = pd.concat([shifted(windows, MODEL_PREDICTORS, 1), shifted(windows, outcome, 1).rename("previous_outcome"),
                              windows[["minutes"]], lecture_dummies, windows[outcome]], axis=1).dropna()
            predictors = [c for c in data.columns[:-1] if data[c].std() > 0]
            if len(data) < MIN_WINDOWS_FOR_STATS or not predictors:
                continue
            fit = sm.OLS(data[outcome], sm.add_constant(data[predictors])).fit()
            for term in predictors:
                model_rows.append({"student_outcome": outcome, "term": term, "coefficient": float(fit.params[term]),
                                   "p_value": float(fit.pvalues[term]), "r_squared": float(fit.rsquared),
                                   "n_windows": int(fit.nobs)})
    return pd.DataFrame(lag_rows), pd.DataFrame(model_rows)


# ---------------------------------------------------------------- stage 7: reporting (FR7)
def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def finite(value):
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value


def describe_lag(row, n_tests):
    """One lag finding as text. The p-value controls for the outcome's own previous value (neighbouring windows are
    alike) when it can, and is judged against a Bonferroni threshold for all n_tests combinations."""
    controlled = bool(np.isfinite(row.p_controlled))
    p = row.p_controlled if controlled else row.p_value
    if np.isfinite(p):
        verdict = "survives" if p < 0.05 / n_tests else "does not survive"
        how = "controlling for the previous window" if controlled else "uncontrolled, so too optimistic"
        p_text = f"{'p < 0.001' if p < 0.001 else f'p = {p:.3f}'} {how}, {verdict} correction for {n_tests} tests"
    else:
        p_text = "p-value unavailable (scipy not installed)"
    return (f"{row.teacher_feature} -> {row.student_outcome} after {row.lag_seconds} s: "
            f"r = {row.correlation:+.2f} ({p_text}; {row.n_windows} windows). Correlation, not proof of causation.")


def build_linkage_lines(lag_df, model_df):
    """The lag-analysis findings as report lines (FR6.4): lag value, compared variables and result."""
    linkage_lines = []
    if lag_df.empty:
        linkage_lines.append(f"Not enough valid windows (need at least {MIN_WINDOWS_FOR_STATS} with variation) for lag analysis.")
    else:
        planned, other = lag_df[lag_df["prespecified"]], lag_df[~lag_df["prespecified"]]
        n_planned = len(PRESPECIFIED) * MAX_LAG_WINDOWS
        linkage_lines.append(f"Hypotheses fixed in advance ({n_planned} tests, so p must be below {0.05 / n_planned:.4f}):")
        for feature, outcome in PRESPECIFIED:
            sub = planned[(planned["teacher_feature"] == feature) & (planned["student_outcome"] == outcome)]
            if sub.empty:
                linkage_lines.append(f"{feature} -> {outcome}: not testable in this lecture (too few windows with readable data, or no variation).")
            else:
                linkage_lines.append(describe_lag(sub.loc[sub["correlation"].abs().idxmax()], n_planned))
        if not other.empty:
            linkage_lines.append(f"Exploratory ({len(other)} tests; treat these as leads, not findings):")
            linkage_lines.append(describe_lag(other.loc[other["correlation"].idxmax()], len(other)).replace("->", "-> strongest positive:", 1))
            linkage_lines.append(describe_lag(other.loc[other["correlation"].idxmin()], len(other)).replace("->", "-> strongest negative:", 1))
    if sm is None:
        linkage_lines.append("statsmodels is not installed: the lagged regression model was skipped.")
    elif not model_df.empty:
        for outcome, grp in model_df.groupby("student_outcome"):
            grp = grp[~grp["term"].str.startswith("lecture_")]   # lecture baselines are not findings
            sig = grp[grp["p_value"] < 0.05]
            terms = ", ".join(f"{t.term} ({t.coefficient:+.2f})" for t in sig.itertuples()) or "no predictor reached p < 0.05"
            linkage_lines.append(f"Lagged model for {outcome} (R2 = {grp['r_squared'].iloc[0]:.2f}, n = {int(grp['n_windows'].iloc[0])}): {terms}.")
    if sm is None or stats is None:
        linkage_lines.append("Install scipy and statsmodels for p-values and the lagged regression (report section 4.1).")
    return linkage_lines


def instructor_phrase(w):
    """Plain-language description of what the instructor was doing in a window (or the mean of several windows)."""
    if w["instructor_present"] < 0.5:
        return "the instructor was mostly out of view"
    if w["explanation_ratio"] >= 0.5:
        return "the instructor was explaining to the class"
    if w["board_writing_ratio"] >= 0.3:
        return "the instructor was at the board"
    return "the instructor was not facing the class"


def dominant_activity(rows, reference):
    """Most common instructor activity in rows, and how common that same activity is across the reference windows."""
    phrases = [instructor_phrase(r) for _, r in rows.iterrows()]
    if not phrases:
        return None, 0.0
    top = Counter(phrases).most_common(1)[0][0]
    return top, float(np.mean([instructor_phrase(r) == top for _, r in reference.iterrows()]))


def pct(value):
    """A fraction as the whole-number percentage the report prints (so facts and sentences always agree)."""
    return int(f"{100 * value:.0f}")


def build_feedback_points(windows, events, use_pose):
    """Builds positive / negative points and the structured facts behind them. Each ties student focus to what the
    instructor was doing at the same time.

    The points describe what happened together, not what caused it: focus is compared between windows, never
    attributed to a single action. Returns (positive, negative, suggestions, evidence, facts); `facts` holds every
    candidate (not only the 3 printed per list) with its numbers, direction and evidence strength, for the LLM step."""
    suggestions, evidence, candidates = [], [], []
    count = lambda label: int((events["label"] == label).sum())
    window_of = lambda t: windows[windows["window_id"] == int(t // WINDOW_SECONDS)]
    phone_events = events[events["label"] == "phone_use"]

    def add(direction, text, fact):
        candidates.append((direction, text, fact))

    if use_pose:
        usable = ((windows["pose_coverage"] >= MIN_POSE_COVERAGE) & (windows["seats_observed"] >= MIN_SEATS_OBSERVED)
                  & windows["attention_level"].notna())
        valid = windows[usable]

        # 1) focus in windows with a given instructor behaviour, against all the other usable windows
        moving_cut = valid["movement_amount"].quantile(0.75)
        near_cut = valid["teacher_student_distance"].quantile(0.25)
        behaviours = [
            ("the instructor was explaining to the class", valid["explanation_ratio"] >= 0.6),
            ("the instructor was moving around", (valid["movement_amount"] >= moving_cut) & (moving_cut > 0)),
            ("the instructor was seated", valid["teacher_seated_ratio"] >= 0.5),
            ("the instructor was close to the students", valid["teacher_student_distance"] <= near_cut),
        ]
        for phrase, mask in behaviours:
            group, rest = valid[mask], valid[~mask]
            if len(group) < MIN_GROUP_WINDOWS or len(rest) < MIN_GROUP_WINDOWS:
                continue
            g, r = group["attention_level"].mean(), rest["attention_level"].mean()
            if abs(g - r) < MIN_ATTENTION_GAP:
                continue
            text = (f"In the {len(group)} windows ({len(group) * WINDOW_SECONDS} s in total) where {phrase}, about {100 * g:.0f}% of "
                    f"the observable students were focused, compared with {100 * r:.0f}% in the other windows.")
            strength, p_gap, p_text = "unknown", None, None
            if sm is not None:  # how much weight the gap deserves, allowing for neighbouring windows being alike (HAC errors)
                fit = sm.OLS(valid["attention_level"], sm.add_constant(mask.astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
                p_gap = float(fit.pvalues.iloc[-1])
            elif stats is not None:  # without statsmodels: a plain t-test, which overstates the certainty
                p_gap = float(stats.ttest_ind(group["attention_level"], rest["attention_level"], equal_var=False).pvalue)
            if p_gap is not None:
                p_text = "< 0.01" if p_gap < 0.01 else f"= {p_gap:.2f}"
                strength = "weak" if p_gap >= 0.05 else "suggestive"
                note = "allowing for neighbouring windows being alike" if sm is not None else "plain test; overstates the certainty"
                text += ((" This gap could easily be chance" if p_gap >= 0.05 else " This gap is unlikely to be pure chance")
                         + f" (p {p_text}, {note}).")
            add("favourable" if g > r else "unfavourable", text, {
                "kind": "comparison", "instructor_activity": phrase,
                "windows": len(group), "seconds": len(group) * WINDOW_SECONDS, "students_focused_pct": pct(g),
                "other_windows": len(rest), "other_windows_focused_pct": pct(r), "gap_points": abs(pct(g) - pct(r)),
                "evidence": {"strength": strength, "test_p": p_text,
                             "notes": ["The test allows for neighbouring windows being alike." if sm is not None
                                       else "Plain test; neighbouring windows are alike, so it overstates the certainty."]}})
            if g <= r:
                suggestions.append("Look at the lower-focus stretches and consider a short check-in question there.")

        # 2) the most and least focused 30-second stretch, only where the instructor was in view
        visible = usable & (windows["instructor_present"] >= 0.5)
        rolling = windows["attention_level"].where(visible).rolling(3, min_periods=3).mean()
        if rolling.notna().any() and rolling.max() - rolling.min() >= 2 * MIN_ATTENTION_GAP:
            for kind, end in (("highest", rolling.idxmax()), ("lowest", rolling.idxmin())):
                stretch = windows.iloc[end - 2:end + 1]
                span = f"{fmt_time(stretch['start_s'].iloc[0])}-{fmt_time(stretch['end_s'].iloc[-1])}"
                attention = 100 * stretch["attention_level"].mean()
                phrase = instructor_phrase(stretch[["instructor_present", "explanation_ratio", "board_writing_ratio"]].mean())
                fact = {"kind": f"{kind}_focus_stretch", "span": span, "stretch_seconds": len(stretch) * WINDOW_SECONDS,
                        "students_focused_pct": int(f"{attention:.0f}"), "instructor_activity": phrase,
                        "evidence": {"strength": "descriptive", "notes": ["A single 30-second stretch is noisy."]}}
                if kind == "highest":
                    add("favourable", f"Focus was highest at {span}: about {attention:.0f}% of the observable students were focused while {phrase}.", fact)
                else:
                    add("unfavourable", f"Focus was lowest at {span}: only about {attention:.0f}% of the observable students were focused while {phrase}.", fact)
                    suggestions.append(f"Review the video around {span} to see what happened.")
                    evidence.append(f"Lowest focus: {span} ({attention:.0f}% of observable students focused).")

        # 3) hand raises: participation, with what the instructor was doing at those moments
        hands = events[events["label"] == "hand_raise"]
        if len(hands):
            hand_windows = pd.concat([window_of(t) for t in hands["start_s"]])
            top, base = dominant_activity(hand_windows, windows)
            examples = [fmt_time(t) for t in hands["start_s"].head(3)]
            add("favourable",
                f"Students took part: {len(hands)} hand raise{'s were' if len(hands) != 1 else ' was'} detected "
                f"(for example at {', '.join(examples)}), most often while {top} (which describes {100 * base:.0f}% of all windows).",
                {"kind": "hand_raises", "count": int(len(hands)), "example_times": examples, "most_common_instructor_activity": top,
                 "that_activity_share_of_all_windows_pct": pct(base),
                 "evidence": {"strength": "descriptive",
                              "notes": ["Detections have not been checked against human labels.",
                                        "If the camera moves, the instructor can be mislabelled as a student and gestures counted as hand raises."]}})
            for e in hands.head(5).itertuples():
                evidence.append(f"Hand raise: {e.entity_id}, {fmt_time(e.start_s)}-{fmt_time(e.end_s)}.")

        # 3b) windows where most observable students were focused, and what the instructor was doing in them
        focused_windows = valid[(valid["attention_level"] >= 0.6) & (valid["instructor_present"] >= 0.5)]
        if len(focused_windows) >= MIN_GROUP_WINDOWS:
            top, base = dominant_activity(focused_windows, valid)
            add("favourable",
                f"In {len(focused_windows)} of {len(valid)} usable windows ({len(focused_windows) * WINDOW_SECONDS} s) at least 60% of the "
                f"observable students were focused, most often while {top} (which describes {100 * base:.0f}% of all usable windows).",
                {"kind": "mostly_focused_windows", "windows": int(len(focused_windows)), "usable_windows": int(len(valid)),
                 "seconds": int(len(focused_windows) * WINDOW_SECONDS), "most_common_instructor_activity": top,
                 "that_activity_share_of_usable_windows_pct": pct(base),
                 "evidence": {"strength": "descriptive", "notes": ["Counts of windows above a fixed 60% line."]}})

        # 4) a long stretch of explanation with no student hand raise (and no interaction, when that rule ever fires)
        has_interaction = count("interacting") > 0
        quiet = (windows["explanation_ratio"] >= 0.6) & (windows["hand_raise_count"] == 0)
        if has_interaction:
            quiet &= windows["interaction_count"] == 0
        what = "interaction or hand raise" if has_interaction else "student hand raise"
        stretch = max(run_segments(quiet.to_numpy()), key=lambda s: s[1] - s[0], default=None)
        if stretch and stretch[1] - stretch[0] + 1 >= LONG_STRETCH_WINDOWS:
            span = f"{fmt_time(windows.loc[stretch[0], 'start_s'])}-{fmt_time(windows.loc[stretch[1], 'end_s'])}"
            n_win = stretch[1] - stretch[0] + 1
            add("unfavourable", f"From {span} ({n_win} windows) the instructor kept explaining to the class and there was no {what}.",
                {"kind": "no_participation_stretch", "span": span, "windows": int(n_win), "missing": what,
                 "instructor_activity": "the instructor was explaining to the class",
                 "evidence": {"strength": "descriptive",
                              "notes": ["Hand raises that the detector missed would make this stretch look quieter than it was."]}})
            suggestions.append("Break long explanations with a question or short activity.")
            evidence.append(f"Stretch with no {what}: {span}.")

    # 5) phone use
    if len(phone_events):
        e = phone_events.iloc[0]
        w = window_of(e["start_s"])
        activity = instructor_phrase(w.iloc[0]) if len(w) else None
        phrase = f", while {activity}" if activity else ""
        more = f" (and {len(phone_events) - 1} more)" if len(phone_events) > 1 else ""
        add("unfavourable", f"Phone use was detected at {e['entity_id']} around {fmt_time(e['start_s'])}{phrase}{more}.",
            {"kind": "phone_use", "count": int(len(phone_events)), "first_seat": e["entity_id"], "first_time": fmt_time(e["start_s"]),
             "instructor_activity": activity,
             "evidence": {"strength": "descriptive", "notes": ["A phone is a small object, and this detection has not been checked against human labels."]}})
        suggestions.append("Check the flagged segments for off-task phone use.")
        for e in phone_events.head(5).itertuples():
            evidence.append(f"Phone use: {e.entity_id}, {fmt_time(e.start_s)}-{fmt_time(e.end_s)}.")

    facts = [{"id": f"F{n}", "direction": direction, **fact} for n, (direction, _, fact) in enumerate(candidates, 1)]
    positive = [t for d, t, _ in candidates if d == "favourable"][:3]
    negative = [t for d, t, _ in candidates if d == "unfavourable"][:3]
    if not use_pose:
        positive = ["Not available: the pose model was not loaded, so student focus could not be measured."]
    return (positive or ["No clear positive point was found in this analysis."],
            negative or ["No clear negative point was found in this analysis."],
            list(dict.fromkeys(suggestions)), evidence, facts)


def generate_and_save_report(info, video_name, people, tl, events, windows, lag_df, model_df, use_pose):
    """Computes lecture-level indicators, applies the findings rules and renders the Jinja report template."""
    step = FRAME_EVERY_SECONDS
    n_frames = tl["n"]
    lecture_s = n_frames * step
    count = lambda label: int((events["label"] == label).sum())
    plural = lambda n: f"{n} event" + ("" if n == 1 else "s")
    seconds = lambda state: float(np.sum(tl["ins_state"] == state)) * step
    per_frame = people.groupby("frame").size().reindex(range(n_frames), fill_value=0)
    students_per_frame = float(people[people["role"] == "student"].groupby("frame").size()
                               .reindex(range(n_frames), fill_value=0).mean())
    seats_per_frame = windows["seats_observed"].mean()
    present_pct = 100 * float(np.mean(tl["ins_state"] != "absent"))
    board_state = "board_writing" if use_pose else "at_board"

    path_length = float(np.nansum(np.hypot(np.diff(tl["ins_cx"]), np.diff(tl["ins_cy"]))))
    zones = [z for z in tl["ins_zone"] if z]
    zone_changes = sum(1 for a, b in zip(zones, zones[1:]) if a != b)

    mean_attention = windows["attention_level"].mean()
    mean_off_task = windows["off_task_rate"].mean()
    coverage = windows["pose_coverage"].mean()

    instructor_lines = [f"Instructor visible in {present_pct:.0f}% of analysed frames."]
    if use_pose:
        instructor_lines += [
            f"Visible explanation time (facing the students, not writing on the board): {fmt_time(seconds('explaining'))} "
            f"({100 * seconds('explaining') / lecture_s:.0f}% of the lecture).",
            f"Board writing: {fmt_time(seconds('board_writing'))} ({100 * seconds('board_writing') / lecture_s:.0f}%).",
            f"Interaction (facing the students and gesturing): {plural(count('interacting'))}, {fmt_time(seconds('interacting'))} in total."
            + ("" if count("interacting") else " The rule never triggered, so the absence of interaction is not established."),
        ]
    else:
        instructor_lines += [
            f"Present and away from the board: {fmt_time(seconds('present'))} ({100 * seconds('present') / lecture_s:.0f}%). "
            "Explanation time is unavailable: facing the students needs the pose model.",
            f"Time at the board: {fmt_time(seconds('at_board'))} ({100 * seconds('at_board') / lecture_s:.0f}%). "
            "Board writing could not be confirmed without pose.",
            "Interaction: unavailable (needs the pose model).",
        ]
    instructor_lines.append(f"Movement: path length {path_length:.2f} frame-widths, {zone_changes} changes between the board and the teacher zone.")

    # catalog indicators: posture, gesture reach, distance to students, stationary share (each with its evidence coverage)
    visible_n = int((tl["ins_state"] != "absent").sum())
    seated_known = int((~np.isnan(tl["ins_seated"])).sum())
    seated_share = nanmean_or_nan(tl["ins_seated"])
    gesture_reach = nanmean_or_nan(tl["ins_gesture"])
    student_distance = nanmean_or_nan(tl["ins_dist"])
    stationary_share = windows["teacher_stationary_ratio"].mean()
    if use_pose:
        if seated_known:
            instructor_lines.append(f"Posture was readable in {100 * seated_known / max(visible_n, 1):.0f}% of the frames where the instructor was visible; "
                                    f"seated in {100 * seated_share:.0f}% of those.")
        else:
            instructor_lines.append("Sitting or standing could not be read (hips and knees were never in view), so no sitting comparison is possible here.")
        if not np.isnan(gesture_reach):
            instructor_lines.append(f"Gesture reach (wrist distance from the shoulder, in shoulder-widths): {gesture_reach:.2f} on average.")
    if not np.isnan(student_distance):
        instructor_lines.append(f"Average distance from the instructor to the nearest student: {student_distance:.2f} frame-widths.")
    if not np.isnan(stationary_share):
        instructor_lines.append(f"The instructor barely moved between samples in {100 * stationary_share:.0f}% of the sample pairs.")

    student_lines = [f"Average students detected per frame: {students_per_frame:.1f}, in {seats_per_frame:.1f} of the "
                     f"{len(tl['seats'])} seats ever occupied (maximum {int(per_frame.max())} people per frame including the instructor)."]
    if use_pose:
        student_lines += [
            f"Attention level (focused seat-frames / seats observed): {100 * mean_attention:.0f}% on average.",
            f"Off-task rate (sustained head-down without a book, or a phone): {100 * mean_off_task:.0f}% on average.",
            f"Hand raises: {plural(count('hand_raise'))}. Pose state was readable for {100 * coverage:.0f}% of seat-frames.",
        ]
    else:
        student_lines.append("Pose model not loaded: attention, off-task and hand-raise indicators are unavailable.")
    student_lines.append(f"Phone use: {plural(count('phone_use'))}. Reading (book detected): {plural(count('reading'))}.")
    if use_pose:
        seen_enough = tl["present"].sum(axis=1) >= 0.5 * n_frames          # seats in view for at least half of the frames
        raised = tl["hand"].any(axis=1)
        if seen_enough.any():
            student_lines.append(f"Hand-raise participation: {int(raised[seen_enough].sum())} of the {int(seen_enough.sum())} seats seen in at least "
                                 "half of the frames had a hand raise.")
        student_lines.append(f"Students read as standing: {windows['students_standing'].mean():.2f} per frame on average "
                             "(legs are often hidden behind desks, so this undercounts).")

    linkage_lines = build_linkage_lines(lag_df, model_df)

    positive, negative, suggestions, evidence, facts = build_feedback_points(windows, events, use_pose)

    limitations = [
        "Behaviour labels come from rule-based heuristics on pose and COCO object cues, not from trained instructor / student behaviour models.",
        *([] if use_pose else ["Pose weights were not loaded, so explanation, interaction, attention, off-task and hand-raise indicators are unavailable."]),
        *([f"The instructor was visible in only {present_pct:.0f}% of frames, so instructor indicators may be under-counted."] if present_pct < 50 else []),
        "Roles come from the calibration zones: anyone inside the teacher zone is treated as the instructor.",
        "The calibration assumes a fixed camera. If the camera pans, zooms or cuts to a close-up, the instructor can fall outside the teacher zone and be labelled a student, so events in those segments are unreliable.",
        "Seat IDs are grid cells, so crowded or moving students can share or switch seats.",
        "Writing cannot be told apart from looking down, so note-taking may count as off-task.",
        "Board writing needs a raised wrist and interaction needs facing plus a raised wrist; both rules are strict and unvalidated, so zero counts can mean a missed behaviour.",
        "Frames are sampled every 2 s, so short actions can be missed.",
    ]

    mode = "pose + object cues" if use_pose else "object cues only (pose weights not loaded)"
    details = Template(DETAILS_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(
        video=video_name, duration=fmt_time(info["duration_s"]), n_frames=n_frames, step=step,
        n_windows=len(windows), window_s=WINDOW_SECONDS, mode=mode, instructor_lines=instructor_lines,
        student_lines=student_lines, linkage_lines=linkage_lines, suggestions=suggestions or ["No specific suggestion."],
        evidence=evidence or ["No flagged events."], limitations=limitations)
    with open(os.path.join(REPORT_DIR, "analysis_details.txt"), "w", encoding="utf-8") as f:
        f.write(details)

    text = Template(FEEDBACK_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(
        positive=positive, negative=negative, note=FEEDBACK_NOTE)
    report_path = os.path.join(REPORT_DIR, "final_feedback_report.txt")
    for path in (report_path, os.path.join(REPORT_DIR, "final_feedback_report_rules.txt")):  # the _rules copy survives the LLM step
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # facts for the language-model step (llm_report.py): the only thing the model is allowed to see
    facts_doc = {
        "schema_version": 1,
        "lecture": {"video": video_name, "duration": fmt_time(info["duration_s"]), "frames_analysed": n_frames,
                    "sampling_seconds": step, "window_seconds": WINDOW_SECONDS, "windows": len(windows)},
        "data_quality": {
            "mode": mode, "instructor_visible_pct": int(f"{present_pct:.0f}"),
            "student_posture_readable_pct": int(f"{100 * coverage:.0f}") if use_pose and not np.isnan(coverage) else None,
            "instructor_posture_readable_pct": int(f"{100 * seated_known / max(visible_n, 1):.0f}") if use_pose else None,
            "limitations": limitations},
        "indicator_validation": {name: {"checked_against_human_labels": False}
                                 for name in ("instructor_role", "head_up_posture", "hand_raise", "phone_use", "instructor_seated")},
        "glossary": GLOSSARY,
        "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": facts,
        "report_note": FEEDBACK_NOTE,
    }
    with open(os.path.join(REPORT_DIR, "facts.json"), "w", encoding="utf-8") as f:
        json.dump(facts_doc, f, indent=2)

    # structured evaluation result (FR7.1)
    result = {
        "video": video_name, "mode": mode, "frames": n_frames, "windows": len(windows),
        "instructor": {"visible_pct": present_pct, "explanation_s": seconds("explaining") if use_pose else None,
                       "board_s": seconds(board_state), "board_is_writing": use_pose,
                       "interaction_events": count("interacting") if use_pose else None,
                       "path_length": path_length, "zone_changes": zone_changes,
                       "posture_readable_pct": 100 * seated_known / max(visible_n, 1) if use_pose else None,
                       "seated_pct_of_readable": finite(100 * seated_share),
                       "gesture_reach": finite(gesture_reach), "distance_to_nearest_student": finite(student_distance),
                       "stationary_pct": finite(100 * float(stationary_share))},
        "students": {"avg_per_frame": float(students_per_frame), "attention_level": finite(float(mean_attention)),
                     "off_task_rate": finite(float(mean_off_task)), "hand_raise_events": count("hand_raise"),
                     "phone_use_events": count("phone_use"), "reading_events": count("reading"),
                     "standing_per_frame": float(windows["students_standing"].mean())},
        "lag_results": json.loads(lag_df.to_json(orient="records")),
        "lagged_model": json.loads(model_df.to_json(orient="records")),
    }
    with open(os.path.join(REPORT_DIR, "evaluation_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Final report successfully written to {report_path}")
    return text


def plot_timeline(windows):
    """Saves teacher and student indicators over the lecture timeline (Matplotlib)."""
    minutes = (windows["start_s"] + WINDOW_SECONDS / 2) / 60
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.plot(minutes, windows["explanation_ratio"], label="explanation ratio")
    top.plot(minutes, windows["board_writing_ratio"], label="board-writing ratio")
    top.set_ylabel("instructor (fraction of window)")
    top.legend(loc="upper right")
    bottom.plot(minutes, windows["attention_level"], label="attention level")
    bottom.plot(minutes, windows["off_task_rate"], label="off-task rate")
    bottom.set_ylabel("students (fraction of seats)")
    bottom.set_xlabel("lecture time (minutes)")
    bottom.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Timeline chart saved to {path}")


def save_previews(frames_df, people, calib, every=10):
    """Saves annotated preview frames (calibration zones, roles, seat IDs, detected cues)."""
    by_frame = dict(list(people.groupby("frame")))
    zone_colors = {"teacher_zone": (0, 200, 0), "board_region": (0, 200, 255), "student_region": (200, 120, 0)}
    saved = 0
    for i in range(0, len(frames_df), every):
        img = cv2.imread(frames_df.loc[i, "path"])
        h, w = img.shape[:2]
        for name, color in zone_colors.items():
            x1, y1, x2, y2 = calib[name]
            cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w) - 1, int(y2 * h) - 1), color, 1)
        for r in by_frame.get(i, pd.DataFrame()).itertuples():
            if r.role == "other":
                continue
            color = (0, 0, 255) if r.role == "instructor" else (255, 100, 0)
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
            tags = [r.role if r.role == "instructor" else r.seat] + [t for t, f in (
                ("hand_up", r.hand_up), ("head_down", r.head_down), ("phone", r.phone), ("book", r.book),
                ("seated", r.seated == 1.0), ("standing", r.seated == 0.0)) if f]
            cv2.putText(img, " ".join(tags), (int(r.x1), max(12, int(r.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(os.path.join(YOLO_OUT_DIR, f"preview_{i:04d}.jpg"), img)
        saved += 1
    print(f"Saved {saved} annotated preview frames to {YOLO_OUT_DIR}")


def run_session(video_path, session_id, det_model, pose_model):
    """Runs the whole pipeline on one lecture recording. Results go to classroom_project/sessions/<session_id>/."""
    set_session_dirs(session_id)
    calib = load_calibration(session_id)
    use_pose = pose_model is not None
    print(f"\n=== Session {session_id}: {video_path} ===")

    print("\n[1/7] Video input")
    info = validate_video(video_path)

    print("\n[2/7] Preprocessing: standardize video, extract frames with timestamps")
    std_path = standardize_video(video_path, info, STD_DIR)
    frames_df = extract_frames(std_path, FRAME_DIR)

    print("\n[3/7] Detection and pose")
    people = detect_people(frames_df, pose_model, det_model)
    objects = detect_objects(frames_df, det_model)
    people = assign_roles(people, objects, calib)
    people.to_csv(os.path.join(TABLE_DIR, "frame_detections.csv"), index=False)
    people.groupby("frame").size().reindex(range(len(frames_df)), fill_value=0).rename("person_count") \
        .rename_axis("frame").to_csv(os.path.join(REPORT_DIR, "people_counts.csv"))
    save_previews(frames_df, people, calib)

    print("\n[4/7] Behaviour recognition and events")
    tl = build_timeline(frames_df, people, calib, use_pose)
    events = make_events(tl)
    events.to_csv(os.path.join(TABLE_DIR, "events.csv"), index=False)
    print(f"{len(events)} events across {len(tl['seats'])} seats and 1 instructor")

    print("\n[5/7] Window-level indicators")
    windows = build_window_features(tl, events, use_pose)
    windows.insert(0, "session_id", session_id)
    windows.to_csv(os.path.join(TABLE_DIR, "window_features.csv"), index=False)

    print("\n[6/7] Fusion: lag analysis")
    lag_df, model_df = analyze_lags(windows)
    lag_df.to_csv(os.path.join(TABLE_DIR, "lag_results.csv"), index=False)
    model_df.to_csv(os.path.join(TABLE_DIR, "lagged_model.csv"), index=False)

    print("\n[7/7] Report")
    generate_and_save_report(info, os.path.basename(video_path), people, tl, events, windows, lag_df, model_df, use_pose)
    plot_timeline(windows)
    return windows


def pool_sessions():
    """Pools the windows of every lecture analysed so far, for a stronger association analysis than one lecture allows.

    Lags never cross from one lecture into the next, each lecture gets its own baseline, and the lectures should share
    a camera layout (or each have its own calibration), or the pooled numbers mix different measurements."""
    paths = sorted(glob.glob(os.path.join(BASE_DIR, "sessions", "*", "tables", "window_features.csv")))
    if len(paths) < 2:
        print("\nOnly one lecture analysed so far; pooling starts with the second.")
        return
    pooled = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    out_dir = os.path.join(BASE_DIR, "pooled")
    os.makedirs(out_dir, exist_ok=True)
    pooled.to_csv(os.path.join(out_dir, "all_windows.csv"), index=False)
    lag_df, model_df = analyze_lags(pooled)
    lag_df.to_csv(os.path.join(out_dir, "lag_results.csv"), index=False)
    model_df.to_csv(os.path.join(out_dir, "lagged_model.csv"), index=False)
    lines = [f"Pooled analysis: {pooled['session_id'].nunique()} lectures, {len(pooled)} windows."] + build_linkage_lines(lag_df, model_df)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"- {line}" for line in lines) + "\n")
    print("\n=== " + lines[0] + " ===")
    for line in lines[1:]:
        print(f"- {line}")


def main():
    parser = argparse.ArgumentParser(description="TeachLens classroom analytics for one or many lecture recordings.")
    parser.add_argument("--video", help="analyse only this file (default: every video in the videos folder)")
    args = parser.parse_args()

    setup_directories()
    if args.video:
        videos = [args.video]
    else:
        videos = sorted(os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    if not videos:
        print(f"Please place your input video(s) (e.g. Video10min.mp4) in the '{VIDEO_DIR}' directory and run again.")
        return

    det_model = load_model(DETECT_WEIGHTS)
    pose_model = load_model(POSE_WEIGHTS, required=False)
    for video_path in videos:
        session_id = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(video_path))[0])
        run_session(video_path, session_id, det_model, pose_model)
    pool_sessions()


if __name__ == "__main__":
    main()
