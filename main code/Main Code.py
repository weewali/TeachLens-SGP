"""TeachLens classroom analytics for two cameras recording the same lecture: one on the instructor, one on the students.

Both videos are lined up on one clock, sampled at the same moments and analysed with their own YOLO26 pose model. The
results are put into 10-second windows, and the feedback ties each student behaviour (head up, phone use, head down
and still, hand raises) to what the instructor was doing at the same time. These are associations, never causes.

Each camera's observations are also written as the team's timestamped events (one JSONL file per camera, same format
as the scripting-frame-extraction pipeline): outputs/events/<session>_instructor_view.jsonl and
outputs/events/<session>_student_view.jsonl. See docs/main_code_events.md for the indicators in them. The team's
fusion layer (src/analysis/fusion.py, lag_analysis.py) then reads those two files and tests whether student changes
follow instructor actions a few seconds later; its results become the "links over time" points
(src/reporting/lag_facts.py, llm_text.py).

Usage (from the repository root):
    python "main code/Main Code.py" --instructor A.mp4 --students B.mp4 [--session NAME] [--offset S] [--overwrite]
    python "main code/Main Code.py"
        (the second form analyses every folder in classroom_project/videos/ that holds an instructor.* and a students.* video)
"""
import os
import re
import sys
import json
import glob
import shutil
import argparse
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The per-camera JSONL files use the team's event format from src/analysis/events.py (the same file as in the
# scripting-frame-extraction pipeline), so both pipelines write identical, validated records (AGENTS.md section 7).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis.events import make_event, with_real_time, write_jsonl  # noqa: E402

# OpenCV, Ultralytics and Jinja2 are imported inside the functions that use them, so the analysis and report stages
# can be tested on saved tables without the video stack installed.

# scipy / statsmodels are only needed for the statistics. If they are not installed the pipeline still runs: the gap
# tests fall back to a plain t-test (which overstates certainty), and the fusion stage reports that it could not run.
try:
    from scipy import stats
except ImportError:
    stats = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

# 1. Path configuration
BASE_DIR = os.environ.get("TEACHLENS_BASE", "classroom_project")   # tests point this somewhere else
VIDEO_DIR = os.path.join(BASE_DIR, "videos")     # one folder per lecture: <lecture>/instructor.mp4 + <lecture>/students.mp4
CONFIG_DIR = os.path.join(BASE_DIR, "config")
STD_DIR = os.path.join(BASE_DIR, "standardized")  # the per-lecture folders below are set by set_session_dirs()
FRAME_DIR = os.path.join(BASE_DIR, "frames")
TABLE_DIR = os.path.join(BASE_DIR, "tables")
YOLO_OUT_DIR = os.path.join(BASE_DIR, "yolo_outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
# One JSONL file of timestamped events per camera recording, named by its anonymous video ID (AGENTS.md: one ID per
# camera recording; outputs/ is local and ignored by Git).
EVENTS_DIR = os.environ.get("TEACHLENS_EVENTS_DIR", os.path.join(REPO_ROOT, "outputs", "events"))
CAMERAS = ("instructor", "students")
CAMERA_VIEWS = {"instructor": "instructor_view", "students": "student_view"}   # the team's camera IDs
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# 2. Pipeline parameters
STANDARD_FPS = 25                  # video standardization: one frame rate ...
STANDARD_MAX_HEIGHT = 720          # ... and one maximum resolution (never upscaled)
FRAME_EVERY_SECONDS = 2            # frame sampling interval, the same moments in both cameras
WINDOW_SECONDS = 10                # window length of the feedback comparisons and charts (a multiple of FRAME_EVERY_SECONDS);
                                   # the team's fusion layer uses its own, fusion.window_seconds in configs/default.yaml (5 s)

# Models (YOLO26, the newest Ultralytics release that could be downloaded in November 2026; YOLO27 was announced but
# not released). The instructor is one large person, so the nano pose model at the normal input size is enough.
# Students are many, small and partly hidden, so they get the medium pose model at a larger input size. Phones and
# books need a detection model. All weights must be local files in the working directory: nothing is downloaded.
INSTRUCTOR_POSE_WEIGHTS = "yolo26n-pose.pt"
INSTRUCTOR_IMGSZ = 640
STUDENT_POSE_WEIGHTS = "yolo26m-pose.pt"
STUDENT_IMGSZ = 960
OBJECT_WEIGHTS = "yolo26s.pt"      # COCO detector for phone / book cues on the student camera
OBJECT_IMGSZ = 960
CONF = 0.25
DEVICE = "cpu"                     # Change to "cuda" or 0 if GPU is available
COCO_OBJECTS = {67: "phone", 73: "book"}
PREDICT_CHUNK = 20                 # frames per predict() call; one call over all frames grew past 3 GB and was killed

# Behaviour heuristics: rule-based stand-ins for trained behaviour models, so treat their outputs as a baseline.
KP_CONF = 0.5                      # minimum keypoint confidence to use a keypoint
HAND_RAISE_MARGIN = 0.05           # wrist must be this fraction of the box height above the shoulder
HAND_RAISE_MIN_FRAMES = 2          # ... in this many consecutive sampled frames
HEAD_DOWN_RATIO = 0.25             # nose within this many shoulder-widths of the shoulder line = head down
OFFTASK_MIN_FRAMES = 3             # head down this many consecutive sampled frames (no book) = off-task
FRONTAL_MAX_OFFSET = 0.35          # nose within this many shoulder-widths of the shoulder midpoint = face toward the camera
MIN_SHOULDER_PX = 4                # narrower shoulders (far away or side-on) are too small to measure stillness
STILL_SHIFT = 0.15                 # shoulders moving less than this many shoulder-widths between samples = still
STILL_DOWN_SECONDS = 30            # head down (or face hidden) and still for this long = "head down and still"
STILL_DOWN_MIN_FRAMES = int(np.ceil(STILL_DOWN_SECONDS / FRAME_EVERY_SECONDS))
LONG_STRETCH_WINDOWS = 6           # explanation with no hand raise for this many windows = flagged
SEATED_THIGH_RATIO = 0.5           # thigh / torso vertical extent below this = seated (needs hips and knees in view)
STATIONARY_MOVE = 0.01             # instructor moving less than this fraction of the frame between samples = stationary
STATIONARY_MIN_SECONDS = 10        # a stationary episode in the JSONL lasts at least this long
STATIONARY_MIN_FRAMES = int(np.ceil(STATIONARY_MIN_SECONDS / FRAME_EVERY_SECONDS))

# Report points only use windows where student posture could be read for enough seats.
MIN_POSE_COVERAGE = 0.4            # share of seat-frames with a readable posture
MIN_SEATS_OBSERVED = 3             # seats seen per frame, on average
MIN_GROUP_WINDOWS = 5              # an instructor activity needs this many windows (and as many others) to be compared

# Seats (student camera): found from where students sat, using the shoulder midpoint. Distances are in shoulder widths.
SEAT_MIN_SHARE = 0.10              # a spot occupied in fewer samples than this is no seat (e.g. a chair, bag or legs seen as a person)
SEAT_SEED_MATCH = 0.7              # while finding seats: within this distance = the same spot, farther = a new spot
SEAT_MATCH = 1.0                   # a person sits in a seat within this distance (moving up counts half: standing up)
SEAT_MERGE_OVERLAP = 0.05          # two close spots occupied together in no more than this share of samples are one seat
SEAT_ROW_GAP = 0.6                 # when numbering, a seat this much lower than the row's first seat starts a new row

# 3. Calibration, one file per camera (FR2.3): fractions of that camera's frame as [x1, y1, x2, y2].
# Edit classroom_project/config/calibration_instructor.json and calibration_students.json for your cameras, or add
# calibration_<session>_<camera>.json for one lecture.
DEFAULT_CALIBRATION = {
    "instructor": {
        "teacher_zone": [0.0, 0.0, 1.0, 1.0],   # where the instructor can be; anyone outside it is ignored
        "board_region": None,                   # None = not calibrated, so time at the board is not measured
        "camera_faces_class": True,             # True when this camera sits on the students' side, facing the instructor
    },
    "students": {
        "student_region": [0.0, 0.0, 1.0, 1.0],
        "seats": "auto",                        # "auto": seats are found from where students sat; "grid": seat_grid cells
        "seat_grid": [3, 4],                    # rows, columns of equal cells inside the student region ("grid" only)
    },
}

# The team's fusion layer (stage 6) reads this pipeline's two event files. Main Code.py's rules measure these of its
# features (src/analysis/fusion.py FEATURES; how each is mapped: docs/main_code_events.md). The others are not measured
# here, so they are left out of the analysis rather than counted as zero, which also means fewer tests. S_looking_down
# is left out as well: here it is the same number as S_head_down (both come from the head-down rule).
FUSION_FEATURES = ("T_facing_class", "T_board_or_screen", "T_stationary", "T_path",
                   "S_on_target", "S_off_target", "S_phone", "S_head_down", "S_sleep", "S_hand_raises")
FUSION_EVENTS = ("turn_to_class", "board_or_screen_start", "raised_hand_start")   # scripts/fuse_lesson.py TEACHER_EVENTS
NOT_MEASURED_FOR_FUSION = ("the instructor's gestures, walking, board writing and pointing, and where their head was "
                           "turned; students talking, smiling, writing, reading, using a laptop, moving, or answering "
                           "after a raised hand")
FUSION_SETTINGS = REPO_ROOT / "configs" / "default.yaml"          # its fusion: section (the team's settings)
REPORT_VOCABULARY = REPO_ROOT / "configs" / "report_vocabulary.yaml"
FUSE_LESSON = REPO_ROOT / "scripts" / "fuse_lesson.py"            # the team's TEACHER_EVENTS and LIMITATIONS
SHARED_CLOCK_ORIGIN = 946684800.0  # 2000-01-01 UTC: a stand-in start for lectures without a recording time (never shown)

# Student behaviours compared across instructor activities. The sentence follows "the students in view" and fills in
# g (this activity, as plain words such as "about 95% of the time") and r (the other times, a whole-number percent).
COMPARED_BEHAVIOURS = [
    {"column": "attention_level", "name": "heads up", "favourable_when_higher": True, "min_gap": 0.05,
     "indicator": "head_up_posture",
     "sentence": "had their heads up {g}, compared with {r}% at other times"},
    {"column": "phone_rate", "name": "phone use", "favourable_when_higher": False, "min_gap": 0.02,
     "indicator": "phone_use",
     "sentence": "had a phone in view {g}, compared with {r}% at other times"},
    {"column": "head_down_still_rate", "name": "head down and still", "favourable_when_higher": False, "min_gap": 0.02,
     "indicator": "head_down_still",
     "sentence": "kept their heads down and stayed still {g}, compared with {r}% at other times"},
]
FACING_STATES = ["explaining", "interacting"]   # instructor states facing the class ("explaining to the class" in the report)
STRENGTH_RANK = {"suggestive": 0, "descriptive": 1, "weak": 2, "unknown": 3}
INDICATOR_NAMES = ("instructor_role", "head_up_posture", "head_down_still", "hand_raise", "phone_use", "instructor_seated")

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

# Written for an instructor, not a statistician: the points speak to them as "you", and the statistics behind each
# point stay in facts.json and analysis_details.txt.
FEEDBACK_NOTE = ("Note: \"heads up\" means a student was looking up rather than down. It is a posture, not a measure of "
                 "attention or learning: taking notes also looks like heads down. \"Head down and still\" means a head kept "
                 f"down, with almost no movement, for {STILL_DOWN_SECONDS} seconds or more; it can be resting, but also quiet "
                 "reading or writing. \"Students in view\" are the students the camera could see clearly. Times are "
                 "minutes:seconds from when both cameras were recording. These are automatic detections that no person has "
                 "checked yet.")

# What the language model is told the words in facts.json mean, and the alternative explanations it may offer (as
# possibilities only). Keeping them here means the model cannot invent its own definitions or science.
GLOSSARY = {
    "heads up": "Head up rather than looking down. A posture; it is not a measure of attention, understanding or learning.",
    "students in view": "The students the student camera could see clearly enough to read their posture. Not every student in the room.",
    "seat": f"A place where a student sat for at least {SEAT_MIN_SHARE:.0%} of the lecture, numbered row by row from the "
            "top left of the student camera's view. Seats are anonymous: nobody is identified.",
    "observable students": "The same as students in view: seats in the student camera where a student's head and shoulders were readable.",
    "observable student time": "Seat-frames in which a student was seen; the denominator of every student percentage.",
    "phone use": "A phone was detected inside a student's box in the student camera. It may have been used for course work.",
    "head down and still": f"Head down (or face hidden) with almost no movement for at least {STILL_DOWN_SECONDS} s. A posture: it can "
                           "mean resting or sleeping, but also quiet reading or writing.",
    "window": f"A {WINDOW_SECONDS}-second slice of the lecture.",
    "explaining to the class": "The instructor was visible to the instructor camera, facing the class, and not at the board facing it.",
    "at the board facing it": "The instructor was inside the calibrated board region, turned toward the board (writing, pointing or reading it).",
    "not facing the class": "The instructor was visible but turned away from the class, and not at the board.",
    "mostly out of view": "The instructor camera did not see the instructor for most of a 10-second part of the lecture.",
    "moving around": "The quarter of the lecture's 10-second parts in which the instructor moved furthest in the instructor camera's view.",
    "seated": "The instructor's thighs looked short beside the torso (needs hips and knees in view).",
    "evidence.strength": "weak = the gap could easily be chance, after allowing for the number of comparisons made; suggestive = unlikely "
                         "to be pure chance even after that; descriptive = a single observation or a count, not a test.",
}
ALTERNATIVE_EXPLANATIONS = {
    "E1": "Students may have looked down to write notes, which counts as head down.",
    "E2": "Students may have been reading a slide or the board rather than watching the instructor.",
    "E3": "The two cameras are lined up to about 1 second, so a moment can be off by one sample.",
    "E4": "The comparison rests on a small number of 10-second windows.",
    "E5": "The type of activity (lecture, seat-work, discussion) was not recorded, and it changes posture on its own.",
    "E6": "A head kept down and still can be quiet reading or writing rather than resting.",
    "E7": "A phone may have been used for course work, such as looking up material.",
}

# The supporting detail behind the report above.
DETAILS_TEMPLATE = """\
Classroom Analytics: analysis details
{{ "=" * 50 }}

Lecture: {{ session_id }}
Cameras: instructor = {{ instructor_video }}  |  students = {{ students_video }}
Alignment: {{ alignment }}
Shared lecture time: {{ duration }}  |  Frames analysed: {{ n_frames }} per camera (one every {{ step }} s)  |  Windows: {{ n_windows }} x {{ window_s }} s

1. Instructor indicators (instructor camera)
{% for line in instructor_lines %}
- {{ line }}
{% endfor %}

2. Student indicators (student camera)
{% for line in student_lines %}
- {{ line }}
{% endfor %}

3. Teacher-student linkage (the team's fusion layer, and the comparisons behind the feedback)
{% for line in linkage_lines %}
- {{ line }}
{% endfor %}

4. Suggestions
{% for line in suggestions %}
- {{ line }}
{% endfor %}

5. Supporting evidence (timestamps are mm:ss from the start of the shared lecture time)
{% for line in evidence %}
- {{ line }}
{% endfor %}

6. Limitations
{% for line in limitations %}
- {{ line }}
{% endfor %}
"""

PEOPLE_COLUMNS = ["camera", "frame", "time_s", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "hand_up", "head_down", "head_up",
                  "frontal", "face_hidden", "sx", "sy", "sw", "seated", "gesture", "hands_up", "nose_dx", "hip_x", "hip_y",
                  "phone", "book", "role", "seat", "candidates", "img_w", "img_h"]
NO_CUES = {"hand_up": False, "head_down": False, "head_up": False, "frontal": False, "face_hidden": False,
           "sx": np.nan, "sy": np.nan, "sw": np.nan, "seated": np.nan, "gesture": np.nan,
           "hands_up": 0, "nose_dx": np.nan, "hip_x": np.nan, "hip_y": np.nan}


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


def load_calibration(camera, session_id=None):
    """Loads one camera's calibration; writes the defaults if missing. calibration_<session>_<camera>.json wins over
    the shared calibration_<camera>.json, for a lecture filmed with a different camera position."""
    path = os.path.join(CONFIG_DIR, f"calibration_{camera}.json")
    specific = os.path.join(CONFIG_DIR, f"calibration_{session_id}_{camera}.json") if session_id else None
    if specific and os.path.exists(specific):
        path = specific
        print(f"Using this lecture's own {camera} calibration: {specific}")
    if not os.path.exists(path):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION[camera], f, indent=2)
        print(f"Default {camera} calibration written to {path} (edit it to match your camera).")
    with open(path, encoding="utf-8") as f:
        return {**DEFAULT_CALIBRATION[camera], **json.load(f)}


# ---------------------------------------------------------------- stage 1: video input and camera alignment (FR1)
def validate_video(video_path):
    """Checks that the video can be opened and read before any analysis starts."""
    import cv2
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
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
    print(f"Video OK: {os.path.basename(video_path)}, {width}x{height}, {fps:.2f} fps, {frame_count / fps:.0f} s")
    return {"fps": fps, "width": width, "height": height, "duration_s": frame_count / fps}


def parse_tag_time(value):
    """A metadata time as an aware datetime, or None (no time zone, unreadable, or an unset camera clock)."""
    text = str(value).strip().replace("Z", "+00:00")
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)          # +0400 -> +04:00
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None or moment.year < 2000:
        return None
    return moment


def start_from_tags(meta):
    """The recording start from ffprobe's JSON: Apple's creationdate first (local time with offset), else the MP4
    creation_time (UTC). Returns (datetime, tag name) or (None, None)."""
    tags = {}
    for block in [meta.get("format", {})] + meta.get("streams", []):
        for key, value in (block.get("tags") or {}).items():
            tags.setdefault(key.lower(), value)
    for name in ("com.apple.quicktime.creationdate", "creation_time"):
        if name in tags:
            moment = parse_tag_time(tags[name])
            if moment is not None:
                return moment, name
    return None, None


def read_recording_start(video_path):
    """The recording start time from the video's metadata, read with ffprobe (the team's agreed sync method).

    Tags hold whole seconds and were measured 0.7-0.95 s after the first frame on iPhones, so this lines two cameras
    up to about 1 s. Downloads, edited and re-encoded files usually have no tag."""
    if shutil.which("ffprobe") is None:
        return None, None
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None, None
    try:
        return start_from_tags(json.loads(out.stdout))
    except json.JSONDecodeError:
        return None, None


def align_cameras(paths, infos, offset=None):
    """Works out where the shared lecture timeline starts in each video and how long both cameras overlap.

    offset = seconds the student camera started after the instructor camera (negative if it started first). Without
    one, it comes from the recording-start metadata; without that, both videos are assumed to start together."""
    starts = {camera: read_recording_start(paths[camera]) for camera in CAMERAS}

    def gap(seconds):
        if seconds == 0:
            return "the cameras started together"
        return f"the student camera started {abs(seconds):.1f} s {'after' if seconds > 0 else 'before'} the instructor camera"

    if offset is not None:
        method = f"manual offset ({gap(offset)})"
    elif all(starts[camera][0] is not None for camera in CAMERAS):
        offset = (starts["students"][0] - starts["instructor"][0]).total_seconds()
        method = f"recording-start time in each video's metadata, about +/-1 s ({gap(offset)})"
    else:
        offset = 0.0
        method = "assumed to start together, because the recording-start time was missing from the metadata"
        print("WARNING: no recording-start time in the metadata of both videos; assuming they started together. "
              "If they did not, rerun with --offset SECONDS.")
    skip = {"instructor": max(0.0, offset), "students": max(0.0, -offset)}
    duration = min(infos[camera]["duration_s"] - skip[camera] for camera in CAMERAS)
    if duration <= FRAME_EVERY_SECONDS:
        raise ValueError(f"The two videos do not overlap in time (offset {offset:+.1f} s). Check that they are the same "
                         "lecture, or pass --offset.")
    print(f"Alignment: {method}. Shared lecture time: {fmt_time(duration)}.")

    # each camera's own recording start, for the real (wall-clock) times of its JSONL events. With a known offset (from
    # --offset, not guessed) a camera without a tag gets the other camera's start shifted by the offset.
    recording = {camera: starts[camera][0] for camera in CAMERAS}
    if not method.startswith("assumed"):
        if recording["instructor"] is None and recording["students"] is not None:
            recording["instructor"] = recording["students"] - timedelta(seconds=offset)
        if recording["students"] is None and recording["instructor"] is not None:
            recording["students"] = recording["instructor"] + timedelta(seconds=offset)
    return {"method": method, "offset_s": round(offset, 2), "skip_s": skip, "duration_s": duration,
            "start_tags": {camera: starts[camera][1] for camera in CAMERAS},
            "recording_starts": {camera: recording[camera].isoformat() if recording[camera] else None for camera in CAMERAS}}


# ---------------------------------------------------------------- stage 2: preprocessing (FR2.1, FR2.2)
def standardize_video(video_path, info, output_dir, camera, skip_s, duration_s):
    """Uses FFmpeg to cut this camera's video to the shared lecture time and normalize its frame rate, codec and
    (maximum) resolution. Afterwards second 0 of both cameras' videos is the same moment."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for video standardization but was not found on PATH.")
    out_path = os.path.join(output_dir, f"{camera}.mp4")
    params_path = os.path.join(output_dir, f"{camera}.json")
    height = min(STANDARD_MAX_HEIGHT, info["height"])
    height -= height % 2
    params = {"source": os.path.abspath(video_path), "source_mtime": os.path.getmtime(video_path), "skip_s": round(skip_s, 3),
              "duration_s": round(duration_s, 3), "fps": STANDARD_FPS, "height": height}
    if os.path.exists(out_path) and os.path.exists(params_path):
        with open(params_path, encoding="utf-8") as f:
            if json.load(f) == params:
                print(f"Using existing standardized {camera} video: {out_path}")
                return out_path

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{skip_s:.3f}", "-i", video_path, "-t", f"{duration_s:.3f}", "-an",
        "-vf", f"fps={STANDARD_FPS},scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", out_path,
    ]
    print(f"Standardizing the {camera} video with FFmpeg...")
    subprocess.run(command, check=True)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return out_path


def extract_frames(video_path, output_dir, every_n_seconds=FRAME_EVERY_SECONDS):
    """Extracts one frame every N seconds and records each frame's time in the lecture."""
    import cv2
    os.makedirs(output_dir, exist_ok=True)
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
    print(f"Extracted {len(rows)} frames to {output_dir}")
    return pd.DataFrame(rows, columns=["frame", "time_s", "path"])


def pair_frames(instructor_frames, student_frames):
    """Pairs the two cameras' frames taken at the same lecture time (both videos were cut to the same start)."""
    n = min(len(instructor_frames), len(student_frames))
    if n == 0:
        raise RuntimeError("No frames could be extracted from one of the videos.")
    gap = float(np.abs(instructor_frames["time_s"].iloc[:n].to_numpy() - student_frames["time_s"].iloc[:n].to_numpy()).max())
    if gap > 0.5:
        print(f"WARNING: paired frames differ by up to {gap:.2f} s; check the frame rates of the standardized videos.")
    return pd.DataFrame({"frame": range(n), "time_s": instructor_frames["time_s"].iloc[:n].to_numpy(),
                         "instructor_path": instructor_frames["path"].iloc[:n].to_numpy(),
                         "students_path": student_frames["path"].iloc[:n].to_numpy()})


# ---------------------------------------------------------------- stage 3: detection + pose
def load_model(weights):
    """Loads local YOLO weights. Never downloads: ultralytics would fetch missing weights automatically."""
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights '{weights}' not found in {os.getcwd()}. Place the file there and run again "
                                "(the pipeline never downloads weights by itself).")
    from ultralytics import YOLO
    return YOLO(weights)


def inside(cx, cy, region):
    """True if the point (fractions of the frame) lies inside the [x1, y1, x2, y2] region (False if no region is set)."""
    if not region:
        return False
    x1, y1, x2, y2 = region
    return x1 <= cx <= x2 and y1 <= cy <= y2


def seat_for(cx, cy, calib):
    """Maps a position inside the student region to an equal grid cell ("seats": "grid" in the calibration)."""
    x1, y1, x2, y2 = calib["student_region"]
    rows, cols = calib["seat_grid"]
    col = min(cols - 1, max(0, int((cx - x1) / (x2 - x1) * cols)))
    row = min(rows - 1, max(0, int((cy - y1) / (y2 - y1) * rows)))
    return f"seat_{row * cols + col + 1:02d}"


def match_to_seats(points, seats, limit, upward=1.0):
    """Pairs people (shoulder midpoints in pixels) with seats, at most one person per seat, by the smallest total
    distance in each seat's shoulder widths. Moving up counts `upward` times its length (standing up). Returns
    [(point index, seat index)] for the pairs within `limit`."""
    if not points or not seats:
        return []
    from scipy.optimize import linear_sum_assignment
    cost = np.empty((len(points), len(seats)))
    for i, (x, y) in enumerate(points):
        for j, s in enumerate(seats):
            dy = y - s["y"]
            cost[i, j] = np.hypot(x - s["x"], dy * (upward if dy < 0 else 1.0)) / s["scale"]
    rows, cols = linear_sum_assignment(np.where(cost <= limit, cost, 1e6))
    return [(i, j) for i, j in zip(rows, cols) if cost[i, j] <= limit]


def find_seats(candidates, n_frames):
    """Finds the seats from where students sat during the lecture (fixed cameras, seated students).

    Every sample's people with readable shoulders are matched to the spots found so far; someone far from all of them
    starts a new spot. Two close spots that were almost never occupied at the same time are one seat (a student who
    shifted). Spots occupied in fewer than SEAT_MIN_SHARE of the samples are dropped: a chair, a bag or legs under a
    desk seen as a person. Returns [{"seat", "x", "y", "scale"}] (pixels), numbered row by row from the top left."""
    spots = []
    for frame, grp in candidates[candidates["sw"].notna()].groupby("frame", sort=True):
        points = list(zip(grp["sx"], grp["sy"]))
        matched = dict(match_to_seats(points, spots, SEAT_SEED_MATCH))
        for i, ((x, y), scale) in enumerate(zip(points, grp["sw"])):
            if i in matched:
                spot = spots[matched[i]]
            else:
                spot = {"sum": np.zeros(3), "n": 0, "frames": set()}
                spots.append(spot)
            spot["sum"] += (x, y, scale)
            spot["n"] += 1
            spot["frames"].add(frame)
            spot["x"], spot["y"], spot["scale"] = spot["sum"] / spot["n"]

    while True:   # merge the closest pair of spots that were (almost) never occupied together
        best = None
        for a in range(len(spots)):
            for b in range(a + 1, len(spots)):
                sa, sb = spots[a], spots[b]
                distance = np.hypot(sa["x"] - sb["x"], sa["y"] - sb["y"]) / max(sa["scale"], sb["scale"])
                together = len(sa["frames"] & sb["frames"])
                if (distance <= SEAT_MATCH and together <= SEAT_MERGE_OVERLAP * min(len(sa["frames"]), len(sb["frames"]))
                        and (best is None or distance < best[0])):
                    best = (distance, a, b)
        if best is None:
            break
        _, a, b = best
        sa, sb = spots[a], spots.pop(b)
        sa["sum"] += sb["sum"]
        sa["n"] += sb["n"]
        sa["frames"] |= sb["frames"]
        sa["x"], sa["y"], sa["scale"] = sa["sum"] / sa["n"]

    kept = sorted((s for s in spots if len(s["frames"]) >= max(2, SEAT_MIN_SHARE * n_frames)), key=lambda s: s["y"])
    rows = []
    for spot in kept:
        if rows and spot["y"] - rows[-1][0]["y"] <= SEAT_ROW_GAP * rows[-1][0]["scale"]:
            rows[-1].append(spot)
        else:
            rows.append([spot])
    ordered = [s for row in rows for s in sorted(row, key=lambda s: s["x"])]
    return [{"seat": f"seat_{k + 1:02d}", "x": float(s["x"]), "y": float(s["y"]), "scale": float(s["scale"])}
            for k, s in enumerate(ordered)]


def pose_cues(kxy, kconf, box_h):
    """Reads posture, face direction and gesture cues from one person's 17 COCO keypoints."""
    def ok(i):
        return kconf[i] >= KP_CONF

    raised = [ok(w) and ok(s) and kxy[w][1] < kxy[s][1] - HAND_RAISE_MARGIN * box_h
              for w, s in ((9, 5), (10, 6))]  # (wrist, shoulder) pairs: left, right
    hand_up = any(raised)
    shoulders = ok(5) and ok(6)
    shoulder_w = max(abs(kxy[5][0] - kxy[6][0]), 1e-6)
    head_down = head_up = False
    if ok(0) and shoulders:
        head_down = (((kxy[5][1] + kxy[6][1]) / 2) - kxy[0][1]) < HEAD_DOWN_RATIO * shoulder_w
        head_up = not head_down

    # Face toward the camera: nose plus an eye visible, with the nose near the middle of the shoulders (a side view
    # puts it outside). Face hidden: shoulders visible but no nose or eyes (turned away, or head down on the desk).
    if shoulders:
        frontal = ok(0) and (ok(1) or ok(2)) and abs(kxy[0][0] - (kxy[5][0] + kxy[6][0]) / 2) < FRONTAL_MAX_OFFSET * shoulder_w
    else:
        frontal = ok(0) and ok(1) and ok(2)
    face_hidden = shoulders and not (ok(0) or ok(1) or ok(2))
    # where the nose sits beside the shoulder midpoint, in shoulder-widths (negative = to the image's left): the side a
    # turned head points to, for the team's facing_direction labels
    nose_dx = float((kxy[0][0] - (kxy[5][0] + kxy[6][0]) / 2) / shoulder_w) if shoulders and ok(0) else np.nan

    # hip centre in pixels (both hips needed): the team's path anchor, steadier than the box centre
    hip_x = hip_y = np.nan
    if ok(11) and ok(12):
        hip_x, hip_y = (kxy[11][0] + kxy[12][0]) / 2, (kxy[11][1] + kxy[12][1]) / 2

    # shoulder midpoint and width in pixels, for the stillness rule
    sx = sy = sw = np.nan
    if shoulders and abs(kxy[5][0] - kxy[6][0]) >= MIN_SHOULDER_PX:
        sx, sy, sw = (kxy[5][0] + kxy[6][0]) / 2, (kxy[5][1] + kxy[6][1]) / 2, abs(kxy[5][0] - kxy[6][0])

    # seated (1.0) or standing (0.0) from how long the thigh looks beside the torso; NaN when hips or knees are hidden
    seated = np.nan
    if all(ok(i) for i in (5, 6, 11, 12, 13, 14)):
        torso = abs((kxy[11][1] + kxy[12][1]) / 2 - (kxy[5][1] + kxy[6][1]) / 2)
        thigh = abs((kxy[13][1] + kxy[14][1]) / 2 - (kxy[11][1] + kxy[12][1]) / 2)
        if torso > 1e-6:
            seated = float(thigh / torso < SEATED_THIGH_RATIO)

    # gesture reach: how far the wrists get from the shoulders, in shoulder-widths
    gesture = np.nan
    if shoulders:
        reach = [np.hypot(*(kxy[w] - kxy[s])) / shoulder_w for w, s in ((9, 5), (10, 6)) if ok(w) and ok(s)]
        if reach:
            gesture = float(max(reach))
    return {"hand_up": bool(hand_up), "head_down": bool(head_down), "head_up": bool(head_up), "frontal": bool(frontal),
            "face_hidden": bool(face_hidden), "sx": float(sx), "sy": float(sy), "sw": float(sw), "seated": seated,
            "gesture": gesture, "hands_up": int(sum(raised)), "nose_dx": nose_dx, "hip_x": float(hip_x),
            "hip_y": float(hip_y)}


def predict_in_chunks(model, paths, imgsz, **kwargs):
    """Yields YOLO results frame by frame, calling predict() on small chunks so memory stays flat."""
    for start in range(0, len(paths), PREDICT_CHUNK):
        yield from model.predict(source=paths[start:start + PREDICT_CHUNK], stream=True, conf=CONF, imgsz=imgsz,
                                 device=DEVICE, verbose=False, **kwargs)


def detect_people(paths, times, pose_model, imgsz, camera):
    """Detects every person with their keypoints in each sampled frame of one camera."""
    rows = []
    for i, r in enumerate(predict_in_chunks(pose_model, list(paths), imgsz)):
        h, w = r.orig_shape
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kxy = kconf = None
        if r.keypoints is not None and r.keypoints.conf is not None:
            kxy, kconf = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            cues = pose_cues(kxy[j], kconf[j], y2 - y1) if kxy is not None else dict(NO_CUES)
            rows.append({"camera": camera, "frame": i, "time_s": float(times[i]), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h, "conf": float(confs[j]), **cues,
                         "phone": False, "book": False, "role": "other", "seat": "", "candidates": 0, "img_w": w, "img_h": h})
    return pd.DataFrame(rows, columns=PEOPLE_COLUMNS)


def detect_objects(paths, det_model, imgsz):
    """Detects phones and books (COCO classes) in the student camera, used as student-behaviour cues."""
    rows = []
    for i, r in enumerate(predict_in_chunks(det_model, list(paths), imgsz, classes=list(COCO_OBJECTS))):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            rows.append({"frame": i, "label": COCO_OBJECTS[int(c)], "px": (x1 + x2) / 2, "py": (y1 + y2) / 2})
    return pd.DataFrame(rows, columns=["frame", "label", "px", "py"])


def assign_instructor(people, calib):
    """Instructor camera: the most confident person inside the teacher zone is the instructor (one instructor
    timeline, FR3.1). Anyone else this camera sees (e.g. front-row students) is ignored. The number of candidates is
    kept so that frames where the choice was ambiguous can be counted."""
    for _, grp in people.groupby("frame"):
        candidates = grp[[inside(r.cx, r.cy, calib["teacher_zone"]) for r in grp.itertuples()]]
        if len(candidates) > 0:
            best = candidates["conf"].idxmax()
            people.loc[best, "role"] = "instructor"
            people.loc[best, "candidates"] = len(candidates)
    return people


def assign_students(people, objects, calib, n_frames=None):
    """Student camera: people inside the student region are put on anonymous seats, then the phone / book cues are
    attached. By default the seats are found from where students sat (find_seats), and a detection that fits no seat
    (a chair, bag or legs seen as a person, or someone walking past) is left out. With "seats": "grid" in the
    calibration, every person inside the region is a student on an equal grid cell instead."""
    candidates = people[[inside(r.cx, r.cy, calib["student_region"]) for r in people.itertuples()]]
    if calib.get("seats", "auto") == "grid":
        for r in candidates.itertuples():
            people.at[r.Index, "role"] = "student"
            people.at[r.Index, "seat"] = seat_for(r.cx, r.cy, calib)
    else:
        if n_frames is None:
            n_frames = int(people["frame"].max()) + 1 if len(people) else 0
        seats = find_seats(candidates, n_frames)
        for _, grp in candidates.groupby("frame"):
            readable = grp[grp["sw"].notna()]
            taken = set()
            for i, j in match_to_seats(list(zip(readable["sx"], readable["sy"])), seats, SEAT_MATCH, upward=0.5):
                people.at[readable.index[i], "role"] = "student"
                people.at[readable.index[i], "seat"] = seats[j]["seat"]
                taken.add(j)
            # shoulders not readable: the student of a free seat if the box covers that seat's usual shoulder spot
            for r in grp[grp["sw"].isna()].sort_values("conf", ascending=False).itertuples():
                free = [j for j, s in enumerate(seats)
                        if j not in taken and r.x1 <= s["x"] <= r.x2 and r.y1 <= s["y"] <= r.y2]
                if free:
                    j = min(free, key=lambda j: np.hypot((r.x1 + r.x2) / 2 - seats[j]["x"], (r.y1 + r.y2) / 2 - seats[j]["y"]))
                    people.at[r.Index, "role"] = "student"
                    people.at[r.Index, "seat"] = seats[j]["seat"]
                    taken.add(j)

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


def with_cue_columns(people):
    """Adds the cue columns that detection tables saved by an older version lack (hands_up follows hand_up), so a
    saved frame_detections.csv can still be read."""
    missing = [column for column in NO_CUES if column not in people.columns]
    if not missing:
        return people
    people = people.copy()
    for column in missing:
        people[column] = people["hand_up"].astype(int) if column == "hands_up" else NO_CUES[column]
    return people


def build_timeline(frames_df, people, calib):
    """Turns per-frame detections into one instructor state timeline (instructor camera) and per-seat student state
    timelines (student camera), on the shared lecture clock."""
    people = with_cue_columns(people)
    n = len(frames_df)
    tl = {"times": frames_df["time_s"].to_numpy(dtype=float), "n": n}

    # instructor: absent / board_work / interacting / explaining / other (visible, not facing the class)
    board = calib["instructor"].get("board_region")
    facing_measured = bool(calib["instructor"].get("camera_faces_class", True))
    cx, cy, conf = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    state = np.array(["absent"] * n, dtype=object)
    zone = np.array([""] * n, dtype=object)
    seated_i, gesture_i = np.full(n, np.nan), np.full(n, np.nan)
    hand_i = np.zeros(n, dtype=bool)
    hands_i = np.zeros(n, dtype=int)                 # raised hands (0, 1, 2), for the team's raised_hand labels
    nose_i, hidden_i = np.full(n, np.nan), np.zeros(n, dtype=bool)   # which way a turned head points
    candidates = np.zeros(n, dtype=int)
    box = np.full((n, 4), np.nan)   # instructor box as fractions of the frame, for the JSONL position events
    hip = np.full((n, 2), np.nan)   # instructor hip centre as fractions of the frame (the team's path anchor)
    instructor = people[(people["camera"] == "instructor") & (people["role"] == "instructor")]
    for r in instructor.itertuples():
        i = r.frame
        cx[i], cy[i], conf[i] = r.cx, r.cy, r.conf
        box[i] = (r.x1 / r.img_w, r.y1 / r.img_h, r.x2 / r.img_w, r.y2 / r.img_h)
        hip[i] = (r.hip_x / r.img_w, r.hip_y / r.img_h)
        seated_i[i], gesture_i[i], hand_i[i], candidates[i] = r.seated, r.gesture, r.hand_up, r.candidates
        hands_i[i] = max(int(r.hands_up), int(bool(r.hand_up)))   # a raised hand counts at least once
        nose_i[i], hidden_i[i] = r.nose_dx, r.face_hidden
        at_board = inside(r.cx, r.cy, board)
        facing = facing_measured and bool(r.frontal)   # the camera is on the students' side, so a visible face faces the class
        if at_board and not facing:
            state[i] = "board_work"           # at the board and turned toward it: writing, pointing or reading it
        elif facing and r.hand_up:
            state[i] = "interacting"          # facing the class and gesturing
        elif facing:
            state[i] = "explaining"           # facing the class
        else:
            state[i] = "other"
        zone[i] = "board" if at_board else "away_from_board"
    tl.update(ins_cx=cx, ins_cy=cy, ins_conf=conf, ins_state=state, ins_zone=zone, ins_seated=seated_i,
              ins_gesture=gesture_i, ins_hand=hand_i, ins_hands=hands_i, ins_nose_dx=nose_i, ins_face_hidden=hidden_i,
              ins_candidates=candidates, ins_box=box, ins_hip=hip,
              facing_measured=facing_measured, board_measured=bool(board))

    # students: one timeline per anonymous seat
    students = people[(people["camera"] == "students") & (people["role"] == "student")]
    seats = sorted(students["seat"].unique())
    shape = (len(seats), n)
    present, hand, down, up, hidden = (np.zeros(shape, dtype=bool) for _ in range(5))
    phone, book, standing, front = (np.zeros(shape, dtype=bool) for _ in range(4))
    sconf = np.zeros(shape)
    sx, sy, sw, seated_s = (np.full(shape, np.nan) for _ in range(4))
    student_count = np.zeros(n, dtype=int)
    seat_index = {s: k for k, s in enumerate(seats)}
    for r in students.itertuples():
        k, i = seat_index[r.seat], r.frame
        student_count[i] += 1
        present[k, i] = True
        hand[k, i] |= r.hand_up
        down[k, i] |= r.head_down
        up[k, i] |= r.head_up
        hidden[k, i] |= r.face_hidden
        front[k, i] |= r.frontal              # face toward the student camera, which sits at the front of the room
        phone[k, i] |= r.phone
        book[k, i] |= r.book
        standing[k, i] |= (r.seated == 0.0)   # NaN (hidden legs) is not standing
        if r.conf > sconf[k, i]:              # stillness and posture follow the clearest person in the seat
            sconf[k, i], sx[k, i], sy[k, i], sw[k, i], seated_s[k, i] = r.conf, r.sx, r.sy, r.sw, r.seated

    # still: the shoulders moved less than STILL_SHIFT shoulder-widths since the previous sample
    shift = np.full(shape, np.nan)
    if n > 1:
        shift[:, 1:] = np.hypot(np.diff(sx, axis=1), np.diff(sy, axis=1)) / sw[:, 1:]
    still = np.zeros(shape, dtype=bool)
    np.less(shift, STILL_SHIFT, out=still, where=~np.isnan(shift))

    hand_s, off_task, down_still = (np.zeros(shape, dtype=bool) for _ in range(3))
    for k in range(len(seats)):
        hand_s[k] = sustained(hand[k], HAND_RAISE_MIN_FRAMES)
        down_still[k] = sustained((down[k] | hidden[k]) & still[k], STILL_DOWN_MIN_FRAMES)
        off_task[k] = sustained(down[k] & ~book[k], OFFTASK_MIN_FRAMES) | phone[k] | down_still[k]
    focused = up & ~off_task
    tl.update(seats=seats, seat_mode=calib["students"].get("seats", "auto"),
              present=present, hand=hand_s, phone=phone, book=book, off_task=off_task,
              focused=focused, standing=standing, sconf=sconf, head_down_still=down_still,
              head_up_raw=up, head_down_raw=down, face_hidden=hidden, facing_front=front, seated_seat=seated_s,
              student_count=student_count)
    return tl


# ---------------------------------------------------------------- stage 4b: per-camera JSONL events (team format)
def value_runs(values):
    """(first, last, value) for each run of equal values, skipping None (inclusive frame indices)."""
    runs, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            if values[start] is not None:
                runs.append((start, i - 1, values[start]))
            start = i
    return runs


def video_id(session_id, camera):
    """The anonymous ID of one camera's recording of this lecture (AGENTS.md: one ID per camera recording)."""
    return f"{session_id}_{CAMERA_VIEWS[camera]}"


def jsonl_paths(session_id):
    return {camera: os.path.join(EVENTS_DIR, f"{video_id(session_id, camera)}.jsonl") for camera in CAMERAS}


def facing_label(tl, i):
    """The team's facing_direction label (docs/list_b_teacher.md) for the instructor in sample i: face toward the
    instructor camera = toward the class (that camera sits on the students' side); shoulders but no face = back to the
    class; otherwise the side of the image the nose points to. Not measured when the camera is elsewhere."""
    if not tl["facing_measured"]:
        return "unknown"
    if tl["ins_state"][i] in FACING_STATES:          # explaining / interacting = the face was toward the camera
        return "toward_class"
    if tl["ins_face_hidden"][i]:
        return "away_from_class"
    dx = tl["ins_nose_dx"][i]
    if np.isnan(dx) or abs(dx) < FRONTAL_MAX_OFFSET:  # nose centred but no eye seen: no side to name
        return "unknown"
    return "sideways_left" if dx < 0 else "sideways_right"


def build_jsonl_events(tl, session_id, alignment):
    """Both cameras' observations as the team's timestamped events: {camera: [Event, ...]}.

    Follows AGENTS.md: role "teacher" (not instructor); times are seconds into each ORIGINAL video (the cut is added
    back) plus real (wall-clock) times from that camera's recording start; confidence is null because these are rule
    states, not calibrated probabilities; unknown is its own value, never counted as no; seats are anonymous places
    (found from where students sat, or grid cells), so they go in `zone` and track_id stays null (there is no tracker). Off-task is not written: it is an
    interpretation, which the fusion layer can derive from these observations.

    Besides this pipeline's own indicators, the ones the team's fusion layer reads are written with the team's names
    and labels (facing_direction, board_interaction, raised_hand, position.hip_center_in_shot; head_down,
    orientation_proxy, alertness, device_use, hand_raise_episode), each from this pipeline's simpler rules
    (docs/main_code_events.md says how). Every state covers all the time a person was seen, "no" included, because the
    fusion layer divides by the known time."""
    times, n, step = tl["times"], tl["n"], FRAME_EVERY_SECONDS
    events = {camera: [] for camera in CAMERAS}

    def span(camera, name, first, last, value, role, coverage="adequate", zone=None):
        skip = alignment["skip_s"][camera]
        events[camera].append(make_event(video_id(session_id, camera), name, skip + times[first], skip + times[last] + step,
                                         value, role=role, coverage=coverage, zone=zone))

    def instant(camera, name, i, value, role, coverage="adequate", zone=None):
        t = alignment["skip_s"][camera] + times[i]
        events[camera].append(make_event(video_id(session_id, camera), name, t, t, value, role=role,
                                         coverage=coverage, zone=zone))

    # instructor camera
    visible = tl["ins_state"] != "absent"
    zone = [z if (tl["board_measured"] and z) else None for z in tl["ins_zone"]]
    for i in range(n):
        if not visible[i]:
            instant("instructor", "teacher_presence_sample", i, "not_detected", "teacher")
            continue
        ambiguous = tl["ins_candidates"][i] > 1   # several people in the teacher zone: the most confident one was taken
        instant("instructor", "teacher_presence_sample", i, "visible", "teacher",
                coverage="limited" if ambiguous else "adequate", zone=zone[i])
        moved = None
        if i > 0 and visible[i - 1]:
            moved = round(float(np.hypot(tl["ins_cx"][i] - tl["ins_cx"][i - 1], tl["ins_cy"][i] - tl["ins_cy"][i - 1])), 4)
        box, hip = tl["ins_box"][i], tl["ins_hip"][i]
        instant("instructor", "position", i, {
            "box_center": [round(float(tl["ins_cx"][i]), 4), round(float(tl["ins_cy"][i]), 4)],
            "box_xyxy": None if np.isnan(box).any() else [round(float(v), 4) for v in box],
            "moved_since_previous_sample": moved,
            # the team's path anchor (fractions of the frame); both cameras are fixed, so there is one shot
            "hip_center_in_shot": None if np.isnan(hip).any() else [round(float(v), 4) for v in hip],
            "camera_shot": 0}, "teacher", coverage="limited" if ambiguous else "adequate", zone=zone[i])
        if not np.isnan(tl["ins_gesture"][i]):
            instant("instructor", "gesture_reach", i, round(float(tl["ins_gesture"][i]), 3), "teacher")
    for first, last, _ in value_runs([True if v else None for v in visible]):
        span("instructor", "teacher_presence", first, last, "visible", "teacher")
    for first, last, value in value_runs([s if s != "absent" else None for s in tl["ins_state"]]):
        span("instructor", "teaching_state", first, last, value, "teacher",
             coverage="adequate" if tl["facing_measured"] else "limited", zone=zone[first])
    posture = [None if not visible[i] else ("unknown" if np.isnan(tl["ins_seated"][i]) else
                                            "sitting" if tl["ins_seated"][i] == 1.0 else "standing") for i in range(n)]
    for first, last, value in value_runs(posture):
        span("instructor", "sitting_standing", first, last, value, "teacher",
             coverage="unknown" if value == "unknown" else "adequate")
    # the team's labels, over all the time the instructor was seen (the fusion layer divides by the known time)
    seen = lambda label: [label(i) if visible[i] else None for i in range(n)]
    for first, last, value in value_runs(seen(lambda i: facing_label(tl, i))):
        span("instructor", "facing_direction", first, last, value, "teacher",
             coverage="unknown" if value == "unknown" else "adequate")
    at_board = (lambda i: "board" if tl["ins_state"][i] == "board_work" else "none") if tl["board_measured"] \
        else (lambda i: "unknown")                    # no board region: not measured
    for first, last, value in value_runs(seen(at_board)):
        span("instructor", "board_interaction", first, last, value, "teacher",
             coverage="unknown" if value == "unknown" else "adequate", zone=zone[first])
    hands = seen(lambda i: ("none", "one", "both")[min(int(tl["ins_hands"][i]), 2)])   # wrist above the shoulder
    for first, last, value in value_runs(hands):
        span("instructor", "raised_hand", first, last, value, "teacher")
    still = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if visible[i] and visible[i - 1]:
            still[i] = np.hypot(tl["ins_cx"][i] - tl["ins_cx"][i - 1], tl["ins_cy"][i] - tl["ins_cy"][i - 1]) < STATIONARY_MOVE
    for first, last in run_segments(still, STATIONARY_MIN_FRAMES):   # still[i] = small step from sample i-1 to i
        span("instructor", "stationary_episode", first - 1, last, "stationary", "teacher", zone=zone[first - 1])

    # student camera
    for i in range(n):
        instant("students", "visible_student_count", i, int(tl["student_count"][i]), "student")
    for k, seat in enumerate(tl["seats"]):
        present = tl["present"][k]
        head = [None if not present[i] else "down" if tl["head_down_raw"][k, i] else "face_hidden" if tl["face_hidden"][k, i]
                else "up" if tl["head_up_raw"][k, i] else "unknown" for i in range(n)]
        for first, last, value in value_runs(head):
            span("students", "head_posture", first, last, value, "student",
                 coverage="unknown" if value == "unknown" else "adequate", zone=seat)
        seated = tl["seated_seat"][k]
        posture = [None if not present[i] else "unknown" if np.isnan(seated[i]) else
                   "sitting" if seated[i] == 1.0 else "standing" for i in range(n)]
        for first, last, value in value_runs(posture):
            span("students", "sitting_standing", first, last, value, "student",
                 coverage="unknown" if value == "unknown" else "adequate", zone=seat)
        for name, mask, value in (("hand_raise", tl["hand"][k], "raised"),
                                  ("head_down_still", tl["head_down_still"][k], "still"),
                                  ("learning_material_use", tl["book"][k], "book")):
            for first, last, _ in value_runs([True if v else None for v in mask]):
                span("students", name, first, last, value, "student", zone=seat)
        for first, last, _ in value_runs([True if v else None for v in tl["hand"][k]]):
            span("students", "hand_raise_episode", first, last, {"samples": int(last - first + 1)}, "student", zone=seat)

        # the team's labels (docs/list_c_students.md) over all the time the seat was seen; "unknown" is its own value
        up, down, hidden = tl["head_up_raw"][k], tl["head_down_raw"][k], tl["face_hidden"][k]
        front, phone, still_down = tl["facing_front"][k], tl["phone"][k], tl["head_down_still"][k]
        seen = lambda label: [label(i) if present[i] else None for i in range(n)]
        labels = {
            "head_down": seen(lambda i: "down" if down[i] else "up" if up[i] else "unknown"),
            # facing the front = head up with the face toward the student camera; a classmate who is answering is not
            # recognised here (the team counts facing them as on target)
            "orientation_proxy": seen(lambda i: "looking_down" if down[i] else "unknown" if hidden[i] else
                                      ("on_target" if front[i] else "off_target") if up[i] else "unknown"),
            # possible_sleep = head down (or face hidden) and still for 30 s; closed eyes cannot be seen here
            "alertness": seen(lambda i: "possible_sleep" if still_down[i] else "awake" if up[i] else "unknown"),
            # phone = a phone detected inside the student's box; none = no phone seen (one under the desk is missed)
            "device_use": seen(lambda i: "phone" if phone[i] else "none"),
        }
        for name, values in labels.items():
            for first, last, value in value_runs(values):
                span("students", name, first, last, value, "student",
                     coverage="unknown" if value == "unknown" else "adequate", zone=seat)

    starts = {camera: datetime.fromisoformat(s) if s else None for camera, s in alignment["recording_starts"].items()}
    return {camera: sorted((with_real_time(e, starts[camera]) for e in events[camera]),
                           key=lambda e: (e.start_time, e.indicator_name, e.zone or ""))
            for camera in CAMERAS}


def write_session_jsonl(events_by_camera, session_id, overwrite=False):
    """Writes one JSONL file per camera; never replaces an existing file unless overwrite is set (AGENTS.md)."""
    paths = jsonl_paths(session_id)
    for camera in CAMERAS:
        write_jsonl(Path(paths[camera]), events_by_camera[camera], overwrite=overwrite)
    return paths


def make_events(tl):
    """Builds the timestamped event table: label, start, end, confidence, entity ID (FR3.3, FR4.3)."""
    times, step = tl["times"], FRAME_EVERY_SECONDS
    rows = []

    def add(entity, entity_id, label, mask, conf):
        for s, e in run_segments(mask):
            rows.append({"entity": entity, "entity_id": entity_id, "label": label, "start_s": times[s],
                         "end_s": times[e] + step, "duration_s": times[e] + step - times[s],
                         "confidence": nanmean_or_nan(conf[s:e + 1])})

    for label in ("explaining", "board_work", "interacting"):
        add("instructor", "instructor", label, tl["ins_state"] == label, tl["ins_conf"])
    for k, seat in enumerate(tl["seats"]):
        for label, mask in (("hand_raise", tl["hand"][k]), ("phone_use", tl["phone"][k]), ("reading", tl["book"][k]),
                            ("head_down_still", tl["head_down_still"][k]), ("off_task", tl["off_task"][k]),
                            ("focused", tl["focused"][k])):
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
def build_window_features(tl, events):
    """Puts both cameras' outputs on one timeline and computes features per fixed window."""
    win = (tl["times"] // WINDOW_SECONDS).astype(int)
    lecture_end = float(tl["times"][-1]) + FRAME_EVERY_SECONDS   # the last window is shorter than WINDOW_SECONDS
    counts = events.groupby(["label", "start_window"]).size()
    cx, cy = tl["ins_cx"], tl["ins_cy"]
    facing, board = tl["facing_measured"], tl["board_measured"]
    rows = []
    for w in range(int(win.max()) + 1):
        idx = np.where(win == w)[0]
        if len(idx) == 0:
            continue
        steps = [float(np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]))
                 for i in idx if i > 0 and not np.isnan(cx[i]) and not np.isnan(cx[i - 1])]
        visible = tl["ins_state"][idx] != "absent"
        observed = tl["present"][:, idx].sum()
        focused, off = tl["focused"][:, idx].sum(), tl["off_task"][:, idx].sum()

        def share(arr):
            return float(arr[:, idx].sum() / observed) if observed else np.nan

        rows.append({
            "window_id": w, "start_s": w * WINDOW_SECONDS, "end_s": min((w + 1) * WINDOW_SECONDS, lecture_end), "n_frames": len(idx),
            "instructor_present": float(np.mean(visible)),
            "explanation_ratio": float(np.mean(np.isin(tl["ins_state"][idx], FACING_STATES))) if facing else np.nan,
            "board_work_ratio": float(np.mean(tl["ins_state"][idx] == "board_work")) if board else np.nan,
            "interaction_count": int(counts.get(("interacting", w), 0)) if facing else np.nan,
            "movement_amount": float(sum(steps)) if steps else np.nan,
            "teacher_stationary_ratio": float(np.mean(np.array(steps) < STATIONARY_MOVE)) if steps else np.nan,
            "teacher_seated_ratio": nanmean_or_nan(tl["ins_seated"][idx]),
            "teacher_posture_readable": float(np.mean(~np.isnan(tl["ins_seated"][idx][visible]))) if visible.any() else np.nan,
            "teacher_gesture": nanmean_or_nan(tl["ins_gesture"][idx]),
            "teacher_hand_up_ratio": float(np.mean(tl["ins_hand"][idx])),
            "minutes": (w * WINDOW_SECONDS + WINDOW_SECONDS / 2) / 60,
            "seats_observed": observed / len(idx),
            "students_standing": tl["standing"][:, idx].sum() / len(idx),
            "attention_level": focused / observed if observed else np.nan,
            "off_task_rate": off / observed if observed else np.nan,
            "pose_coverage": (focused + off) / observed if observed else np.nan,
            "phone_rate": share(tl["phone"]),
            "head_down_still_rate": share(tl["head_down_still"]),
            "hand_raise_count": int(counts.get(("hand_raise", w), 0)),
            "phone_use_count": int(counts.get(("phone_use", w), 0)),
            "head_down_still_count": int(counts.get(("head_down_still", w), 0)),
            "reading_count": int(counts.get(("reading", w), 0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 6: fusion / lag analysis (FR6.3, FR6.4)
# The team's fusion layer (src/analysis/fusion.py and lag_analysis.py, docs/fusion_lag_analysis.md) replaced this
# pipeline's own lag analysis on 2026-11-19 (user decision): 5-second windows, lags of 5-30 s, a circular-shift test,
# Benjamini-Hochberg q-values, lagged models with Newey-West errors, and averages around the moments an action started.
def fuse_lesson_module():
    """The team's scripts/fuse_lesson.py (its TEACHER_EVENTS and LIMITATIONS), loaded without running it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fuse_lesson", FUSE_LESSON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fusion_plan(tl):
    """(features, teacher events) of the fusion layer this lecture measures: facing the class needs the instructor
    camera on the students' side, and time at the board a calibrated board region."""
    skip = set()
    if not tl["facing_measured"]:
        skip |= {"T_facing_class", "turn_to_class"}
    if not tl["board_measured"]:
        skip |= {"T_board_or_screen", "board_or_screen_start"}
    return [name for name in FUSION_FEATURES if name not in skip], [name for name in FUSION_EVENTS if name not in skip]


def clock_origin(alignment):
    """Epoch seconds of the shared lecture clock's 0 (the moment both cameras were recording): from a camera's recording
    start when one is known, otherwise a fixed stand-in that is never shown as a time. Second value: whether it is real."""
    for camera in CAMERAS:
        start = alignment["recording_starts"].get(camera)
        if start:
            return datetime.fromisoformat(start).timestamp() + alignment["skip_s"][camera], True
    return SHARED_CLOCK_ORIGIN, False


def load_lecture_events(path, skip_s, origin):
    """One event file with t0 / t1 on the shared lecture clock (epoch seconds, origin = its 0), as the fusion layer
    reads them. The team's fuse_lesson.py takes each event's real time instead; here both cameras were already lined up
    in stage 1 (by those same recording times, by --offset, or by an assumed joint start), so this clock also works for
    videos without a recording time."""
    events = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                event["t0"] = origin + event["start_time"] - skip_s
                event["t1"] = origin + event["end_time"] - skip_s
                events.append(event)
    return events


def run_fusion(event_files, alignment, tl, session_id):
    """The team's lesson fusion and lag analysis on this lecture's two event files. Returns (the document the team's
    fuse_lesson.py writes as <name>.fusion.json, with what was not measured; the window table). Raises ValueError when
    the lecture cannot be analysed (for example the cameras do not overlap)."""
    from src.analysis.fusion import FEATURES_BY_NAME, build_windows, onsets, overlap_range
    from src.analysis.lag_analysis import analyze
    from src.utils.config import load_config
    fuse = fuse_lesson_module()
    settings = load_config(FUSION_SETTINGS).get("fusion") or {}
    window = float(settings.get("window_seconds", 5.0))
    names, event_names = fusion_plan(tl)
    origin, real = clock_origin(alignment)
    events = {camera: load_lecture_events(event_files[camera], alignment["skip_s"][camera], origin) for camera in CAMERAS}
    table, metadata = build_windows(events["students"], events["instructor"], window_seconds=window,
                                    min_known_fraction=float(settings.get("min_known_fraction", 0.5)),
                                    features=[FEATURES_BY_NAME[name] for name in names])
    start, _ = overlap_range(events["students"], events["instructor"])
    teacher_onsets = {name: onsets(events["instructor"], fuse.TEACHER_EVENTS[name][0], "teacher",
                                   fuse.TEACHER_EVENTS[name][1], start, window, len(table),
                                   float(settings.get("onset_min_gap_seconds", 10)))
                      for name in event_names}
    import warnings
    with warnings.catch_warnings(record=True) as caught:   # e.g. statsmodels: instructor features that move together
        warnings.simplefilter("always")
        results = analyze(table, metadata, settings, teacher_onsets)
    analysis_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    # times: the shared lecture clock (mm:ss in this pipeline's reports) and, when known, real times in the recording's
    # own time zone; a stand-in clock is never shown as a real time
    zone_of = datetime.fromisoformat(next(s for s in alignment["recording_starts"].values() if s)).tzinfo if real else None
    to_real = (lambda t: datetime.fromtimestamp(t, zone_of).isoformat(timespec="milliseconds")) if real else (lambda t: None)
    table.insert(2, "lecture_seconds", [round(start - origin + s, 3) for s in table["lesson_seconds"]])
    table["start_real_time"] = [to_real(start + s) for s in table["lesson_seconds"]]
    metadata.update(start_real_time=to_real(start), end_real_time=to_real(start + len(table) * window))
    limitations = [line for line in fuse.LIMITATIONS if not line.startswith("Cameras are lined up")] + [
        f"The two cameras were lined up by the {alignment['method']}.",
        f"Frames are sampled every {FRAME_EVERY_SECONDS} s, so each {window:g}-second window holds 2 or 3 samples "
        "from each camera.",
        "The indicators come from this pipeline's own rules, which are simpler than the team's: facing the class means "
        "the instructor's face was toward the instructor camera; students facing the front means head up with the face "
        "toward the student camera (a classmate who is answering is not recognised); a phone means one was detected "
        "next to the student (one under the desk is missed); resting means head down or face hidden and still for "
        f"{STILL_DOWN_SECONDS} s (closed eyes cannot be seen).",
        f"Not measured by this pipeline, so not tested: {NOT_MEASURED_FOR_FUSION}.",
    ]
    doc = {"lesson": session_id, "student_run": video_id(session_id, "students"),
           "instructor_run": video_id(session_id, "instructor"), "measured_by": "main_code",
           "clock": {"lecture_seconds_at_start": round(start - origin, 3), "real_times": real},
           **metadata, "teacher_onsets": teacher_onsets, **results,
           "not_measured": {"features": [f.name for f in FEATURES_BY_NAME.values() if f.name not in names],
                            "teacher_events": [e for e in fuse.TEACHER_EVENTS if e not in event_names]},
           "warnings": analysis_warnings, "limitations": limitations}
    return doc, table


def save_fusion(fusion, table):
    """reports/fusion.json (or the reason it could not run), tables/fusion_windows.csv and fusion_links.csv."""
    with open(os.path.join(REPORT_DIR, "fusion.json"), "w", encoding="utf-8") as f:
        json.dump(fusion, f, indent=2)
    paths = {"windows": os.path.join(TABLE_DIR, "fusion_windows.csv"), "links": os.path.join(TABLE_DIR, "fusion_links.csv")}
    if table is None:
        for path in paths.values():       # an older run's tables would no longer match this lecture's report
            if os.path.exists(path):
                os.remove(path)
        return
    table.to_csv(paths["windows"], index=False, float_format="%.4f")
    pd.DataFrame(fusion["links"]).to_csv(paths["links"], index=False)


def write_link_points(fusion):
    """The fusion results as plain-language points for the instructor ("links over time"): reports/links/
    report_points.json and .txt, rule-based (src/reporting/lag_facts.py, llm_text.py). The web app's optional
    language-model step rewrites them with scripts/report.py, from the same fusion.json."""
    from src.reporting import lag_facts, llm_text
    out_dir = Path(REPORT_DIR) / "links"
    if "error" in fusion:
        for name in (llm_text.POINTS_FILE, llm_text.TEXT_FILE):   # no points rather than an older run's
            if (out_dir / name).exists():
                (out_dir / name).unlink()
        return None
    rows, metadata = lag_facts.from_fusion(fusion)
    facts = lag_facts.build_facts(rows, lag_facts.load_vocabulary(REPORT_VOCABULARY), metadata)
    return llm_text.generate(facts, out_dir, use_llm=False, subject="you")


# ---------------------------------------------------------------- stage 7: reporting (FR7)
def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def finite(value):
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value


def whole(value):
    """A number as the whole-number value the report prints, or None when it is not available."""
    return None if value is None or not np.isfinite(value) else int(f"{value:.0f}")


def build_linkage_lines(fusion, shown=5):
    """The fusion layer's results as report lines (FR6.4): the lag, the compared variables and the result."""
    if "error" in fusion:
        return [f"The team's fusion layer could not analyse this lecture: {fusion['error']}"]
    window, settings = fusion["window_seconds"], fusion["settings"]
    alpha, links, findings = settings["alpha"], fusion["links"], fusion["findings"]
    lines = [f"The team's fusion layer (src/analysis/fusion.py, lag_analysis.py): {fusion['windows']} windows of "
             f"{window:g} s; each student feature {window:g}-{settings['max_lag_windows'] * window:g} s after each "
             f"instructor feature, circular-shift test, Benjamini-Hochberg q-values; a finding needs q <= {alpha}. "
             "Associations within this lecture, not causes."]
    rare = ", ".join(f"{name} ({n} windows)" for name, n in fusion["untested_teacher_features"].items())
    lines.append(f"Measured here: {', '.join(fusion['features'])}. Instructor features tested: "
                 f"{', '.join(fusion['tested_teacher_features']) or 'none'}; present in fewer than "
                 f"{settings['min_active_windows']} windows, so not tested: {rare or 'none'}. Student features tested: "
                 f"{', '.join(fusion['outcome_features']) or 'none'}.")
    if links:
        floor = 1 / (1 + max(link["null_shifts"] for link in links))
        lines.append(f"{len(links)} links tested; the smallest p a shift test can give here is {floor:.3f}, so after "
                     f"the correction for {len(links)} links a single link alone reaches q = "
                     f"{min(1.0, floor * len(links)):.2f} at best. Strongest links:")
        for link in links[:shown]:
            reverse = "" if link["r_reverse_max"] is None else f", reverse r = {link['r_reverse_max']:+.2f}"
            lines.append(f"{link['teacher_feature']} -> {link['student_feature']} after {link['lag_seconds']:g} s: "
                         f"r = {link['r']:+.2f} (p = {link['p_shift']:.3f}, q = {link['q']:.3f}{reverse}).")
    else:
        lines.append("No link could be tested (no instructor feature was present in enough windows, or no student "
                     "feature varied).")
    for test in fusion["peri_event"]:
        if "p_random_onsets" not in test:
            lines.append(f"{test['teacher_event']}: {test['events']} starts, too few to average around "
                         f"(need {settings.get('min_events', 3)}).")
        else:
            lines.append(f"After {test['teacher_event']} ({test['events']} starts): {test['student_feature']} changed by "
                         f"{test['mean_change']:+.3f} on average (p = {test['p_random_onsets']:.3f}, q = {test['q']:.3f}).")
    for model in fusion["lagged_models"]:
        terms = [f"{name} ({term['coef']:+.3f}, p = {term['p']:.3f})" for name, term in model["terms"].items()
                 if name != "const" and not name.startswith(model["outcome"] + "[") and term["p"] < 0.05]
        lines.append(f"Lagged model for {model['outcome']}, instructor features {model['lag_windows']} window(s) "
                     f"earlier (R2 = {model['r_squared']:.2f}, {model['windows_used']} windows, Newey-West errors): "
                     + (", ".join(terms) if terms else "no instructor feature reached p < 0.05") + ".")
    if any("rank-deficient" in w for w in fusion.get("warnings", [])):
        lines.append("Some lagged models are unreliable: instructor features in them moved together in this lecture "
                     "(for example facing the class is the opposite of being at the board), so their separate "
                     "coefficients cannot be told apart.")
    for warning in fusion.get("warnings", []):
        lines.append(f"Warning from the analysis: {warning}")
    lines.append(f"Findings (q <= {alpha}): {len(findings)}." + "".join(
        f" {f['kind']}: {f['teacher_feature']} -> {f['student_feature']} ({f['method']}, effect {f['effect']:+.3f}, "
        f"q = {f['q']:.3f})." for f in findings))
    lines.append(f"Not measured by this pipeline, so not tested: {NOT_MEASURED_FOR_FUSION}.")
    return lines


def fusion_summary(fusion):
    """The fusion stage in evaluation_result.json: where its results are and what they say in short."""
    if "error" in fusion:
        return {"error": fusion["error"]}
    return {"file": "reports/fusion.json", "windows": fusion["windows"], "window_seconds": fusion["window_seconds"],
            "features": list(fusion["features"]), "tested_teacher_features": fusion["tested_teacher_features"],
            "untested_teacher_features": fusion["untested_teacher_features"], "links_tested": len(fusion["links"]),
            "findings": fusion["findings"]}


def instructor_phrase(w):
    """Plain-language description of what the instructor was doing in a window (or the mean of several windows)."""
    if not w["instructor_present"] >= 0.5:
        return "the instructor was mostly out of view"
    if w["explanation_ratio"] >= 0.5:
        return "the instructor was explaining to the class"
    if w["board_work_ratio"] >= 0.3:
        return "the instructor was at the board facing it"
    if pd.isna(w["explanation_ratio"]):
        return "the instructor was in view"   # facing the class is not measured with this camera placement
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


# Plain-English pieces for the feedback points, which speak to the instructor directly.
YOU_WERE = {
    "the instructor was explaining to the class": "you were explaining to the class",
    "the instructor was at the board facing it": "you were at the board, facing it",
    "the instructor was not facing the class": "you were not facing the class",
    "the instructor was in view": "you were in the camera's view",
    "the instructor was mostly out of view": "you were mostly out of the camera's view",
    "the instructor was moving around": "you were moving around the most",
    "the instructor was seated": "you were seated",
}


def you_were(phrase):
    """'the instructor was explaining to the class' -> 'you were explaining to the class'."""
    return YOU_WERE.get(phrase, str(phrase).replace("the instructor was", "you were"))


def seat_name(entity_id):
    """'seat_06' -> 'seat 6'."""
    match = re.fullmatch(r"seat_0*(\d+)", str(entity_id))
    return f"seat {match.group(1)}" if match else str(entity_id)


def join_words(items):
    """['a'] -> 'a', ['a', 'b'] -> 'a and b', ['a', 'b', 'c'] -> 'a, b and c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def length_words(seconds):
    """120 -> '2 minutes', 90 -> '1 minute 30 seconds', 40 -> '40 seconds'."""
    minutes, rest = divmod(int(round(seconds)), 60)
    parts = [f"{minutes} minute{'' if minutes == 1 else 's'}"] if minutes else []
    if rest or not minutes:
        parts.append(f"{rest} second{'' if rest == 1 else 's'}")
    return " ".join(parts)


def share_of_time(value):
    """A fraction as plain words: 'all of the time', '0% of the time' or 'about 95% of the time'."""
    p = pct(value)
    return "all of the time" if p == 100 else f"{p}% of the time" if p == 0 else f"about {p}% of the time"


def how_often(n):
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def times_words(times):
    """'at 02:02 and 03:10', or 'twice at 00:14 and once at 03:02' when a time repeats; at most three different times."""
    counts = Counter(times)
    distinct = list(dict.fromkeys(times))
    lead = "" if len(distinct) <= 3 else "for example "
    if all(counts[t] == 1 for t in distinct):
        return lead + "at " + join_words(distinct[:3])
    return lead + join_words(f"{how_often(counts[t])} at {t}" for t in distinct[:3])


def gap_p_value(values, in_group):
    """p-value for the gap in `values` between the windows in `in_group` and the rest, and whether the test allowed for
    neighbouring windows being alike (HAC errors, with statsmodels) or was a plain t-test (which overstates certainty)."""
    if sm is not None:
        fit = sm.OLS(values.to_numpy(dtype=float), sm.add_constant(in_group.to_numpy(dtype=float))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        return float(np.asarray(fit.pvalues)[-1]), True
    if stats is not None:
        return float(stats.ttest_ind(values[in_group], values[~in_group], equal_var=False).pvalue), False
    return None, False


def build_feedback_points(windows, events):
    """Builds positive / negative points and the structured facts behind them. Each ties a student behaviour to what
    the instructor was doing at the same time.

    The points are plain English for the instructor ("you"): no windows, p-values or other statistics words; the
    numbers behind them are in `facts` and the technical details. They describe what happened together, not what
    caused it. Returns (positive, negative, suggestions, evidence, facts); `facts` holds every candidate (not only
    the 3 printed per list) with its numbers, direction, evidence strength and the detections it rests on, for the
    LLM step."""
    suggestions, evidence, candidates = [], [], []
    window_of = lambda t: windows[windows["window_id"] == int(t // WINDOW_SECONDS)]
    all_phrases = [instructor_phrase(row) for _, row in windows.iterrows()]

    def add(direction, text, fact):
        candidates.append((direction, text, fact))

    usable = ((windows["pose_coverage"] >= MIN_POSE_COVERAGE) & (windows["seats_observed"] >= MIN_SEATS_OBSERVED)
              & windows["attention_level"].notna())
    valid = windows[usable]

    # 1) each student behaviour in windows with a given instructor activity, against the other usable windows. Windows
    #    where the activity could not be measured are left out of both sides.
    moving_cut = valid["movement_amount"].quantile(0.75) if valid["movement_amount"].notna().any() else np.nan
    activities = [
        ("the instructor was explaining to the class", "explanation_ratio", lambda s: s >= 0.6),
        ("the instructor was at the board facing it", "board_work_ratio", lambda s: s >= 0.3),
        ("the instructor was moving around", "movement_amount", lambda s: (s >= moving_cut) & (moving_cut > 0)),
        ("the instructor was seated", "teacher_seated_ratio", lambda s: s >= 0.5),
        ("the instructor was mostly out of view", "instructor_present", lambda s: s < 0.5),
    ]
    pairs = []
    for phrase, column, test in activities:
        for behaviour in COMPARED_BEHAVIOURS:
            known = valid[column].notna() & valid[behaviour["column"]].notna()
            in_group = test(valid.loc[known, column])
            if in_group.sum() >= MIN_GROUP_WINDOWS and (~in_group).sum() >= MIN_GROUP_WINDOWS:
                pairs.append((phrase, behaviour, valid.loc[known, behaviour["column"]], in_group))
    n_tests = len(pairs)   # every comparison that could be made counts, reported or not
    for phrase, behaviour, values, in_group in pairs:
        g, r = values[in_group].mean(), values[~in_group].mean()
        if abs(g - r) < behaviour["min_gap"]:
            continue
        n_g, n_r = int(in_group.sum()), int((~in_group).sum())
        text = (f"When {you_were(phrase)} ({length_words(n_g * WINDOW_SECONDS)} in total), the students in view "
                + behaviour["sentence"].format(g=share_of_time(g), r=pct(r)) + ".")
        p, robust = gap_p_value(values, in_group)
        if p is None:
            strength, p_text, note = "unknown", None, "No statistics package was available to test this gap."
            text += " Whether this difference could be chance was not tested."
        else:
            p_text = "< 0.001" if p < 0.001 else f"= {p:.3f}"
            strength = "suggestive" if p < 0.05 / n_tests else "weak"
            # the p-value and the correction stay in the facts and the technical details, not in the instructor's text
            text += " This difference is unlikely to be chance." if strength == "suggestive" else " This difference could easily be chance."
            note = (f"The test allows for neighbouring windows being alike and for the {n_tests} comparisons made." if robust else
                    f"Plain test (neighbouring windows are alike, so it overstates the certainty), judged against the {n_tests} comparisons made.")
        favourable = (g > r) == behaviour["favourable_when_higher"]
        add("favourable" if favourable else "unfavourable", text, {
            "kind": "comparison", "student_behaviour": behaviour["name"], "instructor_activity": phrase,
            "windows": n_g, "seconds": n_g * WINDOW_SECONDS, "student_share_pct": pct(g),
            "other_windows": n_r, "other_windows_share_pct": pct(r), "gap_points": abs(pct(g) - pct(r)),
            "comparisons_made": n_tests, "evidence": {"strength": strength, "test_p": p_text, "notes": [note]},
            "indicators": ["instructor_role", behaviour["indicator"]]})
        if not favourable:
            suggestions.append("Look at the stretches with fewer heads up and consider a short check-in question there."
                               if behaviour["name"] == "heads up" else
                               "Break long stretches of this activity with a question or a short task.")

    # 2) each phone-use and head-down-and-still episode, with what the instructor was doing when it began, against how
    #    common that activity was over the whole lecture
    still = f"kept their head down and stayed still for {STILL_DOWN_SECONDS} seconds or more"
    moment_specs = [
        ("phone_use", "phone use", "A phone was seen", "A phone was seen",
         "Phones can also be used for course work.",
         ["A phone is a small object, and these detections have not been checked against human labels.",
          "A phone may have been used for course work."]),
        ("head_down_still", "head down and still", f"A student {still}",
         f"Students {still.replace('their head', 'their heads')}",
         "This looks like resting, but quiet reading or writing looks the same.",
         ["Quiet reading or writing can look the same as resting.",
          "These detections have not been checked against human labels."]),
    ]
    for label, name, one, many, caveat, notes in moment_specs:
        episodes = events[events["label"] == label].sort_values("start_s")
        if not len(episodes):
            continue
        during = []
        for e in episodes.itertuples():
            w = window_of(e.start_s)
            during.append(instructor_phrase(w.iloc[0]) if len(w) else None)
        known = [d for d in during if d]
        top, k = Counter(known).most_common(1)[0] if known else (None, 0)
        base = float(np.mean([p == top for p in all_phrases])) if top else 0.0
        n = len(episodes)
        examples = list(episodes.head(3).itertuples())
        where = join_words(f"{fmt_time(e.start_s)} in {seat_name(e.entity_id)}" for e in examples)
        if n == 1:
            text = (f"{one} once: at {where}" + (f", while {you_were(top)}" if top else "") + ". "
                    "A single moment is not enough to link it to what you were doing.")
        else:
            text = f"{many} {how_often(n)} ({'at' if n <= 3 else 'for example at'} {where})."
            if top:
                # against the share of the lecture spent on that activity: 3 of 4 during an activity that filled 80%
                # of the lecture is what timing alone would give
                episodes_pct, time_pct = pct(k / n), pct(base)
                if episodes_pct >= time_pct + 10:
                    text += (f" {k} of the {n} began while {you_were(top)}: more than you would expect from the "
                             f"{time_pct}% of the lecture spent on it" + (", though with so few this could be chance." if n < 10 else "."))
                elif episodes_pct <= time_pct - 10:
                    text += (f" Only {k} of the {n} began while {you_were(top)}: fewer than you would expect from the "
                             f"{time_pct}% of the lecture spent on it.")
                else:
                    text += (f" {k} of the {n} began while {you_were(top)}, about what you would expect, since that took up "
                             f"{time_pct}% of the lecture.")
        text += " " + caveat
        add("unfavourable", text, {
            "kind": "behaviour_moments", "student_behaviour": name, "count": int(n),
            "example_times": [fmt_time(e.start_s) for e in examples], "example_seats": [e.entity_id for e in examples],
            "most_common_instructor_activity": top, "began_during_that_activity": int(k),
            "share_of_episodes_pct": pct(k / n) if top else None,
            "that_activity_share_of_lecture_pct": pct(base) if top else None,
            **({"minimum_duration_seconds": STILL_DOWN_SECONDS} if label == "head_down_still" else {}),
            "evidence": {"strength": "descriptive", "notes": notes},
            "indicators": [label, "instructor_role"]})
        if top and n >= 2 and k / n > base:
            suggestions.append(f"Episodes of {name} clustered while {top.removeprefix('the instructor was ')}; "
                               "consider a question or a short task at those points.")
        for e, d in list(zip(episodes.itertuples(), during))[:8]:
            evidence.append(f"{name.capitalize()}: {e.entity_id}, {fmt_time(e.start_s)}-{fmt_time(e.end_s)}"
                            + (f", while {d}." if d else "."))

    # 3) the most and least focused 30-second stretch, only where the instructor was in view
    visible = usable & (windows["instructor_present"] >= 0.5)
    rolling = windows["attention_level"].where(visible).rolling(3, min_periods=3).mean()
    if rolling.notna().any() and rolling.max() - rolling.min() >= 0.10:
        for kind, end in (("highest", rolling.idxmax()), ("lowest", rolling.idxmin())):
            stretch = windows.iloc[end - 2:end + 1]
            start_t, end_t = fmt_time(stretch["start_s"].iloc[0]), fmt_time(stretch["end_s"].iloc[-1])
            span = f"{start_t}-{end_t}"
            share = stretch["attention_level"].mean()
            phrase = instructor_phrase(stretch[["instructor_present", "explanation_ratio", "board_work_ratio"]].mean())
            fact = {"kind": f"{kind}_focus_stretch", "span": span, "stretch_seconds": len(stretch) * WINDOW_SECONDS,
                    "students_focused_pct": pct(share), "instructor_activity": phrase,
                    "evidence": {"strength": "descriptive", "notes": ["A single 30-second stretch is noisy."]},
                    "indicators": ["head_up_posture", "instructor_role"]}
            lead = (f"Between {start_t} and {end_t}, while {you_were(phrase)}, the students in view had their heads up "
                    f"{share_of_time(share)}")
            # "30 seconds" on purpose: the timeline chart labels its single highest and lowest 10-second parts
            length = length_words(len(stretch) * WINDOW_SECONDS)
            if kind == "highest":
                add("favourable", lead + f". This was the highest for any {length} of the lecture.", fact)
            else:
                add("unfavourable", lead + f", the lowest for any {length} of the lecture. Watching this part of the "
                    "video may show what changed.", fact)
                suggestions.append(f"Review the video around {span} to see what happened.")
                evidence.append(f"Fewest heads up: {span} ({pct(share)}% of observable student time).")

    # 4) hand raises: participation, with what the instructor was doing at those moments
    hands = events[events["label"] == "hand_raise"].sort_values("start_s")
    if len(hands):
        hand_windows = pd.concat([window_of(t) for t in hands["start_s"]])
        top, base = dominant_activity(hand_windows, windows)
        k = sum(instructor_phrase(row) == top for _, row in hand_windows.iterrows()) if top else 0
        n = len(hands)
        times = [fmt_time(t) for t in hands["start_s"]]
        examples = list(dict.fromkeys(times))[:3]
        if n == 1:
            text = f"A student raised their hand once ({times_words(times)})" + (f", while {you_were(top)}." if top else ".")
        else:
            text = f"Students raised their hands {how_often(n)} ({times_words(times)})"
            if top:
                how = "all" if k == n else "mostly" if k > n / 2 else "one of them" if k == 1 else f"{k} of them"
                text += f", {how} while {you_were(top)}, which took up {pct(base)}% of the lecture"
            text += "."
        add("favourable", text,
            {"kind": "hand_raises", "count": int(n), "example_times": examples, "most_common_instructor_activity": top,
             "that_activity_share_of_all_windows_pct": pct(base),
             "evidence": {"strength": "descriptive",
                          "notes": ["Detections have not been checked against human labels.",
                                    "If the instructor walks into the student camera's view, their gestures can be counted as hand raises."]},
             "indicators": ["hand_raise"]})
        for e in hands.head(5).itertuples():
            evidence.append(f"Hand raise: {e.entity_id}, {fmt_time(e.start_s)}-{fmt_time(e.end_s)}.")

    # 5) windows where most observable students had their heads up, and what the instructor was doing in them
    focused_windows = valid[(valid["attention_level"] >= 0.6) & (valid["instructor_present"] >= 0.5)]
    if len(focused_windows) >= MIN_GROUP_WINDOWS:
        top, base = dominant_activity(focused_windows, valid)
        if len(focused_windows) < len(valid):
            when = f"For about {pct(len(focused_windows) / len(valid))}% of the lecture that could be measured"
        elif len(valid) == len(windows):
            when = "Throughout the lecture"
        else:
            when = "Throughout the parts of the lecture that could be measured"
        add("favourable", f"{when}, at least 60% of the students in view had their heads up.",
            {"kind": "mostly_focused_windows", "windows": int(len(focused_windows)), "usable_windows": int(len(valid)),
             "seconds": int(len(focused_windows) * WINDOW_SECONDS), "most_common_instructor_activity": top,
             "that_activity_share_of_usable_windows_pct": pct(base),
             "evidence": {"strength": "descriptive", "notes": ["Counts of windows above a fixed 60% line."]},
             "indicators": ["head_up_posture", "instructor_role"]})

    # 6) a long stretch of explanation with no student hand raise (and no interaction, when that rule ever fires)
    has_interaction = int((events["label"] == "interacting").sum()) > 0
    quiet = (windows["explanation_ratio"] >= 0.6) & (windows["hand_raise_count"] == 0)
    if has_interaction:
        quiet &= windows["interaction_count"] == 0
    what = "interaction or hand raise" if has_interaction else "student hand raise"
    stretch = max(run_segments(quiet.to_numpy()), key=lambda s: s[1] - s[0], default=None)
    if stretch and stretch[1] - stretch[0] + 1 >= LONG_STRETCH_WINDOWS:
        start_t, end_t = fmt_time(windows.loc[stretch[0], "start_s"]), fmt_time(windows.loc[stretch[1], "end_s"])
        span = f"{start_t}-{end_t}"
        n_win = stretch[1] - stretch[0] + 1
        nobody = "there was no interaction or hand raise" if has_interaction else "no student raised a hand"
        add("unfavourable", f"From {start_t} to {end_t} ({length_words(n_win * WINDOW_SECONDS)}) you were mostly explaining to "
            f"the class and {nobody}. A short question during long explanations gives students a chance to take part.",
            {"kind": "no_participation_stretch", "span": span, "windows": int(n_win), "missing": what,
             "instructor_activity": "the instructor was explaining to the class",
             "evidence": {"strength": "descriptive",
                          "notes": ["Hand raises that the detector missed would make this stretch look quieter than it was."]},
             "indicators": ["hand_raise", "instructor_role"]})
        suggestions.append("Break long explanations with a question or short activity.")
        evidence.append(f"Stretch with no {what}: {span}.")

    # strongest evidence first (a stable sort keeps the order above within each strength)
    candidates.sort(key=lambda c: STRENGTH_RANK.get(c[2]["evidence"]["strength"], 3))
    facts = [{"id": f"F{n}", "direction": direction, **fact} for n, (direction, _, fact) in enumerate(candidates, 1)]
    positive = [t for d, t, _ in candidates if d == "favourable"][:3]
    negative = [t for d, t, _ in candidates if d == "unfavourable"][:3]
    return (positive or ["No clear positive point was found in this analysis."],
            negative or ["No clear negative point was found in this analysis."],
            list(dict.fromkeys(suggestions)), evidence, facts)


def generate_and_save_report(session, people, tl, events, windows, fusion):
    """Computes lecture-level indicators, applies the findings rules and renders the report templates. fusion is the
    team's fusion layer result from stage 6 (run_fusion), or {"error": reason}."""
    from jinja2 import Template
    step = FRAME_EVERY_SECONDS
    n_frames = tl["n"]
    lecture_s = n_frames * step
    count = lambda label: int((events["label"] == label).sum())
    plural = lambda n: f"{n} event" + ("" if n == 1 else "s")
    seconds = lambda *states: float(np.sum(np.isin(tl["ins_state"], states))) * step
    explaining_s = seconds(*FACING_STATES)
    students = people[(people["camera"] == "students") & (people["role"] == "student")]
    students_per_frame = float(students.groupby("frame").size().reindex(range(n_frames), fill_value=0).mean())
    seats_per_frame = windows["seats_observed"].mean()
    left_out = int(((people["camera"] == "students") & (people["role"] != "student")).sum())
    visible_n = int((tl["ins_state"] != "absent").sum())
    present_pct = 100 * visible_n / max(n_frames, 1)
    ambiguous_pct = 100 * float(np.mean(tl["ins_candidates"] > 1))

    path_length = float(np.nansum(np.hypot(np.diff(tl["ins_cx"]), np.diff(tl["ins_cy"]))))
    zones = [z for z in tl["ins_zone"] if z]
    zone_changes = sum(1 for a, b in zip(zones, zones[1:]) if a != b)

    mean_attention = windows["attention_level"].mean()
    mean_off_task = windows["off_task_rate"].mean()
    coverage = windows["pose_coverage"].mean()
    observed_total = int(tl["present"].sum())
    share_of_observed = lambda arr: 100 * float(arr.sum()) / observed_total if observed_total else np.nan

    instructor_lines = [f"Instructor visible in {present_pct:.0f}% of analysed frames."]
    if ambiguous_pct >= 1:
        instructor_lines.append(f"In {ambiguous_pct:.0f}% of frames more than one person was inside the teacher zone; "
                                "the most confident one was taken as the instructor.")
    if tl["facing_measured"]:
        instructor_lines += [
            f"Explaining to the class (facing the class, not at the board facing it): {fmt_time(explaining_s)} "
            f"({100 * explaining_s / lecture_s:.0f}% of the lecture).",
            f"Interaction (facing the class with a raised hand): {plural(count('interacting'))}, {fmt_time(seconds('interacting'))} in total."
            + ("" if count("interacting") else " The rule never triggered, so the absence of interaction is not established."),
        ]
    else:
        instructor_lines.append("Facing the class was not measured: the calibration says the instructor camera is not on the students' side.")
    if tl["board_measured"]:
        instructor_lines.append(f"At the board facing it (writing, pointing or reading it): {fmt_time(seconds('board_work'))} "
                                f"({100 * seconds('board_work') / lecture_s:.0f}%).")
    else:
        instructor_lines.append("Time at the board was not measured: no board region is set in the instructor camera's calibration.")
    instructor_lines.append(f"Movement: path length {path_length:.2f} frame-widths of the instructor camera"
                            + (f", {zone_changes} moves to or from the board." if tl["board_measured"] else "."))

    # catalog indicators: posture, gesture reach, stationary share (each with its evidence coverage)
    seated_known = int((~np.isnan(tl["ins_seated"])).sum())
    seated_share = nanmean_or_nan(tl["ins_seated"])
    gesture_reach = nanmean_or_nan(tl["ins_gesture"])
    stationary_share = windows["teacher_stationary_ratio"].mean()
    if seated_known:
        instructor_lines.append(f"Posture was readable in {100 * seated_known / max(visible_n, 1):.0f}% of the frames where the instructor was visible; "
                                f"seated in {100 * seated_share:.0f}% of those.")
    else:
        instructor_lines.append("Sitting or standing could not be read (hips and knees were never in view), so no sitting comparison is possible here.")
    if not np.isnan(gesture_reach):
        instructor_lines.append(f"Gesture reach (wrist distance from the shoulder, in shoulder-widths): {gesture_reach:.2f} on average.")
    if not np.isnan(stationary_share):
        instructor_lines.append(f"The instructor barely moved between samples in {100 * stationary_share:.0f}% of the sample pairs.")

    if observed_total:
        student_lines = [
            f"Average students detected per frame: {students_per_frame:.1f}, in {seats_per_frame:.1f} of the {len(tl['seats'])} seats found.",
            f"{left_out} person detection{'' if left_out == 1 else 's'} in the student camera fitted no seat and "
            f"{'was' if left_out == 1 else 'were'} left out (for example a chair, bag or legs seen as a person, or someone walking past).",
            f"Heads up (and not off-task): {100 * mean_attention:.0f}% of observable student time on average.",
            f"Off-task (head down for {OFFTASK_MIN_FRAMES * step} s or more without a book, a phone seen, or head down and still): "
            f"{100 * mean_off_task:.0f}% on average.",
            f"Phone use: {plural(count('phone_use'))}; a phone was seen in {share_of_observed(tl['phone']):.1f}% of observable student time.",
            f"Head down and still for {STILL_DOWN_SECONDS} s or more: {plural(count('head_down_still'))}; "
            f"{share_of_observed(tl['head_down_still']):.1f}% of observable student time.",
            f"Hand raises: {plural(count('hand_raise'))}. Reading (book detected): {plural(count('reading'))}. "
            f"Pose state was readable for {100 * coverage:.0f}% of seat-frames.",
        ]
        seen_enough = tl["present"].sum(axis=1) >= 0.5 * n_frames          # seats in view for at least half of the frames
        raised = tl["hand"].any(axis=1)
        if seen_enough.any():
            student_lines.append(f"Hand-raise participation: {int(raised[seen_enough].sum())} of the {int(seen_enough.sum())} seats seen in at least "
                                 "half of the frames had a hand raise.")
        student_lines.append(f"Students read as standing: {windows['students_standing'].mean():.2f} per frame on average "
                             "(legs are often hidden behind desks, so this undercounts).")
    else:
        student_lines = ["No student was detected inside the student camera's student region, so student indicators are unavailable."]

    linkage_lines = build_linkage_lines(fusion)
    positive, negative, suggestions, evidence, facts = build_feedback_points(windows, events)
    # the statistics behind the comparison points, which the plain-language points leave out
    for f in (f for f in facts if f["kind"] == "comparison"):
        test = f["evidence"]
        linkage_lines.append(
            f"Comparison: {f['student_behaviour']} {f['student_share_pct']}% while {you_were(f['instructor_activity']).removeprefix('you were ')} "
            f"({f['windows']} windows) vs {f['other_windows_share_pct']}% in the other {f['other_windows']} windows; "
            + (f"p {test['test_p']}, {test['strength']}. {test['notes'][0]}" if test.get("test_p") else "not tested."))

    limitations = [
        "Behaviour labels come from rule-based heuristics on pose and COCO object cues, not from trained behaviour models, "
        "and they have not been checked against human labels.",
        f"The two cameras were lined up by the {session['alignment']['method']}. A moment can be off by about one sample ({step} s).",
        "The instructor is the most confident person inside the instructor camera's teacher zone. If the instructor walks "
        "into the student camera's view, they are counted as a student there.",
        "Facing the class means the instructor's face was visible to the instructor camera, which assumes that camera sits on the students' side of the room.",
        *([] if tl["board_measured"] else ["No board region was calibrated, so time at the board was not measured."]),
        *([f"The instructor was visible in only {present_pct:.0f}% of frames, so instructor indicators may be under-counted."] if present_pct < 50 else []),
        ("Seats are equal grid cells of the student camera, so crowded or moving students can share or switch seats."
         if tl.get("seat_mode") == "grid" else
         f"Seats are the places where a student sat in at least {SEAT_MIN_SHARE:.0%} of the samples, so a student who sat "
         "somewhere only briefly is left out, one who changes seats counts as two seats, and nobody is identified."),
        "Writing cannot be told apart from looking down, so note-taking may count as off-task.",
        f"'Head down and still' (head down or face hidden, almost no movement, {STILL_DOWN_SECONDS} s or more) often means resting "
        "or sleeping, but quiet reading or writing can look the same.",
        "Phones are small objects that are easily missed or confused with other objects, and a phone may be used for course work.",
        f"Frames are sampled every {step} s, so short actions can be missed.",
    ]

    details = Template(DETAILS_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(
        session_id=session["id"], instructor_video=session["videos"]["instructor"], students_video=session["videos"]["students"],
        alignment=session["alignment"]["method"], duration=fmt_time(session["duration_s"]), n_frames=n_frames, step=step,
        n_windows=len(windows), window_s=WINDOW_SECONDS, instructor_lines=instructor_lines,
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

    # facts for the language-model step (llm_report.py): the only thing the model is allowed to see. No file names,
    # because they can name the instructor.
    facts_doc = {
        "schema_version": 2,
        "lecture": {"cameras": list(CAMERAS), "duration": fmt_time(session["duration_s"]), "frames_analysed": n_frames,
                    "sampling_seconds": step, "window_seconds": WINDOW_SECONDS, "windows": len(windows)},
        "data_quality": {
            "mode": "two cameras: pose + object cues", "instructor_visible_pct": whole(present_pct),
            "instructor_ambiguous_frames_pct": whole(ambiguous_pct),
            "student_posture_readable_pct": whole(100 * coverage),
            "instructor_posture_readable_pct": whole(100 * seated_known / max(visible_n, 1)),
            "limitations": limitations},
        "indicator_validation": {name: {"checked_against_human_labels": False} for name in INDICATOR_NAMES},
        "glossary": GLOSSARY,
        "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": facts,
        "report_note": FEEDBACK_NOTE,
    }
    with open(os.path.join(REPORT_DIR, "facts.json"), "w", encoding="utf-8") as f:
        json.dump(facts_doc, f, indent=2)

    # structured evaluation result (FR7.1)
    result = {
        "session": session["id"], "videos": session["videos"], "alignment": session["alignment"],
        "events_jsonl": session.get("events_jsonl"),
        "frames": n_frames, "windows": len(windows),
        "instructor": {"visible_pct": present_pct, "ambiguous_frames_pct": ambiguous_pct,
                       "explanation_s": explaining_s if tl["facing_measured"] else None,
                       "board_work_s": seconds("board_work") if tl["board_measured"] else None,
                       "interaction_events": count("interacting") if tl["facing_measured"] else None,
                       "path_length": path_length, "board_zone_changes": zone_changes if tl["board_measured"] else None,
                       "posture_readable_pct": 100 * seated_known / max(visible_n, 1),
                       "seated_pct_of_readable": finite(100 * seated_share),
                       "gesture_reach": finite(gesture_reach), "stationary_pct": finite(100 * float(stationary_share))},
        "students": {"avg_per_frame": students_per_frame, "attention_level": finite(float(mean_attention)),
                     "off_task_rate": finite(float(mean_off_task)), "hand_raise_events": count("hand_raise"),
                     "phone_use_events": count("phone_use"), "phone_pct_of_observed": finite(share_of_observed(tl["phone"])),
                     "head_down_still_events": count("head_down_still"),
                     "head_down_still_pct_of_observed": finite(share_of_observed(tl["head_down_still"])),
                     "reading_events": count("reading"), "standing_per_frame": float(windows["students_standing"].mean())},
        "fusion": fusion_summary(fusion),
    }
    with open(os.path.join(REPORT_DIR, "evaluation_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Final report successfully written to {report_path}")

    links = write_link_points(fusion)
    if links is not None:
        print(f"Links over time ({len(links['positive_points'])} positive, {len(links['negative_points'])} negative "
              f"points) written to {os.path.join(REPORT_DIR, 'links')}")
    return text


def plot_timeline(windows):
    """Saves instructor and student indicators over the lecture timeline (Matplotlib)."""
    minutes = (windows["start_s"] + WINDOW_SECONDS / 2) / 60
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.plot(minutes, windows["instructor_present"], label="instructor visible")
    top.plot(minutes, windows["explanation_ratio"], label="explaining to the class")
    top.plot(minutes, windows["board_work_ratio"], label="at the board facing it")
    top.set_ylabel("instructor (fraction of window)")
    top.legend(loc="upper right")
    bottom.plot(minutes, windows["attention_level"], label="heads up")
    bottom.plot(minutes, windows["phone_rate"], label="phone seen")
    bottom.plot(minutes, windows["head_down_still_rate"], label="head down and still")
    bottom.set_ylabel("students (share of observed)")
    bottom.set_xlabel("lecture time (minutes)")
    bottom.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Timeline chart saved to {path}")


def save_previews(frames_df, people, calib, every=10):
    """Saves annotated preview frames for both cameras (calibration zones, roles, seat IDs, detected cues)."""
    import cv2
    zone_colors = {"teacher_zone": (0, 200, 0), "board_region": (0, 200, 255), "student_region": (200, 120, 0)}
    saved = 0
    for camera in CAMERAS:
        by_frame = dict(list(people[people["camera"] == camera].groupby("frame")))
        for i in range(0, len(frames_df), every):
            img = cv2.imread(frames_df.loc[i, f"{camera}_path"])
            if img is None:
                continue
            h, w = img.shape[:2]
            for name, color in zone_colors.items():
                if calib[camera].get(name):
                    x1, y1, x2, y2 = calib[camera][name]
                    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w) - 1, int(y2 * h) - 1), color, 1)
            for r in by_frame.get(i, pd.DataFrame(columns=PEOPLE_COLUMNS)).itertuples():
                if r.role == "other":
                    continue
                color = (0, 0, 255) if r.role == "instructor" else (255, 100, 0)
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
                tags = [r.role if r.role == "instructor" else r.seat] + [t for t, f in (
                    ("facing", r.role == "instructor" and r.frontal), ("hand_up", r.hand_up), ("head_down", r.head_down),
                    ("face_hidden", r.face_hidden), ("phone", r.phone), ("book", r.book),
                    ("seated", r.seated == 1.0), ("standing", r.seated == 0.0)) if f]
                cv2.putText(img, " ".join(tags), (int(r.x1), max(12, int(r.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(os.path.join(YOLO_OUT_DIR, f"{camera}_preview_{i:04d}.jpg"), img)
            saved += 1
    print(f"Saved {saved} annotated preview frames to {YOLO_OUT_DIR}")


def run_session(paths, session_id, models, offset=None, overwrite=False):
    """Runs the whole pipeline on one lecture filmed by two cameras. Results go to classroom_project/sessions/<session_id>/
    and the two per-camera event files to outputs/events/."""
    existing = [p for p in jsonl_paths(session_id).values() if os.path.exists(p)]
    if existing and not overwrite:   # checked before any slow work (AGENTS.md: overwrite only on purpose)
        raise FileExistsError(f"Event files already exist for '{session_id}': {', '.join(existing)}. "
                              "Choose another --session name, or pass --overwrite to replace them.")
    set_session_dirs(session_id)
    calib = {camera: load_calibration(camera, session_id) for camera in CAMERAS}
    print(f"\n=== Session {session_id}: instructor = {paths['instructor']}, students = {paths['students']} ===")

    print("\n[1/7] Video input and camera alignment")
    infos = {camera: validate_video(paths[camera]) for camera in CAMERAS}
    alignment = align_cameras(paths, infos, offset)

    print("\n[2/7] Preprocessing: cut both videos to the shared time, standardize, extract frames with timestamps")
    extracted = {}
    for camera in CAMERAS:
        std_path = standardize_video(paths[camera], infos[camera], STD_DIR, camera, alignment["skip_s"][camera], alignment["duration_s"])
        extracted[camera] = extract_frames(std_path, os.path.join(FRAME_DIR, camera))
    frames_df = pair_frames(extracted["instructor"], extracted["students"])
    frames_df.to_csv(os.path.join(TABLE_DIR, "frames.csv"), index=False)

    print("\n[3/7] Detection and pose (instructor camera, then student camera)")
    times = frames_df["time_s"].to_numpy()
    instructor_people = assign_instructor(
        detect_people(frames_df["instructor_path"], times, models["instructor_pose"], INSTRUCTOR_IMGSZ, "instructor"),
        calib["instructor"])
    objects = detect_objects(frames_df["students_path"], models["objects"], OBJECT_IMGSZ)
    student_people = assign_students(
        detect_people(frames_df["students_path"], times, models["student_pose"], STUDENT_IMGSZ, "students"),
        objects, calib["students"], n_frames=len(frames_df))
    people = pd.concat([instructor_people, student_people], ignore_index=True)
    people.to_csv(os.path.join(TABLE_DIR, "frame_detections.csv"), index=False)
    people.groupby(["frame", "camera"]).size().unstack(fill_value=0).reindex(range(len(frames_df)), fill_value=0) \
        .to_csv(os.path.join(REPORT_DIR, "people_counts.csv"))
    save_previews(frames_df, people, calib)

    print("\n[4/7] Behaviour recognition and events")
    tl = build_timeline(frames_df, people, calib)
    events = make_events(tl)
    events.to_csv(os.path.join(TABLE_DIR, "events.csv"), index=False)
    print(f"{len(events)} events across {len(tl['seats'])} seats and 1 instructor")
    jsonl_events = build_jsonl_events(tl, session_id, alignment)
    event_files = write_session_jsonl(jsonl_events, session_id, overwrite=overwrite)
    for camera in CAMERAS:
        print(f"{len(jsonl_events[camera])} {CAMERA_VIEWS[camera]} events written to {event_files[camera]}")

    print("\n[5/7] Window-level indicators")
    windows = build_window_features(tl, events)
    windows.insert(0, "session_id", session_id)
    windows.insert(1, "setup", "two_camera")
    windows.to_csv(os.path.join(TABLE_DIR, "window_features.csv"), index=False)

    print("\n[6/7] Fusion: the team's lag analysis (do student changes follow instructor actions?)")
    fusion = analyse_links(event_files, alignment, tl, session_id)

    print("\n[7/7] Report")
    session = {"id": session_id, "videos": {camera: os.path.basename(paths[camera]) for camera in CAMERAS},
               "duration_s": alignment["duration_s"], "alignment": alignment,
               "events_jsonl": {CAMERA_VIEWS[camera]: event_files[camera] for camera in CAMERAS}}
    generate_and_save_report(session, people, tl, events, windows, fusion)
    plot_timeline(windows)
    return windows


def analyse_links(event_files, alignment, tl, session_id):
    """Stage 6: runs the team's fusion layer and saves its results; a lecture it cannot analyse gets the reason
    instead, and the rest of the report is still written."""
    try:
        fusion, table = run_fusion(event_files, alignment, tl, session_id)
    except (ValueError, ImportError) as error:   # e.g. the cameras do not overlap, or statsmodels is missing
        fusion, table = {"error": str(error)}, None
        print(f"The fusion layer could not analyse this lecture: {error}")
    save_fusion(fusion, table)
    if table is not None:
        rare = ", ".join(f"{name} ({n})" for name, n in fusion["untested_teacher_features"].items()) or "none"
        print(f"{fusion['windows']} windows of {fusion['window_seconds']:g} s; {len(fusion['links'])} links tested, "
              f"{len(fusion['findings'])} finding(s) with q <= {fusion['settings']['alpha']}; too rare to test "
              f"(windows present): {rare}")
    return fusion


def safe_id(name):
    """A lecture name that is safe as a folder name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "lecture"


def find_lectures():
    """Every folder in classroom_project/videos/ holding an instructor.* and a students.* video, as (session_id, paths)."""
    lectures = []
    for folder in sorted(glob.glob(os.path.join(VIDEO_DIR, "*"))):
        if not os.path.isdir(folder):
            continue
        found = {}
        for path in glob.glob(os.path.join(folder, "*")):
            stem, ext = os.path.splitext(os.path.basename(path))
            if ext.lower() in VIDEO_EXTENSIONS and stem.lower() in CAMERAS:
                found[stem.lower()] = path
        if len(found) == len(CAMERAS):
            lectures.append((safe_id(os.path.basename(folder)), found))
        elif found:
            print(f"Skipping {folder}: it needs both an instructor.* and a students.* video.")
    return lectures


def main():
    parser = argparse.ArgumentParser(description="TeachLens classroom analytics: two cameras (instructor + students) per lecture.")
    parser.add_argument("--instructor", help="the instructor-camera video of one lecture (use together with --students)")
    parser.add_argument("--students", help="the student-camera video of the same lecture")
    parser.add_argument("--session", help="name for this lecture's results (default: the instructor video's file name)")
    parser.add_argument("--offset", type=float, help="seconds the student camera started AFTER the instructor camera "
                                                     "(negative if it started first); overrides the video metadata")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace this lecture's existing JSONL event files (otherwise the run stops before any work)")
    args = parser.parse_args()

    setup_directories()
    if bool(args.instructor) != bool(args.students):
        parser.error("--instructor and --students go together: give both videos of the same lecture.")
    if args.instructor:
        name = args.session or os.path.splitext(os.path.basename(args.instructor))[0]
        lectures = [(safe_id(name), {"instructor": args.instructor, "students": args.students})]
    else:
        if args.offset is not None or args.session:
            parser.error("--offset and --session need --instructor and --students.")
        lectures = find_lectures()
    if not lectures:
        print(f"No lecture found. Put each lecture in its own folder under '{VIDEO_DIR}' holding an instructor.* and a "
              f"students.* video (e.g. {VIDEO_DIR}/lecture01/instructor.mp4 and students.mp4), or pass --instructor and --students.")
        return

    models = {"instructor_pose": load_model(INSTRUCTOR_POSE_WEIGHTS), "student_pose": load_model(STUDENT_POSE_WEIGHTS),
              "objects": load_model(OBJECT_WEIGHTS)}
    for session_id, paths in lectures:
        try:
            run_session(paths, session_id, models, args.offset if args.instructor else None, overwrite=args.overwrite)
        except FileExistsError as error:
            print(f"\nSkipped {session_id}: {error}")


if __name__ == "__main__":
    main()
