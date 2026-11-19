"""Builds a complete MADE-UP two-camera lecture result (no video) with the real functions of main code/Main Code.py,
for testing and previewing the web app before real recordings are analysed.

    python tests/fixtures/demo_lecture.py BASE_DIR      then: TEACHLENS_BASE=BASE_DIR python web/app.py

Story of the 5-minute demo: the instructor explains (with a few raised-hand moments), works at the board, explains
again, and leaves the camera's view at the end. Eight students; some phone use, a few head-down-and-still stretches,
two hand raises, a book, and one student writing (head down but moving).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
N, W, H = 150, 1280, 720          # 150 samples, one every 2 s = 5 minutes per camera


def _main_code():
    spec = importlib.util.spec_from_file_location("main_code_demo", ROOT / "main code" / "Main Code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_demo_lecture(base_dir, lecture_id="demo_lecture"):
    mc = _main_code()
    base_dir = Path(base_dir)
    mc.BASE_DIR, mc.CONFIG_DIR, mc.EVENTS_DIR = str(base_dir), str(base_dir / "config"), str(base_dir / "events")
    calib = {"instructor": {"teacher_zone": [0.0, 0.0, 1.0, 0.85], "board_region": [0.7, 0.0, 1.0, 0.5], "camera_faces_class": True},
             "students": {"student_region": [0.0, 0.0, 1.0, 1.0], "seat_grid": [2, 4]}}
    (base_dir / "config").mkdir(parents=True, exist_ok=True)
    for camera, values in calib.items():
        (base_dir / "config" / f"calibration_{camera}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")

    def person(camera, i, cx, cy, conf, **cues):
        row = {"camera": camera, "frame": i, "time_s": 2.0 * i, "x1": cx * W - 40, "y1": cy * H - 60, "x2": cx * W + 40,
               "y2": cy * H + 60, "cx": cx, "cy": cy, "conf": conf, **mc.NO_CUES, "phone": False, "book": False,
               "role": "other", "seat": "", "candidates": 0, "img_w": W, "img_h": H}
        row.update(cues)
        return row

    rows_i = []
    for i in range(N):
        rows_i.append(person("instructor", i, 0.5, 0.95, 0.95))                    # a student's head below the teacher zone
        if i >= 140:
            continue                                                                # out of view at the end
        if 60 <= i < 90:                                                            # at the board, back to the class
            rows_i.append(person("instructor", i, 0.85, 0.3, 0.8, frontal=False, face_hidden=True, gesture=1.4,
                                 hip_x=0.85 * W, hip_y=0.3 * H + 30))
        else:
            seated = 1.0 if 20 <= i < 30 else (0.0 if 30 <= i < 40 else np.nan)
            hand = 100 <= i < 104
            rows_i.append(person("instructor", i, 0.3 + 0.002 * i, 0.5, 0.8, frontal=True, hand_up=hand,
                                 hands_up=int(hand), seated=seated, gesture=0.9 + 0.3 * np.sin(i / 7),
                                 hip_x=(0.3 + 0.002 * i) * W, hip_y=0.5 * H + 30))
    sleep = {2: range(100, 126), 5: range(20, 41), 7: range(62, 81)}
    phone = {4: range(30, 36), 1: range(110, 115), 6: range(70, 73)}
    hands = {0: range(61, 64), 3: range(95, 98)}
    writing = {6: range(40, 51)}
    rows_s, objects = [], []
    for k in range(8):
        r, c = divmod(k, 4)
        cx, cy = (c + 0.5) / 4, (r + 0.5) / 2
        for i in range(N):
            asleep, writes = i in sleep.get(k, ()), i in writing.get(k, ())
            jitter = 0.0 if asleep else 12.0 * np.sin(i * 1.7 + k)
            rows_s.append(person("students", i, cx, cy, 0.7 + 0.01 * k, head_up=not (asleep or writes),
                                 head_down=writes or (asleep and i % 2 == 0), face_hidden=asleep and i % 2 == 1,
                                 frontal=not (asleep or writes), hand_up=i in hands.get(k, ()),
                                 sx=cx * W + jitter, sy=cy * H - 20 + jitter, sw=40.0))
            if i in phone.get(k, ()):
                objects.append({"frame": i, "label": "phone", "px": cx * W, "py": cy * H + 10})
            if k == 3 and 120 <= i < 126:
                objects.append({"frame": i, "label": "book", "px": cx * W, "py": cy * H + 10})

    instructor = mc.assign_instructor(pd.DataFrame(rows_i, columns=mc.PEOPLE_COLUMNS), calib["instructor"])
    students = mc.assign_students(pd.DataFrame(rows_s, columns=mc.PEOPLE_COLUMNS),
                                  pd.DataFrame(objects, columns=["frame", "label", "px", "py"]), calib["students"])
    people = pd.concat([instructor, students], ignore_index=True)
    frames = pd.DataFrame({"frame": range(N), "time_s": [2.0 * i for i in range(N)]})

    mc.set_session_dirs(lecture_id)
    tables = Path(mc.TABLE_DIR)
    people.to_csv(tables / "frame_detections.csv", index=False)
    frames.to_csv(tables / "frames.csv", index=False)
    tl = mc.build_timeline(frames, people, calib)
    events = mc.make_events(tl)
    events.to_csv(tables / "events.csv", index=False)
    alignment = {"method": "made-up demo data, no video", "offset_s": 0.0, "skip_s": {"instructor": 0.0, "students": 0.0},
                 "duration_s": N * 2.0, "start_tags": {"instructor": None, "students": None},
                 "recording_starts": {"instructor": "2026-11-17T09:00:00+04:00", "students": "2026-11-17T09:00:00+04:00"}}
    event_files = mc.write_session_jsonl(mc.build_jsonl_events(tl, lecture_id, alignment), lecture_id, overwrite=True)
    windows = mc.build_window_features(tl, events)
    windows.insert(0, "session_id", lecture_id)
    windows.insert(1, "setup", "two_camera")
    windows.to_csv(tables / "window_features.csv", index=False)
    fusion = mc.analyse_links(event_files, alignment, tl, lecture_id)      # the team's fusion layer (stage 6)
    session = {"id": lecture_id, "videos": {"instructor": "instructor.mp4", "students": "students.mp4"},
               "duration_s": N * 2.0, "alignment": alignment,
               "events_jsonl": {mc.CAMERA_VIEWS[c]: event_files[c] for c in mc.CAMERAS}}
    mc.generate_and_save_report(session, people, tl, events, windows, fusion)
    mc.plot_timeline(windows)

    inputs = base_dir / "videos" / lecture_id
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "context.json").write_text(json.dumps({
        "name": "Demo lecture (made-up data)", "topic": "Example only: no real class was recorded",
        "notes": "Generated by tests/fixtures/demo_lecture.py to preview the report page.", "slides": None,
        "created": "2026-11-17 09:00"}, indent=2), encoding="utf-8")
    return lecture_id


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python tests/fixtures/demo_lecture.py BASE_DIR")
    print("Demo lecture written:", make_demo_lecture(sys.argv[1]))
