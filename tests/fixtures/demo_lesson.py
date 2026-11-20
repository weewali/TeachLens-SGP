"""A made-up lesson in the lesson pipeline's folder layout, for tests (no video, no model, no network).

make_demo_lesson(base) writes, under base (the outputs/ root):
    uploads/<id>/instructor.mp4, students.mp4 (a few bytes), context.json
    lessons/2026-11-20_1200_<id>/  fusion/windows.csv + fusion.json, student/ and instructor/ events.jsonl,
                                   summary.json, run.json, lesson.json, facts.json
and then runs the real report stage (src/reporting/feedback.py) with a stand-in for Claude (no network): by default it
answers with points that pass every check, so final_feedback_report.txt, report_status.json and tables/ are the real
stage's output; with claude_ok=False it fails as if no API key were set, so there is no feedback, only the reason.

The numbers are invented and labelled as such: 60 windows of 5 s (5 minutes). The instructor faces the class in
alternating blocks of 6 windows, and students are planted to be more on task (85% vs 60%) and less on the phone while
the instructor faces them, so the "during each activity" comparison has a clear difference to find.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.reporting.facts import ALTERNATIVE_EXPLANATIONS, GLOSSARY, VALIDATION_KEYS  # noqa: E402

WINDOWS = 60
WINDOW = 5.0
START = datetime.fromisoformat("2026-11-20T12:00:00+04:00")
STAMP = "2026-11-20_1200"


def real(seconds):
    return (START + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def event(indicator, start, end, value, track=None, role="student", zone=None, coverage="adequate"):
    return {"confidence": None, "coverage": coverage, "duration": end - start, "end_real_time": real(end),
            "end_time": end, "indicator_name": indicator, "role": role, "start_real_time": real(start),
            "start_time": start, "track_id": track, "value": value, "video_id": "demo", "zone": zone}


def windows_table():
    rows = []
    for w in range(WINDOWS):
        facing = (w // 6) % 2 == 0
        wobble = ((w * 7) % 5) / 100                     # small, deterministic variation
        on_task = (0.85 if facing else 0.60) + wobble
        rows.append({"window": w, "lesson_seconds": w * WINDOW, "start_real_time": real(w * WINDOW),
                     "T_facing_class": 0.9 if facing else 0.1, "T_board_or_screen": 0.6 if 14 <= w <= 16 else 0.0,
                     "T_writing": 0.0, "T_seated": 0.0, "T_gesturing": 0.4, "T_walking": 0.1,
                     "T_reading_notes": 0.0, "T_back_to_class": 0.0 if facing else 0.4, "T_stationary": 0.5,
                     "T_path": 0.1, "S_on_task": on_task, "S_off_task": 1 - on_task,
                     "S_phone": (0.05 if facing else 0.20) + wobble / 2, "S_head_down": (0.10 if facing else 0.30),
                     "S_hand_raises": 1.0 if facing and w % 6 == 1 else 0.0,
                     "teacher_known_seconds": WINDOW, "students_state_known_seconds": 30.0,
                     "students_known_seconds": 30.0})
    return pd.DataFrame(rows)


DEMO_ANSWER = {
    "positive_points": [{"text": "Students raised their hands 5 times, for example at 00:05; these hand raises have "
                                 "not been verified by a person.", "fact_ids": ["F1"], "explanation_ids": []}],
    "negative_points": [{"text": "A phone was seen with 13% of the students in view on average, and with 22% at "
                                 "00:30; this detection has not been verified.", "fact_ids": ["F2"],
                         "explanation_ids": []}]}


def fake_claude(system, user, model):
    """Stands in for llm_report.call_claude: a checked answer and made-up token counts."""
    from types import SimpleNamespace
    return DEMO_ANSWER, SimpleNamespace(usage=SimpleNamespace(input_tokens=2000, output_tokens=400), model=model), None


def make_demo_lesson(base, lecture_id="demo_lecture", run_report=True, claude_ok=True):
    """Writes the demo lesson; returns its lesson folder."""
    base = Path(base)
    upload = base / "uploads" / lecture_id
    upload.mkdir(parents=True, exist_ok=True)
    for camera in ("instructor", "students"):
        (upload / f"{camera}.mp4").write_bytes(b"not a real video")
    (upload / "context.json").write_text(json.dumps({
        "name": "Demo lecture (made-up data)", "topic": "Example only: no real class was recorded",
        "notes": "", "created": "2026-11-20 12:00", "slides": None, "activities": []}), encoding="utf-8")

    lesson = base / "lessons" / f"{STAMP}_{lecture_id}"
    for folder in ("fusion", "student", "instructor"):
        (lesson / folder).mkdir(parents=True, exist_ok=True)
    windows_table().to_csv(lesson / "fusion" / "windows.csv", index=False)
    features = {"S_on_task": {"desirable": 1}, "S_phone": {"desirable": -1}, "S_head_down": {"desirable": -1},
                "S_hand_raises": {"desirable": 1}}
    (lesson / "fusion" / "fusion.json").write_text(json.dumps({
        "window_seconds": WINDOW, "windows": WINDOWS, "start_real_time": real(0), "features": features,
        "settings": {"alpha": 0.05}, "hypotheses": []}), encoding="utf-8")

    # students: six seated students (track = seat number), a few moments
    student = []
    for t in range(5):
        for track in range(1, 7):
            col, row = (track - 1) % 3, (track - 1) // 3
            x1, y1 = 0.15 + 0.25 * col, 0.2 + 0.35 * row
            student.append(event("position", float(t), float(t), {"box_xyxy": [x1, y1, x1 + 0.15, y1 + 0.3],
                                                                 "box_center": [x1 + 0.075, y1 + 0.15]}, track))
    student += [event("hand_raise", 46.0, 49.0, "raised", 2), event("hand_raise", 125.0, 127.0, "raised", 3),
                event("device_use", 20.0, 31.0, "phone", 4), event("device_use", 200.0, 205.0, "phone_likely", 6),
                event("alertness", 100.0, 140.0, "dozing", 5), event("device_use", 0.0, 20.0, "none", 4)]
    (lesson / "student" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in student) + "\n", encoding="utf-8")
    seats = [{"seat_id": f"seat_0{n}", "tracks": [n], "on_target_seconds": 250.0 - 20 * n, "off_target_seconds": 20.0 * n,
              "row": 1 + (n - 1) // 3, "x": 0.2, "y": 0.4} for n in range(1, 7)]
    (lesson / "student" / "summary.json").write_text(json.dumps({"seats": {"basis": "made-up", "seats": seats}}),
                                                     encoding="utf-8")
    instructor = [event("position", float(t), float(t), {"hip_center": [0.3 + 0.002 * t, 0.6],
                                                         "box_xyxy": [0.25, 0.3, 0.35, 0.9]}, 1, role="teacher")
                  for t in range(0, 300, 2)]
    (lesson / "instructor" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in instructor) + "\n",
                                                        encoding="utf-8")
    for camera in ("student", "instructor"):
        (lesson / camera / "run.json").write_text(json.dumps({
            "run_id": f"{lecture_id}_{camera}", "video": str(upload / "no_such_video.mp4"), "analyzed_start_seconds": 0.0,
            "first_sample_seconds": 0.0, "sample_period": 1.0, "samples": 300}), encoding="utf-8")
    (lesson / "lesson.json").write_text(json.dumps({"lesson": lecture_id, "made_up": True}), encoding="utf-8")
    (lesson / "facts.json").write_text(json.dumps({
        "schema_version": 1,
        "lecture": {"lesson": lecture_id, "videos": ["students.mp4", "instructor.mp4"], "duration": "05:00",
                    "start_real_time": real(0), "sampling_seconds": 1, "window_seconds": 5, "windows": WINDOWS},
        "data_quality": {"mode": "made-up test data", "limitations": ["Invented numbers for tests."]},
        "indicator_validation": {key: {"checked_against_human_labels": False} for key in VALIDATION_KEYS},
        "glossary": GLOSSARY, "alternative_explanations": ALTERNATIVE_EXPLANATIONS,
        "facts": [
            {"id": "F1", "direction": "favourable", "kind": "hand_raises", "count": 5, "answered_likely": 2,
             "example_times": ["00:05", "01:05", "02:05"],
             "evidence": {"strength": "descriptive", "notes": ["Hand raises have not been checked."]}},
            {"id": "F2", "direction": "unfavourable", "kind": "phone_use", "students_with_phone_pct": 13,
             "peak_window": "00:30", "peak_pct": 22,
             "evidence": {"strength": "descriptive", "notes": ["Phones are small."]}}],
        "report_note": "Note: made-up numbers for tests."}, indent=2), encoding="utf-8")
    if run_report:
        from src.reporting import llm_report
        from src.reporting.feedback import report_lesson

        def no_key(*_args, **_kwargs):
            raise llm_report.RewriteUnavailable("No Claude API key is set on this computer (ANTHROPIC_API_KEY). "
                                                "Nothing was sent.", llm_report.KEY_PROBLEM)
        report_lesson(lesson, caller=fake_claude if claude_ok else no_key, log=lambda _line: None)
    return lesson
