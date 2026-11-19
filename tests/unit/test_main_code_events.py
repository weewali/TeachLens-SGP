"""Per-camera JSONL events from the two-camera pipeline (main code/Main Code.py): team format, roles, times, coverage."""

from datetime import datetime, timezone, timedelta
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.events import Event

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("main_code", ROOT / "main code" / "Main Code.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

N, W, H = 20, 1280, 720
CALIB = {"instructor": {"teacher_zone": [0.0, 0.0, 1.0, 0.85], "board_region": [0.7, 0.0, 1.0, 0.5], "camera_faces_class": True},
         "students": {"student_region": [0.0, 0.0, 1.0, 1.0], "seat_grid": [1, 2]}}
ALIGNMENT = {"skip_s": {"instructor": 7.0, "students": 0.0},
             "recording_starts": {"instructor": "2026-11-16T09:00:00+04:00", "students": None}}


def person(camera, i, cx, cy, **cues):
    row = {"camera": camera, "frame": i, "time_s": 2.0 * i, "x1": cx * W - 40, "y1": cy * H - 60, "x2": cx * W + 40,
           "y2": cy * H + 60, "cx": cx, "cy": cy, "conf": 0.8, **mc.NO_CUES, "phone": False, "book": False,
           "role": "other", "seat": "", "candidates": 0, "img_w": W, "img_h": H}
    row.update(cues)
    return row


def lecture():
    """20 samples (40 s). Instructor: explains standing still (0-9, one hand up at 3-4), at the board turned to the
    image's right (10-14), out of view (15-19). Seat 1: head up facing the camera, phone 2-4, hand raised 5-7, book
    8-9. Seat 2: head down and still the whole time."""
    rows_i, rows_s, objects = [], [], []
    for i in range(N):
        rows_i.append(person("instructor", i, 0.5, 0.95))                        # a student's head below the teacher zone
        if i < 10:
            rows_i.append(person("instructor", i, 0.3, 0.5, frontal=True, hand_up=i in (3, 4), gesture=1.2,
                                 seated=1.0 if i < 5 else float("nan"), hip_x=0.3 * W, hip_y=0.5 * H + 30))
        elif i < 15:
            rows_i.append(person("instructor", i, 0.85, 0.3, frontal=False, gesture=1.2, nose_dx=0.6))
        rows_s.append(person("students", i, 0.25, 0.5, head_up=True, frontal=True, hand_up=5 <= i <= 7,
                             sx=320.0 + 15 * (i % 2), sy=300.0, sw=40.0))
        rows_s.append(person("students", i, 0.75, 0.5, head_down=True, sx=960.0, sy=300.0, sw=40.0))
        if 2 <= i <= 4:
            objects.append({"frame": i, "label": "phone", "px": 0.25 * W, "py": 0.5 * H})
        if i in (8, 9):
            objects.append({"frame": i, "label": "book", "px": 0.25 * W, "py": 0.5 * H})
    frames = pd.DataFrame({"frame": range(N), "time_s": [2.0 * i for i in range(N)]})
    instructor = mc.assign_instructor(pd.DataFrame(rows_i, columns=mc.PEOPLE_COLUMNS), CALIB["instructor"])
    students = mc.assign_students(pd.DataFrame(rows_s, columns=mc.PEOPLE_COLUMNS),
                                  pd.DataFrame(objects, columns=["frame", "label", "px", "py"]), CALIB["students"])
    tl = mc.build_timeline(frames, pd.concat([instructor, students], ignore_index=True), CALIB)
    return mc.build_jsonl_events(tl, "lec1", ALIGNMENT)


def named(events, name):
    return [e for e in events if e.indicator_name == name]


class JsonlEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = lecture()
        cls.teacher, cls.students = cls.events["instructor"], cls.events["students"]

    def test_one_video_id_and_role_per_camera(self):
        self.assertEqual({e.video_id for e in self.teacher}, {"lec1_instructor_view"})
        self.assertEqual({e.video_id for e in self.students}, {"lec1_student_view"})
        self.assertEqual({e.role for e in self.teacher}, {"teacher"})
        self.assertEqual({e.role for e in self.students}, {"student"})

    def test_rule_states_carry_no_invented_confidence(self):
        self.assertTrue(all(e.confidence is None for e in self.teacher + self.students))

    def test_times_are_seconds_into_each_original_video(self):
        samples = named(self.teacher, "teacher_presence_sample")
        self.assertEqual(len(samples), N)
        self.assertEqual(samples[0].start_time, 7.0)                    # the 7 s cut is added back
        self.assertEqual(named(self.students, "visible_student_count")[0].start_time, 0.0)

    def test_real_times_come_from_each_cameras_recording_start(self):
        first = named(self.teacher, "teacher_presence_sample")[0]
        self.assertEqual(first.start_real_time, "2026-11-16T09:00:07.000+04:00")
        self.assertTrue(all(e.start_real_time is None for e in self.students))   # no tag and no known offset

    def test_instructor_indicators(self):
        states = {e.value: e for e in named(self.teacher, "teaching_state")}
        self.assertEqual(set(states), {"explaining", "interacting", "board_work"})
        self.assertEqual(states["board_work"].zone, "board")
        self.assertEqual(states["board_work"].start_time, 7.0 + 20.0)
        self.assertEqual([e.value for e in named(self.teacher, "teacher_presence")], ["visible"])
        self.assertEqual(named(self.teacher, "teacher_presence")[0].end_time, 7.0 + 30.0)
        self.assertEqual(sum(e.value == "not_detected" for e in named(self.teacher, "teacher_presence_sample")), 5)
        self.assertEqual([(e.value, e.coverage) for e in named(self.teacher, "sitting_standing")],
                         [("sitting", "adequate"), ("unknown", "unknown")])
        stationary = named(self.teacher, "stationary_episode")
        self.assertEqual([(e.start_time, e.end_time) for e in stationary], [(7.0, 7.0 + 20.0)])   # board stretch too short
        position = named(self.teacher, "position")[1]
        self.assertEqual(position.value["box_center"], [0.3, 0.5])
        self.assertEqual(position.value["moved_since_previous_sample"], 0.0)
        self.assertEqual(len(named(self.teacher, "gesture_reach")), 15)

    def test_instructor_indicators_for_the_fusion_layer(self):
        """The team's names and labels, over all the time the instructor was seen (absent time is not 'no')."""
        runs = lambda name: [(e.value, e.start_time - 7.0, e.end_time - 7.0) for e in named(self.teacher, name)]
        self.assertEqual(runs("facing_direction"), [("toward_class", 0.0, 20.0), ("sideways_right", 20.0, 30.0)])
        self.assertEqual(runs("board_interaction"), [("none", 0.0, 20.0), ("board", 20.0, 30.0)])
        self.assertEqual(named(self.teacher, "board_interaction")[1].zone, "board")
        self.assertEqual(runs("raised_hand"), [("none", 0.0, 6.0), ("one", 6.0, 10.0), ("none", 10.0, 30.0)])
        positions = named(self.teacher, "position")
        self.assertEqual(positions[0].value["hip_center_in_shot"], [0.3, round(0.5 + 30 / H, 4)])
        self.assertIsNone(positions[-1].value["hip_center_in_shot"])      # no hips seen at the board
        self.assertEqual({p.value["camera_shot"] for p in positions}, {0})

    def test_unmeasured_facing_and_board_are_unknown_not_no(self):
        calib = {**CALIB, "instructor": {**CALIB["instructor"], "board_region": None, "camera_faces_class": False}}
        frames = pd.DataFrame({"frame": range(3), "time_s": [0.0, 2.0, 4.0]})
        rows = [person("instructor", i, 0.3, 0.5, frontal=True) for i in range(3)]
        tl = mc.build_timeline(frames, mc.assign_instructor(pd.DataFrame(rows, columns=mc.PEOPLE_COLUMNS),
                                                            calib["instructor"]), calib)
        teacher = mc.build_jsonl_events(tl, "lec2", ALIGNMENT)["instructor"]
        for name in ("facing_direction", "board_interaction"):
            self.assertEqual([(e.value, e.coverage) for e in named(teacher, name)], [("unknown", "unknown")], name)

    def test_student_indicators_per_anonymous_seat(self):
        self.assertTrue(all(e.track_id is None for e in self.students))
        by_name = {name: named(self.students, name) for name in
                   ("device_use", "hand_raise", "learning_material_use", "head_down_still", "head_posture", "sitting_standing")}
        self.assertEqual([(e.zone, e.start_time, e.end_time) for e in by_name["device_use"] if e.value == "phone"],
                         [("seat_01", 4.0, 10.0)])
        self.assertEqual([(e.zone, e.value) for e in by_name["hand_raise"]], [("seat_01", "raised")])
        self.assertEqual([e.zone for e in by_name["learning_material_use"]], ["seat_01"])
        self.assertEqual([(e.zone, e.value) for e in by_name["head_down_still"]], [("seat_02", "still")])
        self.assertEqual({(e.zone, e.value) for e in by_name["head_posture"]}, {("seat_01", "up"), ("seat_02", "down")})
        self.assertEqual({(e.value, e.coverage) for e in by_name["sitting_standing"]}, {("unknown", "unknown")})
        self.assertEqual({e.value for e in named(self.students, "visible_student_count")}, {2})
        self.assertFalse(named(self.students, "off_task"))                # an interpretation, left to the fusion layer

    def test_student_indicators_for_the_fusion_layer(self):
        """The team's names and labels per seat, covering all the time the seat was seen ("no" included)."""
        def runs(name, seat):
            return [(e.value, e.start_time, e.end_time) for e in named(self.students, name) if e.zone == seat]
        self.assertEqual(runs("device_use", "seat_01"), [("none", 0.0, 4.0), ("phone", 4.0, 10.0), ("none", 10.0, 40.0)])
        self.assertEqual(runs("device_use", "seat_02"), [("none", 0.0, 40.0)])
        self.assertEqual(runs("head_down", "seat_01"), [("up", 0.0, 40.0)])
        self.assertEqual(runs("head_down", "seat_02"), [("down", 0.0, 40.0)])
        self.assertEqual(runs("orientation_proxy", "seat_01"), [("on_target", 0.0, 40.0)])
        self.assertEqual(runs("orientation_proxy", "seat_02"), [("looking_down", 0.0, 40.0)])
        self.assertEqual(runs("alertness", "seat_01"), [("awake", 0.0, 40.0)])
        # the first sample has no previous one to measure stillness against
        self.assertEqual(runs("alertness", "seat_02"), [("unknown", 0.0, 2.0), ("possible_sleep", 2.0, 40.0)])
        episodes = [e for e in named(self.students, "hand_raise_episode")]
        self.assertEqual([(e.zone, e.start_time, e.end_time, e.value) for e in episodes],
                         [("seat_01", 10.0, 16.0, {"samples": 3})])
        unknown = [e for e in self.students if e.value == "unknown"]
        self.assertTrue(unknown and all(e.coverage == "unknown" for e in unknown))

    def test_files_are_valid_team_events_and_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            old = mc.EVENTS_DIR
            mc.EVENTS_DIR = directory
            try:
                paths = mc.write_session_jsonl(self.events, "lec1")
                self.assertEqual({Path(p).name for p in paths.values()},
                                 {"lec1_instructor_view.jsonl", "lec1_student_view.jsonl"})
                for camera, path in paths.items():
                    lines = Path(path).read_text(encoding="utf-8").splitlines()
                    self.assertEqual(len(lines), len(self.events[camera]))
                    for line in lines:
                        Event(**json.loads(line))                        # every line passes the team's checks
                with self.assertRaises(FileExistsError):
                    mc.write_session_jsonl(self.events, "lec1")
                mc.write_session_jsonl(self.events, "lec1", overwrite=True)
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    mc.run_session({"instructor": "missing.mp4", "students": "missing.mp4"}, "lec1", models={})
            finally:
                mc.EVENTS_DIR = old


class AlignmentStartTests(unittest.TestCase):
    def setUp(self):
        self.original = mc.read_recording_start
        self.start = datetime(2026, 11, 16, 9, 0, 0, tzinfo=timezone(timedelta(hours=4)))
        mc.read_recording_start = lambda path: (self.start, "creation_time") if path == "i.mp4" else (None, None)
        self.infos = {"instructor": {"duration_s": 600.0}, "students": {"duration_s": 600.0}}

    def tearDown(self):
        mc.read_recording_start = self.original

    def test_missing_start_is_derived_from_a_known_offset(self):
        alignment = mc.align_cameras({"instructor": "i.mp4", "students": "s.mp4"}, self.infos, offset=3.0)
        self.assertEqual(alignment["recording_starts"]["students"], "2026-11-16T09:00:03+04:00")
        json.dumps(alignment)                                           # stored in evaluation_result.json

    def test_guessed_alignment_does_not_invent_a_start(self):
        alignment = mc.align_cameras({"instructor": "i.mp4", "students": "s.mp4"}, self.infos)
        self.assertIsNone(alignment["recording_starts"]["students"])
        self.assertEqual(alignment["recording_starts"]["instructor"], "2026-11-16T09:00:00+04:00")


if __name__ == "__main__":
    unittest.main()
