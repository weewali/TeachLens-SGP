"""Stage 6 of the two-camera pipeline (main code/Main Code.py): the team's fusion layer run on this pipeline's own event
files, and its results turned into the "links over time" points. Synthetic detections, no video or model."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import lag_facts

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("main_code_fusion", ROOT / "main code" / "Main Code.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

W, H = 1280, 720
CALIB = {"instructor": {"teacher_zone": [0.0, 0.0, 1.0, 0.85], "board_region": None, "camera_faces_class": True},
         "students": {"student_region": [0.0, 0.0, 1.0, 1.0], "seats": "grid", "seat_grid": [1, 4]}}
# the instructor camera's video was cut 7 s later than the student camera's: a planted link only comes out at the right
# lag if both files are put back on one clock
ALIGNMENT = {"method": "test alignment", "skip_s": {"instructor": 7.0, "students": 0.0},
             "recording_starts": {"instructor": "2026-11-16T09:00:00+04:00", "students": None}}


def person(camera, i, cx, cy, **cues):
    row = {"camera": camera, "frame": i, "time_s": 2.0 * i, "x1": cx * W - 40, "y1": cy * H - 60, "x2": cx * W + 40,
           "y2": cy * H + 60, "cx": cx, "cy": cy, "conf": 0.8, **mc.NO_CUES, "phone": False, "book": False,
           "role": "other", "seat": "", "candidates": 0, "img_w": W, "img_h": H}
    row.update(cues)
    return row


def planted_lecture(n=180, delay=5, seed=3):
    """n samples (2 s apart). The instructor faces the class or turns their back in blocks of 10-30 s. Four students
    keep their heads up and face the front exactly when the instructor faced the class `delay` samples earlier
    (delay 5 = 10 s), otherwise they look aside."""
    rng = np.random.default_rng(seed)
    facing, state = [], True
    while len(facing) < n:
        facing += [state] * int(rng.integers(5, 16))
        state = not state
    facing = facing[:n]
    rows_i, rows_s = [], []
    for i in range(n):
        rows_i.append(person("instructor", i, 0.4, 0.5, frontal=facing[i], face_hidden=not facing[i]))
        follows = i >= delay and facing[i - delay]
        for k in range(4):
            rows_s.append(person("students", i, (k + 0.5) / 4, 0.5, head_up=True, frontal=follows))
    frames = pd.DataFrame({"frame": range(n), "time_s": [2.0 * i for i in range(n)]})
    instructor = mc.assign_instructor(pd.DataFrame(rows_i, columns=mc.PEOPLE_COLUMNS), CALIB["instructor"])
    students = mc.assign_students(pd.DataFrame(rows_s, columns=mc.PEOPLE_COLUMNS),
                                  pd.DataFrame(columns=["frame", "label", "px", "py"]), CALIB["students"])
    return mc.build_timeline(frames, pd.concat([instructor, students], ignore_index=True), CALIB)


class FusionStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.saved = (mc.EVENTS_DIR, mc.BASE_DIR)
        mc.EVENTS_DIR, mc.BASE_DIR = cls.temp.name, cls.temp.name
        cls.tl = planted_lecture()
        cls.files = mc.write_session_jsonl(mc.build_jsonl_events(cls.tl, "lecP", ALIGNMENT), "lecP")
        cls.fusion, cls.table = mc.run_fusion(cls.files, ALIGNMENT, cls.tl, "lecP")

    @classmethod
    def tearDownClass(cls):
        mc.EVENTS_DIR, mc.BASE_DIR = cls.saved
        cls.temp.cleanup()

    def link(self, teacher, student):
        return next(link for link in self.fusion["links"]
                    if (link["teacher_feature"], link["student_feature"]) == (teacher, student))

    def test_planted_link_is_found_at_its_lag_on_the_shared_clock(self):
        link = self.link("T_facing_class", "S_on_target")
        self.assertEqual(link["lag_seconds"], 10.0)
        self.assertGreater(link["r"], 0.8)
        self.assertLess(self.link("T_facing_class", "S_off_target")["r"], -0.8)
        turns = self.fusion["teacher_onsets"]["turn_to_class"]
        self.assertEqual(turns[0], 0)                                    # facing the class from the start
        self.assertGreaterEqual(len(turns), 3)

    def test_only_what_this_pipeline_measures_is_analysed(self):
        self.assertEqual(self.fusion["measured_by"], "main_code")
        self.assertEqual(list(self.fusion["features"]), [f for f in mc.FUSION_FEATURES if f != "T_board_or_screen"])
        self.assertIn("T_board_or_screen", self.fusion["not_measured"]["features"])      # no board region
        self.assertIn("T_gesturing", self.fusion["not_measured"]["features"])
        self.assertNotIn("board_or_screen_start", self.fusion["teacher_onsets"])
        tested = {(l["teacher_feature"], l["student_feature"]) for l in self.fusion["links"]}
        self.assertTrue(all(t in self.fusion["features"] and s in self.fusion["features"] for t, s in tested))
        self.assertEqual(mc.fusion_plan({"facing_measured": False, "board_measured": True}),
                         ([f for f in mc.FUSION_FEATURES if f != "T_facing_class"],
                          [e for e in mc.FUSION_EVENTS if e != "turn_to_class"]))

    def test_times(self):
        self.assertEqual(self.fusion["start_real_time"], "2026-11-16T09:00:07.000+04:00")   # recording start + 7 s cut
        self.assertEqual(self.fusion["clock"], {"lecture_seconds_at_start": 0.0, "real_times": True})
        self.assertEqual(list(self.table["lecture_seconds"][:3]), [0.0, 5.0, 10.0])
        self.assertTrue(any("lined up by the test alignment" in line for line in self.fusion["limitations"]))
        self.assertFalse(any(line.startswith("Cameras are lined up") for line in self.fusion["limitations"]))

    def test_videos_without_recording_times_still_work_and_show_no_invented_time(self):
        alignment = dict(ALIGNMENT, recording_starts={"instructor": None, "students": None})
        files = mc.write_session_jsonl(mc.build_jsonl_events(self.tl, "lecQ", alignment), "lecQ")
        fusion, table = mc.run_fusion(files, alignment, self.tl, "lecQ")
        self.assertEqual((fusion["start_real_time"], fusion["clock"]["real_times"]), (None, False))
        self.assertTrue(table["start_real_time"].isna().all())
        self.assertEqual([l["r"] for l in fusion["links"]], [l["r"] for l in self.fusion["links"]])

    def test_report_lines_and_points(self):
        lines = " ".join(mc.build_linkage_lines(self.fusion))
        for expected in ("windows of 5 s", "T_facing_class -> S_on_target after 10 s", "smallest p",
                         "Not measured by this pipeline", "Findings (q <= 0.05)"):
            self.assertIn(expected, lines)
        summary = mc.fusion_summary(self.fusion)
        self.assertEqual(summary["links_tested"], len(self.fusion["links"]))
        json.dumps(summary)
        mc.set_session_dirs("lecP")
        points = mc.write_link_points(self.fusion)
        saved = json.loads((Path(mc.REPORT_DIR) / "links" / "report_points.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["overview"], points["overview"])
        self.assertIn("between what you did", saved["overview"])
        rows, metadata = lag_facts.from_fusion(self.fusion)
        facts = lag_facts.build_facts(rows, lag_facts.load_vocabulary(mc.REPORT_VOCABULARY), metadata)
        on_target = next(f for f in facts["facts"] if f["indicators"] == ["T_facing_class", "S_on_target"])
        self.assertEqual(on_target["student_behaviour"], "the share of students with their heads up and facing the front")
        self.assertEqual((on_target["direction"], on_target["after_seconds"]), ("favourable", 10))

    def test_a_lecture_that_cannot_be_analysed_gets_the_reason_and_no_stale_results(self):
        mc.set_session_dirs("lecE")
        stale = [Path(mc.TABLE_DIR) / "fusion_windows.csv", Path(mc.REPORT_DIR) / "links" / "report_points.json"]
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("from an older run", encoding="utf-8")
        empty = {camera: str(Path(self.temp.name) / f"empty_{camera}.jsonl") for camera in mc.CAMERAS}
        for path in empty.values():
            Path(path).write_text("", encoding="utf-8")
        fusion = mc.analyse_links(empty, ALIGNMENT, self.tl, "lecE")
        self.assertIn("error", fusion)
        saved = json.loads((Path(mc.REPORT_DIR) / "fusion.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, fusion)
        self.assertIsNone(mc.write_link_points(fusion))
        self.assertFalse(any(path.exists() for path in stale))
        self.assertIn("could not analyse", mc.build_linkage_lines(fusion)[0])
        self.assertEqual(mc.fusion_summary(fusion), fusion)


if __name__ == "__main__":
    unittest.main()
