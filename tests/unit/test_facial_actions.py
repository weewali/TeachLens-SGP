"""Student facial actions: face crops, mesh checks, rules, and wiring; no model needed."""

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.analysis.aggregates import summarize_list_a
from src.analysis.events import write_jsonl
from src.analysis.facial_actions import INDICATOR_NAMES, StudentFaces, head_direction
from src.analysis.visual_indicators import FoundationAnalyzer
from src.inference.face_mesh import FaceLandmarker, FaceMesh, face_crop, head_angles, plausible
from src.inference.pose import Keypoint, PosePerson
from src.reporting.annotated_video import ReviewOverlay, describe_face
from src.utils.config import (
    FACE_INDICATORS, load_config, load_indicator_flags, select_indicators, validate_config,
)
from src.utils.paths import project_path

# Normalized 1000 x 1000 frame; a seated student facing the camera.
HEAD = {0: (.50, .30), 1: (.51, .29), 2: (.49, .29), 3: (.52, .30), 4: (.48, .30),
        5: (.54, .36), 6: (.46, .36), 11: (.53, .56), 12: (.47, .56)}


def person(points=HEAD, track_id=1, drop=()):
    keypoints = tuple(Keypoint(*points[i], .9) if i in points and i not in drop
                      else Keypoint(0, 0, 0.0) for i in range(17))
    return PosePerson(track_id, (.42, .24, .58, .70), .9, keypoints, 1000, 1000)


def mesh(track_id=1, yaw=0.0, pitch=15.0, **scores):
    names = {"smile": ("mouthSmileLeft", "mouthSmileRight"), "blink": ("eyeBlinkLeft", "eyeBlinkRight"),
             "down": ("browDownLeft", "browDownRight"),
             "up": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), "jaw": ("jawOpen",)}
    values = {}
    for key, value in scores.items():
        values.update(dict.fromkeys(names[key], value))
    return FaceMesh(track_id, ((.5, .3),) * 478, values, yaw, pitch, 0.0, 110.0)


class CropAndMeshTests(unittest.TestCase):
    def test_crop_needs_nose_and_an_eye_and_is_centred_on_the_head(self):
        left, top, side = face_crop(person(), .35)
        self.assertAlmostEqual(left + side / 2, 500, delta=2)
        self.assertGreater(side, 40)  # ears 40 px apart -> 64 px span -> 80 px crop
        self.assertIsNone(face_crop(person(drop=(0,)), .35))        # no nose: back of the head
        self.assertIsNone(face_crop(person(drop=(1, 2)), .35))      # no eyes
        self.assertIsNotNone(face_crop(person(drop=(2, 4)), .35))   # three-quarter view

    def test_head_angles_from_pose_matrix(self):
        turned, _ = cv2.Rodrigues(np.array([0.0, math.radians(30), 0.0]))
        matrix = np.eye(4)
        matrix[:3, :3] = 2 * turned  # scaled, as the landmarker's matrix can be
        yaw, pitch, roll = head_angles(matrix)
        self.assertAlmostEqual(abs(yaw), 30, delta=.5)
        self.assertAlmostEqual(pitch, 0, delta=.5)
        nodded, _ = cv2.Rodrigues(np.array([math.radians(20), 0.0, 0.0]))
        matrix[:3, :3] = nodded
        self.assertAlmostEqual(abs(head_angles(matrix)[1]), 20, delta=.5)

    def test_mesh_must_sit_on_the_pose_face(self):
        crop = (100.0, 100.0, 200.0)
        nose = (200.0, 210.0)  # pose nose in pixels
        good = [(.3 + .4 * (i % 10) / 9, .3 + .4 * (i // 10) / 47) for i in range(478)]
        good[1] = (.5, .55)
        self.assertTrue(plausible(good, crop, nose, .15, .3))
        squeezed = [(.75 + .1 * x, y) for x, y in good]  # collapsed onto an ear
        self.assertFalse(plausible(squeezed, crop, nose, .15, .3))
        shifted = [(x, y) for x, y in good]
        shifted[1] = (.9, .9)  # nose tip far from the pose nose
        self.assertFalse(plausible(shifted, crop, nose, .15, .3))

    def test_missing_model_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FaceLandmarker(Path(directory) / "face_landmarker.task")


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def segment(name, track_id, time, label, role, confidence, coverage, zone=None):
            self.calls.append((name, time, label, coverage))
            return []
        self.segment = segment

    def faces(self, *names):
        return StudentFaces("v", dict.fromkeys(names or INDICATOR_NAMES, True), 1.0,
                            {"baseline_min_samples": 5}, self.segment)

    def labels(self, name):
        return [label for n, _, label, _ in self.calls if n == name]

    def warm(self, faces, samples=5, **usual):
        for time in range(samples):
            faces.update(float(time), 1, mesh(**usual), "student")

    def test_labels_wait_for_the_students_usual_face(self):
        faces = self.faces("smile")
        self.warm(faces, smile=.05)
        self.assertEqual(set(self.labels("smile")), {"unknown"})
        faces.update(5.0, 1, mesh(smile=.05), "student")
        faces.update(6.0, 1, mesh(smile=.70), "student")
        self.assertEqual(self.labels("smile")[-2:], ["not_smiling", "smiling"])

    def test_change_from_usual_level_is_needed(self):
        faces = self.faces("smile", "brows")
        self.warm(faces, smile=.5, down=.4)  # a cheerful student; brows read low from above
        faces.update(5.0, 1, mesh(smile=.6, down=.45), "student")
        self.assertEqual(self.labels("smile")[-1], "not_smiling")
        self.assertEqual(self.labels("brows")[-1], "neutral")
        faces.update(6.0, 1, mesh(smile=.6, down=.75), "student")
        self.assertEqual(self.labels("brows")[-1], "lowered")
        faces.update(7.0, 1, mesh(smile=.5, down=.4, up=.6), "student")
        self.assertEqual(self.labels("brows")[-1], "raised")

    def test_resting_level_survives_frequent_smiles(self):
        faces = self.faces("smile")
        for time in range(10):  # smiles in 6 of the first 10 faces
            faces.update(float(time), 1, mesh(smile=.65 if time % 5 in (0, 2, 4) else .05), "student")
        faces.update(10.0, 1, mesh(smile=.65), "student")
        self.assertEqual(self.labels("smile")[-1], "smiling")

    def test_lasting_brow_shift_becomes_the_new_normal(self):
        faces = self.faces("brows")
        self.warm(faces, samples=10, up=.1)
        for time in range(10, 24):  # head angle changes; browUp reads 0.5 from now on
            faces.update(float(time), 1, mesh(up=.5), "student")
        self.assertEqual(self.labels("brows")[-14], "raised")
        self.assertEqual(self.labels("brows")[-1], "neutral")

    def test_eyes_closed_and_yawn_need_minimum_duration(self):
        faces = self.faces("eyes_closed", "yawn")
        self.warm(faces, blink=.3, jaw=.05)
        for time in (5.0, 6.0):
            faces.update(time, 1, mesh(blink=.9, jaw=.8), "student")
        self.assertEqual(self.labels("eyes_closed")[-2:], ["open", "closed"])
        self.assertEqual(self.labels("yawn")[-2:], ["not_yawning", "yawning"])
        faces.update(7.0, 1, None, "student")  # face lost: unknown, and the run restarts
        faces.update(8.0, 1, mesh(blink=.9, jaw=.05), "student")
        self.assertEqual(self.labels("eyes_closed")[-2:], ["unknown", "open"])

    def test_head_direction(self):
        self.assertEqual(head_direction(5, -3, 25, 15), "toward_camera")
        self.assertEqual(head_direction(40, 0, 25, 15), "turned_right")
        self.assertEqual(head_direction(-40, 0, 25, 15), "turned_left")
        self.assertEqual(head_direction(10, -25, 25, 15), "down")
        faces = self.faces("head_direction")
        self.warm(faces, pitch=20)  # the camera looks down: usual pitch +20
        faces.update(5.0, 1, mesh(pitch=0), "student")
        self.assertEqual(self.labels("head_direction")[-1], "down")

    def test_no_face_is_unknown_and_scores_are_saved(self):
        faces = self.faces("smile")
        events, shown = faces.update(0.0, 1, None, "student")
        self.assertEqual((self.labels("smile"), shown["face"]), (["unknown"], False))
        events, _ = faces.update(1.0, 1, mesh(smile=.2), "student")
        scores = [e for e in events if e.indicator_name == "facial_scores"]
        self.assertEqual(scores[0].value["smile"], .2)
        self.assertIsNone(scores[0].value["baseline"])


class WiringTests(unittest.TestCase):
    SETTINGS = {"min_track_samples": 2, "max_track_gap_seconds": 2.5, "keypoint_confidence": .35,
                "edge_margin": .05, "sitting_angle_degrees": 135, "standing_angle_degrees": 160,
                "motion_low_max": .08, "motion_medium_max": .35, "motion_high_min": .8}

    def test_students_only(self):
        enabled = {"smile": True}
        students = FoundationAnalyzer("v", "student", "all_tracks", True, {}, 1.0, self.SETTINGS, enabled)
        events = []
        for time in range(14):  # confirmed at 1 s, then 10 faces to learn the resting level
            events += students.process(float(time), (person(),), None, {1: mesh(smile=.8 if time == 13 else .0)})
        events += students.finish(14.0)
        smiles = [e for e in events if e.indicator_name == "smile"]
        self.assertTrue(smiles and all(e.role == "student" for e in smiles))
        self.assertEqual(smiles[-1].value, "smiling")
        self.assertEqual(students.last_sample.tracks[0].face["smile"], "smiling")
        teacher = FoundationAnalyzer("v", "teacher", "single_visible", True, {}, 1.0, self.SETTINGS, enabled)
        self.assertIsNone(teacher.faces)
        later = []
        for time in range(4):
            later += teacher.process(float(time), (person(),), None, {1: mesh(smile=.8)})
        self.assertFalse([e for e in later if e.indicator_name in ("smile", "facial_scores")])

    def test_config_switches_and_validation(self):
        self.assertEqual(FACE_INDICATORS, INDICATOR_NAMES)
        flags = load_indicator_flags("configs/indicators.yaml")
        self.assertTrue(all(flags[name] for name in FACE_INDICATORS))
        chosen = select_indicators(flags, ["smile"])
        self.assertTrue(chosen["smile"] and not chosen["yawn"] and not chosen["position"])
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["faces"]["model"], "models/face/face_landmarker.task")
        for bad in ({"smile_min": 2}, {"yawn_min_seconds": 0}, {"baseline_min_samples": 0}):
            with self.assertRaises(ValueError):
                validate_config({**config, "faces": {**config["faces"], **bad}})

    def test_overlay_words_and_points(self):
        self.assertEqual(describe_face({"face": False, "smile": "unknown"}), ["face ? (none toward camera)"])
        self.assertEqual(describe_face({"face": True, "smile": "unknown"}), ["face: learning usual look"])
        line = describe_face({"face": True, "smile": "smiling", "head_direction": "turned_left",
                              "values": {"smile": .71, "yaw": -32}})
        self.assertEqual(line, ["SMILING 0.71 | head turned left (yaw -32)"])
        frame = np.zeros((100, 100, 3), np.uint8)
        drawn = ReviewOverlay("v", .35).draw(frame, (), None, {1: mesh()})
        self.assertGreater(int(drawn[30, 50].sum()), 0)  # mesh point at (0.5, 0.3)

    def test_summary_totals_student_seconds(self):
        faces = StudentFaces("v", {"smile": True}, 1.0, None, None)
        from src.analysis.events import make_event
        events = [make_event("v", "smile", 0, 4, "not_smiling", track_id=1, role="student"),
                  make_event("v", "smile", 4, 6, "smiling", track_id=1, role="student"),
                  make_event("v", "facial_scores", 4, 4, {"smile": .7}, track_id=1, role="student")]
        self.assertTrue(faces.active)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v.jsonl"
            write_jsonl(path, events)
            summary = summarize_list_a(path, "student_view", "v", 6.0, 1.0)
        self.assertEqual(summary["student_face_seconds"]["smile"], {"not_smiling": 4.0, "smiling": 2.0})
        self.assertEqual(summary["student_face_samples_with_mesh"], 1)
        self.assertTrue(any("not emotions" in line for line in summary["limitations"]))


if __name__ == "__main__":
    unittest.main()
