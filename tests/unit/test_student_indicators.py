"""List C (students): rules, seats, stitching, faces over time, interactions, cross-camera.

Synthetic skeletons in a 1000 x 1000 frame; no video, model, or download.
"""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.analysis.aggregates import summarize_list_a
from src.analysis.cross_camera import link_cameras
from src.analysis.events import make_event, with_real_time, write_jsonl
from src.analysis.facial_actions import StudentFaces, reversals
from src.analysis.interaction_indicators import StudentInteractions
from src.analysis.student_indicators import (
    DEFAULTS, INDICATOR_NAMES, StudentBehavior, StudentObservation, raise_level,
)
from src.inference.face_mesh import FaceMesh
from src.inference.hands import HandGesture
from src.inference.objects import DetectedObject
from src.inference.pose import Keypoint, PosePerson
from src.inference.tracking import TrackStitcher
from src.preprocessing.video import frames_with_subsamples
from src.utils.config import STUDENT_INDICATORS, load_indicator_flags

# A seated student facing the camera; shoulders 100 px apart. COCO left = image right.
BODY = {0: (.50, .30), 1: (.51, .29), 2: (.49, .29), 3: (.52, .30), 4: (.48, .30),
        5: (.55, .38), 6: (.45, .38), 7: (.57, .48), 8: (.43, .48), 9: (.56, .56), 10: (.44, .56),
        11: (.54, .60), 12: (.46, .60)}


def person(points=None, track_id=1, dx=0.0, dy=0.0, drop=()):
    points = {**BODY, **(points or {})}
    keypoints = tuple(Keypoint(points[i][0] + dx, points[i][1] + dy, .9) if i in points and i not in drop
                      else Keypoint(0, 0, 0.0) for i in range(17))
    return PosePerson(track_id, (.40 + dx, .25 + dy, .60 + dx, .70 + dy), .9, keypoints, 1000, 1000)


def mesh(track_id=1, yaw=0.0, pitch=10.0, mouth=0.0, **scores):
    return FaceMesh(track_id, ((.5, .3),) * 478, dict(scores), yaw, pitch, 0.0, 110.0, mouth)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name, track_id, time, label, role, confidence, coverage, zone=None):
        self.calls.append((name, track_id, time, label, zone))
        return []

    def labels(self, name, track_id=1):
        return [label for n, t, _, label, _ in self.calls if n == name and t == track_id]


class RaiseTests(unittest.TestCase):
    def test_levels_face_box_and_palm(self):
        s = DEFAULTS
        self.assertEqual(raise_level(person(), .35, s)[0], "none")
        self.assertEqual(raise_level(person({7: (.58, .30), 9: (.58, .18)}), .35, s)[:2], ("high", "left"))
        self.assertEqual(raise_level(person({7: (.62, .37), 9: (.62, .27)}), .35, s)[0], "medium")
        self.assertEqual(raise_level(person({7: (.60, .50), 9: (.61, .40)}), .35, s)[0], "low")
        chin = person({7: (.52, .50), 9: (.51, .36)})  # hand under the chin
        self.assertEqual(raise_level(chin, .35, s)[0], "none")
        palm = {"left": HandGesture("left", True, "open_palm", .9)}
        self.assertEqual(raise_level(chin, .35, s, palm)[:3], ("low", "left", True))
        fist = {"left": HandGesture("left", True, "fist", .9)}
        self.assertEqual(raise_level(person({7: (.60, .50), 9: (.61, .40)}), .35, s, fist)[0], "none")
        self.assertEqual(raise_level(person(drop=(5,)), .35, s)[0], "unknown")

    def test_low_raise_must_be_held_and_episodes_close(self):
        rec = Recorder()
        rules = StudentBehavior("v", {"hand_raise": True}, .35, 1.0, None, rec)
        low = person({7: (.60, .50), 9: (.61, .40)})
        palm = {"left": HandGesture("left", True, "open_palm", .9)}
        events = []
        for time, p in enumerate((low, low, low, person())):
            events += rules.update(float(time), [StudentObservation(1, p, "student", gestures=palm)])[0]
        self.assertEqual(rec.labels("hand_raise"), ["none", "low", "low", "none"])  # palm does not skip the hold
        episodes = [e for e in events if e.indicator_name == "hand_raise_episode"]
        self.assertEqual((episodes[0].start_time, episodes[0].end_time), (1.0, 3.0))
        self.assertEqual(episodes[0].value["max_level"], "low")


class SeatTests(unittest.TestCase):
    def test_seat_learnt_out_of_seat_and_reused_seat_id(self):
        rec = Recorder()
        rules = StudentBehavior("v", {"seat_occupancy": True, "out_of_seat": True}, .35, 1.0, None, rec)
        for time in range(7):
            rules.update(float(time), [StudentObservation(1, person(), "student")])
        self.assertEqual(rec.labels("seat_occupancy")[-1], "seat_01")
        rules.update(7.0, [StudentObservation(1, person(dy=-.13), "student", posture="standing")])
        self.assertEqual(rec.labels("out_of_seat")[-1], "in_seat")  # standing up in place
        for time in (8.0, 9.0, 10.0):
            rules.update(time, [StudentObservation(1, person(dx=.25), "student")])  # walks sideways
        self.assertEqual(rec.labels("out_of_seat")[-3:], ["in_seat", "out_of_seat", "out_of_seat"])  # held 2 s
        rules.close(1, "student")
        for time in range(11, 18):  # someone (a new tracker ID) sits in the same place
            rules.update(float(time), [StudentObservation(2, person(track_id=2), "student")])
        self.assertEqual(rec.labels("seat_occupancy", 2)[-1], "seat_01")
        seat_map = rules.seat_map_event(18.0)
        self.assertEqual(seat_map.value["seats"][0]["tracks"], [1, 2])


class HeadAndAlertnessTests(unittest.TestCase):
    def test_pose_head_pitch_from_ears_and_nose(self):
        from src.analysis.student_indicators import pose_looks_down
        self.assertFalse(pose_looks_down(person(), .35, DEFAULTS))  # ears level with the eyes
        ears_up = person({3: (.52, .279), 4: (.48, .279)})  # 0.55 eye gaps above the eyes: head bowed
        self.assertTrue(pose_looks_down(ears_up, .35, DEFAULTS))
        nose_low = person({0: (.50, .307)})  # 0.85 eye gaps below the eyes
        self.assertTrue(pose_looks_down(nose_low, .35, DEFAULTS))
        self.assertIsNone(pose_looks_down(person(drop=(1,)), .35, DEFAULTS))

    def test_head_down_hidden_head_and_sleep(self):
        rec = Recorder()
        rules = StudentBehavior("v", {"head_down": True, "alertness": True}, .35, 1.0, None, rec)
        for time in range(6):
            rules.update(float(time), [StudentObservation(1, person(), "student", motion="still")])
        self.assertEqual(rec.labels("head_down")[-1], "up")
        down = person({0: (.50, .37), 1: (.51, .365), 2: (.49, .365)}, drop=(3, 4))
        rules.update(6.0, [StudentObservation(1, down, "student", motion="still")])
        self.assertEqual(rec.labels("head_down")[-1], "down")
        bowed = person(drop=(1, 2, 3, 4))  # cap brim: nose seen, eyes hidden
        rules.update(6.5, [StudentObservation(1, bowed, "student", motion="still")])
        self.assertEqual(rec.labels("head_down")[-1], "down")
        hidden = person(drop=(0, 1, 2, 3, 4))  # face on the folded arms
        for time in range(7, 30):
            rules.update(float(time), [StudentObservation(1, hidden, "student", motion="still",
                                                          shoulder_rise=-0.5)])
        self.assertEqual(rec.labels("head_down")[-1], "down")
        self.assertEqual(rec.labels("alertness")[-1], "possible_sleep")
        # Bent over a phone: head down and still for as long, but the face is visible.
        rec2 = Recorder()
        phone_user = StudentBehavior("v", {"alertness": True}, .35, 1.0, None, rec2)
        bent = person({0: (.50, .37), 1: (.51, .365), 2: (.49, .365)}, drop=(3, 4))
        for time in range(30):
            phone_user.update(float(time), [StudentObservation(1, person() if time < 6 else bent, "student",
                                                               motion="still")])
        self.assertNotIn("possible_sleep", rec2.labels("alertness"))
        self.assertEqual(rec2.labels("alertness")[-1], "awake")

    def test_dozing_and_tired_from_face_labels(self):
        rec = Recorder()
        rules = StudentBehavior("v", {"alertness": True}, .35, 1.0, None, rec)
        rules.update(0.0, [StudentObservation(1, person(), "student", face=mesh(),
                                              face_labels={"eyes_closed": "closed"})])
        rules.update(1.0, [StudentObservation(1, person(), "student", face=mesh(),
                                              face_labels={"eyes_closed": "open", "yawn": "yawning"})])
        rules.update(2.0, [StudentObservation(1, person(), "student", face=mesh(),
                                              face_labels={"eyes_closed": "open"})])
        self.assertEqual(rec.labels("alertness"), ["dozing", "tired", "awake"])


class DeskActivityTests(unittest.TestCase):
    ON = {"device_use": True, "note_taking": True, "reading": True}
    DOWN = {0: (.50, .37), 1: (.51, .365), 2: (.49, .365)}

    def run_rules(self, frames, objects=()):
        rec = Recorder()
        rules = StudentBehavior("v", self.ON, .35, 1.0, None, rec)
        for time, (p, face) in enumerate(frames):
            rules.update(float(time), [StudentObservation(1, p, "student", face=face)], objects)
        return rec

    def test_phone_needs_looking_down_or_held_up(self):
        phone = DetectedObject("cell phone", .8, (.43, .55, .46, .58))
        warm = [(person(), None)] * 5
        rec = self.run_rules(warm + [(person(), None)], [phone])
        self.assertEqual(rec.labels("device_use")[-1], "none")  # phone by the hand, head up
        rec = self.run_rules(warm + [(person(self.DOWN, drop=(3, 4)), None)], [phone])
        self.assertEqual(rec.labels("device_use")[-1], "phone")

    def test_laptop_hidden_phone_writing_reading(self):
        warm = [(person(), None)] * 5
        laptop = DetectedObject("laptop", .9, (.40, .46, .60, .64))  # open: box 180 x 200 px
        rec = self.run_rules(warm + [(person(self.DOWN, drop=(3, 4)), None)], [laptop])
        self.assertEqual(rec.labels("device_use")[-1], "laptop")
        closed = DetectedObject("laptop", .9, (.40, .55, .60, .58))  # a flat slab
        rec = self.run_rules(warm + [(person(self.DOWN, drop=(3, 4)), None)], [closed])
        self.assertEqual(rec.labels("device_use")[-1], "none")
        held = {**self.DOWN, 9: (.52, .50), 10: (.48, .50)}  # both hands together at the chest
        rec = self.run_rules(warm + [(person(held, drop=(3, 4)), None)] * 3)
        self.assertEqual(rec.labels("device_use")[-1], "phone_held_possible")
        self.assertNotIn("writing", rec.labels("note_taking"))
        low = {**self.DOWN, 9: (.52, .62), 10: (.48, .62)}  # hands together in the lap
        rec = self.run_rules(warm + [(person(low, drop=(3, 4)), None)] * 3)
        self.assertEqual(rec.labels("device_use")[-3:], ["none", "none", "phone_hidden_possible"])
        writing = [(person({**self.DOWN, 10: (.44 + .01 * (i % 2), .50)}, drop=(3, 4)), None) for i in range(4)]
        rec = self.run_rules(warm + writing)
        self.assertEqual(rec.labels("note_taking")[-1], "writing")
        usual = mesh(eyeLookDownLeft=.3, eyeLookDownRight=.3)  # watching the front reads a little down
        eyes_down = mesh(eyeLookDownLeft=.6, eyeLookDownRight=.6)
        rec = self.run_rules([(person(), usual)] * 5 + [(person(), eyes_down)] * 3)
        self.assertEqual(rec.labels("reading")[-1], "reading")
        rec = self.run_rules([(person(), eyes_down)] * 8)  # always this low: the student's normal
        self.assertNotIn("reading", rec.labels("reading"))

    def test_response_gesture(self):
        rec = Recorder()
        rules = StudentBehavior("v", {"response_gesture": True}, .35, 1.0, None, rec)
        thumbs = {"right": HandGesture("right", True, "thumbs_up", .8)}
        rules.update(0.0, [StudentObservation(1, person(), "student", gestures=thumbs)])
        rules.update(1.0, [StudentObservation(1, person(), "student", gestures={})])
        self.assertEqual(rec.labels("response_gesture"), ["thumbs_up", "unknown"])


class FaceSeriesTests(unittest.TestCase):
    def test_reversals(self):
        self.assertEqual(reversals([0, -6, 0], 5), 1)
        self.assertEqual(reversals([0, 10, 0, -10, 0], 8), 2)  # left-right-left
        self.assertEqual(reversals([0, 2, 1, 3], 5), 0)

    def test_talking_nod_and_shake(self):
        rec = Recorder()
        faces = StudentFaces("v", {"talking": True, "head_gesture": True}, 1.0, None, rec)
        for second in range(3):  # mouth opens and closes within each second
            series = [(second - .8 + .2 * k, mesh(mouth=.06 * (k % 2))) for k in range(4)]
            faces.update(float(second), 1, mesh(mouth=0.0), "student", series)
        self.assertEqual(rec.labels("talking")[-1], "talking")
        nod = [(3.2, mesh(pitch=10)), (3.4, mesh(pitch=3)), (3.6, mesh(pitch=10)), (3.8, mesh(pitch=10))]
        faces.update(4.0, 1, mesh(pitch=10), "student", nod)
        self.assertEqual(rec.labels("head_gesture")[-1], "nod")
        shake = [(4.2 + .2 * k, mesh(yaw=y)) for k, y in enumerate((10, 0, -10, 0))]
        faces.update(5.0, 1, mesh(yaw=0), "student", shake)
        self.assertEqual(rec.labels("head_gesture")[-1], "shake")
        quiet = StudentFaces("v", {"talking": True}, 1.0, None, rec)
        quiet.update(0.0, 2, mesh(2), "student", [(-.5, mesh(2)), (-.3, mesh(2))])
        self.assertEqual(rec.labels("talking", 2)[-1], "silent")


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()
        self.rules = StudentInteractions("v", {"head_target": True, "orientation_proxy": True,
                                               "hand_raise_outcome": True}, .35, 1.0, None, self.rec)

    def students(self, a_dir="turned_right", b_talk="silent", a_talk="silent"):
        a = StudentObservation(1, person(dx=-.2), "student", face=mesh(),
                               face_labels={"head_direction": a_dir, "talking": a_talk})
        b = StudentObservation(2, person(track_id=2, dx=.1), "student", face=mesh(2),
                               face_labels={"head_direction": "toward_camera", "talking": b_talk})
        return [a, b]

    def test_targets_and_proxy(self):
        self.rules.update(0.0, self.students(), {})
        self.assertEqual(self.rec.labels("head_target")[-1], "peer")
        self.assertEqual(self.rec.labels("orientation_proxy")[-1], "off_target")  # facing a classmate
        self.rules.update(1.0, self.students(b_talk="talking"), {})
        self.assertEqual(self.rec.labels("head_target")[-1], "talking_peer")
        self.assertEqual(self.rec.labels("orientation_proxy")[-1], "off_target")  # chatting, no raise
        # Student 2 raised a hand and now answers: facing them is on target.
        self.rules.update(2.0, self.students(b_talk="talking"), {2: {"hand_raise_now": "medium"}})
        self.assertEqual(self.rec.labels("orientation_proxy")[-1], "on_target")
        self.assertEqual(self.rec.labels("orientation_proxy", 2)[-1], "speaking")
        self.rules.update(3.0, self.students(a_dir="toward_camera"), {})
        self.assertEqual(self.rec.labels("head_target")[-1], "front")

    def test_hand_raise_outcomes(self):
        events = []
        events += self.rules.update(0.0, self.students(), {2: {"hand_raise_now": "high"}})[0]
        done = [{"track_id": 2, "start": 0.0, "end": 1.0}]
        for time in (1.0, 2.0, 3.0):  # student 2 faces the front and talks for 3 s
            events += self.rules.update(time, self.students(b_talk="talking"), {}, done if time == 1.0 else [])[0]
        outcome = [e for e in events if e.indicator_name == "hand_raise_outcome"]
        self.assertEqual(outcome[0].value["outcome"], "answered_likely")
        # Student 1 raises, then chats with student 2 (turned toward them): not an answer.
        events = self.rules.update(4.0, self.students(), {1: {"hand_raise_now": "low"}})[0]
        for time in range(5, 21):
            done = [{"track_id": 1, "start": 4.0, "end": 5.0}] if time == 5 else []
            events += self.rules.update(float(time), self.students(a_talk="talking"), {}, done)[0]
        outcome = [e for e in events if e.indicator_name == "hand_raise_outcome"]
        self.assertEqual(outcome[0].value["outcome"], "no_answer_seen")


class StitchTests(unittest.TestCase):
    def walker(self, track_id, x, height=.6):
        return PosePerson(track_id, (x, .3, x + .1, .3 + height), .9,
                          tuple(Keypoint(0, 0, 0) for _ in range(17)), 1000, 1000)

    def test_walking_person_keeps_one_id_but_ambiguity_is_left_alone(self):
        stitch = TrackStitcher(2.5, 1.5, 0.5)
        stitch(0.0, [self.walker(1, .10)])
        stitch(1.0, [self.walker(1, .25)])
        joined = stitch(2.0, [self.walker(7, .55)])  # ByteTrack lost the overlap: new raw ID
        self.assertEqual(joined[0].track_id, 1)
        self.assertEqual(stitch(3.0, [self.walker(7, .70)])[0].track_id, 1)
        far = TrackStitcher(2.5, 1.5, 0.5)
        far(0.0, [self.walker(1, .0, .1)])
        self.assertEqual(far(1.0, [self.walker(9, .9, .1)])[0].track_id, 9)  # too far to walk in 1 s
        two = TrackStitcher(2.5, 1.5, 0.5)
        two(0.0, [self.walker(1, .40), self.walker(2, .48)])
        self.assertEqual(two(1.0, [self.walker(9, .44)])[0].track_id, 9)  # either could be it


class FramesTests(unittest.TestCase):
    def test_extra_frames_between_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            for index in range(20):
                writer.write(np.full((24, 32, 3), index * 10, np.uint8))
            writer.release()
            rows = [(round(t, 2), s) for _, t, _, s in frames_with_subsamples(path, 1.0, 0, None, 4)]
        self.assertEqual([t for t, s in rows if s], [0.0, 1.0])
        self.assertEqual([t for t, s in rows if not s], [0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8])


class CrossCameraTests(unittest.TestCase):
    def test_student_to_board_and_teacher_into_student_view(self):
        start = datetime(2026, 11, 3, 19, 0, tzinfo=timezone(timedelta(hours=4)))

        def pos(track, t, role, center=(.5, .5)):
            return with_real_time(make_event("v", "position", t, t, {"box_center": list(center)},
                                             track_id=track, role=role), start)
        student = [pos(3, t, "student") for t in range(0, 11)]
        student.append(with_real_time(make_event("v", "entry_exit", 10, 10, "apparent_exit_at_frame_edge",
                                                 track_id=3, role="student"), start))
        student += [pos(12, t, "student") for t in range(65, 80)]
        student.append(with_real_time(make_event("v", "entry_exit", 65, 65, "apparent_entry_at_frame_edge",
                                                 track_id=12, role="student"), start))
        instructor = [pos(1, t, "teacher", (.2, .5)) for t in range(0, 60)]
        instructor += [pos(9, t, "unknown", (.75, .6)) for t in range(14, 30)]
        instructor.append(with_real_time(make_event("v", "teacher_presence", 0, 60, "visible",
                                                    role="teacher"), start))
        board = ((.65, .25), (.90, .25), (.90, .50), (.65, .50))
        rows = lambda events: [asdict(e) for e in events]  # as read back from the JSONL
        report = link_cameras(rows(student), rows(instructor), board=board)
        to_front = [link for link in report["links"] if link["link"] == "student_to_front"]
        self.assertEqual(to_front[0]["student_view_track"], 3)
        self.assertEqual(to_front[0]["instructor_view_tracks"][0]["track_id"], 9)
        self.assertEqual(to_front[0]["instructor_view_tracks"][0]["at_board_seconds"], 16.0)
        teacher = [link for link in report["links"] if link["link"] == "teacher_in_student_view"]
        self.assertEqual(teacher[0]["student_view_tracks"][0]["track_id"], 12)


class SummaryAndConfigTests(unittest.TestCase):
    def test_switches(self):
        self.assertEqual(set(STUDENT_INDICATORS), set(INDICATOR_NAMES) | {"head_target", "orientation_proxy",
                                                                            "hand_raise_outcome"})
        flags = load_indicator_flags("configs/indicators.yaml")
        self.assertTrue(all(flags[name] for name in STUDENT_INDICATORS))

    def test_seats_and_rows_in_summary(self):
        events = [make_event("v", "seat_occupancy", 0, 10, "seat_01", track_id=1, role="student", zone="seat_01"),
                  make_event("v", "seat_occupancy", 0, 10, "seat_02", track_id=2, role="student", zone="seat_02"),
                  make_event("v", "hand_raise_episode", 3, 5, {"max_level": "high", "side": "left",
                                                               "open_palm_seen": True, "samples": 2},
                             track_id=2, role="student"),
                  make_event("v", "orientation_proxy", 0, 6, "on_target", track_id=1, role="student"),
                  make_event("v", "orientation_proxy", 6, 10, "off_target", track_id=1, role="student"),
                  make_event("v", "seat_map", 10, 10, {"seats": [
                      {"seat_id": "seat_01", "x": .3, "y": .7, "shoulder_width": .05, "tracks": [1]},
                      {"seat_id": "seat_02", "x": .5, "y": .3, "shoulder_width": .04, "tracks": [2]}]},
                      role="classroom")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v.jsonl"
            write_jsonl(path, events)
            summary = summarize_list_a(path, "student_view", "v", 10.0, 1.0, frame_aspect=16 / 9)
        seats = {s["seat_id"]: s for s in summary["seats"]["seats"]}
        self.assertEqual((seats["seat_01"]["row"], seats["seat_02"]["row"]), (1, 2))  # front row first
        self.assertEqual(seats["seat_02"]["hand_raises"], 1)
        self.assertEqual(seats["seat_01"]["on_target_seconds"], 6.0)
        self.assertEqual(summary["student_hand_raises"]["episodes_by_max_level"], {"high": 1})
        self.assertEqual(summary["student_orientation_proxy"]["on_target_fraction_of_known"], .6)


if __name__ == "__main__":
    unittest.main()
