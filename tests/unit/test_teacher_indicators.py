"""List B teacher rules on synthetic skeletons; no video or model needed."""

import unittest

from src.analysis.teacher_indicators import (
    TeacherBehavior, facing_direction, head_orientation, lean_label, pointing, raised_hands,
    torso_tilt,
)
from src.analysis.visual_indicators import FoundationAnalyzer, inside_any_target
from src.inference.pose import Keypoint, PosePerson

# Normalized 1000 x 1000 frame. Teacher standing at x ~0.3 facing the camera:
# the person's left shoulder (5) appears on the image right.
FRONT = {0: (.30, .30), 1: (.31, .29), 2: (.29, .29), 3: (.32, .30), 4: (.28, .30),
         5: (.34, .36), 6: (.26, .36), 7: (.35, .45), 8: (.25, .45), 9: (.35, .53),
         10: (.25, .53), 11: (.33, .56), 12: (.27, .56)}
TARGETS = {"screen": ((.40, .25), (.60, .25), (.60, .50), (.40, .50)),
           "board": ((.65, .25), (.90, .25), (.90, .50), (.65, .50))}
SETTINGS = {"pointing_min_elbow_degrees": 150, "pointing_min_reach": .6,
            "pointing_min_raise_degrees": 40, "raised_max_from_vertical_degrees": 35}


def pose(points, track_id=1, box=(.22, .26, .38, .95), drop=()):
    keypoints = tuple(Keypoint(*points[i], .9) if i in points and i not in drop
                      else Keypoint(0, 0, 0.0) for i in range(17))
    return PosePerson(track_id, box, .9, keypoints, 1000, 1000)


def moved(points, dx=0.0, dy=0.0, only=None):
    return {i: (x + dx, y + dy) if only is None or i in only else (x, y) for i, (x, y) in points.items()}


class FacingTests(unittest.TestCase):
    def test_front_profile_back(self):
        self.assertEqual(facing_direction(pose(FRONT), .35), "toward_class")
        # Profile looking toward the image left: only the person's left eye and ear show.
        self.assertEqual(facing_direction(pose(FRONT, drop=(2, 4)), .35), "sideways_left")
        self.assertEqual(facing_direction(pose(FRONT, drop=(1, 3)), .35), "sideways_right")
        self.assertEqual(facing_direction(pose(FRONT, drop=(0, 1, 2)), .35), "away_from_class")
        self.assertEqual(facing_direction(pose({}), .35), "unknown")

    def test_head_targets(self):
        self.assertEqual(head_orientation(pose(FRONT), "toward_class", .35, TARGETS, 1.0, .03), "students")
        down = moved(FRONT, dy=.03, only=(0,))  # nose drops well below the eyes
        self.assertEqual(head_orientation(pose(down), "toward_class", .35, TARGETS, 1.0, .03), "notes")
        # Side-on toward the image right at head height: the screen is next in line.
        self.assertEqual(head_orientation(pose(FRONT), "sideways_right", .35, TARGETS, 1.0, .03), "screen")
        self.assertEqual(head_orientation(pose(FRONT), "sideways_left", .35, TARGETS, 1.0, .03), "side_left")
        at_board = moved(FRONT, dx=.45)
        self.assertEqual(head_orientation(pose(at_board), "away_from_class", .35, TARGETS, 1.0, .03), "board")
        self.assertEqual(head_orientation(pose(FRONT), "away_from_class", .35, TARGETS, 1.0, .03), "front_wall")


class BodyTests(unittest.TestCase):
    def test_raised_hands_and_lean(self):
        self.assertEqual(raised_hands(pose(FRONT), .35), "none")
        up = {**FRONT, 7: (.35, .30), 9: (.35, .22)}
        self.assertEqual(raised_hands(pose(up), .35), "one")
        both = {**up, 8: (.25, .30), 10: (.25, .22)}
        self.assertEqual(raised_hands(pose(both), .35), "both")
        self.assertAlmostEqual(torso_tilt(pose(FRONT), .35), 0.0)
        tilted = moved(FRONT, dx=.12, only=(0, 1, 2, 3, 4, 5, 6))
        self.assertGreater(torso_tilt(pose(tilted), .35), 25)
        self.assertEqual(lean_label(30.0, 10, 25), "strong_lean_right")
        self.assertEqual(lean_label(-12.0, 10, 25), "lean_left")
        self.assertEqual(lean_label(None, 10, 25), "unknown")

    def test_pointing_targets(self):
        arm_right = {**FRONT, 7: (.43, .36), 9: (.52, .36)}  # left arm straight out toward the screen
        result = pointing(pose(arm_right), .35, "toward_class", TARGETS, SETTINGS)
        self.assertEqual(result.target, "screen")
        no_target = {**FRONT, 8: (.17, .38), 10: (.08, .40)}  # right arm out toward the empty side
        self.assertEqual(pointing(pose(no_target), .35, "toward_class", TARGETS, SETTINGS).target,
                         "students")
        self.assertIsNone(pointing(pose(FRONT), .35, "toward_class", TARGETS, SETTINGS))  # arms down
        straight_up = {**FRONT, 7: (.34, .27), 9: (.34, .18)}
        self.assertIsNone(pointing(pose(straight_up), .35, "toward_class", TARGETS, SETTINGS))


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def segment(name, track_id, time, label, role, confidence, coverage, zone=None):
            self.calls.append((name, time, label, zone))
            return []
        self.segment = segment

    def behavior(self, **on):
        return TeacherBehavior("v", on, TARGETS, {}, .35, 1.0, None, self.segment)

    def labels(self, name):
        return [label for n, _, label, _ in self.calls if n == name]

    def test_writing_needs_back_to_class_and_wrist_on_board(self):
        teacher = self.behavior(board_interaction=True, writing_on_board=True)
        # At the board, back to the class, right wrist raised onto the board.
        writer = moved(FRONT, dx=.45)
        writer = {i: p for i, p in writer.items() if i not in (0, 1, 2)}
        writer.update({8: (.70, .40), 10: (.72, .34)})
        for time in range(4):
            teacher.update(float(time), 1, pose(writer, box=(.67, .26, .83, .95)), "teacher")
        self.assertEqual(self.labels("board_interaction")[-1], "board")
        self.assertEqual(self.labels("writing_on_board")[-2:], ["writing", "writing"])
        zones = [zone for name, _, label, zone in self.calls if name == "writing_on_board" and label == "writing"]
        self.assertEqual(zones[-1], "board_left")
        facing_class = self.behavior(writing_on_board=True)
        for time in range(4):
            facing_class.update(float(time), 1, pose(FRONT), "teacher")
        self.assertNotIn("writing", self.labels("writing_on_board")[-4:])

    def test_gestures_versus_walking(self):
        teacher = self.behavior(gestures=True)
        teacher.update(0.0, 1, pose(FRONT), "teacher")
        teacher.update(1.0, 1, pose({**FRONT, 9: (.42, .45)}), "teacher")  # hand moves, feet still
        teacher.update(2.0, 1, pose(moved(FRONT, dx=.25)), "teacher")  # whole body walks
        self.assertEqual(self.labels("gesture"), ["unknown", "gesturing", "walking"])

    def test_small_raised_hand_movements_count_but_face_and_hidden_hands_do_not(self):
        chest = {**FRONT, 7: (.38, .42), 9: (.36, .40)}  # forearm up, hand at chest height
        nudged = {**chest, 9: (.36, .38)}  # moves 0.1 torso lengths in a second
        teacher = self.behavior(gestures=True)
        for time, points in enumerate((chest, nudged)):
            teacher.update(float(time), 1, pose(points), "teacher")
        self.assertEqual(self.labels("gesture")[-1], "gesturing")
        self.calls.clear()
        face = {**FRONT, 7: (.36, .36), 9: (.31, .32)}  # hand on the chin
        rubbing = {**face, 9: (.32, .33)}  # still within 0.6 shoulder widths of the nose
        teacher = self.behavior(gestures=True)
        for time, points in enumerate((face, rubbing)):
            teacher.update(float(time), 1, pose(points), "teacher")
        self.assertEqual(self.labels("gesture")[-1], "hands_still")
        self.calls.clear()
        hidden = TeacherBehavior("v", {"gestures": True}, TARGETS, {}, .35, 1.0, None, self.segment)
        for time, x in enumerate((.35, .45)):  # a low-confidence wrist that jumps
            points = {**FRONT, 9: (x, .53)}
            keypoints = tuple(Keypoint(*points[i], .4 if i == 9 else .9) if i in points
                              else Keypoint(0, 0, 0.0) for i in range(17))
            hidden.update(float(time), 1, PosePerson(1, (.22, .26, .38, .95), .9, keypoints, 1000, 1000),
                          "teacher")
        self.assertEqual(self.labels("gesture")[-1], "hands_still")

    def test_stationary_episode_needs_minimum_duration(self):
        teacher = self.behavior(stationary_episodes=True)
        events = []
        for time in range(8):  # still for 7 s, then walks away
            events += teacher.update(float(time), 1, pose(FRONT), "teacher")[0]
        # Hip speed is a median over three samples: one jittery step does not
        # end an episode, two walking samples do.
        events += teacher.update(8.0, 1, pose(moved(FRONT, dx=.2)), "teacher")[0]
        events += teacher.update(9.0, 1, pose(moved(FRONT, dx=.4)), "teacher")[0]
        episodes = [e for e in events if e.indicator_name == "stationary_episode"]
        self.assertEqual(len(episodes), 1)
        self.assertEqual((episodes[0].start_time, episodes[0].end_time), (0.0, 8.0))
        short = self.behavior(stationary_episodes=True)
        for time in range(3):
            short.update(float(time), 1, pose(FRONT), "teacher")
        self.assertEqual(short.close(1, "teacher", 3.0), [])  # 2-3 s: too short


class ScreenFilterTests(unittest.TestCase):
    def test_people_shown_on_the_screen_are_ignored_not_people_in_front(self):
        on_screen = pose(FRONT, box=(.45, .30, .50, .45))  # a photo of a person in the slide
        in_front = pose(FRONT, box=(.42, .28, .55, .95))  # a real teacher reaching below it
        self.assertTrue(inside_any_target(on_screen, TARGETS, ("screen",)))
        self.assertFalse(inside_any_target(in_front, TARGETS, ("screen",)))
        self.assertFalse(inside_any_target(on_screen, TARGETS, ()))  # filter off


class AnalyzerTests(unittest.TestCase):
    def test_teacher_rules_run_only_on_teacher_camera_and_close_with_track(self):
        settings = {"min_track_samples": 2, "max_track_gap_seconds": 2.5, "keypoint_confidence": .35,
                    "edge_margin": .05, "sitting_angle_degrees": 135, "standing_angle_degrees": 160,
                    "motion_low_max": .08, "motion_medium_max": .35, "motion_high_min": .8}
        enabled = {"facing_direction": True, "stationary_episodes": True, "teacher_presence": True}
        analyzer = FoundationAnalyzer("v", "teacher", "single_visible", True, {}, 1.0, settings,
                                      enabled, targets=TARGETS)
        events = []
        for time in range(8):
            events += analyzer.process(float(time), (pose(FRONT),))
        events += analyzer.finish(8.0)
        names = {e.indicator_name for e in events}
        self.assertIn("facing_direction", names)
        self.assertIn("stationary_episode", names)
        self.assertIsNotNone(analyzer.last_sample.tracks[0].teacher)
        students = FoundationAnalyzer("v", "student", "all_tracks", True, {}, 1.0, settings, enabled)
        self.assertIsNone(students.teacher)


if __name__ == "__main__":
    unittest.main()
