"""Synthetic geometry and timeline tests; no video or model download needed."""

import unittest

from src.analysis.visual_indicators import (
    FoundationAnalyzer, motion_level, point_in_polygon, posture_with_basis,
    restrict_joints, shoulder_rise_posture, shoulder_sample, sitting_standing,
    visibility, zone_for,
)
from src.inference.pose import Keypoint, PosePerson, joint_indices
from src.preprocessing.camera_motion import IDENTITY, CameraMotion


def person(track_id=1, posture="standing", x_shift=0.0, visible=True, ankles=True, knee_y=None):
    points = [(0.5, 0.1)] * 17
    if posture == "standing":
        for shoulder, hip, knee, ankle, x in ((5, 11, 13, 15, .45),
                                               (6, 12, 14, 16, .55)):
            points[shoulder] = (x, .20)
            points[hip] = (x, .45)
            points[knee] = (x, knee_y or .65)
            points[ankle] = (x, .90)
    elif posture == "sitting":
        for shoulder, hip, knee, ankle, x in ((5, 11, 13, 15, .35),
                                               (6, 12, 14, 16, .45)):
            points[shoulder] = (x, .20)
            points[hip] = (x, .45)
            points[knee] = (x + .18, .45)
            points[ankle] = (x + .18, .75)
    points[7], points[8] = (.4, .35), (.6, .35)
    points[9], points[10] = (.4, .50), (.6, .50)
    keypoints = tuple(Keypoint(x + x_shift, y, (.9 if visible else 0.0) if ankles or i < 15 else 0.0)
                      for i, (x, y) in enumerate(points))
    return PosePerson(track_id, (.25 + x_shift, .15, .75 + x_shift, .95),
                      .9, keypoints, 640, 480)


def student(rise=0.0, x_shift=0.0):
    """Seated student; shoulders 128 px apart, raised by `rise` shoulder widths.

    Legs are placed (with high confidence) as a pose model guesses joints
    hidden by a desk; student rules must not use them.
    """
    y = .5 - rise * 128 / 480
    points = [(.5, y - .1)] * 5 + [(.4, y), (.6, y), (.38, y + .12), (.62, y + .12),
                                   (.4, y + .2), (.6, y + .2), (.42, y + .25), (.58, y + .25),
                                   (.45, .8), (.55, .8), (.45, .95), (.55, .95)]
    return PosePerson(1, (.3 + x_shift, .1, .7 + x_shift, .9), .9,
                      tuple(Keypoint(x + x_shift, py, .9) for x, py in points), 640, 480)


STUDENT_JOINTS = joint_indices(["head", "arms", "hips"])

SETTINGS = {"min_track_samples": 2, "max_track_gap_seconds": .6,
            "keypoint_confidence": .35, "edge_margin": .05,
            "sitting_angle_degrees": 135, "standing_angle_degrees": 160,
            "motion_low_max": .08, "motion_medium_max": .35,
            "motion_high_min": .8}


class GeometryTests(unittest.TestCase):
    def test_clear_postures_and_unknown(self):
        self.assertEqual(sitting_standing(person(), .35, 135, 160), "standing")
        self.assertEqual(sitting_standing(person(posture="sitting"), .35, 135, 160), "sitting")
        self.assertEqual(sitting_standing(person(visible=False), .35, 135, 160), "unknown")

    def test_upper_legs_classify_when_ankles_are_missing(self):
        def check(pose, enabled=True):
            return posture_with_basis(pose, .35, 135, 160, upper_legs=enabled)
        self.assertEqual(check(person(ankles=False)), ("standing", "upper_legs"))
        self.assertEqual(check(person(posture="sitting", ankles=False)), ("sitting", "upper_legs"))
        self.assertEqual(check(person(ankles=False), enabled=False), ("unknown", None))
        self.assertEqual(check(person()), ("standing", "full_legs"))
        # Thigh much shorter than the torso is foreshortened: direction not trusted.
        self.assertEqual(check(person(ankles=False, knee_y=.5)), ("unknown", None))
        # Waist-up view (no knees) cannot separate sitting from standing.
        waist_up = person(ankles=False)
        waist_up = PosePerson(1, waist_up.box, .9, tuple(
            Keypoint(p.x, p.y, 0.0 if i >= 13 else p.confidence)
            for i, p in enumerate(waist_up.keypoints)), 640, 480)
        self.assertEqual(check(waist_up), ("unknown", None))

    def test_ambiguous_full_legs_are_not_overridden_by_upper_legs(self):
        bent = person(knee_y=.68)
        knees = list(bent.keypoints)
        for knee in (13, 14):  # push knees forward: knee angle ~150, between thresholds
            knees[knee] = Keypoint(knees[knee].x + .05, knees[knee].y, .9)
        bent = PosePerson(1, bent.box, .9, tuple(knees), 640, 480)
        self.assertEqual(posture_with_basis(bent, .35, 135, 160, upper_legs=True), ("unknown", None))

    def test_restrict_joints_drops_unused_and_out_of_frame_joints(self):
        pose = person()
        below = PosePerson(1, pose.box, .9, pose.keypoints[:15] + (Keypoint(.45, 1.04, .8),
                                                                   pose.keypoints[16]), 640, 480)
        kept = restrict_joints(below, joint_indices(None))
        self.assertIsNone(kept.point(15, .35))  # feet below the frame are model guesses
        self.assertIsNotNone(kept.point(16, .35))
        seated = restrict_joints(pose, STUDENT_JOINTS)
        self.assertTrue(all(seated.point(i, .35) is None for i in (13, 14, 15, 16)))
        self.assertIsNotNone(seated.point(11, .35))
        self.assertEqual(seated.keypoints[13].x, pose.keypoints[13].x)  # position kept
        parts = visibility(seated, (), .35, STUDENT_JOINTS)
        self.assertTrue(parts["upper_body_visible"])
        self.assertEqual((parts["lower_body_visible"], parts["full_body_visible"]), (None, None))

    def test_shoulder_rise_compares_with_the_tracks_seated_level(self):
        history = [shoulder_sample(student(), .35)] * 15

        def check(sample_history=history, **kwargs):
            return shoulder_rise_posture(shoulder_sample(student(**kwargs), .35),
                                         sample_history, .7, .35, 1.0, 15)
        self.assertEqual(check(), ("sitting", 0.0))
        self.assertEqual(check(rise=1.0), ("standing", 1.0))
        self.assertEqual(check(rise=.5)[0], "unknown")  # between the thresholds
        self.assertEqual(check(rise=-.6)[0], "sitting")  # leaning on the desk
        self.assertEqual(check(rise=1.0, x_shift=.3), ("unknown", None))  # away from the seat
        self.assertEqual(check(history[:5], rise=1.0), ("unknown", None))  # too little history
        # Brief bending does not move the seated level.
        bent = history + [shoulder_sample(student(rise=-1.0), .35)] * 3
        self.assertEqual(check(bent, rise=1.0)[0], "standing")

    def test_stationary_motion_and_fixed_camera_gate(self):
        pose = person()
        self.assertEqual(motion_level(pose, pose, .2, .35, .08, .35, .8, True)[0], "still")
        self.assertEqual(motion_level(pose, pose, .2, .35, .08, .35, .8, False)[0], "unknown")

    def test_zones_and_missing_joints(self):
        square = ((.0, .0), (.5, .0), (.5, .5), (.0, .5))
        self.assertTrue(point_in_polygon((.25, .25), square))
        self.assertEqual(zone_for((.8, .8), {"left": square}), "outside_calibrated_zones")
        self.assertFalse(visibility(person(visible=False), (), .35)["full_body_visible"])


class TimelineTests(unittest.TestCase):
    def test_student_count_track_role_position_and_posture(self):
        analyzer = FoundationAnalyzer("video_01", "student", "all_tracks", True,
                                      {}, .2, SETTINGS)
        observed = []
        for time in (0.0, .2, .4):
            observed.extend(analyzer.process(time, (person(),)))
        observed.extend(analyzer.finish(.6))
        counts = [event.value for event in observed
                  if event.indicator_name == "visible_student_count"]
        self.assertEqual(counts, [0, 1, 1])
        self.assertEqual(len([event for event in observed
                              if event.indicator_name == "persistent_anonymous_identity"]), 1)
        self.assertTrue(any(event.indicator_name == "sitting_standing" and event.value == "standing"
                            for event in observed))
        self.assertTrue(all(event.confidence is None for event in observed
                            if event.indicator_name == "role_classification"))

    def test_instructor_view_with_two_people_is_unknown(self):
        analyzer = FoundationAnalyzer("video_02", "teacher", "single_visible", True,
                                      {}, .2, SETTINGS)
        events = analyzer.process(0, (person(1), person(2, x_shift=.1)))
        events += analyzer.process(.2, (person(1), person(2, x_shift=.1)))
        events += analyzer.finish(.4)
        self.assertFalse(any(event.indicator_name == "teacher_presence" for event in events))
        self.assertTrue(any(event.indicator_name == "role_classification"
                            and event.role == "unknown" for event in events))

    def test_instructor_presence_has_visible_and_unknown_samples(self):
        analyzer = FoundationAnalyzer("video_05", "teacher", "single_visible", True,
                                      {}, .2, SETTINGS)
        events = analyzer.process(0, (person(),))
        events += analyzer.process(.2, (person(),))
        events += analyzer.finish(.4)
        samples = [event.value for event in events
                   if event.indicator_name == "teacher_presence_sample"]
        self.assertEqual(samples, ["unknown", "visible"])
        self.assertTrue(any(event.indicator_name == "teacher_presence"
                            and event.value == "visible" for event in events))

    def test_disabled_position_switch_suppresses_position_only(self):
        enabled = {name: True for name in (
            "teacher_presence", "visible_student_count", "anonymous_tracking",
            "role_classification", "position", "classroom_zone", "entry_exit",
            "visibility_occlusion", "sitting_standing", "motion_level")}
        enabled["position"] = False
        analyzer = FoundationAnalyzer("video_06", "student", "all_tracks", True,
                                      {}, .2, SETTINGS, enabled)
        analyzer.process(0, (person(),))
        events = analyzer.process(.2, (person(),))
        self.assertFalse(any(event.indicator_name == "position" for event in events))
        self.assertTrue(any(event.indicator_name == "visible_student_count" for event in events))

    def test_last_sample_snapshot_marks_provisional_then_rule_states(self):
        analyzer = FoundationAnalyzer("video_07", "student", "all_tracks", True,
                                      {}, .2, SETTINGS)
        analyzer.process(0, (person(),))
        first = analyzer.last_sample
        self.assertFalse(first.tracks[0].confirmed)
        self.assertIsNone(first.tracks[0].posture)
        self.assertEqual((first.visible_student_count, first.student_count_coverage), (0, "limited"))
        analyzer.process(.2, (person(),))
        track = analyzer.last_sample.tracks[0]
        self.assertTrue(track.confirmed)
        self.assertEqual((track.role, track.posture), ("student", "standing"))
        self.assertIsNone(track.zone)  # no calibration: zone not evaluated
        self.assertIsNone(analyzer.last_sample.teacher_presence)

    def test_no_entry_or_exit_for_track_starting_at_video_boundary(self):
        analyzer = FoundationAnalyzer("video_03", "student", "all_tracks", True,
                                      {}, .2, SETTINGS)
        edge = person(x_shift=-.24)
        analyzer.process(0, (edge,))
        analyzer.process(.2, (edge,))
        events = analyzer.finish(.4)
        self.assertFalse(any(event.indicator_name == "entry_exit" for event in events))

    def test_camera_compensation_measures_motion_relative_to_scene(self):
        def run(cameras):
            analyzer = FoundationAnalyzer("video_09", "teacher", "single_visible", False,
                                          {}, .2, SETTINGS)
            events = []
            for index, camera in enumerate(cameras):
                # The camera pans, so a still person drifts across the image.
                events += analyzer.process(index * .2, (person(x_shift=.02 * index),), camera)
            return analyzer.last_sample.tracks[0], events
        pans = [CameraMotion(0, (1, 0, -.02 * i, 0, 1, 0), i > 0 or None) for i in range(3)]
        track, events = run(pans)
        self.assertEqual(track.motion, "still")
        self.assertAlmostEqual(track.foot_point_in_shot[0], .5, places=6)
        self.assertEqual(len([e for e in events if e.indicator_name == "camera_motion"]), 3)
        self.assertEqual(run([None, None, None])[0].motion, "unknown")  # uncompensated moving camera
        cut = pans[:2] + [CameraMotion(1, IDENTITY, False)]
        self.assertEqual(run(cut)[0].motion, "unknown")  # new shot: no comparable previous pose

    def test_student_stands_up_without_legs(self):
        analyzer = FoundationAnalyzer("video_10", "student", "all_tracks", True, {}, .2,
                                      {**SETTINGS, "shoulder_rise_history_seconds": 1.0},
                                      posture_method="shoulder_rise", joints=STUDENT_JOINTS)
        events = []
        for index in range(12):
            pose = restrict_joints(student(rise=1.0 if index >= 9 else 0.0), STUDENT_JOINTS)
            events += analyzer.process(index * .2, (pose,))
        events += analyzer.finish(2.4)
        postures = [event for event in events if event.indicator_name == "sitting_standing"]
        self.assertEqual([event.value for event in postures], ["unknown", "sitting", "standing"])
        self.assertTrue(all(event.coverage == "limited" for event in postures))
        self.assertEqual(postures[-1].start_time, 1.8)
        self.assertEqual(analyzer.last_sample.tracks[0].posture_basis, "shoulder_rise")
        self.assertTrue(any(event.indicator_name == "shoulder_rise" for event in events))
        position = next(event.value for event in events if event.indicator_name == "position")
        self.assertFalse(any(point["index"] >= 13 for point in position["keypoints"]))
        with self.assertRaises(ValueError):  # the seated level needs a static image
            FoundationAnalyzer("video_10", "student", "all_tracks", False, {}, .2, SETTINGS,
                               posture_method="shoulder_rise")

    def test_cut_off_at_frame_bottom_is_not_an_exit_or_a_zone(self):
        waist_up = PosePerson(1, (.3, .2, .7, 1.0), .9, person(ankles=False).keypoints, 640, 480)
        self.assertEqual(waist_up.foot_point(.35)[1], "frame_bottom_cut_off")
        analyzer = FoundationAnalyzer("video_11", "teacher", "single_visible", True,
                                      {"front": ((0, 0), (1, 0), (1, 1), (0, 1))}, .2, SETTINGS)
        events = []
        for time in (1.0, 1.2, 1.4):
            events += analyzer.process(time, (waist_up,))
        events += analyzer.process(2.4, ())  # the tracker loses them in place
        events += analyzer.finish(3.0)
        self.assertFalse(any(event.indicator_name == "entry_exit" for event in events))
        self.assertEqual({event.value for event in events
                          if event.indicator_name == "classroom_zone"}, {"unknown"})

    def test_no_entry_for_track_visible_at_segment_start(self):
        analyzer = FoundationAnalyzer("video_08", "student", "all_tracks", True,
                                      {}, .2, SETTINGS, start_time=30.0)
        edge = person(x_shift=-.25)
        analyzer.process(30.0, (edge,))
        analyzer.process(30.2, (edge,))
        events = analyzer.finish(30.4)
        self.assertFalse(any(event.indicator_name == "entry_exit" for event in events))

    def test_apparent_edge_entry_and_exit_after_confirmed_track_loss(self):
        analyzer = FoundationAnalyzer("video_04", "student", "all_tracks", True,
                                      {}, .2, SETTINGS)
        edge = person(x_shift=-.25)
        analyzer.process(1.0, (edge,))
        analyzer.process(1.2, (edge,))
        events = analyzer.process(2.0, ())
        transitions = [event.value for event in events if event.indicator_name == "entry_exit"]
        self.assertEqual(transitions, ["apparent_entry_at_frame_edge",
                                       "apparent_exit_at_frame_edge"])


if __name__ == "__main__":
    unittest.main()
