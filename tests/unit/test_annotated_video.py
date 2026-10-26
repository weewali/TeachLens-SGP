"""Review overlays and MP4 output on synthetic frames; no model or GUI needed."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.analysis.visual_indicators import SampleSnapshot, TrackSnapshot
from src.inference.pose import Keypoint, PosePerson
from src.preprocessing.camera_motion import IDENTITY, CameraMotion
from src.reporting.annotated_video import (
    AnnotatedVideoWriter, LivePreview, ReviewOverlay, camera_notes, describe_track,
)


def pose(track_id=3):
    points = tuple(Keypoint(.5, .2 + .04 * i, .9 if i < 13 else .1) for i in range(17))
    return PosePerson(track_id, (.3, .15, .7, .95), .82, points, 320, 240)


class FakeWindow:
    """cv2 stand-in that replays key presses; drawing calls use real OpenCV."""

    def __init__(self, keys, visible=1.0):
        self.keys, self.visible, self.shown = list(keys), visible, 0

    def __getattr__(self, name):
        return getattr(cv2, name)

    def namedWindow(self, *args):
        pass

    def resizeWindow(self, *args):
        pass

    def destroyWindow(self, *args):
        pass

    def imshow(self, *args):
        self.shown += 1

    def waitKey(self, delay):
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, *args):
        return self.visible


class OverlayTests(unittest.TestCase):
    def test_labels_separate_provisional_unknown_and_not_evaluated(self):
        self.assertEqual(describe_track(pose(), None), ["#3 provisional 0.82"])
        track = TrackSnapshot(3, True, 5, "student", posture="unknown", motion="still",
                              motion_score=.031, possible_occlusion=True)
        self.assertEqual(describe_track(pose(), track),
                         ["#3 student 0.82", "posture ? | still 0.03 bh/s", "occluded?"])
        bare = TrackSnapshot(3, True, 5, "unknown")  # switches off: no state lines
        self.assertEqual(describe_track(pose(), bare), ["#3 unknown 0.82"])
        partial = TrackSnapshot(3, True, 5, "teacher", posture="standing", posture_basis="upper_legs")
        self.assertEqual(describe_track(pose(), partial)[1], "standing (partial)")
        risen = TrackSnapshot(3, True, 5, "student", posture="standing",
                              posture_basis="shoulder_rise", posture_rise=1.04)
        self.assertEqual(describe_track(pose(), risen)[1], "standing rise 1.04")
        notes = camera_notes({"role_policy": "all_tracks", "pose_joints": ["head", "arms", "hips"]},
                             {}, False)
        self.assertIn("joints not used: knees, ankles", notes)

    def test_hud_reports_counts_coverage_and_camera_notes(self):
        notes = camera_notes({"role_policy": "single_visible", "fixed_camera": False},
                             {"motion_level": True, "classroom_zone": True}, False)
        overlay = ReviewOverlay("v1 | instructor_view", .35, duration=75.0, notes=notes)
        sample = SampleSnapshot(62.4, (TrackSnapshot(3, True, 4, "teacher"),
                                       TrackSnapshot(9, False, 1, "unknown")),
                                teacher_presence="unknown")
        lines = overlay.hud_lines(sample)
        self.assertIn("t 01:02.4 / 01:15.0", lines)
        self.assertIn("tracked 2: 1 confirmed, 1 provisional", lines)
        self.assertIn("teacher presence unknown", lines)
        self.assertIn("motion: unknown, camera not fixed", lines)
        self.assertIn("zones: not calibrated", lines)
        segment = ReviewOverlay("v1 | instructor_view", .35, duration=75.0, time_range=(60.0, None))
        self.assertIn("t 01:02.4 / 01:15.0 | range 01:00.0-end", segment.hud_lines(sample))

    def test_hud_reports_camera_compensation_and_cuts(self):
        overlay = ReviewOverlay("v1 | instructor_view", .35, path="teacher")
        panning = SampleSnapshot(1.0, (), shot=0, camera=CameraMotion(
            0, (1, 0, -.1, 0, 1, 0), True, .012, 1.0, 300))
        lines = overlay.hud_lines(panning)
        self.assertIn("camera: shot 0, pan 1.2% zoom x1.000 (compensated)", lines)
        self.assertIn("path: teacher, hip centre, resets at camera cuts", lines)
        cut = SampleSnapshot(1.2, (), shot=1, camera=CameraMotion(1, IDENTITY, False, None, 1.4, 8))
        self.assertIn("camera: cut or unmeasured -> shot 1, motion unknown", overlay.hud_lines(cut))
        uncompensated = SampleSnapshot(1.4, (), shot=None)
        self.assertIn("path: teacher unavailable (moving camera, not compensated)",
                      overlay.hud_lines(uncompensated))

    def test_teacher_path_follows_hip_centre_and_resets_at_new_shot(self):
        overlay = ReviewOverlay("v1 | instructor_view", .35, path="teacher")
        frame = np.zeros((240, 320, 3), np.uint8)

        def sample(time, x, shot=0, role="teacher", hips=True):
            track = TrackSnapshot(3, True, 5, role, hip_center_in_shot=(x, .9) if hips else None)
            return SampleSnapshot(time, (track,), shot=shot)
        overlay.draw(frame, (), sample(0.0, .2))
        overlay.draw(frame, (), sample(0.2, .3, hips=False))  # no hips: no point
        overlay.draw(frame, (), sample(0.3, .35, role="unknown"))  # not the assumed teacher
        image = overlay.draw(frame, (), sample(0.4, .4))
        self.assertEqual([p[1] for p in overlay._paths["teacher"]], [.2, .4])
        self.assertTrue((image[216, 64:128] == (0, 255, 255)).all(axis=1).any())  # yellow trail
        overlay.draw(frame, (), sample(0.6, .8, shot=1))
        self.assertEqual([p[1] for p in overlay._paths["teacher"]], [.8])

    def test_draw_returns_annotated_copy(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        sample = SampleSnapshot(.4, (TrackSnapshot(3, True, 3, "student", posture="standing"),),
                                visible_student_count=1, student_count_coverage="adequate")
        image = ReviewOverlay("v1 | student_view", .35).draw(frame, (pose(),), sample)
        self.assertEqual(image.shape, frame.shape)
        self.assertFalse(frame.any())
        self.assertTrue(image.any())


class LivePreviewTests(unittest.TestCase):
    frame = np.zeros((240, 320, 3), np.uint8)

    def show(self, preview, window):
        with mock.patch("src.reporting.annotated_video._cv2", return_value=window):
            return preview.show(self.frame)

    def test_q_stops_and_other_keys_continue(self):
        self.assertTrue(self.show(LivePreview(), FakeWindow([-1])))
        self.assertFalse(self.show(LivePreview(), FakeWindow([ord("q")])))
        self.assertFalse(self.show(LivePreview(), FakeWindow([27])))

    def test_pause_then_step_keeps_paused_until_space(self):
        preview = LivePreview()
        self.assertTrue(self.show(preview, FakeWindow([ord(" "), -1, -1, ord("n")])))
        window = FakeWindow([-1, -1, ord(" ")])
        self.assertTrue(self.show(preview, window))
        self.assertEqual(window.keys, [])  # stayed paused until space resumed

    def test_closed_window_lets_analysis_continue_without_preview(self):
        preview = LivePreview()
        self.assertTrue(self.show(preview, FakeWindow([-1], visible=0)))
        window = FakeWindow([ord("q")])
        self.assertTrue(self.show(preview, window))
        self.assertEqual(window.shown, 0)


class WriterTests(unittest.TestCase):
    def test_publishes_complete_mp4_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug" / "v1.list_a.mp4"
            with AnnotatedVideoWriter(path, "mp4v") as writer:
                writer.open(5.0, 64, 48)
                for value in (0, 120, 240):
                    writer.write(np.full((48, 64, 3), value, np.uint8))
            capture = cv2.VideoCapture(str(path))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()
            with self.assertRaises(FileExistsError):
                AnnotatedVideoWriter(path, "mp4v").__enter__()
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])

    def test_failed_run_discards_partial_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.list_a.mp4"
            with self.assertRaises(RuntimeError):
                with AnnotatedVideoWriter(path, "mp4v") as writer:
                    writer.open(5.0, 64, 48)
                    writer.write(np.zeros((48, 64, 3), np.uint8))
                    raise RuntimeError("simulated stop")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
