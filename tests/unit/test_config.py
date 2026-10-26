"""Config guards and local model requirement."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

from src.inference.pose import YoloPoseTracker, joint_indices
from src.utils.config import load_config, load_indicator_flags, load_zones, validate_config
from src.utils.paths import project_path


class ConfigurationTests(unittest.TestCase):
    def test_project_config_loads(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["models"]["pose_estimator"], "models/pose/yolo26n-pose.pt")
        self.assertTrue(load_indicator_flags("configs/indicators.yaml")["sitting_standing"])
        self.assertNotIn("hand_raise", load_indicator_flags("configs/indicators.yaml"))

    def test_camera_zone_id_is_required(self):
        with self.assertRaises(ValueError):
            load_zones("configs/zones.example.yaml", "student_view")

    def test_sparse_sampling_rejected_for_tracking(self):
        config = deepcopy(load_config(project_path("configs/default.yaml")))
        config["video"]["sample_every_seconds"] = 2.0
        with self.assertRaisesRegex(ValueError, "max_track_gap_seconds"):
            validate_config(config)

    def test_review_path_and_posture_fallback_settings_are_validated(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["review"]["path"], "off")  # quoted: bare off would be false
        for section, key, value in (("review", "path", "instructor"),
                                    ("analysis", "thigh_axis_tolerance_degrees", 60),
                                    ("analysis", "shoulder_rise_sitting_max", .9),
                                    ("camera_motion", "max_scale_step", 1.5)):
            broken = deepcopy(config)
            broken[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)

    def test_static_cameras_by_default_and_web_samples_compensated(self):
        config = load_config(project_path("configs/default.yaml"))
        for camera in config["cameras"].values():
            self.assertTrue(camera["fixed_camera"])
            self.assertFalse(camera["compensate_camera_motion"])
        web = load_config(project_path("configs/default.yaml"),
                          project_path("configs/web_sample_cameras.yaml"))
        instructor = web["cameras"]["instructor_view"]
        self.assertFalse(instructor["fixed_camera"])
        self.assertTrue(instructor["compensate_camera_motion"])
        self.assertEqual(instructor["role_policy"], "single_visible")  # merged, not replaced

    def test_student_rules_do_not_use_legs(self):
        config = load_config(project_path("configs/default.yaml"))
        student = config["cameras"]["student_view"]
        self.assertTrue(joint_indices(student["pose_joints"]).isdisjoint({13, 14, 15, 16}))
        self.assertEqual(student["posture_method"], "shoulder_rise")
        self.assertEqual(joint_indices(config["cameras"]["instructor_view"]["pose_joints"]),
                         frozenset(range(17)))
        for key, value in (("pose_joints", ["head", "legs"]),  # unknown group
                           ("pose_joints", ["head", "hips"]),  # no shoulders
                           ("posture_method", "legs"),  # leg angles without knees
                           ("fixed_camera", False)):  # shoulder_rise needs a static camera
            broken = deepcopy(config)
            broken["cameras"]["student_view"][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)

    def test_missing_checkpoint_does_not_auto_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                YoloPoseTracker(Path(directory) / "missing.pt", .25, .45)


if __name__ == "__main__":
    unittest.main()
