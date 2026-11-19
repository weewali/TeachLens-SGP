"""Config guards and local model requirement."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

from src.inference.pose import YoloPoseTracker, joint_indices
from src.utils.config import (
    INDICATORS, LIST_A_INDICATORS, load_config, load_indicator_flags, load_zones, select_indicators,
    validate_config,
)
from src.utils.paths import project_path


class ConfigurationTests(unittest.TestCase):
    def test_project_config_loads(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["models"]["pose_estimator"], "models/pose/yolo26n-pose.pt")
        self.assertTrue(load_indicator_flags("configs/indicators.yaml")["sitting_standing"])
        # Student hand raises are List C, read from the students: section (2026-11-17).
        self.assertTrue(load_indicator_flags("configs/indicators.yaml")["hand_raise"])
        self.assertNotIn("student_movement", load_indicator_flags("configs/indicators.yaml"))

    def test_camera_zone_id_is_required(self):
        with self.assertRaises(ValueError):
            load_zones("configs/zones.example.yaml", "student_view")

    def test_one_sample_per_second_and_sparser_sampling_rejected(self):
        config = deepcopy(load_config(project_path("configs/default.yaml")))
        self.assertEqual(config["video"]["sample_every_seconds"], 1.0)
        self.assertGreaterEqual(config["analysis"]["max_track_gap_seconds"], 2.0)  # one missed sample
        config["video"]["sample_every_seconds"] = 5.0
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

    def test_instructor_camera_ignores_bottom_heads_and_value_is_validated(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["cameras"]["instructor_view"]["ignore_people_starting_below"], .75)
        self.assertIsNone(config["cameras"]["student_view"].get("ignore_people_starting_below"))
        for value in (0, 1.5, "low", True):
            broken = deepcopy(config)
            broken["cameras"]["instructor_view"]["ignore_people_starting_below"] = value
            with self.assertRaises(ValueError):
                validate_config(broken)
        self.assertEqual(config["inference"]["image_size"], 1280)
        for value in (1000, 100, "big", True):
            broken = deepcopy(config)
            broken["inference"]["image_size"] = value
            with self.assertRaises(ValueError):
                validate_config(broken)
        default = deepcopy(config)
        default["inference"]["image_size"] = None
        validate_config(default)
        self.assertEqual(config["analysis"]["duplicate_max_keypoint_distance"], .25)
        for key, value in (("duplicate_max_keypoint_distance", 0),
                           ("duplicate_max_keypoint_distance", 3),
                           ("duplicate_min_box_inside", 1.5)):
            broken = deepcopy(config)
            broken["analysis"][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)
        off = deepcopy(config)
        off["analysis"]["duplicate_max_keypoint_distance"] = None
        validate_config(off)
        self.assertIs(config["cameras"]["instructor_view"]["keep_teacher_track"], True)
        broken = deepcopy(config)
        broken["cameras"]["instructor_view"]["keep_teacher_track"] = "yes"
        with self.assertRaises(ValueError):
            validate_config(broken)

    def test_fusion_settings_are_validated(self):
        config = load_config(project_path("configs/default.yaml"))
        self.assertEqual(config["fusion"]["window_seconds"], 5.0)
        self.assertEqual(config["fusion"]["model_lags"], [1, 2, 3])
        for key, value in (("window_seconds", 0), ("alpha", 1.5), ("min_known_fraction", 0),
                           ("model_lags", [0]), ("model_lags", 2), ("permutations", -1)):
            broken = deepcopy(config)
            broken["fusion"][key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_config(broken)

    def test_select_indicators_keeps_only_chosen_switches(self):
        enabled = load_indicator_flags("configs/indicators.yaml")
        self.assertIs(select_indicators(enabled, None), enabled)
        chosen = select_indicators(enabled, ["sitting_standing"])
        self.assertEqual([name for name, on in chosen.items() if on], ["sitting_standing"])
        self.assertEqual(set(chosen), set(INDICATORS))
        both = select_indicators(dict.fromkeys(LIST_A_INDICATORS, False), ["motion_level", "position"])
        self.assertEqual(sum(both.values()), 2)  # switched on even if the file has them off
        for bad in (["student_movement"], []):
            with self.assertRaises(ValueError):
                select_indicators(enabled, bad)

    def test_missing_checkpoint_does_not_auto_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                YoloPoseTracker(Path(directory) / "missing.pt", .25, .45)


if __name__ == "__main__":
    unittest.main()
