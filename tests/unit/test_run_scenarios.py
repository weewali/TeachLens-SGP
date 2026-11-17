"""Scenario manifest reading and run planning; loads the script without running analysis."""

import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.utils.paths import project_path


spec = importlib.util.spec_from_file_location("run_scenarios",
                                              project_path("scripts/run_scenarios.py"))
run_scenarios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_scenarios)

HEADER = ("video_id,scenario,take,camera,file,recording_start,"
          "sync_start_seconds,sync_end_seconds,notes\n")
ROWS = (
    "sgp_A1_student,A1,1,student_view,data/raw_student_videos/sgp_A1_student.MOV,"
    "2026-11-03T19:15:53.000+04:00,8,292,\n"
    "sgp_A1_instructor,A1,1,instructor_view,data/raw_instructor_videos/sgp_A1_instructor.MP4,"
    "2026-11-03T19:16:01.000+04:00,0,,\n"
    "sgp_B1_student,B1,1,student_view,data/raw_student_videos/sgp_B1_student.MOV,,2,638,\n"
    "IMG_0181,unknown,,instructor_view,data/raw_instructor_videos/IMG_0181.MP4,,,,unlabelled\n"
)


class ManifestTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_reads_rows(self):
        rows = run_scenarios.read_manifest(self.write(HEADER + ROWS))
        self.assertEqual([r["video_id"] for r in rows][:2], ["sgp_A1_student", "sgp_A1_instructor"])

    def test_rejects_unknown_camera_and_missing_fields(self):
        for row in ("x,A1,1,side_view,a.mp4,\n", "x,A1,1,student_view,,\n"):
            with self.assertRaises(ValueError):
                run_scenarios.read_manifest(self.write(HEADER + row))

    def test_committed_manifest_is_valid(self):
        rows = run_scenarios.read_manifest(run_scenarios.DEFAULT_MANIFEST)
        ids = [r["video_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c.isalnum() or c in "-_" for i in ids for c in i))


class PlanTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        handle.write(HEADER + ROWS)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        self.rows = run_scenarios.read_manifest(Path(handle.name))
        self.root = Path("D:/repo")

    def test_default_skips_unlabelled_clips_and_resolves_paths(self):
        runs = run_scenarios.plan_runs(self.rows, self.root)
        self.assertEqual([r.video_id for r in runs],
                         ["sgp_A1_student", "sgp_A1_instructor", "sgp_B1_student"])
        self.assertEqual(runs[0].video, self.root / "data/raw_student_videos/sgp_A1_student.MOV")

    def test_filters_by_scenario_camera_and_tags_ids(self):
        runs = run_scenarios.plan_runs(self.rows, self.root, scenarios=["a1"],
                                       cameras=["instructor_view"], tag="base")
        self.assertEqual([r.video_id for r in runs], ["sgp_A1_instructor_base"])
        self.assertEqual(runs[0].path, "teacher")

    def test_unlabelled_clips_run_when_named(self):
        runs = run_scenarios.plan_runs(self.rows, self.root, scenarios=["unknown"])
        self.assertEqual([r.video_id for r in runs], ["IMG_0181"])

    def test_synced_window_by_default_and_full_files_on_request(self):
        runs = run_scenarios.plan_runs(self.rows, self.root, scenarios=["A1", "unknown"])
        self.assertEqual([(r.start, r.end) for r in runs], [(8.0, 292.0), (0.0, None), (0.0, None)])
        full = run_scenarios.plan_runs(self.rows, self.root, scenarios=["A1"], synced=False)
        self.assertEqual({(r.start, r.end) for r in full}, {(0.0, None)})

    def test_manifest_recording_start_is_passed_on(self):
        runs = run_scenarios.plan_runs(self.rows, self.root, scenarios=["A1", "B1"])
        self.assertEqual(runs[1].real_start.isoformat(), "2026-11-03T19:16:01+04:00")
        self.assertIsNone(runs[2].real_start)  # empty: the file's own tag is read

    def test_student_runs_draw_no_path(self):
        runs = run_scenarios.plan_runs(self.rows, self.root, cameras=["student_view"])
        self.assertEqual({r.path for r in runs}, {"off"})


class ObjectOptionTests(unittest.TestCase):
    def main(self, *argv):
        with mock.patch("sys.argv", ["run_scenarios.py", *argv]), \
                mock.patch("sys.stdout", io.StringIO()) as output, \
                mock.patch("sys.stderr", io.StringIO()):
            try:
                code = run_scenarios.main()
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue()

    def test_objects_runs_are_tagged_and_exclude_pose_options(self):
        # The plan is printed even where the media are missing (exit code 1 then).
        _, output = self.main("--scenario", "A2", "--objects", "--dry-run")
        self.assertIn("sgp_A2_student_objects", output)
        self.assertIn("sgp_A2_instructor_objects", output)
        self.assertEqual(self.main("--objects", "--indicators", "position", "--dry-run")[0], 2)
        self.assertEqual(self.main("--classes", "laptop", "--dry-run")[0], 2)


if __name__ == "__main__":
    unittest.main()
