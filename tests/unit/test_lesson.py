"""The one-command lesson pipeline: planning, alignment, and the step order (no models)."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lesson import (CameraRun, analyze_lesson, limit_range, plan_from_manifest,  # noqa: E402
                        plan_from_videos, shared_range)
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

ZONE = timezone(timedelta(hours=4))
STUDENT_START = datetime(2026, 11, 3, 19, 15, 53, tzinfo=ZONE)
INSTRUCTOR_START = datetime(2026, 11, 3, 19, 16, 1, tzinfo=ZONE)
HEADER = ("video_id,scenario,take,camera,file,duration_seconds,recording_start,recording_start_source,"
          "sync_start_seconds,sync_end_seconds,notes\n")


class AlignmentTest(unittest.TestCase):
    def test_shared_range_matches_the_a1_manifest(self):
        starts = {"student_view": STUDENT_START, "instructor_view": INSTRUCTOR_START}
        begin, length = shared_range(starts, {"student_view": 294.14, "instructor_view": 276.43})
        self.assertEqual(begin, INSTRUCTOR_START)
        self.assertAlmostEqual(length, 276.43)  # the student file runs on after the instructor's stops
        with self.assertRaises(ValueError):
            shared_range(starts, {"student_view": 5.0, "instructor_view": 276.43})

    def test_limit_range_counts_from_the_shared_start(self):
        self.assertEqual(limit_range(8.0, 284.43), (8.0, 284.43))
        self.assertEqual(limit_range(8.0, 284.43, 120, 240), (128.0, 248.0))
        self.assertEqual(limit_range(0.0, None, 30), (30.0, None))
        self.assertEqual(limit_range(8.0, 50.0, 0, 600), (8.0, 50.0))
        with self.assertRaises(ValueError):
            limit_range(8.0, 50.0, 60)

    def test_plan_from_videos_uses_tags_or_given_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            student, instructor = Path(folder) / "s.MOV", Path(folder) / "i.MOV"
            student.touch()
            instructor.touch()
            windows = {student: (STUDENT_START, 294.14, "tag"), instructor: (INSTRUCTOR_START, 276.43, "tag")}
            given = INSTRUCTOR_START + timedelta(seconds=14.067)

            def window(video, real_start=None):
                start, duration, source = windows[video]
                return (real_start or start), duration, ("given" if real_start else source)

            with mock.patch("src.lesson.recording_window", side_effect=window):
                runs = plan_from_videos(student, instructor, "lesson_01")
                self.assertEqual([(r.camera, r.video_id, r.start, r.end, r.real_start) for r in runs],
                                 [("student_view", "lesson_01_student", 8.0, 284.43, None),
                                  ("instructor_view", "lesson_01_instructor", 0.0, None, None)])
                shifted = plan_from_videos(student, instructor, "x", instructor_start=given, lesson_end=60)
                self.assertEqual((shifted[0].start, shifted[0].end), (22.067, 82.067))
                self.assertEqual(shifted[1].real_start, given)
                with self.assertRaises(ValueError):
                    plan_from_videos(student, instructor, "x", synced=False, lesson_start=10)
                with self.assertRaises(ValueError):
                    plan_from_videos(student, instructor, "bad name")


class ManifestTest(unittest.TestCase):
    ROWS = ("sgp_A1_student,A1,1,student_view,s/a1.MOV,294,2026-11-03T19:15:53.000+04:00,tag,8.00,284.43,\n"
            "sgp_A1_instructor,A1,1,instructor_view,i/a1.MOV,276,2026-11-03T19:16:01.000+04:00,tag,0.00,,\n"
            "sgp_A4_t1_student,A4,1,student_view,s/a4.MOV,89,,,6.94,64.61,\n"
            "sgp_A4_t1_instructor,A4,1,instructor_view,i/a4.MOV,57,,,0.00,,\n"
            "sgp_A4_t2_instructor,A4,2,instructor_view,i/a4b.MOV,202,,,,,\n")

    def manifest(self, folder: str, extra: str = "") -> Path:
        path = Path(folder) / "manifest.csv"
        path.write_text(HEADER + self.ROWS + extra, encoding="utf-8")
        return path

    def test_scenario_runs_ids_and_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            name, runs = plan_from_manifest("a1", tag="lesson", manifest=self.manifest(folder),
                                            data_root=Path(folder))
            self.assertEqual(name, "sgp_A1_lesson")
            self.assertEqual([(r.video_id, r.start, r.end) for r in runs],
                             [("sgp_A1_student_lesson", 8.0, 284.43), ("sgp_A1_instructor_lesson", 0.0, None)])
            self.assertEqual(runs[0].video, Path(folder) / "s/a1.MOV")
            self.assertEqual(runs[1].real_start, INSTRUCTOR_START)
            _, short = plan_from_manifest("A1", manifest=self.manifest(folder), lesson_start=60, lesson_end=90)
            self.assertEqual([(r.start, r.end) for r in short], [(68.0, 98.0), (60.0, 90.0)])

    def test_takes(self):
        with tempfile.TemporaryDirectory() as folder:
            # Take 2 has no student video, so take 1 is the lesson.
            name, runs = plan_from_manifest("A4", manifest=self.manifest(folder))
            self.assertEqual((name, runs[1].video_id), ("sgp_A4_t1", "sgp_A4_t1_instructor"))
            both = self.manifest(folder, "sgp_A4_t2_student,A4,2,student_view,s/a4b.MOV,200,,,,,\n")
            with self.assertRaises(ValueError):
                plan_from_manifest("A4", manifest=both)
            self.assertEqual(plan_from_manifest("A4", take="2", manifest=both)[0], "sgp_A4_t2")
            with self.assertRaises(ValueError):
                plan_from_manifest("A9", manifest=both)


def event(video_id, track, role, indicator, value, start, end):
    t0 = INSTRUCTOR_START + timedelta(seconds=start)
    t1 = INSTRUCTOR_START + timedelta(seconds=end)
    return {"video_id": video_id, "track_id": track, "role": role, "indicator_name": indicator,
            "start_time": start, "end_time": end, "duration": end - start, "value": value,
            "confidence": None, "coverage": "adequate", "zone": None,
            "start_real_time": t0.isoformat(timespec="milliseconds"),
            "end_real_time": t1.isoformat(timespec="milliseconds")}


def fake_analyze(video, video_id, camera, config, *, output_dir, **kwargs):
    """Stand-in for pipeline.analyze_video: a minute of evidence with real times."""
    role = "teacher" if camera == "instructor_view" else "student"
    rows = [event(video_id, 1, role, "position", {"hip_center_in_shot": [0.5, 0.5], "camera_shot": 0}, t, t)
            for t in range(61)]
    if role == "teacher":
        rows += [event(video_id, 1, role, "facing_direction", "toward_class" if (t // 10) % 2 else "sideways_left",
                       t, t + 5) for t in range(0, 60, 5)]
    else:
        rows += [event(video_id, 1, role, "orientation_proxy", "on_target" if (t // 15) % 2 else "off_target",
                       t, t + 5) for t in range(0, 60, 5)]
    events = output_dir / "events" / f"{video_id}.list_a.jsonl"
    summary = output_dir / "reports" / f"{video_id}.list_a.json"
    for path in (events, summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    return events, summary, None


class AnalyzeLessonTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(project_path("configs/default.yaml"))

    def test_runs_every_step_then_reuses_saved_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.MOV"
            video.touch()
            runs = [CameraRun("student_view", video, "l_student", 8.0, 68.0),
                    CameraRun("instructor_view", video, "l_instructor")]
            with mock.patch("src.pipeline.analyze_video", side_effect=fake_analyze) as analyze:
                lesson = analyze_lesson(runs, "l", self.config, output_dir=root)
                self.assertEqual(analyze.call_count, 2)
                self.assertEqual(analyze.call_args_list[0].kwargs["start"], 8.0)
                self.assertEqual(analyze.call_args_list[1].kwargs["path"], "teacher")
                self.assertEqual([r["status"] for r in lesson["runs"]], ["analyzed", "analyzed"])
                self.assertEqual(lesson["fusion"]["windows"], 12)
                for key in ("cross_camera", "fusion", "fusion_windows", "fusion_links", "fusion_plot", "lesson"):
                    self.assertTrue(Path(lesson["outputs"][key]).is_file(), key)
                saved = json.loads((root / "reports" / "l.lesson.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["student_run"], "l_student")
                fusion = json.loads((root / "reports" / "l.fusion.json").read_text(encoding="utf-8"))
                self.assertEqual((fusion["lesson"], fusion["instructor_run"]), ("l", "l_instructor"))
                # Existing lesson outputs are refused before any camera is analyzed.
                with self.assertRaises(FileExistsError):
                    analyze_lesson(runs, "l", self.config, output_dir=root)
                self.assertEqual(analyze.call_count, 2)
                again = analyze_lesson(runs, "l", self.config, output_dir=root, reuse=True, overwrite=True,
                                       plot=False)
                self.assertEqual(analyze.call_count, 2)
                self.assertEqual([r["status"] for r in again["runs"]], ["reused", "reused"])
                self.assertNotIn("fusion_plot", again["outputs"])
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_needs_both_cameras_and_a_valid_name(self):
        video = Path(__file__)
        with self.assertRaises(ValueError):
            analyze_lesson([CameraRun("student_view", video, "a")], "l", self.config)
        with self.assertRaises(ValueError):
            analyze_lesson([CameraRun("student_view", video, "a"), CameraRun("instructor_view", video, "b")],
                           "bad name", self.config)


if __name__ == "__main__":
    unittest.main()
