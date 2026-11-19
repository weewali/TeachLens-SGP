"""The MVC web app: models (lectures, jobs, report data, charts) and controllers (pages, uploads, progress, files).

Uses a made-up lecture (tests/fixtures/demo_lecture.py) and a fake pipeline script, so no video or model is needed."""

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from demo_lecture import make_demo_lecture  # noqa: E402
from web.app import create_app  # noqa: E402
from web.models import lecture as lecture_model  # noqa: E402
from web.models import report as report_model  # noqa: E402
from web.models import stat_charts  # noqa: E402
from web.models.job import JobRunner  # noqa: E402
from web.models.lecture import Lecture, LectureRepository, safe_id  # noqa: E402

FAKE_PIPELINE = """
import sys
print("args: " + " ".join(sys.argv[1:]), flush=True)
stages = ["Video input", "Preprocessing", "Detection", "Behaviour", "Windows", "Fusion", "Report"]
for n, name in enumerate(stages, 1):
    print(f"[{n}/7] {name}", flush=True)
    if n == {fail_at}:
        print("Something went wrong", flush=True)
        sys.exit(3)
"""

# An analysis that is still busy: like Main Code.py running FFmpeg, it has started a child program that keeps one of
# the lecture's files open (on Windows an open file cannot be deleted).
SLOW_PIPELINE = """
import os, subprocess, sys, time
session = sys.argv[sys.argv.index("--session") + 1]
folder = os.path.join(os.environ["TEACHLENS_BASE"], "sessions", session)
os.makedirs(folder, exist_ok=True)
held = os.path.join(folder, "held_open.tmp")
subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'w'); time.sleep(60)", held])
while not os.path.exists(held):
    time.sleep(0.05)
print("[1/7] Video input", flush=True)
time.sleep(60)
"""


class FakeUpload:
    def __init__(self, filename, data=b"video"):
        self.filename, self.data = filename, data

    def save(self, path):
        Path(path).write_bytes(self.data)


def fake_script(directory, name, fail_at=0):
    path = Path(directory) / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_PIPELINE.replace("{fail_at}", str(fail_at)), encoding="utf-8")
    return path


class WebTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / "base"
        make_demo_lecture(cls.base)
        cls.app = create_app(BASE_DIR=cls.base, EVENTS_DIR=cls.base / "events", TESTING=True, LLM_MODE="off",
                             MAIN_CODE=fake_script(cls.temp.name, "ok.py"), LLM_REPORT=fake_script(cls.temp.name, "llm.py"))
        cls.client = cls.app.test_client()
        cls.jobs = cls.app.extensions["teachlens"]["jobs"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()


class PageTests(WebTestCase):
    def test_lecture_list(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("Demo lecture (made-up data)", page)
        self.assertIn("Report ready", page)
        self.assertIn('href="/lectures/demo_lecture/delete"', page)
        self.assertIn('aria-label="Delete Demo lecture (made-up data)"', page)

    def test_delete_asks_first(self):
        page = self.client.get("/lectures/demo_lecture/delete").get_data(as_text=True)
        for expected in ("Delete this lecture?", "Demo lecture (made-up data)", "the report and its charts",
                         "This cannot be undone.", 'method="post" action="/lectures/demo_lecture/delete"', "Cancel"):
            self.assertIn(expected, page)
        self.assertNotIn("being analysed right now", page)
        self.assertTrue(Lecture("demo_lecture", self.base).has_report())   # asking deletes nothing
        self.assertEqual(self.client.get("/lectures/nope/delete").status_code, 404)
        self.assertEqual(self.client.get("/lectures/demo_lecture/delete/").status_code, 404)

    def test_report_page(self):
        response = self.client.get("/lectures/demo_lecture")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertNotIn("<footer", page)   # the advisory footer line was removed (user request, 2026-11-18)
        for expected in ("Positive points", "Negative points", "A phone was seen", "Phone in view", "Head down and still",
                         "explaining to the class", "Where you were", "Heads up by seat",
                         "What the words in this report mean", '<a href="#terms">Key terms</a>',
                         '<section id="terms" class="tab-panel terms-panel">', "<dt>heads up</dt>", "05:00 of lecture", "✗",
                         '<nav class="tabs"', '<section id="feedback" class="tab-panel is-active">',
                         '<section id="charts" class="tab-panel charts-panel">', '<section id="moments" class="tab-panel">',
                         '<span class="tab-count">', "uaeu-logo.svg", 'alt="United Arab Emirates University (UAEU)"',
                         "Over the lecture", 'class="chart-data"', "Show the numbers",
                         "Students during each of your activities", "compare-bar act-explaining",
                         # the team's fusion results in plain language (brought back, user decision 2026-11-19)
                         '<a href="#links">Links over time</a>', '<section id="links" class="tab-panel">',
                         "possible links between what you did", "5-second parts", "Too rare to test"):
            self.assertIn(expected, page)
        for developer_detail in ("Claude", ".jsonl", "Written by the rule-based", "analysis_details.txt",
                                 "evidence.strength", "Describe it as",   # not shown to the instructor
                                 'class="fact"', "Frames per camera", "Students seen per frame",
                                 "Cameras lined up by", "timeline.png", "seat_0", "Technical details",
                                 "Instructor indicators", "Student indicators", "Teacher-student linkage",
                                 "grid cell", "fusion", "T_facing_class", "S_on_target", "q-value",
                                 "possible links between what the instructor did", "Suggestions and limitations",
                                 "<summary>Suggestions", "<summary>Limitations", "limitations say",
                                 "<dt>window</dt>", "<dt>observable student time</dt>"):   # removed from the page
            self.assertNotIn(developer_detail, page)
        lecture = Lecture("demo_lecture", self.base)
        self.assertTrue((lecture.charts_dir / "instructor_heatmap.png").exists())
        self.assertTrue((lecture.charts_dir / "seat_heatmap.png").exists())

    def test_charts_are_served_and_nothing_else(self):
        response = self.client.get("/lectures/demo_lecture/charts/seat_heatmap.png")
        self.assertEqual((response.status_code, response.mimetype), (200, "image/png"))
        response.close()
        for path in ("/lectures/demo_lecture/charts/facts.json", "/lectures/demo_lecture/charts/..%2Freports%2Ffacts.json",
                     "/lectures/nope/charts/timeline.png", "/lectures/..%2F..%2Fsecret"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_new_form(self):
        page = self.client.get("/lectures/new").get_data(as_text=True)
        for expected in ("Instructor camera", "Student camera", "What was the class about?", "Lecture slides",
                         "Start analysis"):
            self.assertIn(expected, page)
        for not_for_instructors in ("Claude", "API key", "use_llm", "cents"):
            self.assertNotIn(not_for_instructors, page)

    def test_upload_is_checked(self):
        response = self.client.post("/lectures", data={"name": "x"}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose the instructor-camera video", response.get_data(as_text=True))
        response = self.client.post("/lectures", content_type="multipart/form-data", data={
            "instructor_video": (io.BytesIO(b"v"), "a.mp4"), "students_video": (io.BytesIO(b"v"), "notes.txt")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("student video must be one of", response.get_data(as_text=True))


class JobTests(WebTestCase):
    def test_upload_starts_the_analysis_and_progress_follows_it(self):
        response = self.client.post("/lectures", content_type="multipart/form-data", data={
            "name": "Week 5", "topic": "Recursion", "notes": "quiz at minute 20",
            "instructor_video": (io.BytesIO(b"i"), "IMG_0001.MOV"), "students_video": (io.BytesIO(b"s"), "cam2.mp4")})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/lectures/Week_5/progress"))
        self.jobs.wait(30)
        status = self.client.get("/lectures/Week_5/status").get_json()
        self.assertEqual((status["state"], status["stage"]), ("done", 7))
        self.assertIn("[7/7] Report", status["log"])
        self.assertFalse(status["has_report"])                  # the fake pipeline writes no report
        lecture = LectureRepository(self.base).get("Week_5")
        self.assertEqual(lecture.context()["topic"], "Recursion")
        self.assertEqual(lecture.video_path("instructor").name, "instructor.mov")
        self.assertFalse(status["use_llm"])                     # switched off for this deployment
        page = self.client.get("/lectures/Week_5/progress").get_data(as_text=True)
        self.assertIn("Lagged analysis: linking instructor actions to student behaviour", page)
        self.assertIn('data-stage="8" class="stage" hidden', page)  # the plain-language stage is not shown
        self.assertEqual(self.client.get("/lectures/Week_5").status_code, 302)   # no report yet: back to progress

    def test_plain_language_rewrite_runs_by_itself_when_the_deployment_enables_it(self):
        self.jobs.llm_mode = "on"
        try:
            self.client.post("/lectures", content_type="multipart/form-data", data={
                "name": "with_ai", "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
            self.jobs.wait(30)
            status = self.client.get("/lectures/with_ai/status").get_json()
            self.assertEqual((status["state"], status["stage"], status["use_llm"]), ("done", 8, True))
            self.assertEqual(status["llm_result"], "Feedback written in plain language and checked.")
        finally:
            self.jobs.llm_mode = "off"

    def test_auto_mode_follows_the_servers_api_key(self):
        self.jobs.llm_mode = "auto"
        try:
            with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}):
                self.assertFalse(self.jobs.llm_enabled())
            with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "set-by-the-operator"}):
                self.assertTrue(self.jobs.llm_enabled())
        finally:
            self.jobs.llm_mode = "off"

    def test_a_failing_analysis_is_reported(self):
        self.jobs.main_code = str(fake_script(self.temp.name, "fail.py", fail_at=3))
        try:
            self.client.post("/lectures", content_type="multipart/form-data", data={
                "name": "broken", "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
            self.jobs.wait(30)
            status = self.client.get("/lectures/broken/status").get_json()
            self.assertEqual((status["state"], status["stage"]), ("failed", 3))
            self.assertIn("exit code 3", status["error"])
        finally:
            self.jobs.main_code = str(fake_script(self.temp.name, "ok.py"))

    def test_windows_blocking_the_status_file_never_stops_the_analysis(self):
        """The bug seen on 2026-11-17: Windows refused to replace the status file while the progress page read it."""
        from web.models import job as job_model
        real_replace = job_model.os.replace          # kept before it is patched
        calls = {"n": 0}

        def flaky_replace(source, target):
            calls["n"] += 1
            if calls["n"] % 3:                       # two of every three saves are blocked at first
                raise PermissionError(5, "Access is denied")
            return real_replace(source, target)

        with mock.patch.object(job_model.os, "replace", side_effect=flaky_replace):
            self.client.post("/lectures", content_type="multipart/form-data", data={
                "name": "locked", "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
            while self.jobs.running():
                self.client.get("/lectures/locked/status")   # keep polling, as the progress page does
            self.jobs.wait(30)
        status = self.client.get("/lectures/locked/status").get_json()
        self.assertEqual((status["state"], status["stage"]), ("done", 7))
        saved = json.loads((self.base / "jobs" / "locked.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "done")             # the retries got the final copy saved

    def test_status_file_that_can_never_be_saved_is_not_an_error(self):
        from web.models import job as job_model
        with mock.patch.object(job_model.os, "replace", side_effect=PermissionError(5, "Access is denied")), \
                mock.patch.object(job_model, "FILE_ATTEMPTS", 2):
            self.client.post("/lectures", content_type="multipart/form-data", data={
                "name": "never_saved", "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
            self.jobs.wait(30)
            status = self.client.get("/lectures/never_saved/status").get_json()
        self.assertEqual(status["state"], "done")            # kept in memory, so the page is still right

    def test_run_again_after_a_failure_uses_the_uploaded_videos(self):
        self.jobs.main_code = str(fake_script(self.temp.name, "fail2.py", fail_at=2))
        try:
            self.client.post("/lectures", content_type="multipart/form-data", data={
                "name": "retry", "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
            self.jobs.wait(30)
            page = self.client.get("/lectures/retry/progress").get_data(as_text=True)
            self.assertIn("Run the analysis again", page)
            self.assertNotIn('id="rerun" method="post" action="/lectures/retry/run"\n        hidden', page)
        finally:
            self.jobs.main_code = str(fake_script(self.temp.name, "ok.py"))
        response = self.client.post("/lectures/retry/run")
        self.assertEqual(response.status_code, 302)
        self.jobs.wait(30)
        status = self.client.get("/lectures/retry/status").get_json()
        self.assertEqual(status["state"], "done")
        self.assertTrue(status["log"][0].endswith("--session retry --overwrite"))   # earlier results are replaced
        self.assertEqual(self.client.post("/lectures/nope/run").status_code, 404)

    def test_a_job_left_running_by_a_stopped_app_shows_as_interrupted(self):
        path = self.base / "jobs" / "ghost.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"lecture_id": "ghost", "state": "running", "stage": 3, "log": []}), encoding="utf-8")
        self.assertEqual(self.jobs.status("ghost")["state"], "interrupted")


class DeleteTests(WebTestCase):
    def upload(self, name):
        response = self.client.post("/lectures", content_type="multipart/form-data", data={
            "name": name, "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
        self.assertEqual(response.status_code, 302)

    def test_delete_removes_everything_of_that_lecture_and_nothing_else(self):
        self.upload("Week 9")
        self.jobs.wait(30)
        lecture = LectureRepository(self.base).get("Week_9")
        events = self.base / "events"
        (lecture.reports_dir).mkdir(parents=True, exist_ok=True)
        (lecture.reports_dir / "final_feedback_report.txt").write_text("report", encoding="utf-8")
        events.mkdir(parents=True, exist_ok=True)
        for name in ("Week_9_instructor_view.jsonl", "Week_9_student_view.jsonl", "demo_lecture_instructor_view.jsonl"):
            (events / name).write_text("{}\n", encoding="utf-8")
        (self.base / "validation" / "Week_9").mkdir(parents=True)
        stored = lecture.stored_paths(events)
        self.assertTrue((self.base / "jobs" / "Week_9.json").exists())

        response = self.client.post("/lectures/Week_9/delete")
        self.assertEqual(response.status_code, 302)
        page = self.client.get(response.headers["Location"]).get_data(as_text=True)
        self.assertIn("“Week 9” was deleted.", page)
        self.assertIn('data-autohide="5000"', page)                    # fades out after 5 s (lectures.js)
        self.assertIn("lectures.js", page)
        self.assertNotIn("lectures.js", self.client.get("/").get_data(as_text=True))   # no message, no script
        self.assertNotIn("/lectures/Week_9/", page)
        self.assertFalse([path for path in stored if path.exists()])
        self.assertIsNone(self.jobs.status("Week_9"))
        self.assertEqual(self.client.get("/lectures/Week_9/progress").status_code, 404)
        self.assertEqual(self.client.post("/lectures/Week_9/delete").status_code, 404)
        self.assertTrue((events / "demo_lecture_instructor_view.jsonl").exists())    # another lecture's files stay
        self.assertTrue(Lecture("demo_lecture", self.base).has_report())

    def test_a_running_analysis_is_stopped_before_its_lecture_is_deleted(self):
        slow = Path(self.temp.name) / "scripts" / "slow.py"
        slow.parent.mkdir(parents=True, exist_ok=True)
        slow.write_text(SLOW_PIPELINE, encoding="utf-8")
        self.jobs.main_code = str(slow)
        try:
            self.upload("busy")
            deadline = time.monotonic() + 30
            while (self.jobs.status("busy") or {}).get("stage") != 1:
                self.assertLess(time.monotonic(), deadline, "the slow analysis never started")
                time.sleep(0.05)
            lecture = LectureRepository(self.base).get("busy")
            self.assertTrue((lecture.session_dir / "held_open.tmp").exists())
            page = self.client.get("/lectures/busy/delete").get_data(as_text=True)
            self.assertIn("being analysed right now. Deleting it stops the analysis.", page)

            started = time.monotonic()
            response = self.client.post("/lectures/busy/delete")
            self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
            self.assertLess(time.monotonic() - started, 20)          # stopped, not waited out (the job sleeps 60 s)
            self.assertIsNone(self.jobs.running())
            self.assertFalse(lecture.input_dir.exists())
            self.assertFalse(lecture.session_dir.exists())             # the child holding a file open was stopped too
            self.assertFalse((self.base / "jobs" / "busy.json").exists())
        finally:
            self.jobs.main_code = str(fake_script(self.temp.name, "ok.py"))
        self.upload("after")                                           # the next analysis starts normally
        self.jobs.wait(30)
        self.assertEqual(self.jobs.status("after")["state"], "done")

    def test_a_delete_sent_from_another_website_is_refused(self):
        LectureRepository(self.base).create("kept", FakeUpload("a.mp4"), FakeUpload("b.mp4"))
        for headers in ({"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}):
            response = self.client.post("/lectures/kept/delete", headers=headers)
            self.assertEqual(response.status_code, 403, headers)
            self.assertIn("nothing was changed", response.get_data(as_text=True))
        self.assertIsNotNone(LectureRepository(self.base).get("kept"))
        response = self.client.post("/lectures/kept/delete", headers={"Origin": "http://localhost",
                                                                     "Sec-Fetch-Site": "same-origin"})
        self.assertEqual(response.status_code, 302)                  # the app's own form
        self.assertIsNone(LectureRepository(self.base).get("kept"))

    def test_files_another_program_holds_open_are_reported(self):
        LectureRepository(self.base).create("held", FakeUpload("a.mp4"), FakeUpload("b.mp4"))
        with mock.patch.object(lecture_model.shutil, "rmtree", side_effect=PermissionError(32, "in use")), \
                mock.patch.object(lecture_model, "REMOVE_ATTEMPTS", 1):
            response = self.client.post("/lectures/held/delete")
        self.assertEqual(response.status_code, 409)
        page = response.get_data(as_text=True)
        self.assertIn("Some files could not be deleted (held)", page)
        self.assertIn("Close it and press Delete again.", page)
        self.assertEqual(self.client.post("/lectures/held/delete").status_code, 302)   # works once the file is free
        self.assertIsNone(LectureRepository(self.base).get("held"))

    def test_only_plain_lecture_ids_are_deleted(self):
        with self.assertRaises(ValueError):
            LectureRepository(self.base).delete(Lecture("../base", self.base))


class ModelTests(unittest.TestCase):
    def test_feedback_text_is_parsed(self):
        parsed = report_model.parse_feedback("An overview.\n\npositive points:\n\n- Good one.\n\nnegative points:\n\n"
                                             "- No clear negative point was found in this analysis.\n\nNote: careful.")
        self.assertEqual(parsed, {"overview": "An overview.", "positive": ["Good one."], "negative": [], "note": "Note: careful."})

    def test_activity_rule_matches_the_report(self):
        cases = [({"instructor_present": 0.2}, "out_of_view"),
                 ({"instructor_present": 1.0, "explanation_ratio": 0.6, "board_work_ratio": 0.4}, "explaining"),
                 ({"instructor_present": 1.0, "explanation_ratio": 0.2, "board_work_ratio": 0.4}, "board"),
                 ({"instructor_present": 1.0, "explanation_ratio": float("nan"), "board_work_ratio": float("nan")}, "in_view"),
                 ({"instructor_present": 1.0, "explanation_ratio": 0.2, "board_work_ratio": float("nan")}, "not_facing")]
        for window, key in cases:
            self.assertEqual(stat_charts.activity_of(window), key, window)
            self.assertEqual(report_model.activity_phrase(window), stat_charts.PHRASES[key])

    def test_mean_range(self):
        self.assertIsNone(stat_charts.mean_range([0.5, 0.6, 0.7]))            # too few parts for a range
        self.assertEqual(stat_charts.mean_range([0.9] * 6), (0.9, 0.9))       # no spread: a zero-width range
        values = [0.8, 0.95, 0.7, 0.9, 1.0, 0.85, 0.75, 0.9]
        low, high = stat_charts.mean_range(values)
        self.assertTrue(0 <= low < sum(values) / len(values) < high <= 1)
        self.assertEqual(stat_charts.fmt_pct(0), "0%")
        self.assertEqual(stat_charts.fmt_pct(0.42), "0.4%")
        self.assertEqual(stat_charts.fmt_pct(95.4), "95%")

    def test_chart_data_from_the_demo_lecture(self):
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lecture(directory)
            lecture = Lecture("demo_lecture", Path(directory))
            timeline = stat_charts.timeline(lecture)
            self.assertEqual(len(timeline["data"]["parts"]), 30)                # 5 minutes in 10-second parts
            self.assertEqual(len(timeline["rows"]), 30)
            json.dumps(timeline["data"])                                         # embedded in the page as JSON
            kinds = {lane["kind"]: lane["count"] for lane in timeline["data"]["lanes"]}
            self.assertEqual(kinds, {"hand_raise": 2, "phone_use": 3, "head_down_still": 3})
            self.assertTrue(all(m["seat"].startswith("seat ") for m in timeline["data"]["moments"]))
            self.assertEqual({a["key"] for a in timeline["data"]["activities"]}, {"explaining", "board", "out_of_view"})
            comparison = stat_charts.comparison(lecture)
            self.assertEqual([m["label"] for m in comparison["measures"]], ["Heads up", "Phone in view", "Head down and still"])
            self.assertEqual(comparison["measures"][0]["max"], 100)
            for row in comparison["rows"]:
                for cell in row["cells"]:
                    self.assertTrue(0 <= cell["width"] <= 100, cell)
                    if cell["range"]:
                        self.assertTrue(0 <= cell["low"] and cell["low"] + cell["span"] <= 100.01, cell)
            explaining = next(r for r in comparison["rows"] if r["key"] == "explaining")
            self.assertIsNotNone(explaining["cells"][0]["range"])                 # enough parts for a 95% range

    def test_links_tab_and_its_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lecture(directory)
            lecture = Lecture("demo_lecture", Path(directory))
            shown = report_model.links(lecture)
            self.assertTrue(shown["available"])
            self.assertEqual((shown["window_seconds"], shown["analysed"], shown["ai"]), ("5", "04:35", False))
            self.assertIn("what you did", shown["overview"])

            jobs = JobRunner(directory, Path(directory) / "events", "python", "main.py", "llm.py", links_report="report.py")
            command = jobs.links_command(lecture)
            self.assertEqual(command[2:5], ["report.py", "--lag-results", str(lecture.reports_dir / "fusion.json")])
            self.assertIn("--you", command)
            self.assertIsNone(JobRunner(directory, directory, "python", "main.py", "llm.py").links_command(lecture))

            fusion = lecture.reports_dir / "fusion.json"
            fusion.write_text(json.dumps({"error": "The two runs do not overlap in real time"}), encoding="utf-8")
            self.assertIn("could not run", report_model.links(lecture)["reason"])
            fusion.unlink()
            self.assertIn("Run its analysis again", report_model.links(lecture)["reason"])
            self.assertIsNone(jobs.links_command(lecture))                       # nothing to rewrite

    def test_report_source_only_trusts_checked_llm_text(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            self.assertEqual(report_model.report_source(reports)["kind"], "rules")
            record = {"model": "claude-opus-5-5", "output": {"positive_points": []}, "attempts": [{"problems": ["bad"]}]}
            (reports / "llm_output.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(report_model.report_source(reports)["kind"], "rules")
            record["attempts"].append({"problems": []})
            (reports / "llm_output.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(report_model.report_source(reports)["kind"], "llm")

    def test_lecture_names_and_uploads(self):
        self.assertEqual(safe_id("Week 5: Recursion!"), "Week_5_Recursion")
        self.assertEqual(safe_id("../.."), "lecture")
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(directory)
            first = repo.create("lec", FakeUpload("a.mp4"), FakeUpload("b.MOV"), {"topic": " T "}, FakeUpload("s.pdf"))
            second = repo.create("lec", FakeUpload("a.mp4"), FakeUpload("b.mp4"))
            self.assertEqual((first.id, second.id), ("lec", "lec_2"))
            self.assertEqual(first.context()["topic"], "T")
            self.assertEqual(first.context()["slides"], "slides.pdf")
            self.assertIsNone(repo.get("../lec"))
            with self.assertRaisesRegex(ValueError, "slides file"):
                repo.create("x", FakeUpload("a.mp4"), FakeUpload("b.mp4"), slides_file=FakeUpload("s.exe"))


if __name__ == "__main__":
    unittest.main()
