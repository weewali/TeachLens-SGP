"""The MVC web app: models (lectures, jobs, report data, charts) and controllers (pages, uploads, progress, files).

Uses a made-up lesson (tests/fixtures/demo_lesson.py) and fake analysis / report scripts, so no video, model or
network is needed."""

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from demo_lesson import make_demo_lesson  # noqa: E402
from web.app import create_app  # noqa: E402
from web.models import charts  # noqa: E402
from web.models import lecture as lecture_model  # noqa: E402
from web.models import report as report_model  # noqa: E402
from web.models import stat_charts  # noqa: E402
from web.models.job import JobRunner, stage_of  # noqa: E402
from web.models.lecture import Lecture, LectureRepository, safe_id  # noqa: E402

# Stands in for scripts/analyze_lesson.py: prints the lines the progress page follows and makes the lesson folder.
FAKE_ANALYZE = """
import os, sys
args = sys.argv[1:]
print("args: " + " ".join(args), flush=True)
name, out = args[args.index("--name") + 1], args[args.index("--output-dir") + 1]
lines = ["INFO src.preprocessing.alignment: lining up", "INFO src.pipeline: [1/2] student_view: analyzing s.mp4",
         "INFO src.pipeline: [2/2] instructor_view: analyzing i.mp4", "INFO src.pipeline: Linking the cameras by real time",
         "INFO src.pipeline: Fusing the runs and running the lag analysis"]
for n, line in enumerate(lines, 1):
    print(line, flush=True)
    if n == {fail_at}:
        print("Something went wrong", flush=True)
        sys.exit(3)
lesson = os.path.join(out, "lessons", "2026-11-20_1300_" + name)
os.makedirs(lesson, exist_ok=True)
open(os.path.join(lesson, "facts.json"), "w").write("{}")
"""

# Stands in for scripts/report_lesson.py. FAKE_AI=llm in the environment = Claude wrote the feedback; otherwise it
# failed as if no API key were set (no feedback, only the reason).
FAKE_REPORT = """
import glob, json, os, sys
args = sys.argv[1:]
name, out = args[args.index("--name") + 1], args[args.index("--output-dir") + 1]
lesson = sorted(glob.glob(os.path.join(out, "lessons", "*_" + name)))[-1]
print("[6/7] Preparing the facts and comparisons", flush=True)
print("[7/7] Writing the feedback with Claude", flush=True)
if os.environ.get("FAKE_AI") == "llm":
    open(os.path.join(lesson, "final_feedback_report.txt"), "w").write(
        "positive points:\\n\\n- Made up.\\n\\nnegative points:\\n\\n- Also made up.\\n")
status = {"source": "llm", "reason": None} if os.environ.get("FAKE_AI") == "llm" else \\
    {"source": "none", "reason": "No Claude API key is set on this computer (ANTHROPIC_API_KEY). Nothing was sent.",
     "code": "01"}
json.dump(status, open(os.path.join(lesson, "report_status.json"), "w"))
"""

# An analysis that is still busy: it has started a child program that keeps one of the lesson's files open (on
# Windows an open file cannot be deleted).
SLOW_ANALYZE = """
import os, subprocess, sys, time
args = sys.argv[1:]
name, out = args[args.index("--name") + 1], args[args.index("--output-dir") + 1]
folder = os.path.join(out, "lessons", "2026-11-20_1300_" + name)
os.makedirs(folder, exist_ok=True)
held = os.path.join(folder, "held_open.tmp")
subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'w'); time.sleep(60)", held])
while not os.path.exists(held):
    time.sleep(0.05)
print("INFO src.preprocessing.alignment: lining up", flush=True)
time.sleep(60)
"""


class FakeUpload:
    def __init__(self, filename, data=b"video"):
        self.filename, self.data = filename, data

    def save(self, path):
        Path(path).write_bytes(self.data)


def fake_script(directory, name, text, fail_at=0):
    path = Path(directory) / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("{fail_at}", str(fail_at)), encoding="utf-8")
    return path


def upload(client, name):
    return client.post("/lectures", content_type="multipart/form-data", data={
        "name": name, "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})


class WebTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / "base"
        make_demo_lesson(cls.base)
        cls.ok_analyze = fake_script(cls.temp.name, "analyze.py", FAKE_ANALYZE)
        cls.app = create_app(BASE_DIR=cls.base, TESTING=True, ANALYZE=cls.ok_analyze,
                             REPORT=fake_script(cls.temp.name, "report.py", FAKE_REPORT))
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
        self.assertNotIn("<footer", page)
        for expected in ("Positive points", "Negative points", "Students raised their hands 5 times", "Phone seen",
                         "Dozing posture", "Hand raised", '<h3 class="chart-title">Instructor Movement</h3>',
                         '<h3 class="chart-title">Student Seats</h3>', "What the words in this report mean",
                         '<a href="#terms">Key terms</a>', '<section id="terms" class="tab-panel terms-panel">',
                         "<dt>on task</dt>", "<dt>dozing posture</dt>", "<dt>95% range</dt>",
                         "<dt>30-second average</dt>", "05:00 of lecture", "✗", '<nav class="tabs"',
                         '<section id="feedback" class="tab-panel is-active">',
                         '<section id="charts" class="tab-panel charts-panel">', '<section id="moments" class="tab-panel">',
                         '<span class="tab-count">', "uaeu-logo.svg", "Over the lecture", 'class="chart-data"',
                         "Show the numbers", "Students during each of your activities", "compare-bar act-explaining",
                         # the AI rewrite always runs; when it fails, the page says why (user decision 2026-11-20)
                         "Feedback written by Claude from the measured results"):
            self.assertIn(expected, page)
        for developer_detail in (".jsonl", "evidence.strength", "fusion.json", "T_facing_class", "S_on_task", "q-value",
                                 "Links over time", 'href="#links"', "<figcaption><strong>", 'class="note"',
                                 "<dt>window</dt>", "<dt>later</dt>", "<dt>lesson time</dt>", "timeline.png", "seat_0",
                                 "Note: made-up numbers"):
            self.assertNotIn(developer_detail, page)
        lecture = Lecture("demo_lecture", self.base)
        self.assertTrue((lecture.charts_dir / "instructor_heatmap.png").exists())
        self.assertTrue((lecture.charts_dir / "seat_heatmap.png").exists())

    def test_charts_are_served_and_nothing_else(self):
        self.client.get("/lectures/demo_lecture")
        response = self.client.get("/lectures/demo_lecture/charts/seat_heatmap.png")
        self.assertEqual((response.status_code, response.mimetype), (200, "image/png"))
        response.close()
        for path in ("/lectures/demo_lecture/charts/facts.json", "/lectures/demo_lecture/charts/..%2Ffacts.json",
                     "/lectures/nope/charts/seat_heatmap.png", "/lectures/..%2F..%2Fsecret"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_new_form(self):
        page = self.client.get("/lectures/new").get_data(as_text=True)
        for expected in ("Video of the instructor", "Video of the students", "What was the class about?", "Lecture slides",
                         "Start analysis", '<legend>Activities in class <span class="muted">(optional)</span></legend>',
                         "Additional information:", 'value="" required>', 'data-activity="quiz"',
                         'name="activity_minutes" min="1" max="180"', 'data-activity="group_work"',
                         'data-activity="video"', 'data-activity="break"', '<template id="activity-row">',
                         'id="activity-add" hidden', "<noscript>"):
            self.assertIn(expected, page)
        for not_for_instructors in ("Claude", "API key", "cents"):
            self.assertNotIn(not_for_instructors, page)

    def test_upload_is_checked(self):
        response = self.client.post("/lectures", data={"name": "x"}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose the video of the instructor", response.get_data(as_text=True))
        for title in (None, "   "):   # the title is required too, and checked before any video is saved
            data = {"instructor_video": (io.BytesIO(b"v"), "a.mp4"), "students_video": (io.BytesIO(b"v"), "b.mp4")}
            if title is not None:
                data["name"] = title
            response = self.client.post("/lectures", content_type="multipart/form-data", data=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn("Enter a lecture title.", response.get_data(as_text=True))
        self.assertFalse((self.base / "uploads" / "lecture").exists())
        response = self.client.post("/lectures", content_type="multipart/form-data", data={"name": "x",
            "instructor_video": (io.BytesIO(b"v"), "a.mp4"), "students_video": (io.BytesIO(b"v"), "notes.txt")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("student video must be one of", response.get_data(as_text=True))

    def test_a_wrong_activity_is_refused_before_any_video_is_saved_and_its_rows_come_back(self):
        response = self.client.post("/lectures", content_type="multipart/form-data", data={
            "name": "Activity check", "activity_type": ["quiz", "break"], "activity_minutes": ["15", "half"],
            "instructor_video": (io.BytesIO(b"i"), "a.mp4"), "students_video": (io.BytesIO(b"s"), "b.mp4")})
        self.assertEqual(response.status_code, 400)
        page = response.get_data(as_text=True)
        self.assertIn("Enter how long the break lasted in whole minutes, from 1 to 180.", page)
        for kept in ('value="quiz"', 'value="break"', 'value="15"', 'value="half"', '<span class="activity-name">Break</span>',
                     'aria-label="Duration of the break in minutes"'):
            self.assertIn(kept, page)
        self.assertFalse((self.base / "uploads" / "Activity_check").exists())

    def test_the_report_downloads_as_a_pdf(self):
        from urllib.parse import urlparse
        from urllib.request import url2pathname
        printed = []

        def fake_browser(command, **_kwargs):   # stands in for Edge printing the page
            page = Path(url2pathname(urlparse(command[-1]).path))
            printed.append(page.read_text(encoding="utf-8"))
            out = next(a for a in command if a.startswith("--print-to-pdf=")).split("=", 1)[1]
            Path(out).write_bytes(b"%PDF-1.4 made-up")

        config = self.app.config
        config.update(PDF_BROWSER=sys.executable, PDF_RUNNER=fake_browser)
        try:
            page = self.client.get("/lectures/demo_lecture").get_data(as_text=True)
            self.assertIn('href="/lectures/demo_lecture/report.pdf"', page)
            self.assertIn("Download PDF", page)
            self.assertNotIn(" download>", page)      # an error must show as a page, never be saved as a file
            response = self.client.get("/lectures/demo_lecture/report.pdf")
            self.assertEqual((response.status_code, response.mimetype), (200, "application/pdf"))
            self.assertIn('filename="TeachLens report - Demo lecture (made-up data).pdf"',
                          response.headers["Content-Disposition"])
            self.assertTrue(response.data.startswith(b"%PDF"))
            response.close()
            html = printed[0]
            for embedded in ("<style>", "data:image/png;base64,", "data:image/svg+xml;base64,", 'class="pdf"',
                             "<th>Seats (times)</th>", "Students raised their hands 5 times",
                             'class="card chart print-chart"',
                             "(a 30-second average", " moments are listed below."):
                self.assertIn(embedded, html)
            for external in ('src="/static', 'href="/static', "/charts/", "Download PDF",
                             "Show the numbers", "Point at the chart", 'class="timeline-plot"'):   # nothing to click on paper
                self.assertNotIn(external, html)      # self-contained: the browser loads nothing from the server
            self.client.get("/lectures/demo_lecture/report.pdf").close()
            self.assertEqual(len(printed), 1)         # unchanged report: the saved PDF is served again
            self.assertEqual(self.client.get("/lectures/nope/report.pdf").status_code, 404)

            Lecture("demo_lecture", self.base).lesson_dir.joinpath("report.pdf").unlink()
            config["PDF_BROWSER"] = str(self.base / "no_such_browser.exe")
            response = self.client.get("/lectures/demo_lecture/report.pdf")
            self.assertEqual(response.status_code, 503)
            self.assertIn("Save as PDF", response.get_data(as_text=True))
            config["PDF_BROWSER"] = sys.executable
            with mock.patch("web.controllers.lectures.render_template", side_effect=[RuntimeError("template"), "x"]):
                response = self.client.get("/lectures/demo_lecture/report.pdf")   # any failure: a readable page
            self.assertEqual(response.status_code, 503)
        finally:
            config.update(PDF_BROWSER=None, PDF_RUNNER=None)

    def test_every_moment_is_in_the_pdf(self):
        from web.models import pdf
        moments = [{"kind": "phone_use", "seat": "seat 1", "seconds": s, "duration": 1, "behaviour": "Phone seen",
                    "time": "", "end": "", "activity": "facing the class"} for s in (5, 7, 30)]
        moments.append({"kind": "phone_use", "seat": "seat 2", "seconds": 6, "duration": 2, "behaviour": "Phone seen",
                        "time": "", "end": "", "activity": "facing the class"})
        rows = pdf.joined_moments(moments)
        self.assertEqual([(r["seat"], r["_count"], r["seconds"], r["end"]) for r in rows],
                         [("seat 1", 2, 5, "00:08"), ("seat 2", 1, 6, "00:08"), ("seat 1", 1, 30, "00:31")])
        self.assertEqual(rows[0]["behaviour"], "Phone seen (2 times)")
        self.assertEqual(sum(r["_count"] for r in rows), len(moments))   # nothing is left out

    def test_paper_figures(self):
        from web.models import print_charts
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lesson(directory)
            lecture = Lecture("demo_lecture", Path(directory))
            timeline = print_charts.timeline_png(stat_charts.timeline(lecture)["data"])
            bars = print_charts.comparison_png(stat_charts.comparison_values(lecture)[0])
            for png in (timeline, bars):
                self.assertEqual(png[1:4], b"PNG")
        self.assertIsNone(print_charts.timeline_png(None))
        self.assertIsNone(print_charts.comparison_png([]))

    def test_report_lists_the_activities_the_instructor_marked(self):
        context = {"name": "Demo", "activities": [{"type": "quiz", "minutes": 15}, {"type": "break", "minutes": 1},
                                                  {"type": "video", "minutes": "x"}]}   # a hand-edited row is left out
        with mock.patch.object(Lecture, "context", return_value=context):
            page = self.client.get("/lectures/demo_lecture").get_data(as_text=True)
        self.assertIn("<dt>Activities</dt>", page)
        self.assertIn("<li>Quiz or test (15 minutes)</li>", page)
        self.assertIn("<li>Break (1 minute)</li>", page)
        self.assertNotIn("<li>Video", page)


class JobTests(WebTestCase):
    def test_upload_starts_the_analysis_and_progress_follows_it(self):
        response = self.client.post("/lectures", content_type="multipart/form-data", data={
            "name": "Week 5", "topic": "Recursion", "notes": "first lecture after the midterm",
            "activity_type": ["quiz", ""], "activity_minutes": ["15", ""],   # the 2nd row is empty
            "instructor_video": (io.BytesIO(b"i"), "IMG_0001.MOV"), "students_video": (io.BytesIO(b"s"), "cam2.mp4")})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/lectures/Week_5/progress"))
        self.jobs.wait(30)
        status = self.client.get("/lectures/Week_5/status").get_json()
        self.assertEqual((status["state"], status["stage"]), ("done", 7))
        self.assertIn("[7/7] Writing the feedback with Claude", status["log"])
        self.assertTrue(status["has_report"])
        self.assertIs(status["llm_ok"], False)
        self.assertEqual(status["llm_result"], "Error, Code=01")   # an API key problem: only its code (progress page)
        page = self.client.get("/lectures/Week_5").get_data(as_text=True)          # charts etc., but no points
        self.assertIn("<p>Error, Code=01</p>", page)                  # the short banner (user request 2026-11-20)
        self.assertIn("No analysis done yet.", page)
        self.assertNotIn("API key", page)
        self.assertIn('action="/lectures/Week_5/rewrite"', page)
        self.assertNotIn("Positive points", page)
        with mock.patch.dict("os.environ", {"FAKE_AI": "llm"}):                   # Try again: only the report step
            self.assertEqual(self.client.post("/lectures/Week_5/rewrite").status_code, 302)
            self.jobs.wait(30)
        status = self.client.get("/lectures/Week_5/status").get_json()
        self.assertEqual((status["state"], status["llm_ok"]), ("done", True))
        self.assertFalse(any("student_view: analyzing" in line for line in status["log"]))   # videos not re-analysed
        page = self.client.get("/lectures/Week_5").get_data(as_text=True)
        self.assertIn("Made up.", page)
        self.assertNotIn("Claude could not write the feedback", page)
        self.assertEqual(self.client.post("/lectures/Week_5/rewrite", headers={"Sec-Fetch-Site": "cross-site"}).status_code,
                         403)                                                       # it costs money: own pages only
        lecture = LectureRepository(self.base).get("Week_5")
        self.assertEqual(lecture.context()["topic"], "Recursion")
        self.assertEqual(lecture.context()["activities"], [{"type": "quiz", "minutes": 15}])
        self.assertEqual(lecture.video_path("instructor").name, "instructor.mov")
        self.assertEqual(lecture.lesson_dir.name, "2026-11-20_1300_Week_5")
        page = self.client.get("/lectures/Week_5/progress").get_data(as_text=True)
        for stage in ("Analysing the student camera", "Lagged analysis: linking instructor actions to student behaviour",
                      "Claude writes the feedback"):
            self.assertIn(stage, page)

    def test_a_passing_ai_rewrite_is_reported(self):
        with mock.patch.dict("os.environ", {"FAKE_AI": "llm"}):
            upload(self.client, "with_ai")
            self.jobs.wait(30)
        status = self.client.get("/lectures/with_ai/status").get_json()
        self.assertEqual((status["state"], status["llm_ok"]), ("done", True))
        self.assertIn("passed every automatic check", status["llm_result"])

    def test_commands_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lesson(directory, run_report=False)
            lecture = Lecture("demo_lecture", Path(directory))
            jobs = JobRunner(directory, "python", "analyze_lesson.py", "report_lesson.py", room_config="none")
            command = jobs.analysis_command(lecture)
            for part in ("--student-video", str(lecture.video_path("students")), "--instructor-video", "--name",
                         "demo_lecture", "--output-dir", directory, "--local-config", "none", "--no-side-by-side"):
                self.assertIn(part, command)
            self.assertNotIn("--overwrite", command)
            self.assertEqual(jobs.analysis_command(lecture, overwrite=True)[-1], "--overwrite")
            self.assertEqual(jobs.report_command(lecture)[2:], ["report_lesson.py", "--name", "demo_lecture",
                                                                "--output-dir", directory, "--model", "opus"])
            env = jobs.environment()
            self.assertEqual((env["YOLO_AUTOINSTALL"], env["YOLO_OFFLINE"]), ("False", "True"))   # never downloads
        self.assertEqual(stage_of("2026 INFO src.pipeline: [2/2] instructor_view: analyzing x"), 3)
        self.assertEqual(stage_of("[6/7] Writing the report"), 6)
        self.assertIsNone(stage_of("W0000 face_landmarker_graph.cc:181] ..."))

    def test_a_failing_analysis_is_reported(self):
        self.jobs.analyze = str(fake_script(self.temp.name, "fail.py", FAKE_ANALYZE, fail_at=3))
        try:
            upload(self.client, "broken")
            self.jobs.wait(30)
            status = self.client.get("/lectures/broken/status").get_json()
            self.assertEqual((status["state"], status["stage"]), ("failed", 3))
            self.assertIn("exit code 3", status["error"])
        finally:
            self.jobs.analyze = str(self.ok_analyze)

    def test_windows_blocking_the_status_file_never_stops_the_analysis(self):
        """The bug seen on 2026-11-17: Windows refused to replace the status file while the progress page read it."""
        from web.models import job as job_model
        real_replace = job_model.os.replace
        calls = {"n": 0}

        def flaky_replace(source, target):
            calls["n"] += 1
            if calls["n"] % 3:                       # two of every three saves are blocked at first
                raise PermissionError(5, "Access is denied")
            return real_replace(source, target)

        with mock.patch.object(job_model.os, "replace", side_effect=flaky_replace):
            upload(self.client, "locked")
            while self.jobs.running():
                self.client.get("/lectures/locked/status")   # keep polling, as the progress page does
            self.jobs.wait(30)
        status = self.client.get("/lectures/locked/status").get_json()
        self.assertEqual((status["state"], status["stage"]), ("done", 7))
        saved = json.loads((self.base / "web_jobs" / "locked.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "done")

    def test_status_file_that_can_never_be_saved_is_not_an_error(self):
        from web.models import job as job_model
        with mock.patch.object(job_model.os, "replace", side_effect=PermissionError(5, "Access is denied")), \
                mock.patch.object(job_model, "FILE_ATTEMPTS", 2):
            upload(self.client, "never_saved")
            self.jobs.wait(30)
            status = self.client.get("/lectures/never_saved/status").get_json()
        self.assertEqual(status["state"], "done")

    def test_run_again_after_a_failure_uses_the_uploaded_videos(self):
        self.jobs.analyze = str(fake_script(self.temp.name, "fail2.py", FAKE_ANALYZE, fail_at=2))
        try:
            upload(self.client, "retry")
            self.jobs.wait(30)
            page = self.client.get("/lectures/retry/progress").get_data(as_text=True)
            self.assertIn("Run the analysis again", page)
        finally:
            self.jobs.analyze = str(self.ok_analyze)
        response = self.client.post("/lectures/retry/run")
        self.assertEqual(response.status_code, 302)
        self.jobs.wait(30)
        status = self.client.get("/lectures/retry/status").get_json()
        self.assertEqual(status["state"], "done")
        self.assertTrue(status["log"][0].endswith("--overwrite"))   # earlier results are replaced
        self.assertEqual(self.client.post("/lectures/nope/run").status_code, 404)

    def test_a_job_left_running_by_a_stopped_app_shows_as_interrupted(self):
        path = self.base / "web_jobs" / "ghost.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"lecture_id": "ghost", "state": "running", "stage": 3, "log": []}), encoding="utf-8")
        self.assertEqual(self.jobs.status("ghost")["state"], "interrupted")


class DeleteTests(WebTestCase):
    def test_delete_removes_everything_of_that_lecture_and_nothing_else(self):
        self.assertEqual(upload(self.client, "Week 9").status_code, 302)
        self.jobs.wait(30)
        lecture = LectureRepository(self.base).get("Week_9")
        (self.base / "validation" / "Week_9").mkdir(parents=True)
        stored = lecture.stored_paths()
        self.assertTrue(lecture.lesson_dir.exists())
        self.assertTrue((self.base / "web_jobs" / "Week_9.json").exists())

        response = self.client.post("/lectures/Week_9/delete")
        self.assertEqual(response.status_code, 302)
        page = self.client.get(response.headers["Location"]).get_data(as_text=True)
        self.assertIn("“Week 9” was deleted.", page)
        self.assertIn('data-autohide="5000"', page)
        self.assertNotIn("/lectures/Week_9/", page)
        self.assertFalse([path for path in stored if path.exists()])
        self.assertIsNone(self.jobs.status("Week_9"))
        self.assertEqual(self.client.get("/lectures/Week_9/progress").status_code, 404)
        self.assertEqual(self.client.post("/lectures/Week_9/delete").status_code, 404)
        self.assertTrue(Lecture("demo_lecture", self.base).has_report())   # another lecture's files stay

    def test_a_running_analysis_is_stopped_before_its_lecture_is_deleted(self):
        self.jobs.analyze = str(fake_script(self.temp.name, "slow.py", SLOW_ANALYZE))
        try:
            upload(self.client, "busy")
            deadline = time.monotonic() + 30
            while (self.jobs.status("busy") or {}).get("stage") != 1:
                self.assertLess(time.monotonic(), deadline, "the slow analysis never started")
                time.sleep(0.05)
            lecture = LectureRepository(self.base).get("busy")
            self.assertTrue((lecture.lesson_dir / "held_open.tmp").exists())
            page = self.client.get("/lectures/busy/delete").get_data(as_text=True)
            self.assertIn("being analysed right now. Deleting it stops the analysis.", page)
            folder = lecture.lesson_dir
            started = time.monotonic()
            response = self.client.post("/lectures/busy/delete")
            self.assertEqual(response.status_code, 302, response.get_data(as_text=True))
            self.assertLess(time.monotonic() - started, 20)          # stopped, not waited out (the job sleeps 60 s)
            self.assertIsNone(self.jobs.running())
            self.assertFalse(lecture.input_dir.exists())
            self.assertFalse(folder.exists())                         # the child holding a file open was stopped too
            self.assertFalse((self.base / "web_jobs" / "busy.json").exists())
        finally:
            self.jobs.analyze = str(self.ok_analyze)
        upload(self.client, "after")                                   # the next analysis starts normally
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
        self.assertEqual(response.status_code, 302)
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
        self.assertEqual(self.client.post("/lectures/held/delete").status_code, 302)
        self.assertIsNone(LectureRepository(self.base).get("held"))

    def test_only_plain_lecture_ids_are_deleted(self):
        with self.assertRaises(ValueError):
            LectureRepository(self.base).delete(Lecture("../base", self.base))


class ModelTests(unittest.TestCase):
    def test_claude_is_set_up_from_the_private_env_file(self):
        from web import app as app_module
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("# private\nANTHROPIC_API_KEY=made-up-key-123\nexport TEACHLENS_LLM_MODEL='haiku'\n"
                                "OTHER_SECRET=not-read\n", encoding="utf-8")
            environ = {"TEACHLENS_ROOM_CONFIG": "set-in-terminal"}
            self.assertEqual(app_module.load_env_file(env_file, environ), ["ANTHROPIC_API_KEY", "TEACHLENS_LLM_MODEL"])
            self.assertEqual((environ["ANTHROPIC_API_KEY"], environ["TEACHLENS_LLM_MODEL"]), ("made-up-key-123", "haiku"))
            self.assertNotIn("OTHER_SECRET", environ)                       # only the app's own settings
            env_file.write_text("TEACHLENS_ROOM_CONFIG=from-file\n", encoding="utf-8")
            app_module.load_env_file(env_file, environ)
            self.assertEqual(environ["TEACHLENS_ROOM_CONFIG"], "set-in-terminal")   # the terminal wins
            message = app_module.claude_status(environ)
            self.assertIn("API key found", message)
            self.assertNotIn("made-up-key-123", message)                    # the key is never shown
            self.assertIn("NO API key", app_module.claude_status({}))
            self.assertEqual(app_module.load_env_file(Path(directory) / "missing.env", {}), [])
            env_file.write_text("ANTHROPIC_API_KEY=paste-the-key-here\nTEACHLENS_LLM_MODEL=opus\n", encoding="utf-8")
            self.assertEqual(app_module.load_env_file(env_file, {}), ["TEACHLENS_LLM_MODEL"])   # placeholder: no key

    def test_feedback_text_is_parsed(self):
        parsed = report_model.parse_feedback("An overview.\n\npositive points:\n\n- Good one.\n\nnegative points:\n\n"
                                             "- No clear negative point was found in this analysis.\n\nNote: careful.")
        self.assertEqual(parsed, {"overview": "An overview.", "positive": ["Good one."], "negative": [],
                                  "positive_empty": None,
                                  "negative_empty": "No clear negative point was found in this analysis.",
                                  "note": "Note: careful."})
        left_out = report_model.parse_feedback("positive points:\n\n- No positive point passed the automatic checks "
                                               "this time.\n\nnegative points:\n\n- A real point.\n")
        self.assertEqual((left_out["positive"], left_out["positive_empty"], left_out["negative"]),
                         ([], "No positive point passed the automatic checks this time.", ["A real point."]))

    def test_report_source_shows_why_the_ai_was_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            lesson = Path(directory)
            self.assertIn("has not written the feedback", report_model.report_source(lesson)["problem"])
            (lesson / "report_status.json").write_text(json.dumps({"source": "none", "reason": "No network."}))
            self.assertEqual(report_model.report_source(lesson),
                             {"kind": "none", "label": None, "problem": "No network.", "code": None})
            (lesson / "report_status.json").write_text(json.dumps({"source": "none", "code": "01",
                                                                   "reason": "Claude rejected the API key."}))
            self.assertEqual(report_model.report_source(lesson)["problem"], "Code=01")   # never the key reason
            (lesson / "report_status.json").write_text(json.dumps({"source": "none", "code": "02",
                                                                   "reason": "Your credit balance is too low."}))
            self.assertEqual(report_model.report_source(lesson)["problem"], "Code=02")   # never the billing reason
            (lesson / "report_status.json").write_text(json.dumps({"source": "llm", "reason": None}))
            (lesson / "final_feedback_report.txt").write_text("positive points:\n\n- A point.\n")
            source = report_model.report_source(lesson)
            self.assertEqual((source["kind"], source["problem"]), ("llm", None))
            (lesson / "llm_output.json").write_text(json.dumps({"prompt_version": "4"}))   # the earlier wording
            source = report_model.report_source(lesson)
            self.assertEqual((source["kind"], source.get("outdated")), ("none", True))   # hidden; "Try again" shown
            (lesson / "llm_output.json").write_text(json.dumps({"prompt_version": report_model.PROMPT_VERSION}))
            self.assertEqual(report_model.report_source(lesson)["kind"], "llm")

    def test_chart_data_and_moments_from_the_demo_lesson(self):
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lesson(directory)
            lecture = Lecture("demo_lecture", Path(directory))
            timeline = stat_charts.timeline(lecture)
            self.assertEqual(len(timeline["data"]["parts"]), 60)                # 5 minutes in 5-second parts
            json.dumps(timeline["data"])                                         # embedded in the page as JSON
            kinds = {lane["kind"]: lane["count"] for lane in timeline["data"]["lanes"]}
            self.assertEqual(kinds, {"hand_raise": 2, "phone_use": 2, "dozing": 1})
            self.assertTrue(all(m["seat"].startswith("seat ") for m in timeline["data"]["moments"]))
            self.assertEqual({a["key"] for a in timeline["data"]["activities"]}, {"explaining", "board", "not_facing"})
            comparison = stat_charts.comparison(lecture)
            self.assertEqual([m["label"] for m in comparison["measures"]], ["On task", "Phone", "Head down"])
            self.assertEqual(comparison["measures"][0]["max"], 100)
            for row in comparison["rows"]:
                for cell in row["cells"]:
                    self.assertTrue(0 <= cell["width"] <= 100, cell)
                    if cell["range"]:
                        self.assertTrue(0 <= cell["low"] and cell["low"] + cell["span"] <= 100.01, cell)
            explaining = next(r for r in comparison["rows"] if r["key"] == "explaining")
            self.assertIsNotNone(explaining["cells"][0]["range"])
            moments = report_model.moments(lecture)
            phone = next(m for m in moments if m["kind"] == "phone_use")
            self.assertEqual((phone["time"], phone["seat"], phone["activity"]), ("00:20", "seat 4", "facing the class"))

    def test_lecture_names_and_uploads(self):
        self.assertEqual(safe_id("Week 5: Recursion!"), "Week_5_Recursion")
        self.assertEqual(safe_id("../.."), "lecture")
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(directory)
            first = repo.create("lec", FakeUpload("a.mp4"), FakeUpload("b.MOV"), {"topic": " T "}, FakeUpload("s.pdf"))
            second = repo.create("lec", FakeUpload("a.mp4"), FakeUpload("b.mp4"))
            self.assertEqual((first.id, second.id), ("lec", "lec_2"))
            self.assertEqual(first.input_dir, Path(directory) / "uploads" / "lec")
            self.assertEqual(first.context()["topic"], "T")
            self.assertEqual(first.context()["slides"], "slides.pdf")
            self.assertIsNone(first.lesson_dir)
            (Path(directory) / "lessons" / "2026-11-20_1300_my_lec").mkdir(parents=True)   # another lecture's folder
            self.assertIsNone(first.lesson_dir)                                  # "*_lec" must not match "my_lec"
            self.assertIsNone(repo.get("../lec"))
            with self.assertRaisesRegex(ValueError, "slides file"):
                repo.create("x", FakeUpload("a.mp4"), FakeUpload("b.mp4"), slides_file=FakeUpload("s.exe"))
            for no_title in (None, "", " \t"):
                with self.assertRaisesRegex(ValueError, "Enter a lecture title"):
                    repo.create(no_title, FakeUpload("a.mp4"), FakeUpload("b.mp4"))
            self.assertEqual(second.context()["activities"], [])
            self.assertEqual(repo.create(" Week 5 ", FakeUpload("a.mp4"), FakeUpload("b.mp4")).context()["name"], "Week 5")

    def test_planned_activities_are_checked(self):
        parse = lecture_model.parse_activities
        self.assertEqual(parse(["quiz", "break"], ["15", " 5 "]),
                         [{"type": "quiz", "minutes": 15}, {"type": "break", "minutes": 5}])
        self.assertEqual(parse(["", ""], ["", " "]), [])
        self.assertEqual(parse([], []), [])
        self.assertEqual(parse(["break", "break"], ["10", "10"]), [{"type": "break", "minutes": 10}] * 2)
        self.assertEqual(parse(["video"], ["180"]), [{"type": "video", "minutes": 180}])
        for (types, durations), message in (
                ((["quiz"], [""]), "Fill in how many minutes the quiz or test lasted, or remove it"),
                ((["break"], ["0"]), "whole minutes, from 1 to 180"),
                ((["break"], ["181"]), "whole minutes, from 1 to 180"),
                ((["break"], ["-5"]), "whole minutes"),
                ((["break"], ["7.5"]), "whole minutes"),
                ((["break"], ["10:30"]), "whole minutes"),
                ((["break"], ["1e1"]), "whole minutes"),
                ((["nap"], ["15"]), "Choose what kind of activity"),
                ((["", "quiz"], ["10", "15"]), "Choose what kind of activity"),
                ((["quiz", "group_work"], ["120", "61"]), "Together the activities last more than 180 minutes"),
                ((["break"] * 11, ["1"] * 11), "at most 10 activities")):
            with self.assertRaisesRegex(ValueError, message):
                parse(types, durations)
        self.assertEqual(lecture_model.describe_activity({"type": "group_work", "minutes": 20}),
                         "Group or pair work (20 minutes)")


class FaceBlurTests(unittest.TestCase):
    """Both heatmaps pixelate the heads in their camera frame; the videos and the analysis are never changed."""

    def test_head_areas_and_pixelation(self):
        seated, cut_off = (0.25, 0.2, 0.5, 0.8), (0.6, 0.85, 0.8, 1.0)   # fractions of the frame
        (h1, h2) = charts.head_areas([seated, cut_off])
        np.testing.assert_allclose(h1, (0.25 - 0.05, 0.2 - 0.042, 0.5 + 0.05, 0.2 + 0.21 + 0.042))   # top 35%, padded
        self.assertAlmostEqual(h2[3], 1.0)                 # touching the bottom edge: the whole box is blurred
        image = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        out = charts.pixelate(image, [h1])
        self.assertTrue((out[140:] == image[140:]).all())     # everything below the head is untouched
        head = out[60:120, 90:190]
        self.assertLessEqual(len(np.unique(head.reshape(-1, 3), axis=0)), 49)   # a few flat blocks, not the face
        self.assertTrue((charts.pixelate(image, []) == image).all())

    def test_seats_used_only_briefly_are_not_drawn(self):
        summary = {"analyzed_duration_seconds": 600, "seats": {"seats": [
            {"seat_id": "seat_01", "occupied_seconds": 590}, {"seat_id": "seat_07", "occupied_seconds": 3},   # someone passing
            {"seat_id": "seat_08", "occupied_seconds": 59}, {"seat_id": "seat_09"}]}}                  # no time given: kept
        self.assertEqual([s["seat_id"] for s in charts.real_seats(summary)], ["seat_01", "seat_09"])

    def test_the_instructor_is_not_blurred_but_everyone_else_is(self):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        teacher_box, student_box = (0.25, 0.3, 0.35, 0.9), (0.6, 0.7, 0.8, 1.0)   # the demo instructor's box
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(charts, "grab_frame", return_value=(frame, 0.0)) as grab, \
                mock.patch.object(charts, "people_in", return_value=[teacher_box, student_box]), \
                mock.patch.object(charts, "pixelate", wraps=charts.pixelate) as blurred:
            lesson = make_demo_lesson(directory, run_report=False)
            charts.blurred_background(lesson, "instructor", blur_teacher=False)
            areas = blurred.call_args.args[1]
            self.assertEqual(areas, charts.head_areas([student_box]))          # the student only
            self.assertEqual(grab.call_args.args[2], 0.0)                     # the frame of the instructor's first sample
            charts.blurred_background(lesson, "instructor")                   # the default blurs everyone
            areas = blurred.call_args.args[1]
            self.assertIn(charts.head_areas([teacher_box])[0], areas)
            self.assertIn(charts.head_areas([student_box])[0], areas)

    def test_a_failure_never_shows_the_unblurred_frame(self):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(charts, "grab_frame", return_value=(frame, 0.0)), \
                mock.patch.object(charts, "people_in", side_effect=RuntimeError("model broken")):
            lesson = make_demo_lesson(directory, run_report=False)
            self.assertIsNone(charts.blurred_background(lesson, "student"))

    def test_charts_are_drawn_and_old_ones_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            make_demo_lesson(directory, run_report=False)
            lecture = Lecture("demo_lecture", Path(directory))
            lecture.charts_dir.mkdir(parents=True)
            old = charts.chart_path(lecture, "seat_heatmap.png")
            old.write_bytes(b"an old chart in an old style")
            shown = charts.ensure_charts(lecture)
            self.assertEqual([c["file"] for c in shown], ["instructor_heatmap.png", "seat_heatmap.png"])
            self.assertNotEqual(old.read_bytes(), b"an old chart in an old style")
            self.assertEqual((lecture.charts_dir / "style.txt").read_text(encoding="utf-8"), charts.CHART_STYLE)
            (lecture.charts_dir / "style.txt").write_text("1", encoding="utf-8")
            with mock.patch.object(charts, "draw_seat_heatmap", return_value=False):
                shown = charts.ensure_charts(lecture)
            self.assertFalse(old.exists())                                   # cannot be redrawn: not shown at all
            self.assertEqual([c["file"] for c in shown], ["instructor_heatmap.png"])


if __name__ == "__main__":
    unittest.main()
