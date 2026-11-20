"""Analysis job model: runs the lesson pipeline (scripts/analyze_lesson.py: both cameras, camera links, fusion,
facts.json) and then the report (scripts/report_lesson.py: comparisons, tables, rule-based points and the AI rewrite,
which always runs) in the background, and records which stage it has reached, so the progress page can show it.
One job runs at a time: the analysis uses the whole CPU.

While a job runs, its status lives in memory and the progress page reads it from there. A copy is saved as a small
JSON file (<BASE_DIR>/web_jobs/<id>.json) so the page still works after the run, or after the web app restarts; a job
that was running when the web app stopped is shown as interrupted.

Windows will not replace a file that another program or thread has open for a moment (the progress page, an antivirus
scan, OneDrive), so saving retries briefly, and a missed save never stops the analysis: the status file is only a copy.

A running analysis can be stopped (when its lecture is deleted). Stopping ends the pipeline and every program it
started, such as FFmpeg, because on Windows any of them still running would keep the lecture's files open.
"""
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

STAGES = [
    (1, "Lining up the two cameras by their recording times"),
    (2, "Analysing the student camera (the longest step)"),
    (3, "Analysing the instructor camera"),
    (4, "Linking the two cameras"),
    (5, "Lagged analysis: linking instructor actions to student behaviour"),
    (6, "Preparing the facts and comparisons"),
    (7, "Claude writes the feedback"),
]
# Lines of the two programs' output that start a stage (the lesson pipeline logs, report_lesson.py prints [6/7] ...).
STAGE_PATTERNS = [
    (re.compile(r"src\.preprocessing\.alignment:|analyze_lesson: Lesson \d"), 1),
    (re.compile(r"\[\d/2\] student_view: analyzing"), 2),
    (re.compile(r"\[\d/2\] instructor_view: analyzing"), 3),
    (re.compile(r"Linking the cameras"), 4),
    (re.compile(r"Fusing the runs"), 5),
    (re.compile(r"^\[6/7\]"), 6),
    (re.compile(r"^\[7/7\]"), 7),
]
LOG_LINES = 80
SAVE_EVERY_SECONDS = 1.0           # the status copy on disk is refreshed at most this often (and at every stage)
FILE_ATTEMPTS = 20                 # retries when Windows briefly blocks the status file
STOP_WAIT_SECONDS = 30             # how long stop() waits for a stopped analysis to end


def stop_process_tree(process):
    """Ends a program and every program it started (the analysis runs FFmpeg), so none keeps a lecture file open."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)   # the pipeline runs in its own process group (start_new_session)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass


def read_report_status(lecture):
    """report_status.json of the lecture's lesson ({"source": "llm" | "rules", "reason": ...}), or {}."""
    path = lecture.lesson_dir / "report_status.json" if lecture.lesson_dir else None
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def stage_of(line):
    """The stage a line of output starts, or None."""
    return next((stage for pattern, stage in STAGE_PATTERNS if pattern.search(line)), None)


class JobRunner:
    def __init__(self, base_dir, python, analyze, report, ffmpeg_dir=None, llm_model="opus", room_config="none"):
        self.base_dir = Path(base_dir)
        self.python, self.analyze, self.report = str(python), str(analyze), str(report)
        self.ffmpeg_dir = ffmpeg_dir
        self.llm_model, self.room_config = llm_model, str(room_config)
        self._lock = threading.Lock()
        self._running = None           # id of the lecture being analysed
        self._live = None              # its status, kept in memory
        self._finished = {}            # final status of jobs that ran since the app started, in case a save was missed
        self._thread = None
        self._process = None           # the program running right now (pipeline or rewrite), so it can be stopped
        self._cancel = False           # set by stop(): the running job ends as soon as possible
        self._last_save = 0.0

    # ------------------------------------------------------------------ status
    def _path(self, lecture_id):
        return self.base_dir / "web_jobs" / f"{lecture_id}.json"

    def _write(self, status):
        """Saves the status copy; True if it worked. Retries while Windows blocks the file, never raises."""
        path = self._path(status["lecture_id"])
        temporary = path.with_suffix(".tmp")
        for attempt in range(FILE_ATTEMPTS):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(temporary, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
                os.replace(temporary, path)   # readers never see a half-written file
                return True
            except OSError:                   # PermissionError on Windows while the file is open elsewhere
                time.sleep(0.02 * (attempt + 1))
        return False

    def _save(self, status, force=False):
        now = time.monotonic()
        if not force and now - self._last_save < SAVE_EVERY_SECONDS:
            return
        with self._lock:
            snapshot = json.loads(json.dumps(status))
        if self._write(snapshot):
            self._last_save = now

    def _read(self, path):
        for attempt in range(FILE_ATTEMPTS):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.02 * (attempt + 1))
        return None

    def status(self, lecture_id):
        with self._lock:
            if self._running == lecture_id and self._live is not None:
                return json.loads(json.dumps(self._live))   # the running job: from memory, no file access
            if lecture_id in self._finished:
                return json.loads(json.dumps(self._finished[lecture_id]))
        path = self._path(lecture_id)
        status = self._read(path) if path.exists() else None
        if status and status["state"] in ("queued", "running"):
            status["state"] = "interrupted"   # the web app was stopped while this job ran
        return status

    def running(self):
        return self._running

    def wait(self, timeout=None):
        """Waits for the current job to finish (used by tests)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self, lecture_id, timeout=STOP_WAIT_SECONDS):
        """Stops this lecture's analysis if it is the one running, and waits until it has ended, so that no program
        still has the lecture's files open. True when it is not running (any more)."""
        with self._lock:
            if self._running != lecture_id:
                return True
            self._cancel = True
            process = self._process
        if process is not None:
            stop_process_tree(process)
        self.wait(timeout)
        return self._running != lecture_id

    def forget(self, lecture_id):
        """Drops what this app remembers about a lecture's last job (after the lecture was deleted)."""
        with self._lock:
            self._finished.pop(lecture_id, None)

    # ------------------------------------------------------------------ running
    def environment(self):
        # YOLO_AUTOINSTALL / YOLO_OFFLINE: Ultralytics must never download or install anything during an analysis
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="False",
                   YOLO_OFFLINE="True")
        if self.ffmpeg_dir:
            env["PATH"] = str(self.ffmpeg_dir) + os.pathsep + env.get("PATH", "")
        return env

    def analysis_command(self, lecture, overwrite=False):
        command = [self.python, "-u", self.analyze, "--student-video", str(lecture.video_path("students")),
                   "--instructor-video", str(lecture.video_path("instructor")), "--name", lecture.id,
                   "--output-dir", str(self.base_dir), "--local-config", self.room_config, "--no-side-by-side"]
        return command + (["--overwrite"] if overwrite else [])

    def report_command(self, lecture):
        return [self.python, "-u", self.report, "--name", lecture.id, "--output-dir", str(self.base_dir),
                "--model", self.llm_model]

    def start(self, lecture, overwrite=False, report_only=False):
        """Starts the analysis in the background. overwrite=True replaces the lecture's earlier results (a rerun);
        report_only=True runs only the report step again (Claude writes the feedback; "Try again" on the report)."""
        status = {"lecture_id": lecture.id, "state": "running", "stage": 5 if report_only else 0,
                  "stage_name": "Starting", "report_only": report_only,
                  "llm_result": None, "llm_ok": None, "error": None, "log": [],
                  "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None}
        with self._lock:
            if self._running:
                raise RuntimeError(f"An analysis is already running ({self._running}). Wait for it to finish.")
            self._running, self._live, self._cancel = lecture.id, status, False
        self._save(status, force=True)
        self._thread = threading.Thread(target=self._run, args=(lecture, overwrite, status, report_only), daemon=True)
        self._thread.start()

    def _set(self, status, **changes):
        with self._lock:
            status.update(changes)
        self._save(status, force=True)

    def _stream(self, command, status, stage_from_output=True):
        with subprocess.Popen(command, cwd=str(Path(self.analyze).parents[1]), env=self.environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1, start_new_session=os.name != "nt") as process:
            with self._lock:
                self._process = process
                cancelled = self._cancel
            try:
                if cancelled:                 # stop() came while this program was starting
                    stop_process_tree(process)
                for line in process.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    stage = stage_of(line) if stage_from_output else None
                    with self._lock:
                        status["log"] = (status["log"] + [line])[-LOG_LINES:]
                        if stage and stage >= status["stage"]:
                            status["stage"] = stage
                            status["stage_name"] = STAGES[stage - 1][1]
                    self._save(status, force=bool(stage))
                return process.wait()
            finally:
                with self._lock:
                    self._process = None

    def _stopped(self, status):
        """True (and the status says so) when stop() ended this job."""
        if self._cancel:
            self._set(status, state="cancelled", stage_name="Stopped", error=None)
        return self._cancel

    def _run(self, lecture, overwrite, status, report_only=False):
        try:
            commands = [self.report_command(lecture)] if report_only else                 [self.analysis_command(lecture, overwrite), self.report_command(lecture)]
            for command in commands:
                code = self._stream(command, status)
                if self._stopped(status):
                    return
                if code != 0:
                    self._set(status, state="failed", error=f"The analysis stopped with an error (exit code {code}). "
                                                            "The technical log below shows where.")
                    return
            # the charts, moments and key terms are complete even when Claude could not write the feedback
            report = read_report_status(lecture)
            if report.get("source") == "llm":
                self._set(status, llm_ok=True, llm_result="Claude wrote the feedback and it passed every automatic "
                                                          "check.")
            elif report.get("code"):   # key or billing problem: only the code (user requests 2026-11-20)
                self._set(status, llm_ok=False, llm_result=f"Error, Code={report['code']}")
            else:
                self._set(status, llm_ok=False, llm_result="Claude could not write the feedback, so the report has "
                          "none yet; use \"Try again\" on the report. Reason: "
                          f"{report.get('reason') or 'unknown (no report_status.json).'}")
            self._set(status, state="done", stage_name="Finished")
        except Exception as error:   # shown on the progress page instead of disappearing in a background thread
            with self._lock:
                status.update(state="failed", error=f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save(status, force=True)
            with self._lock:
                self._finished[lecture.id] = json.loads(json.dumps(status))
                self._running, self._live = None, None
