"""Analysis job model: runs the two-camera pipeline (and, when the deployment enables it, the plain-language rewrite
of the feedback with Claude) in the background and records which stage it has reached, so the progress page can show
it. One job runs at a time: the analysis uses the whole CPU.

While a job runs, its status lives in memory and the progress page reads it from there. A copy is saved as a small
JSON file (<BASE_DIR>/jobs/<id>.json) so the page still works after the run, or after the web app restarts; a job
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
    (1, "Checking both videos and lining up the two cameras"),
    (2, "Cutting both videos to the shared time and extracting frames"),
    (3, "Detecting people and body poses in both cameras"),
    (4, "Recognising behaviours and writing the two event files"),
    (5, "Building 10-second windows"),
    (6, "Lagged analysis: linking instructor actions to student behaviour"),
    (7, "Writing the report"),
    (8, "Writing the feedback in plain language"),   # only when the deployment enables it
]
STAGE_RE = re.compile(r"^\[(\d)/7\]")
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


class JobRunner:
    def __init__(self, base_dir, events_dir, python, main_code, llm_report, ffmpeg_dir=None, llm_mode="auto",
                 llm_model="opus", links_report=None):
        self.base_dir, self.events_dir = Path(base_dir), Path(events_dir)
        self.python, self.main_code, self.llm_report = str(python), str(main_code), str(llm_report)
        self.links_report = str(links_report) if links_report else None   # scripts/report.py: the links over time
        self.ffmpeg_dir = ffmpeg_dir
        self.llm_mode, self.llm_model = llm_mode, llm_model
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
        return self.base_dir / "jobs" / f"{lecture_id}.json"

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
        env = dict(os.environ, TEACHLENS_BASE=str(self.base_dir), TEACHLENS_EVENTS_DIR=str(self.events_dir),
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        if self.ffmpeg_dir:
            env["PATH"] = str(self.ffmpeg_dir) + os.pathsep + env.get("PATH", "")
        return env

    def analysis_command(self, lecture, overwrite=False):
        command = [self.python, "-u", self.main_code, "--instructor", str(lecture.video_path("instructor")),
                   "--students", str(lecture.video_path("students")), "--session", lecture.id]
        return command + (["--overwrite"] if overwrite else [])

    def llm_command(self, lecture):
        return [self.python, "-u", self.llm_report, "--facts", str(lecture.reports_dir / "facts.json"),
                "--model", self.llm_model]

    def links_command(self, lecture):
        """The plain-language rewrite of the links over time (the team's fusion results), or None when this lecture
        has none to rewrite. It keeps the rule-based points when the rewrite fails its checks."""
        fusion = lecture.reports_dir / "fusion.json"
        if not self.links_report or not fusion.exists():
            return None
        return [self.python, "-u", self.links_report, "--lag-results", str(fusion), "--out-dir",
                str(lecture.reports_dir / "links"), "--lecture-id", lecture.id, "--model", self.llm_model, "--you"]

    def llm_enabled(self):
        """Whether the plain-language rewrite runs: a deployment setting (TEACHLENS_LLM), never an instructor choice."""
        if self.llm_mode == "off":
            return False
        if self.llm_mode == "on":
            return True
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))   # auto

    def start(self, lecture, overwrite=False):
        """Starts the analysis in the background. overwrite=True replaces the lecture's earlier results (a rerun)."""
        status = {"lecture_id": lecture.id, "state": "running", "stage": 0, "stage_name": "Starting",
                  "use_llm": self.llm_enabled(), "llm_result": None, "error": None, "log": [],
                  "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None}
        with self._lock:
            if self._running:
                raise RuntimeError(f"An analysis is already running ({self._running}). Wait for it to finish.")
            self._running, self._live, self._cancel = lecture.id, status, False
        self._save(status, force=True)
        self._thread = threading.Thread(target=self._run, args=(lecture, overwrite, status), daemon=True)
        self._thread.start()

    def _set(self, status, **changes):
        with self._lock:
            status.update(changes)
        self._save(status, force=True)

    def _stream(self, command, status, stage_from_output):
        with subprocess.Popen(command, cwd=str(Path(self.main_code).parents[1]), env=self.environment(),
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
                    match = STAGE_RE.match(line) if stage_from_output else None
                    with self._lock:
                        status["log"] = (status["log"] + [line])[-LOG_LINES:]
                        if match:
                            status["stage"] = int(match.group(1))
                            status["stage_name"] = STAGES[status["stage"] - 1][1]
                    self._save(status, force=bool(match))
                return process.wait()
            finally:
                with self._lock:
                    self._process = None

    def _stopped(self, status):
        """True (and the status says so) when stop() ended this job."""
        if self._cancel:
            self._set(status, state="cancelled", stage_name="Stopped", error=None)
        return self._cancel

    def _run(self, lecture, overwrite, status):
        try:
            code = self._stream(self.analysis_command(lecture, overwrite), status, stage_from_output=True)
            if self._stopped(status):
                return
            if code != 0:
                self._set(status, state="failed", error=f"The analysis stopped with an error (exit code {code}). "
                                                        "The technical log below shows where.")
                return
            if status["use_llm"]:
                self._set(status, stage=8, stage_name=STAGES[7][1])
                code = self._stream(self.llm_command(lecture), status, stage_from_output=False)
                if self._stopped(status):
                    return
                links = self.links_command(lecture)
                if links:   # the links over time: falls back to its rule-based points by itself
                    self._stream(links, status, stage_from_output=False)
                    if self._stopped(status):
                        return
                # a failed rewrite is not a failed analysis: the standard feedback is complete on its own
                self._set(status, llm_result="Feedback written in plain language and checked." if code == 0 else
                          "The plain-language rewrite did not pass its checks, so the standard feedback is shown.")
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
