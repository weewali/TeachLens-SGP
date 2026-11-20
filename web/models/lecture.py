"""Lecture model: where one lecture's inputs and results are stored, and which lectures exist.

Everything lives under BASE_DIR, the lesson pipeline's output root (outputs/, ignored by Git):
Inputs:  <BASE_DIR>/uploads/<id>/instructor.<ext>, students.<ext>, optional slides.<ext>, context.json
Results: <BASE_DIR>/lessons/<YYYY-MM-DD_HHMM>_<id>/ (scripts/analyze_lesson.py, then scripts/report_lesson.py),
         <BASE_DIR>/web_jobs/<id>.json (the analysis status), <BASE_DIR>/validation/<id>/ (review sheets)
Deleting a lecture removes exactly these.
"""
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LESSON_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_(?P<id>[A-Za-z0-9_-]{1,64})$")   # output_layout's folder names
CAMERAS = ("instructor", "students")
REMOVE_ATTEMPTS = 8               # retries while Windows briefly keeps a file open (antivirus scan, OneDrive)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
SLIDE_EXTENSIONS = (".pdf", ".pptx", ".ppt", ".odp", ".key")

# Planned activities the instructor can mark on the upload form (optional). Stored in context.json as
# {"type", "start", "end"} with start/end in seconds from the start of the instructor video. Not used by the analysis
# yet: they are kept with the report for reference.
ACTIVITY_TYPES = {"quiz": "Quiz or test", "group_work": "Group or pair work", "video": "Video", "break": "Break"}
MAX_ACTIVITIES = 10
MAX_ACTIVITY_MINUTES = 180         # all activities together; lectures are 55-75 minutes of teaching, recordings may run longer
MINUTES_RE = re.compile(r"^\d{1,3}$")


def parse_activities(types, durations):
    """The upload form's activity rows -> [{"type", "minutes"}] in the order given. Only how long each activity lasted
    is asked, not when it happened. Rows left completely empty are skipped. Raises ValueError with a message for the
    instructor."""
    rows = []
    for n in range(max(len(types), len(durations))):
        kind, text = ((values[n] if n < len(values) else "") or "" for values in (types, durations))
        kind, text = kind.strip(), text.strip()
        if not (kind or text):
            continue
        if kind not in ACTIVITY_TYPES:
            raise ValueError("Choose what kind of activity each row is (quiz or test, group or pair work, video or break).")
        name = ACTIVITY_TYPES[kind].lower()
        if not text:
            raise ValueError(f"Fill in how many minutes the {name} lasted, or remove it.")
        if not MINUTES_RE.match(text) or not 1 <= int(text) <= MAX_ACTIVITY_MINUTES:
            raise ValueError(f"Enter how long the {name} lasted in whole minutes, from 1 to {MAX_ACTIVITY_MINUTES}.")
        rows.append({"type": kind, "minutes": int(text)})
    if len(rows) > MAX_ACTIVITIES:
        raise ValueError(f"Add at most {MAX_ACTIVITIES} activities.")
    if sum(row["minutes"] for row in rows) > MAX_ACTIVITY_MINUTES:
        raise ValueError(f"Together the activities last more than {MAX_ACTIVITY_MINUTES} minutes. Please check the "
                         "durations.")
    return rows


def describe_activity(activity):
    """{"type": "quiz", "minutes": 15} -> "Quiz or test (15 minutes)"."""
    name = ACTIVITY_TYPES.get(activity.get("type"), "Activity")
    minutes = int(activity["minutes"])
    return f"{name} ({minutes} minute{'' if minutes == 1 else 's'})"


def safe_id(name):
    """A lecture name that is safe as a folder name and as the lesson pipeline's --name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:48] or "lecture"


def _extension(filename, allowed, what):
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise ValueError(f"The {what} must be one of: {', '.join(allowed)} (got '{filename or 'nothing'}').")
    return ext


def _remove(path):
    """Deletes a file or a whole folder; True when it is gone. Retries briefly, because Windows will not delete a
    file that another program has open for a moment."""
    for attempt in range(REMOVE_ATTEMPTS):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(0.1 * (attempt + 1))
    return not path.exists()


@dataclass(frozen=True)
class Lecture:
    id: str
    base_dir: Path

    @property
    def input_dir(self):
        return self.base_dir / "uploads" / self.id

    def lesson_dirs(self):
        """Every lesson folder made for this lecture (one per analysis with a new date), oldest first."""
        root = self.base_dir / "lessons"
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir()
                      if p.is_dir() and (m := LESSON_DIR_RE.match(p.name)) and m.group("id") == self.id)

    @property
    def lesson_dir(self):
        """The newest lesson folder, or None before the first analysis."""
        found = self.lesson_dirs()
        return found[-1] if found else None

    @property
    def tables_dir(self):
        return self.lesson_dir / "tables" if self.lesson_dir else self.input_dir / "no_tables"

    @property
    def charts_dir(self):
        """The report page's heatmaps (drawn by web/models/charts.py), kept with the lesson."""
        return self.lesson_dir / "web_charts" if self.lesson_dir else self.input_dir / "no_charts"

    def video_path(self, camera):
        for ext in VIDEO_EXTENSIONS:
            for path in (self.input_dir / f"{camera}{ext}", self.input_dir / f"{camera}{ext.upper()}"):
                if path.exists():
                    return path
        return None

    def context(self):
        path = self.input_dir / "context.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def result(self):
        """The lesson pipeline's lesson.json (what ran, ranges, findings), or None before the first analysis."""
        path = self.lesson_dir / "lesson.json" if self.lesson_dir else None
        if path is None or not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def has_report(self):
        lesson = self.lesson_dir
        return bool(lesson and (lesson / "final_feedback_report.txt").exists() and (lesson / "facts.json").exists())

    def analysed_at(self):
        path = self.lesson_dir / "final_feedback_report.txt" if self.lesson_dir else None
        if path is None or not path.exists():
            return None
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))

    def stored_paths(self):
        """Everything kept on this computer for this lecture alone: uploads, lesson folders, analysis status and any
        review sheets."""
        jobs = self.base_dir / "web_jobs"
        return [self.input_dir, *self.lesson_dirs(), jobs / f"{self.id}.json", jobs / f"{self.id}.tmp",
                self.base_dir / "validation" / self.id]


class LectureRepository:
    """Finds, lists, creates and deletes lectures under one base folder (the lesson pipeline's outputs/)."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def delete(self, lecture):
        """Removes the lecture and everything stored for it (stop its analysis first). Returns the paths that could
        not be removed, e.g. a video that another program still has open; empty when everything is gone."""
        if not ID_RE.match(lecture.id):   # never build a path from anything but a plain lecture ID
            raise ValueError(f"Not a lecture ID: {lecture.id!r}")
        return [path for path in lecture.stored_paths() if not _remove(path)]

    def get(self, lecture_id):
        if not lecture_id or not ID_RE.match(lecture_id):
            return None
        lecture = Lecture(lecture_id, self.base_dir)
        return lecture if lecture.input_dir.exists() or lecture.lesson_dir else None

    def all(self):
        """Every lecture uploaded through the web app, newest first. Lessons analysed from the command line without
        an upload are not listed (their videos are not here to re-run)."""
        root = self.base_dir / "uploads"
        if not root.exists():
            return []
        lectures = [Lecture(p.name, self.base_dir) for p in root.iterdir() if p.is_dir() and ID_RE.match(p.name)]
        lectures = [lec for lec in lectures if lec.video_path("instructor")]

        def newest(lec):
            paths = [lec.input_dir / "context.json"]
            if lec.lesson_dir:
                paths.append(lec.lesson_dir / "final_feedback_report.txt")
            return max((p.stat().st_mtime for p in paths if p.exists()), default=0)
        return sorted(lectures, key=newest, reverse=True)

    def new_id(self, name):
        base = safe_id(name)
        candidate, n = base, 2
        while Lecture(candidate, self.base_dir).input_dir.exists() or Lecture(candidate, self.base_dir).lesson_dir:
            candidate, n = f"{base}_{n}", n + 1
        return candidate

    def create(self, name, instructor_file, students_file, context=None, slides_file=None, activities=None):
        """Saves the two videos (and optional slides, context and planned activities from parse_activities) for a
        new lecture. The file objects only need .filename and .save(path), like Flask's uploads. Raises ValueError
        with a message for the instructor."""
        title = (name or "").strip()
        if not title:
            raise ValueError("Enter a lecture title.")
        if not instructor_file or not getattr(instructor_file, "filename", ""):
            raise ValueError("Choose the video of the instructor.")
        if not students_file or not getattr(students_file, "filename", ""):
            raise ValueError("Choose the video of the students.")
        extensions = {"instructor": _extension(instructor_file.filename, VIDEO_EXTENSIONS, "instructor video"),
                      "students": _extension(students_file.filename, VIDEO_EXTENSIONS, "student video")}
        slides_ext = _extension(slides_file.filename, SLIDE_EXTENSIONS, "slides file") if slides_file else None

        lecture = Lecture(self.new_id(title), self.base_dir)
        lecture.input_dir.mkdir(parents=True)
        instructor_file.save(str(lecture.input_dir / f"instructor{extensions['instructor']}"))
        students_file.save(str(lecture.input_dir / f"students{extensions['students']}"))
        if slides_file:
            slides_file.save(str(lecture.input_dir / f"slides{slides_ext}"))
        context = {k: (v or "").strip() for k, v in (context or {}).items()}
        context.update(name=title, created=time.strftime("%Y-%m-%d %H:%M"),
                       slides=f"slides{slides_ext}" if slides_file else None, activities=list(activities or []),
                       original_files={"instructor": instructor_file.filename, "students": students_file.filename})
        with open(lecture.input_dir / "context.json", "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
        return lecture
