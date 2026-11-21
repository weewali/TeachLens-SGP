"""Default settings for the TeachLens web app. create_app() accepts overrides (tests use temporary folders)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    # the lesson pipeline's output root: uploads/<id>/, lessons/<stamp>_<id>/, web_jobs/, validation/ (ignored by Git)
    "BASE_DIR": Path(os.environ.get("TEACHLENS_OUTPUTS", REPO_ROOT / "outputs")),
    "PYTHON": sys.executable,
    # step 1: both cameras, camera links, fusion and facts.json (System A's one-command lesson pipeline)
    "ANALYZE": REPO_ROOT / "scripts" / "analyze_lesson.py",
    # step 2: comparisons, tables, and Claude writes the feedback (always run; no rule-based text: if Claude fails,
    # the report says why and offers "Try again")
    "REPORT": REPO_ROOT / "scripts" / "report_lesson.py",
    # room file with the board, screen and desk areas (configs/*.yaml); "none" = no room areas, so "at the board"
    # and "at the desk" are not measured. The pipeline's own default is the scenario room, which is wrong elsewhere.
    "ROOM_CONFIG": os.environ.get("TEACHLENS_ROOM_CONFIG", "none"),
    # optional folder holding ffmpeg and ffprobe, put first on PATH for analysis runs (if they are not on PATH already)
    "FFMPEG_DIR": os.environ.get("TEACHLENS_FFMPEG_DIR"),
    # browser used to print the report to PDF (run hidden); empty = Microsoft Edge or Google Chrome where installed
    "PDF_BROWSER": os.environ.get("TEACHLENS_PDF_BROWSER"),
    # uploads are lecture videos (often several GB), so no size limit; the app only listens on this computer
    "MAX_CONTENT_LENGTH": None,
    # The AI rewrite of the feedback always runs (user decision 2026-11-20), about 4 US cents per lecture with the
    # default model. It sends only facts.json (numbers, no video, slides or names). It needs ANTHROPIC_API_KEY set for
    # this server; without it, or on any error, the rule-based feedback is shown and the page says why.
    "LLM_MODEL": os.environ.get("TEACHLENS_LLM_MODEL", "opus"),
}
HOST, PORT = "127.0.0.1", 5000   # local only: the videos show real people
