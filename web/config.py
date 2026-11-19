"""Default settings for the TeachLens web app. create_app() accepts overrides (tests use temporary folders)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    # where lectures' inputs (videos/<id>/) and results (sessions/<id>/) live; the same folder the pipeline uses
    "BASE_DIR": Path(os.environ.get("TEACHLENS_BASE", REPO_ROOT / "classroom_project")),
    # where the pipeline writes the two per-camera JSONL event files
    "EVENTS_DIR": Path(os.environ.get("TEACHLENS_EVENTS_DIR", REPO_ROOT / "outputs" / "events")),
    "PYTHON": sys.executable,
    "MAIN_CODE": REPO_ROOT / "main code" / "Main Code.py",
    "LLM_REPORT": REPO_ROOT / "main code" / "llm_report.py",
    # the plain-language rewrite of the "links over time" points (the team's fusion results), same deployment setting
    "LINKS_REPORT": REPO_ROOT / "scripts" / "report.py",
    # optional folder holding ffmpeg and ffprobe, put first on PATH for analysis runs (if they are not on PATH already)
    "FFMPEG_DIR": os.environ.get("TEACHLENS_FFMPEG_DIR"),
    # uploads are lecture videos (often several GB), so no size limit; the app only listens on this computer
    "MAX_CONTENT_LENGTH": None,
    # The plain-language rewrite of the feedback (Claude) is a deployment setting, not an instructor choice:
    #   auto (default) = run it when an Anthropic API key is set for this server, otherwise keep the standard feedback
    #   on  = always try it (it falls back to the standard feedback if no key works)
    #   off = never call it
    # Each lecture gets up to two rewrites (the feedback, and the links over time when that analysis ran), each about
    # 4 US cents with the default model; they send only the measured numbers, never video, slides or names.
    "LLM_MODE": os.environ.get("TEACHLENS_LLM", "auto"),
    "LLM_MODEL": os.environ.get("TEACHLENS_LLM_MODEL", "opus"),
}
HOST, PORT = "127.0.0.1", 5000   # local only: the videos show real people
