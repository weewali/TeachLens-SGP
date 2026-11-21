"""Starts the web app (web/) on the evaluation demo lecture folder, local only.

    python evaluation/run_webapp.py      then open http://127.0.0.1:5000
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEACHLENS_OUTPUTS", str(ROOT / "outputs" / "evaluation" / "2026-11-21_webapp_demo"))
os.environ.setdefault("TEACHLENS_ROOM_CONFIG", "none")
sys.argv = [str(ROOT / "web" / "app.py")]
runpy.run_path(sys.argv[0], run_name="__main__")
