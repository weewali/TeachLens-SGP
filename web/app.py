"""TeachLens web application (MVC with Flask and Jinja2).

    python web/app.py          then open http://127.0.0.1:5000 in a browser

Model:      web/models/       lectures and their files, analysis jobs, report data, charts
View:       web/views/        Jinja2 templates and static CSS / JavaScript
Controller: web/controllers/  Flask routes

The app only listens on this computer (127.0.0.1): the videos show real people.
"""
import os
from pathlib import Path
import sys

# Allow direct `python web/app.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask  # noqa: E402

from web.config import DEFAULTS, HOST, PORT  # noqa: E402
from web.controllers.lectures import bp as lectures  # noqa: E402
from web.models.job import JobRunner  # noqa: E402
from web.models.lecture import LectureRepository  # noqa: E402

VIEWS = Path(__file__).resolve().parent / "views"


def create_app(**overrides):
    app = Flask(__name__, template_folder=str(VIEWS / "templates"), static_folder=str(VIEWS / "static"))
    app.config.update(DEFAULTS)
    # Re-read changed templates without a server restart (Flask only does this in debug mode by default).
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config.update(overrides)
    config = app.config
    app.extensions["teachlens"] = {
        "lectures": LectureRepository(config["BASE_DIR"]),
        "jobs": JobRunner(config["BASE_DIR"], config["PYTHON"], config["ANALYZE"], config["REPORT"],
                          ffmpeg_dir=config["FFMPEG_DIR"], llm_model=config["LLM_MODEL"],
                          room_config=config["ROOM_CONFIG"]),
    }
    app.register_blueprint(lectures)
    return app


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"   # private settings file, never uploaded (.gitignore)
ENV_KEYS = ("ANTHROPIC_API_KEY",)                            # plus every TEACHLENS_* setting
PLACEHOLDER = "paste-the-key-here"                           # the old template's stand-in: never used as a key


def load_env_file(path=ENV_FILE, environ=None):
    """Reads KEY=VALUE lines from the private .env file into the environment (a value already set in the terminal
    wins), so Claude is set up every time the web app starts. Returns the names it set; never prints a value."""
    environ = os.environ if environ is None else environ
    loaded = []
    if not Path(path).exists():
        return loaded
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip('"').strip("'")
        if (key in ENV_KEYS or key.startswith("TEACHLENS_")) and value and value != PLACEHOLDER \
                and not environ.get(key):
            environ[key] = value
            loaded.append(key)
    return loaded


def claude_status(environ=None):
    """One line for the terminal: whether Claude can write the feedback (the key itself is never shown)."""
    environ = os.environ if environ is None else environ
    if environ.get("ANTHROPIC_API_KEY") or environ.get("ANTHROPIC_AUTH_TOKEN"):
        model = environ.get("TEACHLENS_LLM_MODEL", DEFAULTS["LLM_MODEL"])
        return f"Claude: API key found; Claude ({model}) will write each report's feedback."
    return ("Claude: NO API key found, so reports will have no written feedback. Put the line "
            "ANTHROPIC_API_KEY=<key> in the file .env in the project folder (see web/README.md) and start again.")


if __name__ == "__main__":
    loaded = load_env_file()
    if loaded:
        print(f"Settings read from .env: {', '.join(loaded)}")
    print(claude_status())
    print(f"TeachLens is running: open http://{HOST}:{PORT} in your browser (Ctrl+C stops it).")
    # settings in .env were read after web.config, so pass them on (the API key reaches the report step through the
    # environment the analysis jobs inherit)
    from_env = {"LLM_MODEL": "TEACHLENS_LLM_MODEL", "ROOM_CONFIG": "TEACHLENS_ROOM_CONFIG",
                "PDF_BROWSER": "TEACHLENS_PDF_BROWSER", "FFMPEG_DIR": "TEACHLENS_FFMPEG_DIR",
                "BASE_DIR": "TEACHLENS_OUTPUTS"}
    overrides = {setting: (Path(os.environ[name]) if setting == "BASE_DIR" else os.environ[name])
                 for setting, name in from_env.items() if os.environ.get(name)}
    create_app(**overrides).run(host=HOST, port=PORT, debug=False, threaded=True)
