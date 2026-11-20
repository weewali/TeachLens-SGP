"""TeachLens web application (MVC with Flask and Jinja2).

    python web/app.py          then open http://127.0.0.1:5000 in a browser

Model:      web/models/       lectures and their files, analysis jobs, report data, charts
View:       web/views/        Jinja2 templates and static CSS / JavaScript
Controller: web/controllers/  Flask routes

The app only listens on this computer (127.0.0.1): the videos show real people.
"""
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


if __name__ == "__main__":
    print(f"TeachLens is running: open http://{HOST}:{PORT} in your browser (Ctrl+C stops it).")
    create_app().run(host=HOST, port=PORT, debug=False, threaded=True)
