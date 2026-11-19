"""Lecture controller: list lectures, start a new analysis, follow its progress, show the report, delete a lecture."""
from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for

from web.models.charts import CHARTS, chart_path, ensure_charts
from web.models.job import STAGES
from web.models.report import build_report

bp = Blueprint("lectures", __name__)


def _models():
    return current_app.extensions["teachlens"]


def _lecture_or_404(lecture_id):
    lecture = _models()["lectures"].get(lecture_id)
    if lecture is None:
        abort(404)
    return lecture


def _from_this_site():
    """False when the browser says a form was sent from another website: a page elsewhere must not be able to make
    this app delete lectures. Browsers send these headers on every form they post."""
    if request.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
        return False
    origin = request.headers.get("Origin")
    return origin is None or origin == request.host_url.rstrip("/")


@bp.get("/")
def index():
    models = _models()
    rows = [{"lecture": lecture, "context": lecture.context(), "status": models["jobs"].status(lecture.id),
             "has_report": lecture.has_report(), "analysed_at": lecture.analysed_at()}
            for lecture in models["lectures"].all()]
    deleted = (request.args.get("deleted") or "")[:120]
    return render_template("index.html", rows=rows, running=models["jobs"].running(), deleted=deleted)


@bp.get("/lectures/new")
def new():
    return render_template("new.html", form={}, error=None, running=_models()["jobs"].running())


@bp.post("/lectures")
def create():
    models = _models()
    form = request.form
    if models["jobs"].running():
        return render_template("new.html", form=form, running=models["jobs"].running(),
                               error="Another analysis is still running. Start this one when it has finished."), 409
    slides = request.files.get("slides")
    try:
        lecture = models["lectures"].create(
            form.get("name"), request.files.get("instructor_video"), request.files.get("students_video"),
            context={"topic": form.get("topic"), "notes": form.get("notes")},
            slides_file=slides if slides is not None and slides.filename else None)
    except ValueError as error:
        return render_template("new.html", form=form, error=str(error), running=None), 400
    models["jobs"].start(lecture)
    return redirect(url_for("lectures.progress", lecture_id=lecture.id))


@bp.get("/lectures/<lecture_id>/progress")
def progress(lecture_id):
    lecture = _lecture_or_404(lecture_id)
    return render_template("progress.html", lecture=lecture, context=lecture.context(),
                           status=_models()["jobs"].status(lecture.id), stages=STAGES)


@bp.post("/lectures/<lecture_id>/run")
def rerun(lecture_id):
    """Runs the analysis again on the videos already uploaded (after a failure or an interruption)."""
    lecture = _lecture_or_404(lecture_id)
    jobs = _models()["jobs"]
    if not (lecture.video_path("instructor") and lecture.video_path("students")):
        abort(404)
    if jobs.running():
        return render_template("progress.html", lecture=lecture, context=lecture.context(), stages=STAGES,
                               status=jobs.status(lecture.id), notice="Another analysis is still running. "
                               "Try again when it has finished."), 409
    jobs.start(lecture, overwrite=True)   # the instructor asked for a fresh run, so earlier results are replaced
    return redirect(url_for("lectures.progress", lecture_id=lecture.id))


def _delete_page(lecture, error=None, status=200):
    return render_template("delete.html", lecture=lecture, context=lecture.context(), error=error,
                           running=_models()["jobs"].running() == lecture.id, has_report=lecture.has_report()), status


@bp.get("/lectures/<lecture_id>/delete")
def confirm_delete(lecture_id):
    """Asks before deleting: what will be removed, and that it cannot be undone."""
    return _delete_page(_lecture_or_404(lecture_id))


@bp.post("/lectures/<lecture_id>/delete")
def delete(lecture_id):
    """Deletes a lecture: stops its analysis if it is running, then removes its videos, results and status."""
    if not _from_this_site():
        abort(403)
    lecture = _lecture_or_404(lecture_id)
    models = _models()
    name = lecture.context().get("name") or lecture.id
    if not models["jobs"].stop(lecture.id):
        return _delete_page(lecture, "The analysis of this lecture could not be stopped. Wait a minute and try again.",
                            409)
    left = models["lectures"].delete(lecture)
    models["jobs"].forget(lecture.id)
    if left:
        names = ", ".join(path.name for path in left)
        return _delete_page(lecture, f"Some files could not be deleted ({names}), probably because another program "
                                     "has them open, such as a video player. Close it and press Delete again.", 409)
    return redirect(url_for("lectures.index", deleted=name))


@bp.get("/lectures/<lecture_id>/status")
def status(lecture_id):
    lecture = _lecture_or_404(lecture_id)
    status = _models()["jobs"].status(lecture.id) or {"state": "none"}
    return jsonify({**status, "has_report": lecture.has_report(),
                    "report_url": url_for("lectures.report", lecture_id=lecture.id)})


@bp.get("/lectures/<lecture_id>")
def report(lecture_id):
    lecture = _lecture_or_404(lecture_id)
    if not lecture.has_report():
        return redirect(url_for("lectures.progress", lecture_id=lecture.id))
    return render_template("report.html", **build_report(lecture))


@bp.get("/lectures/<lecture_id>/charts/<name>")
def chart(lecture_id, name):
    lecture = _lecture_or_404(lecture_id)
    if name not in CHARTS:   # only the known chart files, never an arbitrary path
        abort(404)
    path = chart_path(lecture, name)
    if not path.exists():
        ensure_charts(lecture)   # opened directly, before the report page drew it
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@bp.app_errorhandler(404)
def not_found(error):
    return render_template("error.html", message="That page or lecture does not exist."), 404


@bp.app_errorhandler(403)
def forbidden(error):
    return render_template("error.html", title="Not allowed",
                           message="This request did not come from TeachLens itself, so nothing was changed."), 403
