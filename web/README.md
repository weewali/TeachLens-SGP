# TeachLens web app

A local website for the instructor: upload the two videos of a lecture, add context, start the analysis, follow it
stage by stage, and read the report.

```
python web/app.py            then open http://127.0.0.1:5000
```

It only listens on this computer, because the videos show real people. Since 2026-11-20 it runs the team's lesson
pipeline (System A) and then the report stage:

1. `scripts/analyze_lesson.py --student-video ... --instructor-video ... --name <id> --output-dir outputs
   --local-config <room> --no-side-by-side`: lines the cameras up by their recording times, analyses each camera
   (YOLO26n-pose + ByteTrack, MediaPipe face and hand models, the phone/book detector), links the cameras, runs the
   fusion and lag analysis, and writes `facts.json` (docs/lesson_pipeline.md).
2. `scripts/report_lesson.py --name <id>`: adds the "during each activity" comparisons, writes `tables/`, and has
   Claude write the positive and negative points from facts.json (src/reporting/feedback.py, llm_report.py). There
   is no rule-based text. The points are plain teaching advice: what went well and what to keep doing, and for
   each negative point one idea to try next time, with no caveats about the measurements (user decision
   2026-11-21; feedback written under the earlier rules is hidden until "Try again"). Points that still fail the automatic checks after one retry are left out and the points
   that passed are shown; if no point passes or Claude cannot write them, the page says why and offers
   "Try again" (report step only).
   **Code=01** on the page means an API key problem (no key in `.env`, or the key was rejected or not allowed):
   check `.env`, restart the web app, then press "Try again". **Code=02** means a billing problem of the key's
   account (credit balance too low, spend limit reached, account on hold): add credits or raise the limit in the
   Claude Console, then press "Try again". The full reason is in the lesson's `report_status.json`.

At start-up `web/app.py` reads the private `.env` file in the project folder, so the API key is set every time, and
prints whether a key was found (never the key itself). Create `.env` yourself (Git ignores it, so it is never
uploaded) with these two lines; keys are made at platform.claude.com/settings/keys:

    ANTHROPIC_API_KEY=<your key>
    TEACHLENS_LLM_MODEL=opus

The model files must be in `models/` (pose, detection, face, hands; see AGENTS.md); nothing is ever downloaded:
the job sets `YOLO_AUTOINSTALL=False` and `YOLO_OFFLINE=True`.

## Deployment settings (set by whoever runs the server, never shown to instructors)

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | The API key Claude's feedback is charged to (about 4 US cents per lecture with opus); put it in `.env`. It sends only `facts.json` (numbers; never video, slides or names). Without a key the report has no written feedback and shows "Code=01". |
| `TEACHLENS_LLM_MODEL` | `opus` (default), `sonnet` or `haiku` |
| `TEACHLENS_ROOM_CONFIG` | Room file with the board, screen and desk areas (`configs/*.yaml`); `none` (default) = those are not measured |
| `TEACHLENS_OUTPUTS` | Output root (default `outputs/`) |
| `TEACHLENS_FFMPEG_DIR` | Folder with ffmpeg / ffprobe if they are not on PATH |
| `TEACHLENS_PDF_BROWSER` | Browser that prints the report to PDF (default: Microsoft Edge or Google Chrome where installed) |

## Where things are stored (all under `outputs/`, ignored by Git)

| Path | What |
|---|---|
| `uploads/<id>/` | The two videos, optional slides, `context.json` (title, topic, activities in class, additional information) |
| `lessons/<YYYY-MM-DD_HHMM>_<id>/` | The lesson pipeline's folder plus the report: `final_feedback_report.txt`, `report_status.json`, `llm_output.json`, `tables/`, `web_charts/` |
| `web_jobs/<id>.json` | The analysis status shown on the progress page |
| `validation/<id>/` | Review sheets (`scripts/review_sheets.py make/score`) |

Deleting a lecture removes exactly these.

## MVC

| Part | File | What it does |
|---|---|---|
| Model | `web/models/lecture.py` | Where a lecture's uploads and lesson folders are; lists, creates and deletes lectures; checks the upload form (title, videos, activities in class) |
| Model | `web/models/job.py` | Runs the two commands in the background, follows their output to show stages 1-7, records the AI result (or why it was not used), can stop a running analysis with every program it started |
| Model | `web/models/report.py` | Gathers the report page's data from the lesson folder (no video is reprocessed) |
| Model | `web/models/stat_charts.py` | The lecture timeline (students on task per 5-second window, your activity, moments) and the comparison chart, from `fusion/windows.csv` and `src/analysis/activity_comparison.py` |
| Model | `web/models/pdf.py` | "Download PDF": a self-contained print version of the report, printed by a hidden local browser to `report.pdf` in the lesson folder (made again only when the report changes) |
| Model | `web/models/charts.py` | Instructor Movement and Student Seats heatmaps over a frame of each video, with every student's head pixelated (also people the analysis did not track, found with the local pose model); the instructor is not blurred; seats used for under a minute are not drawn |
| View | `web/views/templates/`, `static/` | Pages: lecture list, upload form, progress, report (Feedback, Charts, Moments, Key terms), delete confirmation |
| Controller | `web/controllers/lectures.py` | Routes; refuses a delete sent from another website |

Tests: `tests/unit/test_web_app.py` (made-up lesson from `tests/fixtures/demo_lesson.py`, fake analysis scripts).
