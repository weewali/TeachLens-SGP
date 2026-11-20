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
2. `scripts/report_lesson.py --name <id>`: adds the "during each activity" comparisons, writes `tables/`, the
   rule-based points, and always asks Claude to rewrite them (src/reporting/feedback.py, llm_report.py). If the
   rewrite fails, the rule-based points are shown and the page says why.

The model files must be in `models/` (pose, detection, face, hands; see AGENTS.md); nothing is ever downloaded:
the job sets `YOLO_AUTOINSTALL=False` and `YOLO_OFFLINE=True`.

## Deployment settings (set by whoever runs the server, never shown to instructors)

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | The organisation's key for the AI rewrite, which always runs: about 4 US cents per lecture with the default model. It sends only `facts.json` (numbers; never video, slides or names). Without a key the rule-based points are used and the report says so. |
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
| Model | `web/models/charts.py` | Instructor Movement and Student Seats heatmaps over a frame of each video, with every head pixelated (also people the analysis did not track, found with the local pose model) |
| View | `web/views/templates/`, `static/` | Pages: lecture list, upload form, progress, report (Feedback, Charts, Moments, Key terms), delete confirmation |
| Controller | `web/controllers/lectures.py` | Routes; refuses a delete sent from another website |

Tests: `tests/unit/test_web_app.py` (made-up lesson from `tests/fixtures/demo_lesson.py`, fake analysis scripts).
