# TeachLens web app

A local website for the instructor: upload the two videos of a lecture, add context, start the analysis, follow it
stage by stage, and read the report.

```
python web/app.py            then open http://127.0.0.1:5000
```

It only listens on this computer, because the videos show real people. FFmpeg must be on PATH (or set
`TEACHLENS_FFMPEG_DIR` to its folder) and the three YOLO26 weight files must be in the repository root, as for
`main code/Main Code.py`.

## Deployment settings (set by whoever runs the server, never shown to instructors)

| Variable | Meaning |
|---|---|
| `TEACHLENS_LLM` | `auto` (default): rewrite the feedback and the links over time in plain language with Claude when an Anthropic API key is set for the server; `on`: always try; `off`: never |
| `ANTHROPIC_API_KEY` | The organisation's key. Each lecture gets up to two rewrites (the feedback, and the links over time), each about 4 US cents with the default model; they send only the measured numbers, never video, slides or names |
| `TEACHLENS_LLM_MODEL` | `opus` (default), `sonnet` or `haiku` |
| `TEACHLENS_FFMPEG_DIR` | Folder with ffmpeg and ffprobe, if they are not on PATH |

Without a key, or if the rewrite fails its checks, instructors get the standard feedback; nothing on their side
changes except a small note on reports whose text was written with AI.

## MVC layout

| Part | Folder | What it does |
|---|---|---|
| Model | `web/models/lecture.py` | Where a lecture's videos, context and results are stored; lists, creates and deletes lectures (a delete removes the uploads, `sessions/<id>/`, the job status, the two event JSONL files and any review sheets; shared calibration and pooled results stay) |
| Model | `web/models/job.py` | Runs `main code/Main Code.py` (and the plain-language rewrite when the deployment enables it) in the background; records the stage reached (1-7, 8 = rewrite) and the program's output; can stop a running analysis together with the programs it started (FFmpeg), so its files can be deleted on Windows |
| Model | `web/models/report.py` | Gathers the report page's data from the saved results (no video is reprocessed) |
| Model | `web/models/report.py` (links) | The "Links over time" tab: the team's fusion layer results for this lecture (`reports/fusion.json`) as positive and negative points (`reports/links/report_points.json`, rule-based, or rewritten by `scripts/report.py --you` when the AI step is enabled), or why there are none (an older analysis, or a lecture the fusion layer could not analyse) |
| Model | `web/models/charts.py` | Instructor position heatmap and heads up by seat (a box where each student usually sat, coloured by the share of readable time with the head up), drawn from the saved tables |
| Model | `web/models/stat_charts.py` | Data for the two statistical charts: the lecture timeline (heads up per 10-second part, instructor activity, moments; drawn as SVG by `report.js` in the page's theme, with a table view) and the activity comparison (averages with a 95% HAC range, HTML bars) |
| View | `web/views/templates/*.html` | Jinja2 pages: lecture list (with a Delete link per lecture), new analysis form, progress, report, delete confirmation, not found / not allowed |
| View | `web/views/static/` | CSS (the UAEU website's colour and font tokens; always a white page), the UAEU logo (`uaeu-logo.svg`, the official file from uaeu.ac.ae, used unchanged), and small scripts for live progress, the report tabs (one section at a time, kept in the address as `#charts`; printing shows every section), the moments filter and the upload message |
| Controller | `web/controllers/lectures.py` | Flask routes: `/`, `/lectures/new`, `POST /lectures`, `/lectures/<id>/progress`, `/lectures/<id>/status`, `/lectures/<id>`, `/lectures/<id>/charts/<name>`, `/lectures/<id>/delete` (GET asks, POST deletes; a POST that the browser marks as sent from another website is refused) |
| App | `web/app.py`, `web/config.py` | Builds the app from the parts above; default paths |

## Where things are stored

- Uploads: `classroom_project/videos/<id>/` (`instructor.*`, `students.*`, optional `slides.*`, `context.json`)
- Results: `classroom_project/sessions/<id>/` (written by the pipeline; charts in `reports/charts/`)
- Job status: `classroom_project/jobs/<id>.json`
- Event files: `outputs/events/<id>_instructor_view.jsonl` and `<id>_student_view.jsonl`

None of these are committed to Git.

## Not done yet

PDF export (a later stage; the report page already prints cleanly), and using the context (topic, notes, slides) in
the analysis: it is stored and shown with the report only.

## Tests and preview

```
python -m unittest discover -s tests/unit -v
python tests/fixtures/demo_lecture.py SOME_FOLDER      # a made-up lecture to preview the report page
set TEACHLENS_BASE=SOME_FOLDER && python web/app.py
```
