# TeachLens: handoff notes for Claude

First written 2026-11-11; rewritten 2026-11-20 after the switch to System A (user's laptop, Windows). Lets a new Claude
session continue without re-deriving anything. **Verify against the actual files before relying on any claim here.**
Update it at the end of every work session.

## What this project is
TeachLens (senior graduation project, UAEU, Project ID 11): analyse a recorded lecture (two static cameras: one on the
instructor, one on the students) and give the instructor an evidence-based report on observable behaviour. Advisory.
It reports observable behaviour, never inner states (attention, engagement, understanding, sleep), and associations,
not causes. Report format the user wants: `positive points:` / `negative points:`, about 3 each, never padded.

## The architecture since 2026-11-20 (user decision: "System A" processing + "System B" report and web app)
Branch `pipeline_creation` (this user's). The user ordered, row by row: code layout, input, camera alignment, frames,
pose model, phones/books, extra models, telling people apart, instructor and student behaviours, analysis and outputs
from **System A** (the teammate's pipeline, branch `scripting-frame-extraction`); plus from **System B** (this
branch's old pipeline): CSV tables, facts.json, the feedback report, review sheets, the web app design; **Claude
always rewrites** the report (not optional); privacy = anonymous track IDs + blurred faces; tests = System A's plus
adapted B tests; accuracy checks = A's + B's review sheets. Everything else of System B was removed.

Pipeline (one lecture):
1. `scripts/analyze_lesson.py` (System A, `src/pipeline.py analyze_lesson`): line up both videos by recording-start
   time (ffprobe metadata, about +/-1 s; `--student-start/--instructor-start` overrides), analyse each camera at
   1 frame/s with YOLO26n-pose @1280 + ByteTrack (+ stitching), MediaPipe face landmarks (5/s) and hand gestures,
   the YOLO26n phone/laptop/book detector @1920; rules for Lists A/B/C and facial actions (docs/list_*.md); link the
   cameras; fusion + lag analysis (5 s windows, lags 5-30 s, circular-shift test, Benjamini-Hochberg over 9
   hypotheses + exploratory pairs, comparisons with HAC, peri-event); `facts.json` (src/reporting/facts.py, schema 1).
   Output: `outputs/lessons/<YYYY-MM-DD_HHMM>_<name>/` (docs/outputs_guide.md).
2. `scripts/report_lesson.py` (new 2026-11-20, `src/reporting/feedback.py`): adds "during each activity" comparisons
   (`src/analysis/activity_comparison.py`: teacher feature >= 0.5 of a window vs other windows, HAC p, Bonferroni,
   >= 5 windows each side) to facts.json as kind `activity_comparison` (idempotent, ids renumbered); writes
   `tables/events.csv`, `tables/activity_comparisons.csv`, rule-based points (`final_feedback_report_rules.txt`, spoken
   as "you"); then ALWAYS calls Claude (`src/reporting/llm_report.py`, moved from `main code/llm_report.py`, prompt
   version 4, same checks: numbers from cited facts, direction, no causal/inner-state words, hedges, "not verified"
   for unchecked detections via KIND_INDICATORS -> indicator_validation keys; one retry; cache by hash). On any
   failure (no key, network, refusal, failed checks) the rules text is kept and `report_status.json` records
   `{"source": "rules", "reason": ...}` (user: "tell me if an error occurred with why"); the web app shows the reason.
3. Web app (`web/`, Flask MVC, design unchanged): job runs 1 then 2; stages 1-7 parsed from System A's log lines
   (`STAGE_PATTERNS` in web/models/job.py) and report_lesson's `[6/7]`/`[7/7]`. Storage under `outputs/` (ignored):
   `uploads/<id>/`, `lessons/<stamp>_<id>/`, `web_jobs/<id>.json`, `validation/<id>/`. Report tabs: Feedback (with an
   "AI rewrite was not used ... Reason:" banner when it failed) | Charts | Moments | Key terms. Charts:
   `stat_charts.py` (timeline of S_on_task per 5 s window + activity strip explaining/board/not_facing/out_of_view +
   moment lanes hand_raise/phone_use/dozing; comparison of On task / Phone / Head down per activity with 95% HAC
   ranges) and `charts.py` (Instructor Movement from teacher hip_center positions; Student Seats from summary.json
   seats, % facing the front = on_target/(on_target+off_target)). Both heatmaps sit on a frame of the uploaded video
   with every head pixelated: position boxes of the first 5 samples, each seat's median box, plus people found by
   the local pose model on that frame (catches untracked students in the instructor view); a box touching the
   bottom edge is blurred whole; any failure = no background, never the raw frame.
4. `scripts/review_sheets.py make|score` (`src/reporting/review_sheets.py`, moved from validate_indicators.py):
   frames from review.mp4 + sheet.csv to fill in; score -> precision with Wilson interval per indicator ->
   facts.json indicator_validation updated (then re-run report_lesson.py).

Removed 2026-11-20 (recoverable from git, last in commit 352d13a): `main code/` (Main Code.py, llm_report.py,
validate_indicators.py), System B's seats/2 s sampling/own rules, `docs/main_code_events.md`, the "links over time"
reporting (`src/reporting/lag_facts.py`, `llm_text.py`, `scripts/report.py`, `configs/report_vocabulary.yaml`) and
their tests/fixtures, `tests/fixtures/demo_lecture.py`. The old prototype folder `Downloaded from prototype/` was
deleted the same day (user request). `classroom_project/` (B's data: the two real lectures' videos and old results,
Video10min outputs, SGP1 report, calibration) was KEPT; the web app no longer reads it.

## Team files and updates
System A's files are byte-identical copies of `origin/scripting-frame-extraction`, last updated to **b4583ff**
(2026-11-20: bc092bd brought 41 files: one-command lesson pipeline, output layout, facts.json, desk posture,
hypotheses, pooling; b4583ff only docs: AGENTS.md sections 16-18 and docs/llm_integration.md / project_structure.md).
**AGENTS.md section 17 is the teammate's plan for our LLM and report stages**; still missing on our side (2026-11-20):
a third checked list `suggestions` (booklet codes T-STAND/T-WALK/T-READ/T-WRITE/T-GEST/T-ASK), their KIND_INDICATORS
mapping for instructor_behaviour_share (instructor_seated, instructor_facing, instructor_gestures; ours uses
instructor_role), `scripts/llm_report.py --lesson/--facts/--dry-run`, use of `src/utils/output_layout.find_lesson()`,
new file names in LESSON_FILES + docs/outputs_guide.md, an `llm:` config section, a standalone report.html renderer.
The team deleted its 95-image labelled dataset (`data/student_instructor_imported/`) in bc092bd; the local copy is
still here (a safety check blocked the folder removal; the user may delete it). To check for later team updates:
fetch, `git diff --name-only bc092bd origin/scripting-frame-extraction`, check each local copy against bc092bd
first, then `git restore --source=origin/scripting-frame-extraction --worktree -- <files>`. Their `.gitignore`
differs (ours is merged by hand). AGENTS.md is the teammate's handoff (System A details).

## Installed on this laptop (with the user's OK, 2026-11-20)
mediapipe 1.2.2 with `--no-deps` + absl-py, flatbuffers, sounddevice (team rule: never reinstall it with deps);
`models/face/face_landmarker.task` (3.8 MB, Google), `models/hands/gesture_recognizer.task` (8.4 MB, Google),
`models/detection/yolo26n.pt` (5.5 MB, Ultralytics v8.4.0); `models/pose/yolo26n-pose.pt` copied from the root.
**Ultralytics auto-installed `lap` (1.5 MB) during the first run without asking** (told the user); the web job now
sets `YOLO_AUTOINSTALL=False` and `YOLO_OFFLINE=True`. Earlier: ultralytics 8.4.215, torch CPU, opencv 5.0,
statsmodels, jinja2, anthropic, Flask 3.1.3, FFmpeg 9.0.2. Root weights yolo26m-pose.pt / yolo26s.pt are B leftovers.

## Measured
- 30 s of the real "Mixed gender classes" lecture: 1.2 min (student camera ~56 s, instructor ~10 s), i.e. about
  2.2 min per lecture minute on this laptop now (the teammate measured 2.5-9.5 depending on the power mode). A 60-min
  lecture: roughly 2-2.5 h (or much more on battery).
- Full "Mixed gender classes" lecture (10.4 min) through the web job (2026-11-20, no API key): done in 16.4 min (student
  camera ~13 min), 124 windows, 0 fusion findings, 10 facts (2 weak activity comparisons), AI fallback with reason.
  Lesson `outputs/lessons/2026-11-20_1714_Mixed_gender_classes`, upload linked (hard links) in
  `outputs/uploads/Mixed_gender_classes/`. To check by eye: 30 hand raises for 6 students; 8 seats for 6 students.
- Tests: `python -m unittest discover -s tests/unit` -> **210 tests** (170 System A, 10 `test_report_stage.py`,
  30 `test_web_app.py`), no network, no video.

## Working agreements with this user (follow these)
- **Ask before downloading anything** and **before anything that spends money or sends data out**. The AI rewrite in
  the web app always runs by the user's decision, but only with a key set by the operator; never call Claude from
  Claude's own shell without asking (~4 cents/lecture).
- **Never ask for or accept an API key in chat.**
- Short, plain answers; state limitations honestly; say when something is unverified.
- Do not commit or push unless asked. Do not delete the user's files without being asked.
- **After every change, update `SGP2 objectives/`** and, if the pipeline changed, STAGES in
  `scripts/make_pipeline_diagram.py` + `python scripts/make_pipeline_diagram.py` (look at the PNG).
- Never press Delete on the user's real lectures while testing: only made-up lectures in temporary folders.
- On this laptop `git` is not on PATH: `%LOCALAPPDATA%\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe`.
- Git history: on 2026-11-20 the then-unpushed commits were rebuilt twice (without unblurred real-lecture images; then
  with the GitHub noreply email, because GitHub blocks pushes that expose the user's private email, GH007) and pushed:
  origin/pipeline_creation = 083c440, 73d6417, b7de6c5, 90befe6, 6c845ac (+ later). This repo's `user.email` is now the
  noreply address. `.gitignore` keeps outputs/, models/, videos and report charts local. The old Video10min frames
  stay tracked and on GitHub (user's choice).

## PDF export (2026-11-20, user request)
"Download PDF" on the report page -> `GET /lectures/<id>/report.pdf` (web/models/pdf.py): report.html rendered with
`pdf=True` (CSS, report.js, logo and charts embedded as data URIs; `.pdf` class shows every tab; Moments = a summary
per kind with seats + the first 40), printed by hidden Edge/Chrome (`--headless=new --print-to-pdf`, temp profile,
`--virtual-time-budget` so the SVG charts draw), saved as `<lesson>/report.pdf`, rebuilt only when the report files,
context or chart style changed. No browser -> 503 page suggesting Ctrl+P "Save as PDF". Setting
`TEACHLENS_PDF_BROWSER`. Real lecture: 7 pages, 0.9 MB, about 3 s. The Read tool cannot show PDF pages here
(no poppler); check the print HTML with a headless Edge screenshot instead.

## Web app details still valid (from the earlier sessions)
New-analysis form: Lecture (title required) | Videos | Additional context (optional: topic, slides) | Activities in
class (optional: buttons Quiz or test / Group or pair work / Video / Break, one "Duration [..] minutes" box each,
1-180, total <= 180, <= 10 rows; then "Additional information:"). Activities and context are stored, shown in the
report, NOT used by the analysis (a duration alone cannot say which minutes; options explained, not decided). Delete
flow (confirmation, stops a running job with its process tree, refuses cross-site POSTs, reports files held open).
UAEU look (logo, colours, Inter/Roboto with fallbacks, white background only). Charts titled "Instructor Movement"
and "Student Seats", no captions (alt text only). The lead sentence promises "recommendations for future lessons";
the report has positive/negative points only (raised with the user, not decided).

## Open, in order
1. Re-run the user's real lectures through the web app (Mixed gender classes was started 2026-11-20 as a check;
   How_much_..._2 has no videos on disk any more).
2. Human labelling with `scripts/review_sheets.py`, then re-run the report.
3. The first real Claude call (needs the user's key on the server side).
4. Room file for the user's classroom (`--local-config`, board/screen/desk areas) so board and desk are measured.
5. Statistical power: with ~112 pairs a single fusion link can never pass BH; the "during each activity" comparisons
   (Bonferroni over the comparisons made) carry most feedback points. Worth raising with the teammate.
6. Update `docs/SGP2_report_context.md` / `SGP2_report_corrections.txt` (describe the old System B) and the Overleaf
   texts given to the user on 2026-11-20 (they describe two implementations; now there is one).
