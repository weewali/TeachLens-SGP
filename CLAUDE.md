# TeachLens: handoff notes for Claude

First written 2026-11-11 in the original Codespace; last updated 2026-11-19 on the user's Windows laptop. Lets a new
Claude session (another profile, laptop or Codespace) continue without re-deriving anything. **Verify against the actual
files before relying on any claim here; this file can go stale.** Update it at the end of every work session.

## What this project is
TeachLens (senior graduation project, UAEU, Project ID 11): analyse a recorded lecture and give the instructor an
evidence-based report on observable instructor and student behaviour. It is advisory. It reports observable behaviour,
never inner states (attention, engagement, understanding, sleep), and it describes associations, not causes.

## Two implementations (read this before the rest)
- **System A, the team's main pipeline**, is on branch `scripting-frame-extraction` (`src/`, `scripts/analyze.py`,
  `configs/*.yaml`, `tests/unit/`, its own `AGENTS.md`): two static cameras, one recording per run, YOLO26n-pose +
  ByteTrack at 1 frame/s, role from camera context, JSONL events + JSON summary + annotated review MP4. Indicators
  ("List A"): presence, student count, position, zone, entry/exit, visibility, sitting/standing, motion. Hand raises,
  orientation and audio are out of its scope. The SGP2 report draft (chapter 3) describes this system.
  **Copied into this working tree on 2026-11-18** (user request "all of them"): every file of
  `origin/scripting-frame-extraction` at 29f5f57 that this branch lacked (227 files, about 11.5 MB, including AGENTS.md,
  System A's `src/`, `scripts/`, `configs/`, `docs/`, its tests, and a 95-image labelled dataset in
  `data/student_instructor_imported/`). Byte-identical to that commit, written with `git restore --worktree`, not staged
  or committed. The 4 files both sides had (`src/__init__.py`, `src/analysis/__init__.py`, `src/analysis/events.py`,
  `src/reporting/__init__.py`) were already identical and left alone. Files both branches track (README, .gitignore,
  requirements.txt, ...) were NOT taken from System A.
  **Updated the same day to ee5b6bb** (user: "check for new files ... fusion layer ... implement it with my files"):
  `git fetch` found 3 new team commits (List C student indicators + track stitching + camera linking; phone use from a
  phone in hand or a sustained head-down; lesson fusion + lag analysis). Their 27 changed/new files were written the
  same way (none had been edited locally; byte-identical to ee5b6bb; nothing staged). New: `src/analysis/fusion.py`,
  `lag_analysis.py`, `student_indicators.py`, `interaction_indicators.py`, `cross_camera.py`, `src/inference/hands.py`,
  `tracking.py`, `scripts/fuse_lesson.py`, `link_cameras.py`, `docs/fusion_lag_analysis.md`, `docs/list_c_students.md`,
  tests. mediapipe is NOT installed on this laptop (the team installs it with `--no-deps`, see AGENTS.md); their tests
  still pass because they do not load it. To check for later team updates: fetch, then compare
  `git diff --name-only ee5b6bb origin/scripting-frame-extraction` and check each local copy against ee5b6bb first.
- **System B, this branch** (`main code/`): since 2026-11-16 also **two cameras** (instructor + students
  of the same lecture), with its own detection, its own simpler student indicators (head-up, phone, head down and
  still, hand raise) and the LLM report. Still separate from System A; merging them is an open question.
  **Since 2026-11-19 System B runs the team's fusion layer** (user decisions, asked with three options: "map your
  detections" (no downloads; not the team's List C code, which needs mediapipe), "replace" its own lag analysis, and
  "its own tab again" in the web app). Main Code.py writes the indicators the fusion reads under the team's names and
  labels from its own rules, then calls `build_windows` / `onsets` / `analyze` on its two event files (stage 6). Details
  in "Fusion in System B" below and `docs/main_code_events.md`.
- **Camera sync decision** (team, 2026-10-26, in System A's `AGENTS.md`; System B follows it): recording-start time from
  each video's metadata (ffprobe), good to about 1 s. An audio-sync tool was built and removed as too complicated; do
  not reintroduce audio or clock-decoding sync unless asked. System B adds a manual `--offset` override.
- SGP2 report help: `docs/SGP2_report_context.md` (context for another model; describes System B as of 2026-11-16
  *before* the two-camera rewrite, so update it) and `docs/SGP2_report_corrections.txt` (fixes to the chapter 3 draft).

## Team pipeline and this user's part (decided 2026-11-16)
Two raw videos (instructor, students) -> the teammate's model (System A) writes one JSONL per camera -> the team's
fusion layer (`scripts/fuse_lesson.py`, built 2026-11-18, docs/fusion_lag_analysis.md) writes
`outputs/reports/<name>.fusion.json` -> **this user's part: the LLM that turns the lagged-analysis numbers into
plain-language positive/negative points** -> later a Jinja2 page in a web app (charts, lecture info, interactive), with
PDF export at a later stage. The user said not to worry about the JSONL files.
Placement follows System A's `docs/project_structure.md` (`src/`, `scripts/`, `configs/`, `tests/unit` with unittest).
- `src/reporting/lag_facts.py`: fusion output -> facts, no model. **Connected to the real fusion format on 2026-11-18**:
  `load_results` recognises `<name>.fusion.json` (`from_fusion`: each link -> a `lagged_link` row with r, p_shift, q;
  each tested peri-event -> an `event_change` row "after the instructor turned to the class (N times), X rose by
  ... percentage points over the next 20 s vs the 10 s before"; untested/too-rare actions -> `analysis.too_rare_to_test`;
  the fusion's limitations kept) and its `.fusion_links.csv` (aliases). Strength: with a q-value, suggestive = q <= alpha
  (exactly the fusion's own findings; Benjamini-Hochberg, less strict than the Bonferroni rule used before), weak = p <
  alpha < q; without q (old formats) the Bonferroni rule still applies. One fact per pair at its strongest lag; weak links
  capped at 6. Behaviours that are neither good nor bad news (`favourable_when_higher: null`, e.g. talking, smiling) are
  counted (`neutral_links_found`), not turned into points. A link as strong in reverse gets a "shared trend" note; an
  event change on < 5 starts gets a "based on only N times" note. The report refuses to run if the vocabulary's
  direction disagrees with the fusion's `desirable`. Facts schema 2; window length comes from the fusion (5 s).
- `configs/report_vocabulary.yaml`: plain names for indicator IDs (fusion T_*/S_* features, the fusion's teacher events
  under `instructor_events`, and System B's IDs) and whether a rise is good news (true/false/null). Unknown IDs are left
  out and listed as `not_interpreted`, never described with guessed words. A test checks it covers every fusion
  feature/event with the same direction and that every phrase passes the LLM word checks with no number in it.
  Section `measured_by: main_code` (2026-11-19) rewords features System B measures more narrowly (S_on_target "heads up
  and facing the front", S_phone "a phone seen near them", S_sleep "head down and still for half a minute or more",
  ...); the loader only allows phrases there, never directions. `build_facts` picks it from the fusion doc's
  `measured_by`. Rule-based sentences take `subject="you"` (web report; `scripts/report.py --you`).
- **Statistical power of the fusion (found 2026-11-18, team's design, not changed):** with all indicators on it tests
  about 112 teacher x student pairs; the shift test's smallest p is 1/(rotations + 1) (150 windows: 1/138) or 1/501
  (500 permutations, longer lessons). After Benjamini-Hochberg over 112 tests a single link can never reach q <= 0.05
  (even a 50-min lecture: 112/501 = 0.22); about 5 links at the floor at once are needed. So most real reports will have
  no "unlikely to be chance" points, only "could easily be chance" ones. Raising `fusion.permutations` helps only for
  long lessons (short ones are already exact). Worth raising with the teammate; not raised yet. System B tests far
  fewer pairs (at most 4 teacher x 6 student = 24), so there a single link can pass in a lecture of about 43 minutes or
  more (24/501 < 0.05), still not in a short one.
- `tests/fixtures/fusion_example.json` (made-up window table with planted links, analysed by the team's real code;
  regenerate with `python tests/fixtures/make_fusion_example.py`, about 6 s; uses 10 of the 24 features so the example
  has findings).
- `src/reporting/llm_text.py`: facts (kept in memory; saved only with `--keep-facts`) -> Claude -> checked points ->
  `report_points.json` / `.txt`. Falls back to rule-based points on any API problem or after 2 failed attempts.
  Prompt version `lag-3` (explains both fact kinds and the window length; speaks to the instructor as "you").
- `scripts/report.py` (CLI; `--lag-results outputs/reports/<name>.fusion.json`, report folder `outputs/reports/<name>/`),
  `tests/unit/test_lag_facts.py`, `tests/unit/test_llm_text.py` (37 tests, no network; one runs the team's `analyze()`
  fresh, so a change in their output format fails a test), `tests/fixtures/lag_results_example.json` (old format,
  made-up numbers, labelled as such).
- **Fusion in System B (2026-11-19).** Main Code.py writes, besides its own indicators, the team's: instructor
  `facing_direction` (frontal -> toward_class; face hidden -> away_from_class; else the nose side -> sideways_left/
  right; all unknown when `camera_faces_class` is false), `board_interaction` (board_work -> board, else none; all
  unknown without a board region), `raised_hand` none/one/both (new cue `hands_up`), `position.hip_center_in_shot` (new
  cues `hip_x`, `hip_y`) and `camera_shot: 0`; per seat `head_down`, `orientation_proxy` (head up + face toward the
  student camera = on_target, else off_target; down = looking_down), `alertness` (head_down_still -> possible_sleep,
  head up -> awake), `device_use` phone/none (was phone only) and `hand_raise_episode`. Every state covers all the time
  the person was seen, because the fusion divides by known time. New cues: `hands_up`, `nose_dx`, `hip_x`, `hip_y` in
  PEOPLE_COLUMNS; `with_cue_columns` lets older frame_detections.csv still load. Stage 6 (`run_fusion`) reads the two
  files on System B's shared clock (start_time minus the cut, epoch from a recording start or a stand-in 2000-01-01
  that is never shown), with only `FUSION_FEATURES` (T_facing_class, T_board_or_screen, T_stationary, T_path,
  S_on_target, S_off_target, S_phone, S_head_down, S_sleep, S_hand_raises; facing/board dropped when not measured;
  S_looking_down left out because it equals S_head_down here) and events turn_to_class, board_or_screen_start,
  raised_hand_start. Settings are the team's `fusion:` section. Outputs: `reports/fusion.json` (team format +
  `measured_by: main_code`, `not_measured`, `clock`, `warnings` (e.g. statsmodels rank-deficient models: facing and
  board are opposites), own limitations), `tables/fusion_windows.csv`, `fusion_links.csv`, and rule-based points in
  `reports/links/report_points.json/.txt` (subject "you"). A lecture it cannot analyse gets `{"error": ...}` and stale
  files are removed; the rest of the report still runs. `analysis_details.txt` section 3 now lists the fusion results
  (incl. the smallest possible p and best reachable q). `evaluation_result.json` has `fusion` instead of
  `lag_results`/`lagged_model`. Removed: `analyze_lags`, `shifted`, `describe_lag`, PRESPECIFIED and the other old lag
  constants, and `pool_sessions` (pooled lag analysis used the same old code; `classroom_project/pooled/` is no longer
  written, old files there are untouched). Tests: `tests/unit/test_main_code_fusion.py` (planted 10 s link found only on
  the shared clock with a 7 s cut difference, calibration-dependent features, no recording times, error path, report
  lines/points) and new cases in `test_main_code_events.py`.
- `src/__init__.py` and `src/reporting/__init__.py` copy System A's exact contents so a later merge does not conflict.
- Run: `python -m unittest discover -s tests/unit -v`; `python scripts/report.py --lag-results FILE --dry-run`.
- The user said to "skip" the question of having the LLM state causes: the current rule stands (links in time, hedged
  possible explanations, causal wording rejected by the checks).

## Web app (built 2026-11-17, MVC with Flask + Jinja2, user request)
`python web/app.py` -> http://127.0.0.1:5000 (local only). Models `web/models/` (lecture storage, background job that
runs `Main Code.py` and records stages 1-7 (+8 = optional Claude rewrite), report data, heatmaps), views
`web/views/templates` + `static`, controller `web/controllers/lectures.py`. Details: `web/README.md`. Flask 3.1.3
installed with the user's OK. The Claude rewrite is a deployment setting, not an instructor option (user decision
2026-11-17: "like the real product"): `TEACHLENS_LLM=auto|on|off`, auto = on when ANTHROPIC_API_KEY is set for the
server; instructors only see a small "written with AI" note. Context (topic, notes, slides) is stored and shown but NOT used by the analysis yet; PDF
export is a later stage. Tested with Flask's test client on a made-up lecture (`tests/fixtures/demo_lecture.py`) and
fake pipeline scripts, plus a real server start; never with real lecture videos yet.
Delete (2026-11-18, user request "add option to delete the report or work in progress"): a Delete link on each row of
the lecture list -> confirmation page -> POST removes the uploads, `sessions/<id>/`, `jobs/<id>.json`, the two event
JSONL files and `validation/<id>/` (calibration and any old `pooled/` folder stay; pooling is no longer run). A
running analysis is stopped first with its whole process tree (`taskkill /T` on Windows; FFmpeg would otherwise keep
files open). Files another program holds open are listed with a "close it and press Delete again" message. A POST the
browser marks as cross-site (Origin / Sec-Fetch-Site) is refused with 403. Tests in `DeleteTests` (test_web_app.py).
The "... was deleted." message fades out after 5 s (`data-autohide="5000"` in index.html, `static/lectures.js`, user
request), and `?deleted=` is removed from the address at once, so a reload does not show it again.
Never press Delete on the user's real lectures while testing: test only with made-up lectures in temporary folders.
New-analysis form (2026-11-19, user request: the instructor only uploads videos, does not record them): hints rewritten
in plain, ethical wording (no "people's names", no tripods); labels "Lecture title" (required since 2026-11-19, user
request: `required` in the form, "Enter a lecture title." from `LectureRepository.create` before any file is saved),
"Video of the instructor", "Video of the students", "Additional information:"; lead sentence "...highlights the strengths of your lecture and recommendations for future lessons"
(user's wording; the report itself has positive/negative points, no recommendations yet). The user removed the
"Before you start" ethics box and the original-files/messaging-apps hint. Do not claim the videos stay private: the
instructor heatmap is drawn on a real video frame. The Student Seats chart's frame has every head pixelated since
2026-11-19 (user request; `head_areas` / `pixelate` / `_student_background` in `web/models/charts.py`: heads of
everyone detected in the first 5 samples, any role, plus each seat's usual head spot, from box top to shoulder line,
1.3 shoulder widths wide, padded 20%; a failure gives no background, never the raw frame). Only the chart is changed,
never the analysis frames, which stay unblurred on disk. `CHART_STYLE` + `charts/style.txt` redraw older saved charts;
one that cannot be redrawn is deleted. A person missed in all those samples and in no seat would stay visible. Section titles are navy headings with a red bar
(`.form legend` in style.css). `TEMPLATES_AUTO_RELOAD = True` in `create_app` so template edits show
on a refresh; Python code changes still need a server restart.
Planned activities (2026-11-19, user chose "point 2" of: optional / quick buttons / ask after the report / automatic
suggestions / reuse): "Activities in class" (was "Activities that were not teaching"; hint says "Add any part of the
lecture that was not teaching ...") in the optional section, buttons Quiz or test, Group or pair
work, Video, Break; each adds a row with ONE box, "Duration [..] minutes" (user request 2026-11-19: duration only, no
"from/to" times; the first version had from/to minutes). `parse_activities(types, durations)` in
`web/models/lecture.py` checks rows before any video is saved (type, whole minutes 1-180, total <= 180, <= 10 rows;
empty rows skipped) and `context.json` stores `activities: [{type, minutes}]` in the order given; the report's "Context
you gave" lists them ("Quiz or test (15 minutes)"). No-JS fallback: one row with a select. The notes label is
"Additional information:" (placeholder "e.g. first lecture after the midterm"). Form order (user requests
2026-11-19): Lecture | Videos | Additional context (optional: topic, slides) | Activities in class (optional; its own
section with the same red-bar heading as the others: activity buttons and rows, then "Additional information:" last) |
Start analysis. **Not used by the analysis yet** (the user's lectures are 55-75 min of teaching).
Consequence of duration-only (told to the user): the analysis cannot know WHICH minutes to leave out. The earlier plan
(convert to the shared clock, mark student states unknown so the fusion skips them, shade the timeline) needs the
times; with a duration alone the options are a note beside the affected figures, or finding a stretch of matching
length automatically and letting the instructor confirm it (point 4). Not decided. Points 4 (automatic suggestions
from class-wide patterns, confirmed by the instructor) and 5 ("same as last time") were explained, not built.

Report page changes (2026-11-17, user request): the five "fact" boxes at the top are gone (length is in the date line).
Later the same day the user also removed the instructor, student and linkage sections from the bottom of the page, and
then the suggestions and limitations too (after asking why they never change: the suggestions are 4 fixed rule
sentences and 8 of 10 limitations are fixed method text). Everything removed stays in `analysis_details.txt`; the web
app no longer reads that file. The feedback note no longer points to "the report's limitations" (saved reports patched).
The last tab is "Key terms" (the glossary from facts.json, shown open; the data-only terms "window", "observable
students" and "observable student time" are left out in `web/models/report.py` DATA_TERMS). Negative points use "✗"
instead of "!". The fixed note under the feedback points (FEEDBACK_NOTE in Main Code.py: definitions, the time base,
"automatic detections that no person has checked yet") is no longer shown on the page (user request 2026-11-19; it
stays in final_feedback_report.txt and facts.json), so the page itself no longer says the detections are unverified.
The two heatmaps in the Charts tab have a title above them ("Instructor Movement",
"Student Seats"; user's wording, the second asked as "Students Seats") and no caption under them (user request 2026-11-19); the old caption text is now only the image's
alt text (`CHARTS` in `web/models/charts.py`), and the head-up caveat stays in Key terms. The report is now tabs instead of one long page (user request): Feedback (with the lecture context) |
Charts | Moments (count) | Key terms. The old "Links over
time" tab (System B's own lag analysis in words) was removed on 2026-11-17 at the user's request: on one short lecture
its "unlikely to be chance" links rested on 3 hand raises and 1 phone moment. It came back on 2026-11-19 (user chose
"its own tab again", from the team's fusion layer) and **was removed again the same evening** (user: "remove links over
time tab. idk what was the purpose of this"; the real lectures only showed "Run its analysis again to add it"). Tabs
now: Feedback | Charts | Moments | Key terms; `links()` was deleted from `web/models/report.py`. The fusion still runs
(stage 6) and its files are still written (`reports/fusion.json`, `reports/links/report_points.json`,
analysis_details.txt section 3). Stage 8 (AI on) also runs `scripts/report.py --lag-results reports/fusion.json --out-dir reports/links --you`
(`JobRunner.links_command`, config `LINKS_REPORT`): up to two Claude calls per lecture (~4 cents each, opus); with the
tab gone nobody sees the second one's text (asked the user whether to switch it off; not answered yet).
`report.js` renames each section id to
`panel-<name>` so `#charts` in the address never makes the browser jump; without scripts the tabs are plain links and
every section shows; printing shows every section.
UAEU design (2026-11-17, user request): header = red brand line, the official UAEU logo (`web/views/static/uaeu-logo.svg`,
fetched from uaeu.ac.ae/assets/images/uaeu-logo.svg with the user's OK; the site blocks scripts, so it was read with
headless Edge), a divider, then "TeachLens". The UAEU Brand Guidelines 2025 (uaeu.ac.ae/en/vc/uod/brand/pdf/
uaeu-brand-oct-2025.pdf) forbid merging another name into the logo or altering it, hence the divider. Colours from the
guide: red #d3202e, Gray 10 #e5e6e6, white; secondary navy #00263e, tan #c2ae8b, gold #d5ad6b, grey #c8c8c8, dark grey
#484342, cream #f2f0eb. Later the same day the user gave the UAEU website's own CSS tokens, which now drive
`style.css` (kept under their own names: `--clr-primary-*`, `--clr-neutral-*`, `--uaeu-red` #c01d2a, `--uaeu-red-hover`
#961721, success/warning/danger, `--shadow-sm`, page-title size/weight, fonts Inter (headings) and Roboto (body) with
Segoe UI/Arial fallback; the web fonts are NOT bundled, which would be a download needing the user's OK). Buttons are
#ae1830 (the user's explicit choice, hover #961721); red line and active tab #c01d2a; headings navy #142E4A; page title
#232528; no footer (its advisory line was removed at the user's request, 2026-11-18). The site's green #2C973F and amber #BC760C are too light for small text on white (3.7:1), so
text uses darker steps #237a33 and #8a5a06; the site colours stay for lines and backgrounds. The user asked for a white
background (their Windows is in dark mode), so there is no dark mode: `color-scheme: light` keeps the page white.
Two statistical charts added (`web/models/stat_charts.py`
+ `report.js`): "Over the lecture" (SVG timeline: heads-up line, activity strip, moment lanes, crosshair tooltip,
table view) and "Students during each of your activities" (averages with a 95% HAC range). Activity colours are
categorical slots 1-3 (blue/orange/aqua), checked with a Python port of the dataviz validator (Node is not
installed); tokens `--viz-*` in `style.css` for light and dark. The old pipeline `timeline.png` is no longer shown.
Feedback wording (Main Code.py `build_feedback_points`) is now plain English for every lecture: "you", "heads up",
"students in view", no windows/p-values (comparison statistics are in `analysis_details.txt` section 3). The LLM prompt
uses the same words (PROMPT_VERSION 3). Tests: `tests/unit/test_feedback_wording.py` + new web tests.

Automatic seats (2026-11-17, after the user saw wrong seats on the real lecture): the old default 3x4 grid over the
whole frame put two back-row students in one cell, gave one of them a second cell for 2 frames, and turned legs under
a desk and a chair (2 weak detections each, no shoulders) into "seats" shown as 0%. Seats are now found from where
students sat (`find_seats` / `assign_students` in Main Code.py, constants `SEAT_*`; rules in `docs/main_code_events.md`);
`"seats": "grid"` in the student calibration brings the grid back. The seat heatmap draws a box where each student sat,
coloured by head-up share of the readable time. On the real lecture: 6 seats = the 6 students, 4 ghosts left out.
Tests: `tests/unit/test_seats.py` (63 tests in total after the details-parser test was removed).
The real lecture `How_much_do_you_rely_on_AI_in_your_daily_life_2` had its feedback text refreshed from its saved
tables (same output as a rerun; its analysis_details.txt is still the older version until a rerun).

## The user's objectives (in their words, roughly)
1. Better **association** analysis (instructor behaviour vs student behaviour over time). Cause comes later. Main focus
   (2026-11-16): link student phone use and sleeping-like posture to what the instructor was doing at that moment.
2. More **accuracy** (nothing has been validated against human labels yet).
3. An **LLM that turns the numbers into plain English** positive/negative points, with good reasoning.
Report format they want: `positive points:` then bullets, `negative points:` then bullets (about 3 each, fewer if the
evidence is thin; never pad).

## Working agreements with this user (follow these)
- **Ask before downloading anything** (pip packages, model weights, FFmpeg) and **before anything that spends money or
  sends data out** (Claude API calls). Say what, where from, and the size/cost, then wait for a yes.
- Approved in the Codespace: pip `scipy statsmodels anthropic`; weights `yolo11n-pose.pt`; **one** real Claude call
  (about $0.04), not made yet (no `llm_output.json` exists). Approvals do not carry over to a new machine or new files:
  the YOLO26 weights and the laptop installs still need a yes.
- **Never ask for or accept an API key in chat.** The user exports `ANTHROPIC_API_KEY` in the terminal themselves
  (a key exported in their terminal is not visible to Claude's own shell; they can instead save it to a private file
  and Claude passes it to the single call).
- The user likes **short, plain answers**, and wants limitations stated honestly. Do not overclaim. Say plainly when
  something is unverified or when an earlier statement turned out to be too generous.
- Do not commit or push unless asked. Do not delete the user's files.
- **After every change to the project, update the SGP2 report chapters in `SGP2 objectives/`** (user request
  2026-11-19, "everytime"). If the pipeline changed, edit STAGES in `scripts/make_pipeline_diagram.py` and run
  `python scripts/make_pipeline_diagram.py` to redraw `SGP2 objectives/pipeline.png` (its 8 steps match the web app's
  stages), then look at the PNG. Last full update: 2026-11-19 (test counts, fusion results of both lectures, second
  lecture's numbers, web form and report tabs, planned activities).

## Code map
The working scripts moved from `Downloaded from prototype/` to `main code/` on 2026-11-17 (user request). The folder
name contains a space; always quote paths.
- The team's ORIGINAL single-camera prototype (`Downloaded from prototype/classroom_analytics.py` + `requirements.txt`)
  was deleted from the working tree on 2026-11-20 (user: "we don't want that ... implementation because we have
  switched to System B"); not committed yet; recoverable from commit 6a035be. The user chose to KEEP the old
  Video10min outputs (frames, yolo_outputs, sessions/validation Video10min, old reports/tables/pooled) and to leave the
  GitHub history as it is (no force-push).
- `main code/Main Code.py`: the real pipeline, two cameras (renamed by the user from `classroom_analytics copy.py`;
  keep the name). Since 2026-11-17 it also writes one JSONL per camera in the team's event format
  (`outputs/events/<session>_instructor_view.jsonl`, `..._student_view.jsonl`) using `src/analysis/events.py`, an exact
  copy of System A's file. Indicators and rules: `docs/main_code_events.md`. Tests: `tests/unit/test_main_code_events.py`.
  Since 2026-11-19 stage 6 runs the team's fusion layer on those two files (`run_fusion`, `analyse_links`) and stage 7
  writes the links points (`write_link_points`); tests `tests/unit/test_main_code_fusion.py`.
- `main code/llm_report.py`: facts.json -> Claude -> checked report. `main code/validate_indicators.py`: review sheets
  (one image per camera per sampled moment) + scoring.
- Root `requirements.txt`; `CLAUDE.md` (this file). Team docs in `docs/`.
- The team's design docs `project_structure.md` and `video_indicator_catalog.md` are on System A's branch under `docs/`
  (read with `git show origin/scripting-frame-extraction:docs/<file>`). Catalog rule: never report attention /
  engagement / emotion as fact; report narrow observable proxies with denominator + confidence + limitation.
- On this laptop `git` is not on PATH; use `%LOCALAPPDATA%\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe`.

## How to run (from the repo root)
```
python "main code/Main Code.py" --instructor I.mp4 --students S.mp4 [--session lecture01] [--offset 2.5] [--overwrite]
python "main code/Main Code.py"     # every classroom_project/videos/<lecture>/ with instructor.* + students.*
python -m unittest discover -s tests/unit -v                         # 251 tests (100 System B + 151 System A), no network, no video
python "main code/llm_report.py" --session lecture01 --dry-run       # shows prompt + cost, calls nothing
python "main code/llm_report.py" --session lecture01 --model haiku   # real call: haiku ~0.1c, sonnet ~2c, opus ~4c
python "main code/validate_indicators.py" make-sheets --session lecture01
python "main code/validate_indicators.py" score --session lecture01  # after a person fills the sheets
```
`--offset` = seconds the student camera started after the instructor camera (negative if before); only needed when the
metadata has no recording time. Not in git, so recreate on a new machine (each download needs the user's OK): the
videos, weights `yolo26n-pose.pt`, `yolo26m-pose.pt` and `yolo26s.pt` in the repo root (`*.pt` is gitignored; the
script never downloads and says when a file is missing), FFmpeg + ffprobe, and `pip install -r requirements.txt`
(`ultralytics>=8.4` for YOLO26).

## Pipeline in one paragraph
Validate both videos -> align them (metadata recording time, or `--offset`, else assume a joint start with a warning)
-> FFmpeg cuts both to the shared time and standardises (25 fps, max 720p) -> one frame every **2 s** from each camera,
paired by time -> per-camera calibration (`classroom_project/config/calibration_instructor.json`: teacher zone, board
region, `camera_faces_class`; `calibration_students.json`: student region, `seats` auto/grid; optional
`calibration_<session>_<camera>.json`) -> instructor camera: `yolo26n-pose` at 640, most confident person in the teacher
zone = instructor -> student camera: `yolo26m-pose` at 960 + `yolo26s` (phones, books) at 960, automatic seats -> rule-based
states and events -> two JSONL files -> 10 s windows (comparisons, charts) -> the team's fusion layer on the JSONL
files (5 s windows, lags 5-30 s) -> rule-based report + `facts.json` + links points -> optional Claude rewrites with
automatic checks. Outputs: `classroom_project/sessions/<id>/`; review sheets: `classroom_project/validation/<id>/`.

## Decisions and why (do not undo without a reason)
- **Models** (user accepted 2026-11-16): instructor `yolo26n-pose` @640 (one large person); students `yolo26m-pose`
  @960 (many small people; m is the best accuracy per CPU time on Ultralytics' table); `yolo26s` @960 for phones (my
  choice, small objects; untested). YOLO27 was announced Sept 2026 but not downloadable; it uses the same API.
- **`PREDICT_CHUNK = 20`**: one long YOLO call grew past 3 GB on a 2-core / 8 GB Codespace and was killed.
- **Facing the class** = nose + an eye visible with the nose near the shoulder midpoint, seen by the instructor camera,
  which must sit on the students' side. "Interacting" (facing + raised hand) counts as explaining to the class.
  **Board work** = inside the calibrated board region and not facing the class (writing with the hand below the
  shoulder was never caught by the old raised-wrist rule). No board region = not measured, reported as such.
- **"Head down and still"** (the user's "sleeping"): head down or face hidden, shoulders moving < 0.15 shoulder-widths
  between samples, for >= 30 s. Reported as a resting posture that quiet reading/writing can mimic. The LLM check
  rejects "sleep/asleep/tired/distracted" so the model must use the posture wording.
- **"focused" means head up**, a posture proxy. Writing looks like head-down, so it can count as off-task.
- **PRESPECIFIED** (explanation -> phone rate, explanation -> head-down-still rate, instructor seated -> head-up) was
  retired with System B's own lag analysis on 2026-11-19: the team's fusion has no pre-registered pairs and corrects
  over every pair with Benjamini-Hochberg. The first two pairs are tested there anyway (T_facing_class -> S_phone,
  S_sleep); "seated" is not a fusion feature.
- **Comparisons** (student behaviour while the instructor did X vs other windows) use a HAC test and are labelled
  suggestive only if p < 0.05 / number of comparisons that could be made. Windows where the activity is unknown are
  left out of both sides.
- **Lag analysis** (since 2026-11-19, user decision "replace it"): the team's fusion layer, not System B's own
  code. Neighbouring windows are alike, which its circular-shift test keeps; q-values (Benjamini-Hochberg) over all
  pairs; teacher features in fewer than 5 windows and actions with fewer than 3 starts are not tested. One lecture at a
  time (no pooling).
- **`facts.json`** (schema 2) is the only thing the LLM sees: no video file names (they can name the instructor).
  Each fact lists the detections it rests on (`indicators`); `llm_report.py` demands an "unverified" caveat for them.
  Every number in the answer must come from the cited facts; no causal or inner-state words; one retry, else the
  rule-based report stays; answers cached by hash (prompt version 2).
- **Claude models**: `opus` (claude-opus-5-5, default), `sonnet`, `haiku`. Opus/Sonnet send the server-side fallback
  option; **Haiku must not**. Opus 5.5 rejects temperature and cannot disable thinking; effort `medium`.

## The old single-camera test video (retired 2026-11-16)
The user asked to stop using `Video10min.mp4` (edited footage, moving camera; it is still at
`classroom_project/video/Video10min.mp4` and its old outputs under `sessions/Video10min/` and `validation/Video10min/`,
not deleted). Lessons from it: zone-based roles failed when the camera moved (instructor gestures counted as hand
raises); posture readability depended on apparent person size; the strongest lead (distance -> off-task) did not survive
correction for 42 tests, and earlier naive p-values (e.g. 0.007) were overstated.

## Status and what is next
Done 2026-11-17: per-camera JSONL output following System A's AGENTS.md (role `teacher`, times into each original
video, real times per camera, confidence null for rule states, explicit unknowns, seats in `zone`, no overwrite without
`--overwrite`, checked before any slow work). Off-task is deliberately not written. Verified by 9 unit tests and the
end-to-end run (both files' first events show the same wall-clock time).
Done 2026-11-16: two-camera rewrite of `classroom_analytics copy.py` (now `Main Code.py`), plus matching changes in `llm_report.py` and
`validate_indicators.py`. Installed on the laptop with the user's OK: ultralytics 8.4.215 (torch 2.15 CPU, opencv 5.0),
statsmodels, jinja2, anthropic, FFmpeg 9.0.2 (winget; on the user PATH for new terminals, at
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin`),
and the three YOLO26 weights in the repo root. Tests (scripts in the session scratchpad, not the repo): 36 checks on
synthetic detections with known answers, including the statsmodels paths; plus an end-to-end run on two generated
videos (Ultralytics sample photos, recording times 3 s apart): alignment from metadata, FFmpeg cut, both pose models,
detector, roles, previews, report, facts, LLM dry run and review sheets all worked. **No real classroom video yet**, so
nothing about accuracy is known.
Found 2026-11-18 (user asked "is the lag statistical analysis working?"): System B's `analyze_lags` had (a) an empty
combined model without a board region, (b) p-values driven by one window for rare outcomes, (c) an untestable "seated"
pair, (d) no time control per pair, (e) no tests. **Resolved 2026-11-19 by replacing it with the team's fusion layer**
(guardrails for rare actions, shift test, tests; see "Fusion in System B").
Checked 2026-11-19 on both recorded lectures, re-run with the new code into the session scratchpad only (the user's
`classroom_project/sessions/` and `outputs/events/` were NOT touched, so the web app still shows their old results and
"Run its analysis again" in the Links tab). How_much_..._2 (55 windows): tested T_facing_class, T_stationary, T_path
against S_on_target, S_head_down (others too rare: off-target/phone/hand raises in 2 windows, sleep 0); 6 links + 2
peri-event tests, nothing below p 0.18 (smallest possible 1/43); instructor never raised a hand. Mixed_gender_classes
(124 windows): 18 links + 6 peri-event tests; one weak lead, standing still -> more students turned away 20 s later
(r +0.39, p 0.009, q 0.16; the reverse direction is almost as strong, +0.36, so it may be a shared trend; no note is
added because the reverse is not stronger). No findings in either. Board never measured (no board region calibrated).
Open, in order:
1. The user will share an instructor video and a student video of the same lecture. Put them in
   `classroom_project/videos/<lecture>/instructor.* + students.*`, look at a few frames, and write the instructor
   calibration (teacher zone, board region) before the first run. Seats are found automatically. Still open for the
   real lecture: no board region, so "at the board" is never measured.
2. (done) installs and weights.
3. Human labelling with the new review sheets, then `score`.
4. The one approved Claude call; optional model comparison.
5. Update `docs/SGP2_report_context.md` for the two-camera System B; decide with the team how A and B merge.
6. Frame interval is 2 s. If changed to 1 s, double `HAND_RAISE_MIN_FRAMES` and `OFFTASK_MIN_FRAMES`;
   `STILL_DOWN_MIN_FRAMES` follows automatically; `STATIONARY_MOVE` and `STILL_SHIFT` change meaning.
7. Git: `.gitignore` now keeps new videos, frames, standardized videos, previews, review images and (since
   2026-11-20) `sessions/*/reports/charts/` + `reports/timeline.png` local. The old `Video10min` frames are still
   tracked and already on GitHub; the user chose to keep them and not rewrite GitHub history (2026-11-20).
   On 2026-11-20 (user OK) the 3 then-unpushed commits were rebuilt without the unblurred heatmaps/timeline of
   Mixed_gender_classes and How_much_..._2 (same messages, authors, dates; nothing else changed): 1ed154b/4839b1d/
   ada0428 -> 82b1d39/d925aac/352d13a. The old ones stay reachable locally via the reflog. The images stay on disk.
