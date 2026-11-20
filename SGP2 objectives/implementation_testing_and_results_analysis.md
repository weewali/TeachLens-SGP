# Implementation, testing and results analysis

Status: 20 November 2026. On that day the project became one pipeline: the team's lesson pipeline (`src/`,
`scripts/analyze_lesson.py`) does the processing, and the report stage (`scripts/report_lesson.py`) and the web
application (`web/`) sit on it. The earlier second implementation's own processing was removed (it stays in the Git
history); its results are kept in section 3.3 as history.

## 1. Implementation

### 1.1 Tools and environment

| Area | Tools |
|---|---|
| Language | Python 3.13 on a Windows laptop (13th-gen Intel Core i7, integrated graphics, CPU only for the models) |
| Vision | Ultralytics 8.4 (YOLO26n-pose, YOLO26n detector, ByteTrack), PyTorch (CPU), OpenCV 5.0 (video decoding on the graphics chip where possible), MediaPipe 1.2.2 (face landmarks, hand gestures) |
| Video | FFmpeg / ffprobe 9.0.2 (recording-start times) |
| Analysis | NumPy, pandas, SciPy, statsmodels (HAC errors, lagged models) |
| Report and web | matplotlib (heatmaps), Flask 3.1.3 + Jinja2, SVG charts drawn in the browser, Anthropic Python SDK |
| Settings | YAML (`configs/default.yaml`, a room file per classroom) |
| Tests | Python `unittest` |

### 1.2 Code structure

| Path | What it does |
|---|---|
| `src/preprocessing/` | Video reading and sampling, recording-start times, lining up the two cameras, camera-motion check |
| `src/inference/` | Pose (YOLO26n-pose), tracking and stitching (ByteTrack), objects, face landmarks, hand gestures |
| `src/analysis/` | Behaviour rules (instructor, students, facial actions, interactions), events, summaries, camera links, fusion and lag analysis, and `activity_comparison.py` (students during each instructor activity) |
| `src/reporting/` | `facts.py` (facts for the AI step), `feedback.py` (comparison facts, tables, rule-based points, the report stage), `llm_report.py` (Claude rewrite and its checks), `review_sheets.py` (human review), review and side-by-side videos, charts |
| `scripts/` | `analyze_lesson.py` (whole lesson), `report_lesson.py` (report), `review_sheets.py` (human review), `make_pipeline_diagram.py` (this report's figure), and the team's single-step scripts |
| `web/` | Flask application (models, views, controller) |
| `tests/unit/` | 217 automated tests |

### 1.3 Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| Sampling | 1 frame per second (faces 5 per second) | Short actions between samples can be missed |
| Pose model input | 1280 px | Smaller inputs lost back-row and head-down students on the 4K student camera |
| Object detector input | 1920 px | Phones in hands are small |
| Window | 5 s | Unit of the lag analysis, the comparisons and the charts |
| Lags | 1–6 windows (5–30 s) | How long after an instructor action a student change is looked for |
| Minimum evidence | A teacher feature in ≥ 5 windows; ≥ 3 starts for rare actions; ≥ 5 windows on each side of a comparison | Smaller cases are not tested |
| Shift test | 500 circular shifts; q ≤ 0.05 (Benjamini–Hochberg) | A link counts as a finding only below this |
| Comparisons | Teacher feature ≥ half a window; p < 0.05 / number of comparisons (Bonferroni) | "Unlikely to be chance" in the feedback |
| Claude feedback | Claude Opus 5.5; at most 3 points per side, 80 words each; 2 attempts | Otherwise no feedback is shown, only the reason |

All behaviour-rule thresholds are unvalidated starting values. They are where accuracy can be improved later: once the
review sheets are labelled, each threshold can be tuned to agree best with the human labels and then checked on
lectures that were not used for tuning. The statistical settings stay fixed in advance, because tuning them on the data
would turn chance patterns into "findings".

### 1.4 Implementation notes

- **No downloads at run time:** models are loaded only from local files; the web job also stops Ultralytics from
  installing or downloading anything. (Ultralytics did install one small tracking package, `lap`, on the very first
  run before this was switched off.)
- **Overwrite protection:** an existing lesson folder is never replaced without `--overwrite`; a rerun from the web
  app replaces it on purpose.
- **Rebuilding from saved files:** the report, charts and web page are built from the lesson folder only.
- **Claude writes the feedback, no rule-based text** (user decision, 20 November 2026): without an API key, without
  a network, on a refusal or when its text fails the checks twice, no feedback is written; `report_status.json` and
  the web page give the reason (an API key problem is shown only as "Code=01", a billing problem only as "Code=02"), and "Try again" re-runs only Claude's step. The web app reads the API key from a
  private `.env` file every time it starts (never uploaded to GitHub).
- **Privacy in pictures:** the two heatmaps are drawn on a video frame with every student's head pixelated (tracked
  people, each seat's usual position, and anyone the pose model finds in that frame); the instructor stays visible,
  because the report is theirs. On any error the chart has no background rather than an unblurred one.
- **Brief seats hidden:** a "seat" used for less than a minute or under 10% of the lecture is someone passing close
  to the camera (on the real lecture: two such "seats" at the right edge, 3-4 s each) and is not drawn.

## 2. Testing

### 2.1 Automated tests: 217 tests, all passing on 20 November 2026

| Test files | Tests | What they cover |
|---|---|---|
| The team's pipeline (`test_config`, `test_events`, `test_visual_indicators`, `test_student_indicators`, `test_teacher_indicators`, `test_facial_actions`, `test_fusion`, `test_lesson`, `test_lesson_outputs`, `test_video`, ...) | 170 | Settings, the event format, behaviour rules on synthetic people, tracking, camera alignment, fusion and lag analysis, lesson folders |
| `test_report_stage.py` | 12 | A planted difference found by the "during each activity" comparison; facts added once and renumbered; no rule-based text; the checks accept a faithful answer and reject a bad one; no key: no feedback, the reason kept and "Code=01"; only API key problems get Code=01 and only billing problems Code=02; a passing answer used and cached; two failed answers: no feedback, with the reason; review sheets made, scored and marked in `facts.json` |
| `test_web_app.py` | 35 | Pages, the API key read from `.env` at start-up (never shown), "Try again" for Claude's step, the PDF download (self-contained page, saved copy reused, a clear message without a browser), upload checks (title, videos, activities), the two commands and their stages, the AI result or its reason, failed and interrupted analyses, Windows file locks, rerun, delete (including a running analysis and cross-site requests), report and chart data, face blurring |

They run without network access or video, in about 8 seconds, on a made-up lesson (`tests/fixtures/demo_lesson.py`).

### 2.2 End-to-end runs

1. **30 seconds of a real two-camera lecture** ("Mixed gender classes", 20 November 2026): both cameras lined up
   from their metadata (the student camera started 2 s earlier), analysed, linked and fused in 1.2 minutes; the
   report stage then ran and, with no API key on that computer, recorded
   "No Claude API key is set on this computer. Nothing was sent." Both heatmaps were checked by eye: every face was
   pixelated after two fixes (untracked students in the instructor view; a person cut off by the bottom edge).
2. **The whole 10.4-minute lecture through the web application's job** (20 November 2026): all seven stages ran
   without errors in 16.4 minutes (about 1.6 minutes of computing per lecture minute; the student camera took 13 of
   them). 124 five-second windows; no fusion finding; 10 facts, including two "during each activity" comparisons
   (both weak). With no API key on that computer, Claude's step recorded its reason, so the report has no written
   feedback yet. To check by eye: 30 hand raises in 10 minutes for six
   students may be an over-count, and the seat map shows 8 seats for 6 students (two seats at the right edge may be
   people passing the camera) **[TODO: review sheets]**.
3. **The team's seven scripted scenario lessons** (1–10 minutes each, the same six students): every lesson ran start
   to finish. The student camera dominated the time (6–9.5 minutes per lesson minute at the time; 2.2 minutes on
   20 November in a faster power mode); the instructor camera took under 2 minutes per minute.

### 2.3 Human-labelled evaluation

Not done yet. `scripts/review_sheets.py make` writes one image per sampled moment (from the review videos, with boxes
and track numbers) and a sheet listing what the pipeline detected; a person writes yes or no; `score` gives the
precision of each detection with a 95% Wilson interval and records it in `facts.json`, so the report stops calling a
checked detection "not verified". **[TODO: labelling]**

## 3. Results analysis

### 3.1 Scenario lessons (team pipeline)

No scenario lesson gave a confirmed link between instructor actions and student behaviour. Board work, writing and
pointing happened in at most 4 windows of any lesson, so they could not be tested, and the instructor faced the class
90–100% of the time, which leaves little to correlate. Two observations shaped the design: note-taking had read as
"not facing the front" (fixed by the on-task measure, which counts writing and reading as on task), and a seated
instructor was detected only after the desk rule was added. Pooled over seven lessons, every testable hypothesis
pointed the way the scenario booklet expects except "facing the class", but none passed q ≤ 0.05.

### 3.2 Why most lessons give no confirmed link

With all indicators the lag analysis tests about 112 instructor × student pairs. The shift test's smallest possible
p-value is 1 / (number of shifts + 1), at best 1/501. After Benjamini–Hochberg over 112 tests a single link can never
reach q ≤ 0.05 (112 / 501 ≈ 0.22); several strong links at once are needed. The 9 hypotheses fixed in advance are
corrected among themselves, which helps, but short lessons still have few windows. This is why the report also uses
the simpler comparison of students during each instructor activity, which needs fewer windows; its points are still
hedged ("could easily be chance") unless they pass the Bonferroni threshold.

### 3.3 Earlier results (the second implementation, retired on 20 November 2026)

Before the merge, a separate two-camera pipeline with its own detection was run on two real lectures (4.6 and 10.4
minutes). Kept as history:
- Both lectures were aligned from metadata (the student camera started 8.0 s and 2.0 s earlier).
- A fixed seat grid merged two students and turned a chair and a pair of legs into "seats"; seats found from where
  students sat corrected this. The team's pipeline learns seats the same way.
- The team's fusion layer, run on that pipeline's events, found no link that passed the correction in either lecture.

### 3.4 What the results show, and what they do not

**They show:** the complete chain runs on a laptop CPU from two videos to a checked report in a browser; the
statistics behave conservatively; privacy measures work on real frames.

**They do not show:** accuracy (no human labels yet); general associations (only short, mostly scripted lessons);
the quality of the AI wording (no real call yet).

**Limitations:**
- The rules use unvalidated thresholds; note-taking can still look like looking down.
- Phones are small and can be missed; one frame per second can miss short actions.
- The cameras are aligned to about 1 second.
- Board, screen and desk need to be marked once per classroom; without a room file they are not measured.
- A one-hour lecture takes two hours or more to analyse on this laptop.
- The web page no longer says that the detections are unverified; only the saved files do.
- Activities in class are stored with their duration but not when they happened, so they cannot yet be left out.

**Next steps:**
- Label the review sheets, report precision and recall, then tune the behaviour thresholds (not the statistics).
- Record full-length lectures (55–75 minutes) with consent.
- Make the first real Claude report and review its wording.
- Write a room file for the lecture room.
- Decide how to use the activities instructors note **[TODO]**.
