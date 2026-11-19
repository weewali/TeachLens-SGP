# Implementation, testing and results analysis

## 1. Implementation

### 1.1 Environment

| Item | Version or detail |
|---|---|
| Language | Python 3.13.9 |
| Computer vision | Ultralytics 8.4.215 (YOLO26) on PyTorch 2.15 (CPU), OpenCV 5.0 |
| Video | FFmpeg and ffprobe 9.0.2 |
| Data and statistics | NumPy 2.3.4, pandas 2.3.3, SciPy 1.16.3, statsmodels 0.15.0 |
| Charts | Matplotlib 3.10.7 (heatmaps); SVG drawn in the browser (timeline, comparison) |
| Web application | Flask 3.1.3, Jinja2 3.1.6 |
| Language model | Anthropic Python SDK 2.9.0, Claude Opus 5.5 |
| Testing | Python `unittest` |
| Hardware | Team laptop, CPU only, Windows 11 **[TODO: processor and memory]** |

### 1.2 Code structure (System B)

| File | Role |
|---|---|
| `main code/Main Code.py` | The two-camera pipeline, stages 1–7 |
| `main code/llm_report.py` | Optional AI rewrite with checks (stage 8) |
| `main code/validate_indicators.py` | Review sheets for human labelling, and scoring |
| `src/analysis/events.py` | The team's event format (identical to System A) |
| `src/analysis/fusion.py`, `lag_analysis.py` | The team's fusion layer (System A's code, unchanged), run by stage 6 |
| `src/reporting/lag_facts.py`, `llm_text.py` | Turn fusion output into facts and checked plain-English points |
| `web/` | Flask application: models, views, controller |
| `scripts/make_pipeline_diagram.py` | Draws the pipeline figure of this report |
| `tests/unit/` | 100 automated tests for System B (plus System A's 151) |

### 1.3 Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| Frame interval | 2 s | Same moments in both cameras |
| Window (comparisons, charts) | 10 s | Unit of the activity comparisons and the report charts |
| Window (links over time) | 5 s | Unit of the team's fusion layer |
| Lags | 1–6 windows of 5 s | Student change 5–30 s after an instructor feature |
| Minimum evidence for a link | Instructor feature in ≥ 5 windows; action started ≥ 3 times | Rarer features and actions are listed as too rare to test |
| Shift test | 500 circular shifts; q ≤ 0.05 (Benjamini–Hochberg) | A link is "unlikely to be chance" only below this |
| Usable window | Posture readable for at least 40% of seat samples and at least 3 seats seen | Feedback uses usable windows only |
| Keypoint confidence | ≥ 0.5 | Below this, a keypoint is ignored |
| Hand raise | Wrist 5% of box height above the shoulder, ≥ 2 samples | About 4 s or more |
| Head down and still | Shoulders move < 0.15 shoulder widths between samples, ≥ 30 s | A resting posture that quiet reading can mimic |
| Seat | Occupied in ≥ 10% of samples; match within 1 shoulder width | Automatic anonymous seats |
| Model batch | 20 frames per call | Keeps memory flat on small machines |

All rule thresholds are unvalidated starting values. They are where accuracy can be improved later: once the review
sheets are labelled, each threshold can be tuned to agree best with the human labels and then checked on lectures
that were not used for tuning. The statistical settings (windows, lags, q ≤ 0.05, minimum evidence) stay fixed in
advance, because tuning them on the data would turn chance patterns into "findings".

### 1.4 Implementation notes

- **Model files:** model weights are loaded only from local files, and a missing file stops the run with a clear
  message instead of a silent download.
- **Overwrite protection:** existing event files are never replaced without `--overwrite`. This is checked before any
  slow work starts.
- **Rebuilding from saved files:** the report, charts and web page are built from saved tables only, so they can be
  rebuilt without the video.
- **Plain-English feedback:** the instructor is addressed as "you", with no statistical terms. The statistics behind
  each point are kept in `analysis_details.txt` and `facts.json`.

## 2. Testing

### 2.1 Automated tests (System B): 100 tests, all passing on 19 November 2026

| Test file | Tests | What it checks |
|---|---|---|
| `test_main_code_events.py` | 12 | JSONL events: one video ID and role per camera, times in each original video, real times from recording metadata, instructor and student indicators (including the ones the fusion layer reads), null confidence for rule states, no silent overwrite, alignment from an offset |
| `test_main_code_fusion.py` | 6 | Stage 6: a planted 10 s link is found only on the shared clock when the two cameras were cut 7 s apart; features that need calibration are dropped when not measured; no recording times; the error path; report lines and points |
| `test_seats.py` | 7 | Automatic seats: a classroom like the real one (false detections excluded, never two people in one seat, row-by-row numbering), close neighbours kept apart, a shifting student keeps one seat, standing up briefly, hidden shoulders, brief places ignored, grid mode still available |
| `test_feedback_wording.py` | 4 | Feedback is plain English (no "window", "p =", "observable", raw seat IDs) and addresses the instructor as "you"; exact sentences for small cases |
| `test_lag_facts.py` | 26 | Fusion output → facts: the real fusion format (run fresh through the team's code), every test counted in the correction, strongest lag per pair, chance links and unknown indicators left out, weak links capped, the vocabulary covers every fusion feature with the same direction |
| `test_llm_text.py` | 14 | AI-text checks with simulated answers: each kind of mistake caught, weak links need a hedge, numbers must come from the analysis, retry then cache, fallback to rules when the service is unreachable or two answers fail |
| `test_web_app.py` | 31 | Pages, upload checks (including a required title and the activity rows: whole minutes from 1 to 180, at most 180 minutes in total, checked before any video is saved), progress, failed and interrupted analyses, delete, the AI deployment setting, report contents (no developer details or removed sections), chart data, 95% ranges, safe file serving, faces pixelated in the seat chart (and old unblurred charts replaced) |

All 251 tests (System B and System A) run without network access or video, in about 16 seconds.

System A (the team's evidence pipeline, copied into this working tree at commit ee5b6bb) has 151 tests, which pass in
the same run.

### 2.2 End-to-end tests

1. **Synthetic two-camera run (16 November 2026).** Two generated videos with recording times 3 s apart. Alignment from
   metadata, the FFmpeg cut, both pose models, the object detector, roles, previews, the report, `facts.json`, an AI
   dry run and the review sheets all worked.
2. **Real two-camera lecture (17 November 2026).** A 4 min 36 s test lecture ("How much do you rely on AI in your daily
   life") with one instructor and six students. It ran end to end, including the web report. **[TODO: confirm the
   recording details and consent.]**
3. **Second real two-camera lecture.** A 10 min 26 s test lecture ("Mixed gender classes", recorded on 3 November
   2026) with one instructor and about six students. It ran end to end. **[TODO: confirm consent.]**
4. **Fusion re-run (19 November 2026).** Both lectures were analysed again with the new stage 6 (the team's fusion
   layer) in a separate folder. The lectures' saved results in the web app still show the earlier analysis until they
   are analysed again.
5. **Visual checks of the web pages** at desktop and phone width (screenshots in a headless browser).

### 2.3 Planned accuracy evaluation **[TODO]**

`validate_indicators.py make-sheets` creates one image per camera for sampled moments and events. A person marks each
one as right or wrong, and `score` computes precision and recall per indicator. No human labels exist yet, so this
report makes no accuracy claim.

## 3. Results

### 3.1 Real lecture: what was measured

All values come from automatic detections that have not been checked by a person.

| Measure | Result |
|---|---|
| Shared lecture time | 04:36, 139 frame pairs (one every 2 s), 28 windows of 10 s |
| Camera alignment | Recording-start metadata: the student camera started 8.0 s before the instructor camera (about ±1 s) |
| Instructor visible | 100% of samples; more than one person in the teacher zone in 5% of samples |
| Instructor facing the class | 03:42 (80% of samples) |
| Instructor posture readable | 98% of visible samples; seated 0% |
| Students | 6.0 per sample in 6 seats; 4 false detections excluded |
| Student posture readable | 99% of seat samples |
| Heads up | 96% of observed student time |
| Hand raises | 3 (two students at 00:14, one at 03:02), all while the instructor was explaining |
| Phone in view | 1 moment (00:14, 2 s) |
| Head down and still | 0 |
| Highest / lowest 30 s | 100% heads up at 00:20–00:50; 84% at 04:00–04:30 |
| Stretch without a hand raise | 01:00–03:00 (2 minutes of explanation) |
| Processing time | About 4 minutes from detection to report (videos already converted, laptop CPU). **[TODO: time a full run from the raw videos.]** |

**Second lecture ("Mixed gender classes"), from its saved run:**

| Measure | Result |
|---|---|
| Shared lecture time | 10:26, 313 frame pairs (one every 2 s) |
| Camera alignment | Recording-start metadata: the student camera started 2.0 s before the instructor camera |
| Instructor visible | 100% of samples; more than one person in the teacher zone in 39% of samples |
| Instructor facing the class | 08:56 (86% of samples) |
| Instructor posture readable | 76% of visible samples; seated 0% |
| Students | 5.9 per sample |
| Heads up | 78% (average over 10-second windows) |
| Hand raises | 11 |
| Phone in view | 17 moments (2.4% of observed student time) |
| Head down and still | 6 moments (8.1% of observed student time) |

The 39% of samples with more than one person in the teacher zone needs a look at the video before these instructor
figures are trusted **[TODO]**.

### 3.2 Association results

Since 19 November 2026 the links over time come from the team's fusion layer (System B's own lag analysis was
replaced). Both lectures were analysed again with it:

| Lecture | 5-second windows | What could be tested | Result |
|---|---|---|---|
| How much do you rely on AI (4:36) | 55 | Instructor facing the class, standing still and distance walked, against students' heads up (facing the front) and heads down: 6 links and 2 averages around action starts. Phones, hand raises and students turning away were too rare. | Nothing below p = 0.18. The smallest possible p was 1/43. |
| Mixed gender classes (10:26) | 124 | 18 links and 6 averages around action starts | One weak lead: when the instructor stood still, more students were turned away 20 s later (r = +0.39, p = 0.009, q = 0.16). The reverse direction is almost as strong (r = +0.36), so it may be a shared trend. |

No link passed the correction (q ≤ 0.05) in either lecture. This is expected for lectures this short. System B tests
up to 24 instructor × student pairs, and the shift test's smallest p is 1/501, so one link alone can pass only in a
lecture of about 43 minutes or more (24/501 < 0.05). The team's real lectures (55–75 minutes) are long enough. The
team's own setting with all indicators tests about 112 pairs, where one link alone can never pass (112/501 = 0.22).
**[TODO: discuss with the team.]**

**Why the guardrails matter.** System B's earlier lag analysis had reported "instructor movement → hand raises 10 s
later" (r = +0.74, p < 0.001) on the first lecture. It rested on only three hand raises in 27 windows. The fusion
layer does not test actions seen fewer than 3 times or features present in fewer than 5 windows, so such a result can
no longer appear as a finding.

### 3.3 A defect found and fixed during results analysis

The first version placed students in a fixed 3×4 grid of equal boxes over the whole student-camera image. Inspecting
the seat heatmap of the real lecture showed three problems:
- Two back-row students fell into the same box.
- One of them spilled into a second box for two frames.
- Legs under a desk and a chair, each seen in only 2 of 139 frames with low confidence (0.25–0.37) and no visible head
  or shoulders, became "seats". The heatmap showed them as 0% heads up.

The pose model itself had found all six students in every frame. The seat step was replaced by automatic seats (see
System design, section 3), and the lecture was rerun.

| Measure | Fixed grid | Automatic seats |
|---|---|---|
| Seats | 8 labels (5 occupied per sample on average) | 6 (6.0 occupied per sample) |
| False detections counted as students | 4 | 0 (excluded) |
| Students with a hand raise / seats seen in at least half the lecture | 2 of 5 | 3 of 6 |
| Heads up, lecture average | 95% | 96% |
| Lowest 30 s | 81% | 84% |

Validating against the real recording found an error that the synthetic tests had not. Seven tests now cover the new
seat logic.

### 3.4 Earlier development lessons (single-camera test video, now retired)

A 10-minute edited test video from a moving camera showed why the design changed:
- Roles assigned by image zone failed when the framing changed: the instructor's gestures were counted as student hand
  raises.
- Posture readability depended strongly on how large people appeared.
- The strongest association lead did not survive correction for 42 tests, and the earlier uncorrected p-values had
  been overstated.

These lessons led to the two static cameras and the corrected statistics (first three hypotheses fixed in advance,
and since 19 November 2026 the team's fusion layer, which corrects over every pair it tests).

## 4. Analysis and discussion

**What the results show:**
- The full system works end to end on a real two-camera recording on a laptop CPU: alignment, detection, behaviour
  rules, events, statistics, feedback and the web report.
- On this recording the detection step was complete: every student in every sample.
- The analysis behaves conservatively: weak or untestable hypotheses are reported as such rather than as findings.

**What they do not show:**
- Accuracy. Without human labels, precision and recall of each indicator are unknown.
- General associations. Two short lectures (4.6 and 10.4 minutes) with about six students each, in which the
  instructor faced the class 80–86% of the time, leave little contrast between activities and few events.

**Limitations:**
- The rules use unvalidated thresholds.
- Note-taking looks like heads down.
- Phones are small and can be missed.
- Short actions between 2-second samples can be missed.
- No board region was calibrated, so board work was not measured.
- The cameras are aligned to about 1 s.
- The AI rewrite has not yet been tried on a real call.
- Faces in the seat chart are pixelated only where a person was detected (in the first 10 seconds or in a seat);
  someone never detected would stay visible. The original frames used by the analysis are not blurred.
- The web page no longer says that the detections are unverified; only the saved report files do.
- Planned activities look like off-task behaviour: during a quiz, students writing look like "heads down". The
  upload form now records such activities (quiz, group work, video, break) and how many minutes each lasted, but not
  when it happened, so the analysis cannot tell which minutes to leave out. It does not use them yet.

**Next steps:**
- Label the review sheets and report precision and recall.
- Tune the behaviour-rule thresholds against the labels (never the statistical settings) and check them on other
  lectures.
- Record full-length lectures (55–75 minutes) with consent.
- Decide how to use the noted activities **[TODO]**. Since only their length is known, options are a note beside the
  student figures they affect, or finding a stretch of matching length automatically (for example many heads down
  while the instructor is not facing the class) and asking the instructor to confirm it. Once the minutes are known,
  they can be left out of the student figures and the links, shaded on the timeline, and listed in the report.
- Calibrate the board region.
- Make the first real AI-rewrite call and compare it with the rule-based text.
- Merge with System A's evidence pipeline.
- Add PDF export of the report.
