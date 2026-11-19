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
| `src/reporting/lag_facts.py`, `llm_text.py` | Turn fusion output into facts and checked plain-English points |
| `web/` | Flask application: models, views, controller |
| `tests/unit/` | 64 automated tests |

### 1.3 Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| Frame interval | 2 s | Same moments in both cameras |
| Window | 10 s | Unit of the association analysis |
| Lags | 1–3 windows | Student change 10–30 s after an instructor feature |
| Minimum windows for statistics | 8 | Below this, no lag statistic is reported |
| Usable window | Posture readable for at least 40% of seat samples and at least 3 seats seen | Feedback uses usable windows only |
| Keypoint confidence | ≥ 0.5 | Below this, a keypoint is ignored |
| Hand raise | Wrist 5% of box height above the shoulder, ≥ 2 samples | About 4 s or more |
| Head down and still | Shoulders move < 0.15 shoulder widths between samples, ≥ 30 s | A resting posture that quiet reading can mimic |
| Seat | Occupied in ≥ 10% of samples; match within 1 shoulder width | Automatic anonymous seats |
| Model batch | 20 frames per call | Keeps memory flat on small machines |

All rule thresholds are unvalidated starting values.

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

### 2.1 Automated tests (System B): 64 tests, all passing on 17 November 2026

| Test file | Tests | What it checks |
|---|---|---|
| `test_main_code_events.py` | 9 | JSONL events: one video ID and role per camera, times in each original video, real times from recording metadata, instructor and student indicators, null confidence for rule states, no silent overwrite, alignment from an offset |
| `test_seats.py` | 7 | Automatic seats: a classroom like the real one (false detections excluded, never two people in one seat, row-by-row numbering), close neighbours kept apart, a shifting student keeps one seat, standing up briefly, hidden shoulders, brief places ignored, grid mode still available |
| `test_feedback_wording.py` | 4 | Feedback is plain English (no "window", "p =", "observable", raw seat IDs) and addresses the instructor as "you"; exact sentences for small cases |
| `test_lag_facts.py` | 13 | Fusion output → facts: input formats, every test counted in the correction, pre-registered rows as their own family, strongest lag per pair, chance links and unknown indicators left out, weak links capped |
| `test_llm_text.py` | 11 | AI-text checks with simulated answers: each kind of mistake caught, weak links need a hedge, numbers must come from the analysis, retry then cache, fallback to rules when the service is unreachable or two answers fail |
| `test_web_app.py` | 20 | Pages, upload checks, progress, failed and interrupted analyses, the AI deployment setting, report contents (no developer details or removed sections), chart data, 95% ranges, safe file serving |

All tests run without network access or video, in about 23 seconds.

System A (the team's evidence pipeline) has its own 9 test files. The team reported 53 passing tests on 26 October
2026. **[TODO: rerun and give the current count.]**

### 2.2 End-to-end tests

1. **Synthetic two-camera run (16 November 2026).** Two generated videos with recording times 3 s apart. Alignment from
   metadata, the FFmpeg cut, both pose models, the object detector, roles, previews, the report, `facts.json`, an AI
   dry run and the review sheets all worked.
2. **Real two-camera lecture (17 November 2026).** A 4 min 36 s test lecture ("How much do you rely on AI in your daily
   life") with one instructor and six students. It ran end to end, including the web report. **[TODO: confirm the
   recording details and consent.]**
3. **Visual checks of the report page** in light mode, dark mode and at phone width (screenshots in a headless
   browser).

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

### 3.2 Association results

- **Pre-registered hypotheses (9 tests, limit p < 0.0056):**
  - "Instructor explaining → phone rate 10 s later": r = −0.31, p = 0.114 after controlling for the previous window.
    Not significant.
  - The other two hypotheses could not be tested: there was no head-down-and-still moment, and the instructor never
    sat down.
- **Exploratory (39 tests):**
  - "Instructor movement → hand raises 10 s later": r = +0.74, p < 0.001. This survives the correction, but it rests on
    only three hand-raise events in 27 windows, so it is a lead to check in more lectures, not a finding.
  - "Instructor stationary → heads up 30 s later": r = −0.49, p = 0.030. Does not survive the correction.

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

These lessons led to the two static cameras, the hypotheses fixed in advance, and the corrected statistics.

## 4. Analysis and discussion

**What the results show:**
- The full system works end to end on a real two-camera recording on a laptop CPU: alignment, detection, behaviour
  rules, events, statistics, feedback and the web report.
- On this recording the detection step was complete: every student in every sample.
- The analysis behaves conservatively: weak or untestable hypotheses are reported as such rather than as findings.

**What they do not show:**
- Accuracy. Without human labels, precision and recall of each indicator are unknown.
- General associations. One 4.6-minute lecture with six students, in which the instructor explained 80% of the time,
  leaves little contrast between activities and very few events.

**Limitations:**
- The rules use unvalidated thresholds.
- Note-taking looks like heads down.
- Phones are small and can be missed.
- Short actions between 2-second samples can be missed.
- No board region was calibrated, so board work was not measured.
- The cameras are aligned to about 1 s.
- The AI rewrite has not yet been tried on a real call.

**Next steps:**
- Label the review sheets and report precision and recall.
- Record more lectures with consent and pool them.
- Calibrate the board region.
- Make the first real AI-rewrite call and compare it with the rule-based text.
- Merge with System A's evidence pipeline.
- Add PDF export of the report.
