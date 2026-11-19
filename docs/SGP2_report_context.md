# TeachLens: context for writing the SGP2 report

Prepared 2026-11-16 from the repository (branches `scripting-frame-extraction` and `pipeline_creation`), the SGP1
report, the team's design docs (`AGENTS.md`, `docs/project_structure.md`, `docs/list_a_pose.md`,
`docs/video_indicator_catalog.md`) and the current SGP2 draft (sections 3.1.4 to 3.3.7).
Give this whole file to the model that helps write the SGP2 report.

---

## 0. Instructions for the model reading this

You are helping a 5-person UAEU team write the **SGP2 final report** (second half of the senior graduation project).
SGP1 was the planning and design report. SGP2 reports implementation, testing, evaluation and results.

Rules:
1. **Never invent results.** Use only numbers in this file or numbers the team gives you. Where a result does not exist
   yet, write `[TODO: result from <what>]`.
2. **Keep status honest.** Components are tagged **DONE**, **PARTLY DONE**, **PLANNED**, **DEFERRED** or **DROPPED**.
   Never describe planned work as finished. Smoke runs are not accuracy results.
3. **Observable behaviour only.** Never write that the system measures attention, engagement, emotion, understanding,
   learning or intent. Say what is seen: "visible", "sitting/standing", "motion level", "head-up posture".
4. **Association, not cause.** "Was associated with", "in the windows where". Never "caused", "led to", "improved".
5. **Advisory tool**, not a score and not surveillance. Unknown is not zero: unknown posture is not sitting, no
   detection is not absence, a tracker ID is not a unique person.
6. **Explain changes from SGP1** (section 5).
7. **Stay consistent with the existing draft.** Chapter 3 of the draft already describes the main pipeline (section 4,
   system A) with YOLO26 Pose, ByteTrack, YAML, JSONL and unittest. Do not contradict it.
8. **References:** the SGP2 draft numbers its own list: [1] Li et al. 2025, [2] Trabelsi et al. 2023, [3] Liu et al.
   2025, [4] ByteTrack, [5] Ultralytics YOLO. Continue from [6]. SGP1 references are listed in section 16 for reuse.
9. Write SGP1-style: numbered chapters and sections, requirements as "The system shall ...".
10. Where team input is needed, write `[TODO: ...]` rather than guessing.

---

## 1. Project in one summary

**TeachLens: AI-Based Classroom Teaching Evaluation System** (UAEU, College of IT, Senior Graduation Project, Project ID 11).

End-of-semester student surveys are often biased, rushed or uninformative, so instructors receive little useful
feedback. TeachLens analyses **recorded lectures** offline to give instructors **evidence-based, advisory feedback** on
observable classroom behaviour. In SGP2 the recording setup uses **two static cameras on tripods**: an
**instructor camera** and a **student camera**. Each recording is processed by a Python pipeline that samples one frame
per second, runs **YOLO26 Pose** (person boxes and 17 body keypoints) with **ByteTrack** (temporary anonymous IDs),
and applies transparent rules to produce **timestamped evidence events** (JSONL): instructor presence, visible student
count, position, sitting/standing, motion level, visibility, and optional classroom zones. The camera view decides the
assumed role; ambiguous cases stay **unknown**. Every event also carries the **real (wall-clock) time** from the
video's metadata, so the two cameras can be aligned to about one second. A summary module turns the events into a JSON
summary, and an annotated review video shows exactly what the rules saw. A separate prototype, built on an earlier
single-camera version, adds the planned upper layers: **10-second windows, lag analysis** between instructor and student
behaviour (associations, not causes, with corrections for autocorrelation and multiple tests), and an **LLM (Claude)
step** that turns pre-computed facts into a short report of **positive and negative points**, checked automatically in
code. Connecting these layers to the two-camera evidence is the main remaining integration work.

---

## 2. Team and identity (from SGP1)

- Course (SGP1): ITBP480 Senior Graduation Project I, Spring 2026. SGP2 course code and term: `[TODO]`.
- Advisor: Dr. Hesham El Sayed. Team notes also mention Dr. Mohammed Amin regarding LLM integration: `[TODO: confirm role]`.
- Team and SGP1 roles:
  - Nahyan Al Ali: Project Coordinator (meetings, supervisor contact, progress, security, risk, feasibility)
  - Ali Mohammad Zainal: AI / Model Developer (YOLO models)
  - Mohammad Alqaidy: Video Processing and Data Management (recording, data)
  - Ali Algaithi: System Design and Integration
  - Abdallah Al Kaabi: Documentation and System Evaluation
- Who built which SGP2 component, contribution table and signatures: `[TODO]`.

---

## 3. What SGP1 delivered and promised

SGP1 (April 2026) was a planning and design report: introduction, background and literature review, requirements
(FR/NFR, standards, tools, risk), conceptual design (architecture, ER diagram, interface), ethics and sustainability,
project management, conclusion.

SGP1 promised for SGP2: collect and annotate data, train instructor and student models, integrate one pipeline,
implement fusion and lag analysis, produce timestamped outputs and indicators, connect report generation, and test on
recorded lectures (model accuracy, lag linking, report clarity). Optional: interface, more data, audio/multilingual.

SGP1 functional requirements: FR1 video input; FR2 preprocessing (frames, timestamps, classroom configuration); FR3
instructor analysis (single instructor timeline, actions, events); FR4 student analysis (seat regions, behaviours,
events); FR5 indicators; FR6 fusion (shared timeline, windows, lag analysis, recorded results); FR7 results (structured
result, readable report, evidence). NFR1 performance and timestamp alignment; NFR2 authorised access and anonymous IDs;
NFR3 usability; NFR4 scalability; NFR5 reproducibility and modular stages. Status in section 13.

**Important:** SGP1 argued for a **single fixed camera** (sections 4.4 and 5.3, citing Kumar: simpler data, lower
compute), while its problem statement (1.1) named the single vs multi-camera choice as an open issue. SGP2 uses two
cameras, so the report must justify the change (section 5).

---

## 4. What exists now: two implementations

| | **System A: main pipeline ("List A")** | **System B: association and LLM prototype** |
|---|---|---|
| Where | Branch `scripting-frame-extraction`: `src/`, `scripts/analyze.py`, `configs/`, `tests/unit/` | Branch `pipeline_creation`: `main code/Main Code.py` |
| Input | One recording per run, from `instructor_view` or `student_view` (two static cameras) | One single-camera video covering both instructor and students |
| Detection | YOLO26n-pose + ByteTrack, 1 frame/s | YOLO11n-pose + YOLOv8n (phones, books), 1 frame per 2 s |
| Roles | Camera context; ambiguous = unknown | Calibration zones (teacher zone, student region) |
| Identity | Temporary anonymous tracker IDs | 3x4 seat grid, anonymous seat IDs |
| Indicators | Presence, student count, position, zone, entry/exit, visibility, sitting/standing, motion level | Head-up / head-down, hand raise, phone, explaining toward class, board writing, gesture, movement, instructor seated, distance to students |
| Output | JSONL events, JSON summary, annotated review MP4 | CSV tables, window features, lag results, rule-based report, `facts.json`, optional Claude report |
| Analysis | Per-recording summary only | 10 s windows, lag analysis with statistical controls |
| Tests | 9 unit-test files (unittest); 53 tests passed on 2026-10-26 per the team handoff | Offline tests of the LLM checks (simulated answers); no unit tests |
| Status | DONE as a working prototype; validation pending | DONE as a single-camera prototype; not connected to System A |

**How the report should present this:** System A is the implemented system and is what Chapter 3 describes. System B
is a prototype of the fusion, lag-analysis and reporting layers that SGP1 planned (FR6, FR7). It shows these layers
work end to end on a single-camera test video, and it is the design for the layers to be added on top of System A's
evidence. Porting it is **PLANNED** (`[TODO: confirm whether done before submission]`).

**Honest consequence:** with System A's current indicators, the student-side outcomes available for lag analysis are
visible student count, student sitting/standing and student motion level. The richer student indicators (head-up,
hand raise, phone) exist only in System B's single-camera rules. Hand raises and orientation are deliberately outside
System A's current scope (they are "List C" in the team's indicator catalog).

---

## 5. Changes from the SGP1 design (explain each in the report)

| Topic | SGP1 plan | SGP2 | Reason |
|---|---|---|---|
| Cameras | One fixed camera covering instructor and students | **DONE (as separate recordings):** two static cameras, `instructor_view` and `student_view`, each processed separately. **PLANNED:** joint analysis of both | A single view gives either a clear instructor or clear students, not both. In System B's single-camera test, zone-based roles failed when framing changed (the instructor's gestures were counted as student hand raises), and camera changes affected instructor and student measurements at the same moment, which can create false associations. Cost: a second camera, alignment, twice the processing. |
| Detection model | Train YOLO11m (instructor) and YOLO11l (students) action models | **DONE:** pretrained YOLO26n-pose with transparent rules; no action model trained | No labelled temporal-action dataset; pose keypoints support posture and motion rules that can be explained and tested. Team rule: evaluate rules on the pretrained pose model first, train only when a labelled dataset exists. |
| Role assignment | Calibration zones | **DONE:** camera context. `student_view`: every confirmed track is a student. `instructor_view`: teacher only when exactly one confirmed person is visible; otherwise unknown | Avoids training a teacher-vs-student classifier; uncertainty preserved. Not a verified role. |
| Tracking | Seat-based IDs; DeepSORT mentioned | **DONE:** ByteTrack temporary IDs (confirmed after 2 samples, tolerate one missed sample) | Works for both views; IDs can switch, so they are not unique people. |
| Dataset | Two-class instructor/student labels | **PARTLY DONE:** one-class `person` dataset prepared and audited (split leakage found); **not used** by the running pipeline | Camera context replaces the role class. |
| Sampling / windows | Windows of e.g. 5 s | System A: 1 frame/s (5/s was too slow). System B: 1 frame per 2 s, 10 s windows | CPU-only hardware. |
| Camera sync | Not designed | **DONE (coarse):** each event stores real time from the recording-start tag in the video metadata (read with ffprobe); about +/-1 s | Team decision 2026-10-26: an audio cross-correlation sync tool was built and then removed as too complicated. |
| Camera motion | Not considered | **DONE (experimental):** background feature matching estimates pan/zoom and detects cuts, for operated web videos only | Development used web videos filmed by operated cameras. Own recordings use static cameras and do not need it. |
| Emotion recognition (AffectNet, MaskTheFace) | Support feature | **DROPPED** | Team catalog rule: never report emotion, attention or engagement as fact. |
| "Attention level" | Student attention score | System A: not measured. System B: head-up posture proxy | Only visible posture can be measured; note-taking looks like head-down. |
| Data storage | Relational design (ER diagram) | **DONE:** JSONL events + JSON summary (A); CSV tables (B). No database | Simple, reusable files; reports can be regenerated without rerunning the model. |
| Visual review | Dashboard video panel with overlays | **DONE:** annotated review MP4 and optional live window (boxes, IDs, roles, posture, motion) from the same run | Lets a person check what the rules saw. Contains original footage, so it stays local. |
| Statistics | Cross-correlation + lagged OLS (SGP1 figure 4.2) | **DONE in B:** lagged model controlling for the previous window and time in lecture, Bonferroni correction, pre-registered vs exploratory pairs, HAC tests | Neighbouring windows are alike (autocorrelation about 0.67), so naive p-values were too optimistic. |
| Report generation | Jinja2 template; LLM only fills placeholders | **DONE in B:** rule-based report + `facts.json` + optional Claude rewrite with automatic checks. **PLANNED in A:** charts and HTML report | Keeps the report traceable to measured facts, as SGP1 intended. |
| Dashboard | Web/desktop dashboard | **NOT BUILT** (`[TODO: in SGP2 scope?]`) | Core pipeline prioritised. |
| Audio / speech | Optional extension | **DEFERRED** | Outside the agreed physical-indicator scope. |
| Hand raise, orientation, objects | Student indicators | **DEFERRED in A** (List C). Rule-based versions exist only in B | Scope decision; needs validation. |

---

## 6. System A architecture (for the design and implementation chapters)

Layers (team rule: models never write report prose; evidence, then analysis, then reporting):

1. **Preprocessing** (`src/preprocessing/video.py`): OpenCV reads the video and samples one frame per second, keeping
   source timestamps; `--start/--end` analyse one segment by seeking directly. `recording_time.py` reads the recording
   start time with ffprobe. `camera_motion.py` handles moving cameras (web samples only).
2. **Inference** (`src/inference/pose.py`): local YOLO26n-pose checkpoint (never auto-downloaded) with ByteTrack; returns
   normalised boxes and COCO-17 keypoints with confidences.
3. **Rules** (`src/analysis/visual_indicators.py`): per camera, only allowed joints are used (students: head, arms, hips,
   because desks hide legs; instructor: all joints). Posture: instructor by hip/knee angles (sitting <= 135 degrees,
   standing >= 160 degrees) with an upper-leg fallback; students by shoulder rise above their own seated level (standing
   >= 0.7 shoulder widths, sitting <= 0.35). Motion: smoothed joint displacement in body-heights per second
   (still / low / medium / high, boundaries 0.08 / 0.35 / 0.8). All thresholds are unvalidated starting values.
4. **Events** (`src/analysis/events.py`): validated records in JSONL: video ID, track ID, role, indicator name,
   start/end/duration, value, confidence (null for rule-based states), coverage (adequate / limited / unknown),
   optional zone, real start/end time.
5. **Summary** (`src/analysis/aggregates.py`): JSON summary from saved events: instructor visible time and fraction of
   valid samples, student count min/median/max, posture and motion durations, zone time, camera-motion counts,
   limitations. Unknown time is reported, not counted as zero.
6. **Review** (`src/reporting/annotated_video.py`): annotated MP4 and optional live window.
7. **Configuration** (`configs/*.yaml`, `src/utils/config.py`): sampling, thresholds, camera policies, zones,
   indicator switches; local overrides without code changes.
8. **Planned upper layers** (from System B): align both cameras' events by real time, build 10 s windows, lag analysis,
   `facts.json`, LLM report, charts and HTML report.

Outputs per run: `outputs/events/<id>.list_a.jsonl`, `outputs/reports/<id>.list_a.json`,
`outputs/debug/<id>.list_a.mp4`. Each camera recording gets its own anonymous video ID.

Tools: Python, OpenCV, Ultralytics YOLO26 Pose (on PyTorch), ByteTrack, NumPy, PyYAML, FFmpeg/ffprobe, JSON/JSONL,
unittest, Git/GitHub. System B adds pandas, SciPy, statsmodels, Matplotlib and the Anthropic SDK.

---

## 7. Indicators

**System A (implemented; all observable, all unvalidated thresholds)**

| Indicator | Camera | What it records | Limits |
|---|---|---|---|
| Teacher presence | instructor | Visible / not detected / unknown per sample, and visible intervals | Unknown when several people appear |
| Visible student count | student | Count of confirmed tracks per sample | Not attendance; hidden students missed |
| Anonymous identity | both | Temporary tracker ID episodes | ID switches split one person into several IDs |
| Role | both | Camera-assumed role, `verified_by_model = false` | Not a learned classifier |
| Position | both | Box, foot point, hip-centre path | Image coordinates, not metres on the floor |
| Classroom zone | both | Time in calibrated polygons (optional) | No zone events until a real calibration file is set |
| Entry / exit | both | Apparent appearance or disappearance at frame edges | Not proof of entering or leaving the room |
| Visibility / occlusion | both | Which body parts are visible, truncation, overlap | Cannot tell occlusion from blur or model failure |
| Sitting / standing | both | Posture intervals, or unknown | Instructor behind a lectern or cut off at the knees stays unknown |
| Motion level | both | Still / low / medium / high / unknown | Apparent image motion, not speed or engagement |
| Camera motion | moving cameras only | Pan, zoom, shots | Web samples only |

**System B (single-camera prototype, rule-based)**: head-up rate ("focused"), head-down / off-task rate (head down for
3+ samples with no book, or a phone), hand-raise count, phone detections, instructor explaining toward class, board
writing, interaction (facing + gesture), movement, instructor seated, instructor-to-student distance.

**Deferred** (team catalog): hand raises and orientation on two cameras, board writing, pointing, note-taking, device
use, peer discussion, all audio and transcript indicators, recording-quality indicators.

**Never claimed** (team catalog, section G): emotion, attention, engagement, motivation, understanding, intent,
personality, protected attributes, misconduct, official attendance. Use narrow proxies with a denominator, e.g.
"oriented toward the board for 62% of valid visible time", never "62% attentive".

---

## 8. Two-camera alignment

- Each event stores `start_real_time` / `end_real_time` = recording-start tag + position in the video (ISO 8601).
- Clock tests with a filmed atomic-synced clock: the iPhone start tag was **0.7 to 0.95 s after the first frame**,
  so real times are good to **about 1 second**. Two iPhones started together agreed within about 0.02 s, with no
  measurable drift over 3 s.
- Limits: whole-second tags; videos without the tag (downloads, edited or re-encoded files) get null real times.
  Coarse alignment only, not frame-level sync.
- With 10 s analysis windows, a 1 s alignment error is small, but it should be stated as an assumption in the lag
  analysis.
- Cross-camera identity and floor-plane positions are not implemented. An indicator that needs both views in one
  coordinate system (e.g. instructor-to-student distance) needs geometric calibration or a coarse zone instead:
  `[TODO: team decision]`.

---

## 9. Analysis and LLM layers (System B; for methodology and evaluation)

**Lag analysis**
- 10 s windows; at least 8 valid windows before any statistic is reported.
- Instructor feature in window t vs student outcome in windows t+1 to t+3 (10 to 30 s later).
- p-values from a regression controlling for the outcome's previous window and time in lecture (SGP1 figure 4.2),
  judged against a Bonferroni limit; pre-registered pairs (instructor seated vs head-up; instructor-to-student distance
  vs hand raises) separated from exploratory pairs. Gap comparisons use HAC (autocorrelation-robust) tests.
- Multi-lecture pooling; lags never cross from one lecture to the next.
- Evidence labels: suggestive, weak, descriptive.

**LLM report layer**
- Only `facts.json` reaches the LLM: anonymised pre-computed facts with ids, direction (favourable / unfavourable),
  evidence strength, glossary, data-quality notes, alternative explanations, and per-indicator validation status.
  No video, frames or identities are sent.
- Output: `positive points:` and `negative points:`, at most 3 each, never padded, plain sentences, max 80 words each.
- Checks in code: every number and time must appear in the cited facts; positive points cite only favourable facts;
  no causal wording; no inner-state words; no judgement of the instructor as a person; hedges required where evidence is
  weak or a detection is unverified (or validated precision below 0.7). One retry; on failure the rule-based report is
  kept. Answers cached for reproducibility.
- Models: Claude Opus 5.5 (default), Sonnet 5.5, Haiku 5.5. About USD 0.04 per report with Opus.
- Status: tested offline with simulated answers, including 18+ deliberately wrong ones that the checks caught.
  **No real API call made yet** `[TODO: first real call result]`.

---

## 10. Testing done so far (label everything as prototype testing)

**System A**
- Unit tests (unittest, 9 files): posture rules and unknown cases, motion, zones, visibility, track confirmation and
  roles, event schema and atomic writing, real-time fields, summary keys, configuration validation, video segment
  seeking, camera-motion maths, review overlays. 53 tests passed on 2026-10-26 (team handoff). `[TODO: rerun and give
  the current count]`. These check code behaviour, not accuracy on real classrooms.
- Smoke runs on web sample videos (diagnostic, **not benchmarks**):
  - 63:42 instructor lecture filmed by an operated camera (before camera compensation): instructor visible in 87% of
    valid samples, but 113 teacher-role and 1,442 unknown-role tracker IDs and 938 apparent entries. This shows heavy
    tracker fragmentation on a moving camera and supports the static-camera decision.
  - 21.5 s student clip: visible student count min 6, median 8, max 9 (12 tracker IDs).
  - 20 s instructor segment with camera compensation: 95 of 98 camera steps measured; real zoom cuts detected.
    About 0.4 s per analysed 1080p frame on a CPU (99 frames in about 38 s).
- Clock tests for alignment (section 8).

**System B** (one single-camera test video, 10:01, 854x480, `[TODO: source]`; every detection unverified)
- The video is edited footage with a moving camera: about 21% of frames match the calibration framing, 49% are pans or
  zooms, 30% are cuts or close-ups. Zone-based roles fail there; two "hand raises" were the instructor gesturing.
- Instructor visible 84% of frames; instructor posture readable 26%; student posture readable 57%. Readability depends
  strongly on apparent person size (r about -0.65).
- Example facts: head-up rate 46% while the instructor explained toward the class vs 54% otherwise (HAC p < 0.01,
  suggestive); 44% vs 52% when the instructor was closest to students (p = 0.02).
- Strongest exploratory lag lead (instructor-to-student distance vs later off-task rate, r about -0.33 after controls,
  p about 0.02) **does not survive correction for 42 tests**. Earlier naive p-values (e.g. 0.007) were overstated.

**Missing** `[TODO]`: accuracy against human labels on own two-camera recordings; lag analysis on two-camera data; real
LLM output and check pass rate; full-lecture runtime.

---

## 11. Evaluation plan (for the testing chapter)

- **Recordings:** short static-camera test lectures with 4 to 6 students, following `docs/test_recording_scenarios.pdf`
  (camera setup, behaviour codes, per-student scenarios, calibration clips, ground-truth log template). Consent required.
- **Detection and tracking accuracy (both cameras):** person recall, missed and duplicate people, tracker ID switches,
  posture unknown rate, event precision and recall with a declared time tolerance; 95% intervals.
- **Threshold tuning** on calibration clips, then testing on held-out clips (never tune on the test set).
- **Alignment:** compare real times of the same moment in both cameras.
- **Association:** pooled lag analysis over several lectures once System B's layers run on System A's events.
- **LLM layer:** share of reports passing all checks first time; failure types caught; optional model comparison.

---

## 12. Limitations, risks and ethics (update SGP1 chapters 3.5 and 5)

Limitations: thresholds unvalidated; pose-only rules; roles assumed from camera view; tracker ID switches; 1 s sampling
misses short actions and fragments tracks for fast walkers; positions are image-plane, not floor distances; no
synchronised fusion yet; activity type (lecture, seat-work, discussion) not recorded although it changes posture by
itself; only web samples and one edited test video analysed so far.

Two-camera risks: alignment error (about 1 s); the instructor entering the student view is counted as a student (the
policy must then be switched to unknown); a front student camera captures faces (higher sensitivity); about twice the
processing; public web videos of different lectures cannot be paired, so own synchronised recordings are required.

Privacy measures in place: local processing; anonymous IDs; no face crops or identity matching; review MP4s contain
original footage and stay local, never committed or shared without permission; iPhone metadata contains GPS tags,
which the tools never print (strip metadata before sharing media); recordings, frames and weights are kept out of Git.

Ethics frameworks (as SGP1): ACM and IEEE codes, UAE PDPL (Federal Decree-Law No. 45 of 2021), NIST Privacy Framework,
ISO/IEC 27001 and 25010 principles (no formal certification). Informed consent from instructor and students; advisory
use only; no inner-state claims; association not cause; the LLM sees anonymised facts only.

Cost update (SGP1 table 5.1): second camera (500-1200 AED) and tripod (100-250 AED) per SGP1 estimates, or two phones;
LLM about USD 0.04 per report with Opus, less with Sonnet or Haiku.

---

## 13. Requirements status

| Req | Status | Note |
|---|---|---|
| FR1.1, 1.2 | DONE | CLI input with validation; no upload UI |
| FR2.1, 2.2 | DONE | Timestamped frames, source-relative times, plus real times |
| FR2.3 | DONE | YAML camera configuration and optional zone calibration |
| FR3.1 | DONE | Camera-context instructor role with tracking; unknown when ambiguous |
| FR3.2, 3.3 | PARTLY | Presence, posture, motion, position events with start/end/value/coverage. Explaining, board writing, interaction only in System B |
| FR4.1 | CHANGED | Tracker IDs instead of seat regions in System A (seat grid in System B) |
| FR4.2, 4.3 | PARTLY | Count, posture, motion in A; hand raise, head posture, phone only in B |
| FR5.1, 5.2 | PARTLY | Per-recording summary in A; window indicators in B |
| FR6.1-6.4 | PARTLY | Implemented in B on one camera; two-camera alignment by real time exists, joint analysis PLANNED |
| FR7.1-7.3 | PARTLY | JSON summary and review MP4 in A; readable report with evidence and LLM layer in B |
| NFR1 | PARTLY | Offline CPU processing; runtime measured only on smoke runs |
| NFR2 | PARTLY | Anonymous IDs, local processing; no access control (no UI) |
| NFR3 | PARTLY | CLI with clear errors; no dashboard |
| NFR4 | DONE | Any length, segments, indicator switches |
| NFR5 | DONE | Modular `src/` layers, saved evidence reusable without rerunning the model, deterministic configuration |

Suggested new requirements (SGP1 wording):
- FR 1.3 The system shall accept recordings from an instructor camera and a student camera of the same lecture,
  each with its own anonymous video identifier.
- FR 2.4 The system shall record, for each event, the real (wall-clock) time derived from the recording's metadata,
  and shall mark it unavailable when the metadata is missing.
- FR 3.4 The system shall assign roles from the camera context and shall keep a role unknown when it is ambiguous.
- FR 7.4 The system shall produce an annotated review video showing the detections and rule results behind its events.
- NFR 1.3 Events from the two cameras shall be alignable to within about one second.

---

## 14. Suggested SGP2 report outline (`[TODO: check against the official template and the existing draft]`)

1. Introduction: problem, objectives, deliverables achieved, structure
2. Background and related work: SGP1 review plus pose-based behaviour recognition, multi-object tracking, multi-camera
   classroom analytics, LLMs for grounded report writing
3. System design and implementation (draft exists: modules, technologies and standards, challenges and solutions)
4. Analysis and reporting layers: windows, lag analysis, statistics, LLM report with checks (System B), and the plan
   to run them on two-camera evidence
5. Testing and evaluation: unit tests, smoke runs, alignment tests, accuracy plan and results
6. Results and discussion: what can and cannot be concluded; limitations
7. Professional, ethical and sustainability considerations (updated for two cameras, review videos, LLM use)
8. Project management: planned vs actual timeline, task split
9. Conclusion and future work: validation on own recordings, hand raises and orientation on two cameras, fusion of
   both cameras, floor calibration, trained action models, charts and HTML report, dashboard, audio
10. Statement of contribution

---

## 15. Open decisions for the team

1. Whether System B's lag analysis and LLM layer will be ported onto System A's events before submission, or reported
   as a prototype plus plan.
2. Which student indicators the association analysis will use (System A's posture/motion/count now, or hand raises and
   orientation added first).
3. Replacement for the instructor-to-student distance indicator (floor calibration, coarse zone, or drop).
4. Own synchronised recordings with consent, and how many lectures.
5. Human labelling for accuracy figures.
6. Whether a dashboard or HTML report is in SGP2 scope.
7. Which Claude model to use, and whether to compare models.

---

## 16. SGP1 references available for reuse (renumber to fit the SGP2 list)

- Sharpe, Trotter, Hale (2025), micro-breaks and student concentration, Frontiers in Psychology.
- Brandmo, Gamlem (2025), students' perceptions of teacher feedback, Frontiers in Education.
- Li, Weng, Li, Zhang (2025), multimodal learning engagement assessment, Int. J. Human-Computer Interaction. (SGP2 [1])
- Kumar (2024), single camera vs multi-camera systems, e-con Systems blog (cite for the two-camera trade-off).
- Hui Wang (2025), AI visual analysis in classroom English teaching evaluation, Discover Artificial Intelligence.
- Trabelsi et al. (2023), real-time attention monitoring with YOLOv5 + DeepSORT, Big Data and Cognitive Computing. (SGP2 [2])
- Liu, Jiang, Jiang (2025), classroom behaviour recognition with computer vision: systematic review, Sensors. (SGP2 [3])
- Wang, Wu (2025), behavioural analysis for classroom instruction with computer vision.
- UNESCO, EMIS progress assessment tool.
- Savva, Scarinci, Dillenbourg (2018), recognising student facial expressions: a web application, EC-TEL.
- Broussard et al. (2021), teacher awareness interface in a VR classroom, IEEE VR.
- Osterholm, Widlund, Korhonen (2025), engagement and burnout crossover from teachers to students, Teaching and Teacher Education.

Still to add with full citations `[TODO]`: HAC / Newey-West, Bonferroni, Wilson interval, the Claude model used, the
COCO keypoint format.
