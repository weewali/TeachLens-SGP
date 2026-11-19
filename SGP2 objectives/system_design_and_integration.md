# System design and integration

![TeachLens pipeline](pipeline.png)

*Figure: TeachLens pipeline (System B), in the eight stages the web application shows while a lecture is analysed.
Two recordings of one lecture are aligned and sampled, people and objects are detected by pretrained YOLO26 models,
transparent rules turn the detections into observable behaviour, 10-second windows compare student behaviour across
instructor activities, the team's fusion layer tests links over time, and the result is reported as plain-English
feedback in a local web application. Drawn by `scripts/make_pipeline_diagram.py`.*

## 1. Design principles

1. **Evidence, then analysis, then reporting.** Each layer reads the saved output of the previous one. Models never
   write report text.
2. **Observable behaviour only.** Indicators describe what the camera sees ("heads up"), never inner states
   ("attentive").
3. **Unknown is not zero.** Hidden or unreadable posture stays unknown and is left out of percentages.
4. **Associations, not causes.** Statistics are corrected for autocorrelation and multiple testing.
5. **Local and anonymous.** Video never leaves the computer. Students are anonymous seat numbers, and faces are
   pixelated in the seat chart. Only anonymised numbers may reach the language model.
6. **Rebuildable.** Every report can be regenerated from saved files without processing the video again.

## 2. Architecture

| Stage | Input | Processing | Output |
|---|---|---|---|
| 1. Input and alignment | Instructor video and student video of the same lecture | Validate both files. Read each recording-start time from the metadata (ffprobe), or take a manual `--offset`. | Shared lecture time and offset (about ±1 s) |
| 2. Preprocessing | Both videos | FFmpeg cuts both to the shared time and standardises them (25 fps, at most 720p). One frame every 2 s is taken from each camera and paired by time. | `frames.csv`, frame images |
| 3. Detection | Paired frames | Instructor camera: YOLO26n-pose at 640 px; the most confident person in the teacher zone is the instructor. Student camera: YOLO26m-pose at 960 px, plus YOLO26s at 960 px for phones and books. Automatic anonymous seats. | `frame_detections.csv`, preview images |
| 4. Behaviour recognition | Detections | Rules on keypoints (section 3) turn frames into a state timeline for the instructor and for each seat. Runs of frames become events. | `events.csv`, two JSONL event files |
| 5. Window indicators | Timelines and events | 10-second windows: share of time explaining, at the board, moving or seated; student heads-up, phone and head-down-and-still rates; hand-raise counts; posture coverage | `window_features.csv` |
| 6. Association analysis | Windows; the two JSONL event files | Comparisons during each instructor activity (HAC test). The team's fusion layer on the two event files (since 19 November 2026; it replaced System B's own lag analysis): 5-second windows, student changes 5–30 s after each instructor feature, circular-shift test, Benjamini-Hochberg q-values, lagged OLS with Newey-West errors, averages around the moments an action started. Only the indicators System B measures are tested. | `fusion.json`, `fusion_windows.csv`, `fusion_links.csv` |
| 7. Report | All of the above | Rule-based positive and negative points, suggestions, evidence, limitations, glossary. `facts.json` holds anonymised numbers only. The fusion results become "links over time" points (`lag_facts.py`). | `final_feedback_report.txt`, `analysis_details.txt`, `facts.json`, `links/report_points.json` |
| 8. Optional AI rewrite | `facts.json`, `fusion.json` | Claude Opus 5.5 rewrites the feedback points and the links over time. Code checks the numbers and the wording, with one retry and a fallback. | `llm_output.json`, `links/report_points.json`, checked report text |
| Web application | A lecture title, the two videos and optional context; saved results | Upload (lecture title and the two videos; optional topic, slides, activities in class with their duration, additional information), progress by stage, tabbed report, delete | Pages served on the local computer; `context.json` with the instructor's context |

## 3. Behaviour rules (System B)

**Instructor (instructor camera):**

| State | Rule |
|---|---|
| Facing the class | Nose and an eye visible, with the nose near the middle of the shoulders. The camera sits on the students' side, so a visible face faces the class. |
| Interacting | Facing the class with a hand raised |
| At the board | Inside the calibrated board region and not facing the class. Not measured when no board region is set. |
| Moving | Distance travelled between samples; "stationary" below 1% of the frame per sample |
| Seated | Thigh short beside the torso (ratio < 0.5); needs hips and knees in view |

**Students (student camera):**

| State | Rule |
|---|---|
| Heads up | Nose clearly above the shoulder line (at least 0.25 shoulder widths) |
| Head down and still | Head down or face hidden, with shoulders moving less than 0.15 shoulder widths between samples, for 30 s or more |
| Hand raised | Wrist above the shoulder by 5% of the person's height, in 2 consecutive samples or more |
| Phone or book in view | A YOLO26s phone or book whose centre lies inside the student's box |

**Anonymous seats:**
- **Finding seats:** shoulder midpoints are grouped over the whole lecture. A place becomes a seat when someone sat
  there in at least 10% of the samples.
- **Matching people to seats:** one person per seat per sample, within one shoulder width. Moving up counts half, so
  a student who stands up stays in their seat.
- **Leaving out false detections:** a detection that fits no seat is excluded, for example legs under a desk or a
  chair seen as a person.
- **Numbering:** seats are numbered row by row from the top left of the student camera's view.

## 4. Data design

| File | Contents |
|---|---|
| `sessions/<lecture>/tables/frames.csv` | Paired frame times of both cameras |
| `.../tables/frame_detections.csv` | Every detection: box, cues, role, seat |
| `.../tables/events.csv` | Events with start, end and seat |
| `.../tables/window_features.csv` | 10-second window indicators |
| `.../reports/fusion.json`, `.../tables/fusion_windows.csv`, `fusion_links.csv` | Association statistics (the team's fusion layer format, plus what System B did not measure) |
| `.../reports/links/report_points.json` | The links over time in plain language |
| `.../reports/facts.json` | Anonymised facts for the AI step (schema 2): each fact's numbers, direction, evidence strength and the detections it rests on |
| `outputs/events/<lecture>_instructor_view.jsonl`, `..._student_view.jsonl` | Team event format: role, indicator, start and end in the original video, real (wall-clock) time, value, confidence (null for rule states), coverage, seat in `zone` |

## 5. Web application design (Model–View–Controller, Flask + Jinja2)

- **Models:**
  - `lecture.py`: where each lecture's videos, context and results are stored; checks the activities an instructor
    marks before any video is saved.
  - `job.py`: runs the pipeline in the background and records stages 1–7, plus stage 8 for the AI rewrite.
  - `report.py`: gathers the report data from saved files.
  - `charts.py`: the instructor position heatmap and heads up by seat. The camera frame behind the seat chart has
    every head it can locate pixelated: everyone detected in the first 10 seconds, plus where each seat's head usually
    was. Only this picture is changed; the analysis reads the original frames.
  - `stat_charts.py`: data for the lecture timeline and the activity comparison.
- **Views:** Jinja2 pages for the lecture list, the upload form, live progress, the report and the delete
  confirmation.
- **Upload form:**
  - Required: a lecture title (so the lecture can be found later) and the two videos. A missing title or video is
    refused before anything is saved.
  - Optional, in two sections: "Additional context" (the topic and the lecture slides) and "Activities in class"
    (activities that were not teaching, then additional information as free text). Each activity (quiz or test, group
    or pair work, video, break) is added with one button and one box for how many minutes it lasted (whole minutes, at
    most 180 in total).
  - The optional context is kept with the report and is not yet used by the analysis **[TODO]**.
- **Report page tabs:**
  - Feedback, with the context the instructor gave;
  - Charts: two statistical charts with table views and two heatmaps ("Instructor Movement", "Student Seats");
  - Moments: a filterable table;
  - Key terms.

  Suggestions and limitations stay in `analysis_details.txt` and are not shown on the page. The links over time (the
  fusion layer's results) are still computed and saved (`fusion.json`, `links/report_points.json`, and section 3 of
  `analysis_details.txt`), but since 19 November 2026 they have no tab of their own **[TODO: decide whether and how
  instructors see them]**.
- **Controller:** `lectures.py` with the routes `/`, `/lectures/new`, `POST /lectures`, `/lectures/<id>/progress`,
  `/lectures/<id>/status`, `POST /lectures/<id>/run` (run again), `/lectures/<id>/delete` (confirmation page, then
  POST), `/lectures/<id>` and `/lectures/<id>/charts/<name>`.
- **Security:**
  - The server listens only on 127.0.0.1, and chart files are served by exact name only.
  - A delete sent from another website is refused.
  - The AI rewrite is a deployment setting, so instructors never handle API keys.

## 6. Integration

| Interface | How it is integrated | Status |
|---|---|---|
| Instructor camera ↔ student camera | Recording-start metadata, or a manual offset. Frames are paired by time on one lecture clock. | Done; about ±1 s |
| System A ↔ System B | System B writes the same JSONL event format, using an identical copy of System A's `events.py`: role `teacher`, times in each original video, real times, seats in `zone`. | Format shared; the two pipelines are not merged **[TODO: team decision]** |
| Evidence ↔ fusion layer ↔ reporting | System B writes the indicators the team's fusion layer reads (its names and labels, from System B's own rules) and runs that layer on its two event files. The reporting layer (`src/reporting/lag_facts.py`) reads the fusion output (`<name>.fusion.json`). | Connected (18–19 November 2026); tested on synthetic data with planted links and on the two recorded test lectures |
| Pipeline ↔ web application | The web job runs `Main Code.py` as a separate process and follows its stage messages. The report page reads saved files only. | Done |
| Instructor's context ↔ analysis | The upload form stores each activity that was not teaching and its duration in `context.json`, but not when it happened, so the analysis cannot tell which minutes it covered. Options: a note beside the student figures it affects, or a stretch of matching length found automatically and confirmed by the instructor. Once the minutes are known, student states there can be marked unknown (the fusion layer already skips unknown time) and left out of the comparisons and moments. | Stored and shown; not yet used **[TODO: decide]** |
| Reporting ↔ language model | `facts.json` → Claude Opus 5.5 → checks → `llm_output.json`. Rule-based text on any failure. | Implemented; first real call pending **[TODO]** |

## 7. Main design decisions

| Decision | Alternative | Reason |
|---|---|---|
| Two static cameras | One camera covering the whole room | One view rarely shows the instructor and the students clearly at the same time, and a moving or edited view broke zone-based roles in a single-camera test. |
| Pretrained pose models with rules | Training action-recognition models | No labelled classroom action dataset. Rules are explainable and testable, and the team can train later once labels exist. |
| Different model sizes per camera | One model for both | The instructor is one large person (nano is enough). Students are many, small and partly hidden (medium at 960 px). |
| 2-second sampling, 10-second windows | 1-second sampling | CPU time. Hand raises still need 2 samples (4 s). |
| Automatic seats | A fixed 3×4 grid | The grid merged two students and created false seats on the real lecture. |
| HAC tests; for the lags, the team's circular-shift test with Benjamini-Hochberg q-values | Plain correlations and t-tests | Neighbouring windows are alike, so plain p-values overstate certainty. Shifting the instructor's timeline keeps that likeness and breaks only the alignment. |
| Rule-based text plus a checked AI rewrite | The AI writes the report freely | Every sentence must trace back to a measured number. |
| Local web application | Cloud hosting | Videos show real people, so processing stays on the local computer. |
| Seat chart on a camera frame with pixelated faces | The raw frame, or no picture | The instructor can still see where each seat is, but no student can be recognised. Only the chart is changed; the analysis reads the original frames. |
| Fixed statistical settings; behaviour thresholds tuned later from human labels | Letting the system tune everything on each lecture | Without human labels there is no correct answer to tune towards, and tuning the statistics on the data would produce false findings. |
