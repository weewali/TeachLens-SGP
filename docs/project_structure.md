# TeachLens Project Structure

This document is the repository map for TeachLens: where every kind of work
belongs, its language or format, and the responsibility of each current or
planned file. Updated 2026-11-20 to match the repository after the lesson
pipeline, fusion and lag analysis, LLM facts, and dated output folders.

Status in one line: the video pipeline works end to end, from a lesson's two
recordings (student camera, instructor camera) to fused lag statistics and a
`facts.json` ready for an LLM; the **LLM feedback and report-generation
stages are the next work** (AGENTS.md section 17, docs/llm_integration.md).
Nothing has been validated against human labels yet.

## 1. Technology choices

| Area | Language or format | Why |
| --- | --- | --- |
| Pipeline, models, analysis, statistics, command-line tools | Python 3.10 (`.py`), Conda environment `TeachLens` | One executable language; easy to test. |
| Runtime settings, calibration, hypotheses | YAML (`.yaml`) | Human-readable; changes without editing Python. |
| Evidence | JSON Lines (`events.jsonl`) | One timestamped event per line; streamable and debuggable. |
| Summaries, analysis results, LLM input | JSON (`summary.json`, `fusion.json`, `facts.json`, `lesson.json`) | Structured, readable, versionable schemas. |
| Tables | CSV (`windows.csv`, `links.csv`, manifests) | Simple exchange with spreadsheets. |
| Review media | MP4 (H.264 through MSMF on Windows) | Visual checks of what the rules saw. |
| Charts | PNG from Matplotlib | Regenerated from saved tables. |
| Dataset annotations | YOLO text (`.txt`) | Ultralytics format (training is not part of the current pipeline). |
| Reports (planned) | Jinja2 HTML (`.html.j2`) + CSS, later PDF | Jinja2 is already a dependency. |
| Documentation | Markdown (`.md`), Mermaid diagrams | Renders on GitHub. |

External tools: FFmpeg/ffprobe (recording-time tags, lossless trims). A later
interactive dashboard belongs in a separate `web/` application, not mixed
into the Python analysis code. Audio is out of scope for the SGP's
physical-indicator work.

## 2. Architecture and data flow

```text
student_view video ----\                                    (src/preprocessing/alignment.py)
                         >-- line up by real time, keep the stretch both cameras recorded
instructor_view video -/
        |  per camera (src/pipeline.analyze_video)
        v
preprocessing  src/preprocessing/video.py        sampled frames (1/s, faces 5/s), GPU decoding (D3D11)
inference      src/inference/pose.py             YOLO26 pose + ByteTrack -> boxes, COCO-17 joints
               src/inference/tracking.py         stitch IDs of people moving between samples
               src/inference/face_mesh.py        MediaPipe face landmarks (student camera only)
               src/inference/hands.py            MediaPipe gesture recognizer on wrist crops
               src/inference/objects.py          COCO detector: phone, laptop, book
rules          src/analysis/visual_indicators.py List A (presence, count, posture, motion, zones ...)
               src/analysis/teacher_indicators.py List B (teacher facing, board, pointing, gestures ...)
               src/analysis/student_indicators.py List C (hand raises, seats, device use, alertness ...)
               src/analysis/interaction_indicators.py, facial_actions.py
evidence       src/analysis/events.py            events.jsonl (+ real wall-clock times)
summary        src/analysis/aggregates.py        summary.json
review         src/reporting/annotated_video.py  review.mp4
        |  per lesson (src/pipeline.analyze_lesson)
        v
links          src/analysis/cross_camera.py      cross_camera.json
fusion         src/analysis/fusion.py            fusion/windows.csv (5 s windows, T_ and S_ features)
statistics     src/analysis/lag_analysis.py      fusion/fusion.json (hypotheses, lags, comparisons, models)
charts         src/reporting/charts.py           fusion/timeline.png
LLM input      src/reporting/facts.py            facts.json  (schema 1 of the LLM stage)
video          src/reporting/side_by_side.py     side_by_side.mp4
index          src/pipeline.py                   lesson.json
        |  several lessons (src/pipeline.pool_lessons)
        v
pooled         lag_analysis.pooled_analyze       outputs/pooled/<stamp>_<name>/pooled.json, facts.json
        |
        v  NEXT (not built here yet)
LLM feedback   reads facts.json -> strengths, weak points, suggestions, checked in code
report         renders lesson.json + facts + LLM text + charts/video links -> HTML/PDF
```

The boundaries: models never write prose; rules write evidence; analysis
reads saved evidence; the LLM reads only `facts.json`; reports regenerate
from saved files without rerunning models.

Every event (`events.jsonl` line) has `video_id`, `track_id` (anonymous, per
recording), `role` (teacher, student, unknown, both, classroom, recording),
`indicator_name`, `start_time`/`end_time`/`duration` (seconds into the video),
`start_real_time`/`end_real_time` (wall clock, ISO 8601), `value`,
`confidence` (model confidence or null for rules), `coverage` (adequate,
limited, unknown), and `zone`.

## 3. Repository tree

`[git]` = committed; `[local]` = on disk only, ignored by Git (copy it to a
new machine by hand); `[planned]` = to be created when the feature is built.

```text
TeachLens-SGP/
|-- AGENTS.md                         [git] handoff and rules for AI agents (read first)
|-- README.md                         [git] short project description
|-- requirements.txt                  [git] Python dependencies (unpinned)
|-- data.yaml                         [git] one-class person dataset contract (training only)
|-- .gitignore                        [git]
|
|-- configs/
|   |-- default.yaml                  [git] all runtime defaults: cameras, inference, analysis, teacher,
|   |                                       faces, students, tracking, objects, fusion (+ hypotheses), review
|   |-- indicators.yaml               [git] on/off switches for every indicator (Lists A, B, C, faces)
|   |-- sgp_room.yaml                 [git] the scenario room: instructor-camera zones/targets
|   |-- bytetrack_1fps.yaml           [git] tracker settings for 1 sample per second
|   |-- web_sample_cameras.yaml       [git] override for downloaded clips with moving cameras
|   |-- howtospeak_4500/4718/4748.yaml [git] zone overrides for web-sample segments
|   `-- zones.example.yaml            [git] calibration template
|
|-- data/
|   |-- manifests/sgp_scenario_videos.csv  [git] scenario recordings: files, real starts, sync windows
|   |-- calibration/
|   |   |-- sgp_room_instructor_view.yaml  [git] zones (front_left, front_center, desk_area) and
|   |   |                                        targets (screen, board, desk) of the scenario room
|   |   `-- howtospeak_*_instructor_view.yaml [git] web-sample calibrations
|   |-- raw_student_videos/           [local] own scenario recordings, student camera (sgp_<scenario>_student.MOV)
|   |-- raw_instructor_videos/        [local] own scenario recordings, instructor camera (+ unedited/)
|   |-- raw_videos/                   [local] web samples and clock tests (clock_test_02.MOV is private)
|   |-- person_detection/combined.yaml [git] dataset YAML; the images are [local]
|   |-- raw_frames/, samples/         [local] / placeholders
|
|-- models/                           [local] weights, never downloaded by the code
|   |-- pose/yolo26n-pose.pt          YOLO26 pose (all pose rules)
|   |-- detection/yolo26n.pt          COCO detector (phones, laptops, books; isolated object runs)
|   |-- face/face_landmarker.task     MediaPipe face landmarks (students only)
|   `-- hands/gesture_recognizer.task MediaPipe gestures (hand raises, response gestures)
|
|-- src/
|   |-- pipeline.py                   [git] analyze_video, detect_objects, analyze_lesson, pool_lessons,
|   |                                       link_runs, fuse_and_write, review_timing
|   |-- preprocessing/
|   |   |-- video.py                  [git] sampled frames, segments, hardware decoding
|   |   |-- alignment.py              [git] line up a lesson's two recordings; CameraRun plans; manifest
|   |   |-- recording_time.py         [git] recording-start tags via ffprobe (GPS never printed)
|   |   |-- camera_motion.py          [git] pan/zoom compensation (web samples only)
|   |   `-- dataset.py                [planned] dataset audits and splits
|   |-- inference/
|   |   |-- pose.py                   [git] YOLO26 pose + ByteTrack adapter, PosePerson, joint groups
|   |   |-- tracking.py               [git] TrackStitcher
|   |   |-- face_mesh.py              [git] MediaPipe face landmarks on pose-based crops
|   |   |-- hands.py                  [git] MediaPipe gesture recognizer
|   |   `-- objects.py                [git] COCO object detector
|   |-- analysis/
|   |   |-- events.py                 [git] Event schema, atomic JSONL, real times
|   |   |-- visual_indicators.py      [git] List A rules, FoundationAnalyzer, desk posture
|   |   |-- teacher_indicators.py     [git] List B teacher rules
|   |   |-- student_indicators.py     [git] List C student rules
|   |   |-- interaction_indicators.py [git] head targets, orientation proxy, hand-raise outcomes
|   |   |-- facial_actions.py         [git] smile, brows, eyes closed, yawn, talking, nods
|   |   |-- aggregates.py             [git] summary.json from saved events
|   |   |-- cross_camera.py           [git] timing links between the cameras
|   |   |-- fusion.py                 [git] window table on real time; on/off-task states
|   |   |-- lag_analysis.py           [git] hypotheses, lag tests, comparisons, models, pooling
|   |   `-- quality.py                [planned] media/evidence quality denominators
|   |-- reporting/
|   |   |-- annotated_video.py        [git] review.mp4 overlays (small grey HUD), live window
|   |   |-- side_by_side.py           [git] side_by_side.mp4
|   |   |-- charts.py                 [git] fusion/timeline.png
|   |   |-- facts.py                  [git] facts.json for the LLM stage
|   |   |-- llm_report.py             [planned] LLM feedback from facts.json (AGENTS.md section 17)
|   |   |-- renderer.py               [planned] HTML/PDF report
|   |   `-- templates/                [git placeholder] report.html.j2, report.css [planned]
|   |-- utils/
|   |   |-- config.py                 [git] YAML loading, overrides, validation, switches, zones
|   |   |-- output_layout.py          [git] dated output folders and file names
|   |   |-- paths.py                  [git] repository paths, file-in-use check
|   |   `-- logging.py                [planned]
|   `-- training/                     [git placeholder] detector training [planned]
|
|-- scripts/                          [git] thin command-line entry points (section 8)
|-- tests/
|   |-- unit/                         [git] 20 test files, 170 tests (unittest), no video or GPU needed
|   |-- integration/, fixtures/       [git placeholders]
|-- docs/                             [git] section 11
|-- notebooks/                        [git placeholder]
|-- outputs/                          [local] generated; only the .gitkeep placeholders are in Git
|   |-- lessons/<YYYY-MM-DD_HHMM>_<lesson>/   one lesson (section 10)
|   |-- runs/<YYYY-MM-DD_HHMM>_<run>/         single-camera runs
|   |-- pooled/<YYYY-MM-DD_HHMM>_<name>/      several lessons tested together
|   `-- logs/                                 batch logs
`-- runs/                             [local] Ultralytics/demo experiment output (repository root)
```

## 4. Root files

| File | Responsibility |
| --- | --- |
| `AGENTS.md` | The handoff for any AI agent or new teammate: decisions, rules, environment, every module, results, and what comes next. Read it first; keep it current. |
| `README.md` | Short project description and links. |
| `requirements.txt` | Python dependencies (ultralytics, opencv-python, numpy, pandas, matplotlib, pyyaml, scipy, statsmodels, jinja2). mediapipe is installed separately without dependencies (AGENTS.md section 5). |
| `data.yaml` | One-class `person` dataset contract for detector training (not used by the running pipeline). |
| `.gitignore` | Keeps recordings, frames, weights, outputs, and caches out of Git; keeps `.gitkeep` placeholders. `/runs/` (root only) is ignored; `outputs/**` content is ignored. |

## 5. Configuration files

| File | Responsibility |
| --- | --- |
| `configs/default.yaml` | Every runtime default: `video` (1 sample/s, hardware decoding), `models`, `inference` (image size 1280), `cameras` (student_view, instructor_view policies, joints, posture method, filters), `analysis` (List A thresholds, desk posture), `teacher` (List B thresholds), `faces`, `students` (List C), `tracking`, `objects`, `fusion` (windows, lags, permutations, alpha, the 9 hypotheses), `review` (MP4, HUD, paths). Validated by `src/utils/config.py`. |
| `configs/indicators.yaml` | On/off switch per indicator; all implemented ones on by default. `--indicators` on the CLIs turns on only the named ones. |
| `configs/sgp_room.yaml` | Local override for the scenario room: points the instructor camera at `data/calibration/sgp_room_instructor_view.yaml`. Default `--local-config` of the scenario and lesson scripts. |
| `configs/bytetrack_1fps.yaml` | ByteTrack with looser matching for 1 sample/s. |
| `configs/web_sample_cameras.yaml`, `configs/howtospeak_*.yaml` | Overrides for the downloaded web samples (moving cameras, segment zones). |
| `configs/zones.example.yaml` | Template for a new room's calibration. |
| `configs/*.local.yaml` | Ignored machine-specific overrides (never credentials). |

Precedence: command-line value > `--local-config` YAML > `configs/default.yaml`.

## 6. Data directories and file contracts

| Path | Responsibility |
| --- | --- |
| `data/manifests/sgp_scenario_videos.csv` | One row per scenario recording: `video_id`, scenario (A1-A6, B1), take, camera, file, duration, `recording_start` (corrected real start), sync window (`sync_start_seconds`, `sync_end_seconds`). Read by `alignment.plan_from_manifest()` and `run_scenarios.py`. |
| `data/calibration/sgp_room_instructor_view.yaml` | Normalized polygons of the scenario room's instructor camera: zones (where the teacher stands) and targets (screen, board, desk). Recalibrate if the camera moves. |
| `data/raw_student_videos/`, `data/raw_instructor_videos/` | Local recordings of the scripted scenarios (docs/test_recording_scenarios.pdf, docs/scenario_test_videos.md). Never commit. |
| `data/raw_videos/` | Local web samples and clock tests. `clock_test_02.MOV` shows an identifiable presenter: private, not test data. |
| `data/person_detection/` | Roboflow exports (local) and `combined.yaml` for future detector training; audit before training. |

## 7. Source-code responsibilities

### `src/pipeline.py`

Orchestration only (no model or rule logic):

- `analyze_video()`: one camera recording -> run folder with `events.jsonl`,
  `summary.json`, `review.mp4`, `run.json`. Loads the models a camera's
  switches need, runs every sample through the rules, writes atomically.
- `detect_objects()`: the isolated COCO detector -> `objects.jsonl`,
  `objects_summary.json`, `objects_review.mp4`, `objects_run.json`.
- `analyze_lesson()`: two `CameraRun` plans -> both camera runs into
  `outputs/lessons/<stamp>_<lesson>/student|instructor/` (or `reuse` saved
  runs, `redo` one camera) -> `link_runs()` -> `fuse_and_write()` (fusion/ and
  facts.json) -> `side_by_side.mp4` -> `lesson.json`. Output clashes are
  checked before any analysis.
- `pool_lessons()`: the hypotheses over several fused lessons ->
  `outputs/pooled/<stamp>_<name>/pooled.json`, `facts.json`.
- `review_timing()`: when each review frame was taken (from `run.json`).

### `src/preprocessing/`

| File | Responsibility |
| --- | --- |
| `video.py` | `frames()` / `frames_with_subsamples()`: sampled frames with source timestamps, segments that seek frame-exactly, extra face frames, grab-only for unused frames, OpenCV hardware decoding with software fallback. |
| `alignment.py` | `CameraRun`; `read_manifest()`; `recording_window()`, `shared_range()`, `limit_range()`; `plan_from_videos()` (two files, tag starts or given starts) and `plan_from_manifest()` (scenario, tag, take). |
| `recording_time.py` | Recording-start/duration tags through ffprobe (Apple creationdate, 3GPP udta date, creation_time). |
| `camera_motion.py` | Moving-camera compensation for web samples (own recordings use static cameras). |

### `src/inference/`

| File | Responsibility |
| --- | --- |
| `pose.py` | YOLO26 pose + ByteTrack adapter; `PosePerson`, `Keypoint`, `joint_indices`; refuses missing weights (never downloads). |
| `tracking.py` | `TrackStitcher`: one ID for a person who moved between samples. |
| `face_mesh.py` | MediaPipe face landmarks (478 points, blendshapes, head angles) on crops from pose head joints; student camera only; crops stay in memory. |
| `hands.py` | MediaPipe gesture recognizer on wrist crops. |
| `objects.py` | COCO YOLO26 detector restricted to chosen classes. |

### `src/analysis/`

| File | Responsibility |
| --- | --- |
| `events.py` | `Event` schema, rounding, atomic JSONL writer, `with_real_time()`. |
| `visual_indicators.py` | List A rules (presence, count, roles, position, zones, entry/exit, visibility, sitting/standing incl. the desk rule, motion), duplicate filter, `FoundationAnalyzer` (calls the List B/C/face rule sets), `last_sample` for overlays. |
| `teacher_indicators.py` | List B: facing, head orientation, board/screen, writing, pointing, gestures, raised hand, torso lean, stationary episodes. |
| `student_indicators.py` | List C: hand raises, seats, out of seat, head down, alertness, device use, note-taking, reading, response gestures. |
| `interaction_indicators.py` | Head targets, orientation proxy (on/off target), hand-raise outcomes. |
| `facial_actions.py` | Per-student facial actions against each student's own baseline. |
| `aggregates.py` | `summary.json` from saved events. |
| `cross_camera.py` | Students who left for the front; the teacher in the student view (timing only). |
| `fusion.py` | `FEATURES` (T_ teacher and S_ student columns with desirable direction), `student_state()` (on/off task per second), `build_windows()`, `onsets()`, `fuse_runs()`. |
| `lag_analysis.py` | `cross_correlation`, `shift_test`, `benjamini_hochberg`, `comparison`, `lagged_model`, `peri_event`, `analyze()` (hypotheses first, exploratory links, findings), `between_lessons`, `pooled_analyze()`. |

### `src/reporting/`

| File | Responsibility |
| --- | --- |
| `annotated_video.py` | Review overlays (boxes, joints, IDs, roles, every rule state), small translucent HUD, MP4 writer, live window. |
| `side_by_side.py` | `ReviewTiming`, `write_side_by_side()`. |
| `charts.py` | `plot_fusion_timelines()`. |
| `facts.py` | `lesson_facts()`, `pooled_facts()`, `comparison_facts()`, `lesson_comparison_facts()`: facts.json schema 1. |
| `llm_report.py` | [planned] the LLM stage (AGENTS.md section 17). |
| `renderer.py`, `templates/` | [planned] the HTML/PDF report (AGENTS.md section 17). |

### `src/utils/`

| File | Responsibility |
| --- | --- |
| `config.py` | YAML loading, recursive overrides, validation, indicator switches, zones and targets. |
| `output_layout.py` | `<YYYY-MM-DD_HHMM>_<name>` folders, `RUN_FILES`, `LESSON_FILES`, `find_run`, `find_lesson`, `run_folder`, `new_folder`, `events_file`. |
| `paths.py` | `project_path()`, `file_in_use()`. |

## 8. Scripts

Thin entry points; reusable behaviour belongs in `src/`.

| Script | Responsibility |
| --- | --- |
| `scripts/analyze_lesson.py` | **The main command**: a whole lesson (two videos + `--name`, or `--scenario`/`--all-scenarios` + `--tag`) -> lesson folder; `--reuse`, `--redo`, `--pool`, `--start/--end`, `--dry-run`. |
| `scripts/analyze.py` | One camera recording -> run folder (`--indicators`, `--start/--end`, `--live`, `--path`). |
| `scripts/run_scenarios.py` | Camera runs over the scenario manifest (indicator tests, `--objects`). |
| `scripts/fuse_lesson.py` | Redo a lesson's fusion, statistics, and facts.json from saved runs. |
| `scripts/link_cameras.py` | Redo a lesson's cross-camera links. |
| `scripts/pool_lessons.py` | Test the hypotheses over several lessons (`--lessons` or `--all`). |
| `scripts/show_indicator.py` | Print one indicator of a saved run as a table (no video). |
| `scripts/detect_objects.py` | Isolated COCO object detection on one video. |
| `scripts/recording_time.py` | Show recording start/end and clock times of videos. |
| `scripts/trim_video.py` | Lossless cut that keeps the real recording start. |
| `scripts/organize_outputs.py` | One-off mover of old flat outputs into the dated layout. |
| `scripts/extract_frames_ffmpeg.py`, `scripts/pose_action_demo.py` | Dataset frame extraction; old YOLO11 hand-raise demo (not the pipeline). |
| `scripts/llm_report.py`, `scripts/report.py` | [planned] thin CLIs for the LLM and report stages. |

## 9. Tests

| Path | Responsibility |
| --- | --- |
| `tests/unit/` | 170 unittest tests, synthetic data, no video, model, or GPU: config, video sampling, recording time, events, aggregates, List A/B/C rules, faces, objects, annotated video, paths, scripts, fusion, lag analysis, lesson pipeline (mocked models), output layout, desk posture, facts.json schema, side-by-side timing. Run: `python -m unittest discover -s tests/unit`. |
| `tests/integration/`, `tests/fixtures/` | Placeholders; never put classroom recordings in fixtures. |

Test files mirror modules (`src/analysis/fusion.py` -> `tests/unit/test_fusion.py`); the 2026-11-19 additions are in `test_lesson.py` and `test_lesson_outputs.py`.

## 10. Model and generated-output directories

| Path | Responsibility |
| --- | --- |
| `models/pose/`, `models/detection/`, `models/face/`, `models/hands/` | Local weights (section 3); copy them to a new machine, the code never downloads them. |
| `outputs/lessons/<YYYY-MM-DD_HHMM>_<lesson>/` | One lesson: `lesson.json` (index), `student/` and `instructor/` (each `events.jsonl`, `summary.json`, `review.mp4`, `run.json`), `cross_camera.json`, `fusion/` (`fusion.json`, `windows.csv`, `links.csv`, `timeline.png`), `facts.json`, `side_by_side.mp4`; later the LLM and report outputs. Contains classroom footage. |
| `outputs/runs/<YYYY-MM-DD_HHMM>_<run>/` | One camera run made on its own (same files as a lesson's camera folder; `objects.*` for object runs). |
| `outputs/pooled/<YYYY-MM-DD_HHMM>_<name>/` | `pooled.json`, `facts.json` over several lessons. |
| `outputs/logs/` | Logs of long batches. |
| `runs/` (root) | Ultralytics experiment output; disposable. |

Every output file and JSON key is explained in docs/outputs_guide.md.

## 11. Documentation files

| File | Responsibility |
| --- | --- |
| `AGENTS.md` | Handoff for agents: decisions, environment, every module, results, next stages (sections 16-18). |
| `docs/project_structure.md` | This map. |
| `docs/outputs_guide.md` | Every output file, what uses it later, the lag formula, every JSON symbol. |
| `docs/llm_integration.md` | The LLM/UI stage on the pipeline_creation branch and the facts.json contract. |
| `docs/lesson_pipeline.md` | The one-command lesson pipeline: steps, options, run times, scenario results. |
| `docs/fusion_lag_analysis.md` | Fusion and lag analysis: features, hypotheses, tests, guardrails, pooling, results. |
| `docs/list_a_pose.md` | List A run guide, indicators, camera assumptions, desk posture, calibration. |
| `docs/list_b_teacher.md` | List B teacher indicators. |
| `docs/list_c_students.md` | List C student indicators, per-indicator test commands. |
| `docs/facial_actions.md` | Student facial actions from MediaPipe face landmarks. |
| `docs/object_detection.md` | Isolated COCO object detection. |
| `docs/scenario_test_videos.md` | The scenario recordings, their sync windows, test commands. |
| `docs/test_recording_scenarios.pdf` | The filming booklet: scenarios A1-D2, behaviour codes (T-SIT, T-READ ...), expected outcomes. |
| `docs/video_indicator_catalog.md` | Catalog of defensible indicators and claims to avoid. |
| `docs/team_workflow.md` | Team Git workflow. |
| `docs/annotation_guidelines.md`, `docs/annotation_review.md`, `docs/dataset_preparation_notes.md`, `docs/model_training_notes.md`, `docs/pose_training_rules.md` | Dataset and training notes (historical parts marked in AGENTS.md section 3). |
| `docs/selected_videos.md`, `docs/video_sources_instructor.md`, `docs/video_sources_students.md` | Source registers (candidates are not approvals). |

## 12. Naming and ownership rules

- `snake_case.py` modules, `test_<module>.py` tests; scripts thin, logic in `src/`, settings in `configs/`.
- Outputs only through `src/utils/output_layout.py`: one dated folder per lesson or run, never loose files in `outputs/`.
- Anonymous IDs only; no names in files, IDs, or facts; no face recognition or identity matching.
- Report observable proxies ("on task", "facing the class") with denominators and limitations, never attention, engagement, emotion, or intent as fact; associations, never causes.
- Never commit recordings, frames, weights, or generated outputs (they show people).

## 13. Implementation order from here

1. Done: per-camera pipeline (Lists A, B, C, faces, objects), one-command lessons, fusion and lag analysis, pooling, facts.json, side-by-side review, dated outputs.
2. Next: the LLM feedback stage and the HTML/PDF report (AGENTS.md section 17).
3. Then: validate the indicators the facts rest on against human labels of the scenario videos; run more and longer lessons; optional OpenVINO inference on the Intel Iris Xe.
