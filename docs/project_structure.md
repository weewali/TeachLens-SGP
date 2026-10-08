# TeachLens Project Structure

This document is the repository map for TeachLens. It defines where every type of work belongs, which language or format it uses, and the responsibility of each current or planned file.

The repository is currently an early computer-vision prototype. Existing datasets, recordings, generated runs, and the pose demo are preserved in place. 

## 1. Technology choices

| Area | Language or format | Why |
| --- | --- | --- |
| Pipeline, model inference, analysis, training, and command-line tools | Python 3.11+ (`.py`) | The current code and the Ultralytics, OpenCV, NumPy, pandas, SciPy, statsmodels, and Jinja2 libraries are Python-based. One executable language keeps the project easier to test and maintain. |
| Runtime and model configuration | YAML (`.yaml`) | Human-readable settings can change without editing Python. |
| Dataset annotations | YOLO text (`.txt`) | Ultralytics expects one normalized annotation row per object. |
| Event interchange | JSON Lines (`.jsonl`) | One timestamped event per line is streamable, debuggable, and easy to process with Python or external tools. |
| Summary tables | CSV (`.csv`) | Simple exchange format for metrics and manual review. |
| Report templates | Jinja2 HTML (`.html.j2`) and CSS (`.css`) | Jinja2 is already a dependency and produces portable reports that can later be printed to PDF. |
| Documentation | Markdown (`.md`) | Renders on GitHub and remains easy for the team to edit. |
| Diagrams | Mermaid inside Markdown; PNG only when required | Text diagrams remain reviewable in Git. |

JavaScript or TypeScript is not part of the initial pipeline. If TeachLens later gets an interactive web dashboard, it should be introduced as a separate `web/` application rather than mixed into the Python analysis code.

FFmpeg is an external command-line dependency used for media conversion and frame/audio extraction; it is not a project language.

## 2. Architecture and data flow

```text
recording + camera zones + run configuration
                    |
                    v
            preprocessing layer
       video metadata, frames, and audio
                    |
                    v
              inference layer
 detector -> tracker -> pose model
                    |
                    v
               event layer
 timestamped observations + confidence + coverage
                    |
                    v
              analysis layer
 visual, interaction, and quality metrics
                    |
                    v
             reporting layer
       charts + HTML/PDF-ready report
```

The important boundary is the event layer. Models should not write report prose directly. They should create timestamped evidence records; analysis code aggregates those records; reporting code presents the results and their limitations.

Every event should eventually include:

- `video_id`
- anonymous `track_id`, when applicable
- `role`: `teacher`, `student`, `both`, `classroom`, or `recording`
- `indicator_name`
- `start_time`, `end_time`, and `duration`
- observed value or event label
- model or rule confidence
- evidence-coverage or media-quality flag
- optional calibrated classroom zone

## 3. Repository tree

`[current]` means the path already contains project work. `[scaffold]` means the directory or configuration was added to reserve its role. `[planned]` means the named implementation file should be created when that feature is built; an empty Python file is deliberately not committed.

```text
TeachLens-SGP/
|-- README.md                              [current]
|-- requirements.txt                      [current]
|-- data.yaml                              [current]
|-- .gitignore                             [current]
|
|-- configs/
|   |-- default.yaml                       [scaffold]
|   |-- indicators.yaml                    [scaffold]
|   `-- zones.example.yaml                 [scaffold]
|
|-- data/
|   |-- raw_videos/                        [current, local only]
|   |-- raw_frames/                        [current, local only]
|   |-- calibration/                       [scaffold]
|   |-- manifests/                         [scaffold]
|   |-- samples/                           [current]
|   |-- dataset_v1/                        [current]
|   |-- dataset_v2/                        [current]
|   `-- Instructor_Student/                [current imported dataset]
|
|-- models/
|   |-- detection/                         [scaffold, local artifacts]
|   |-- pose/                              [scaffold, local artifacts]
|   `-- audio/                             [scaffold, local artifacts]
|
|-- src/
|   |-- preprocessing/
|   |   |-- video.py                       [planned]
|   |   `-- dataset.py                     [planned]
|   |-- inference/
|   |   |-- detection.py                   [planned]
|   |   |-- tracking.py                    [planned]
|   |   |-- pose.py                        [planned]
|   |   `-- speech.py                      [planned]
|   |-- analysis/
|   |   |-- events.py                      [planned]
|   |   |-- visual_indicators.py           [planned]
|   |   |-- speech_indicators.py           [planned]
|   |   |-- interaction_indicators.py      [planned]
|   |   |-- quality.py                     [planned]
|   |   `-- aggregates.py                  [planned]
|   |-- reporting/
|   |   |-- charts.py                      [planned]
|   |   |-- renderer.py                    [planned]
|   |   `-- templates/                     [scaffold]
|   |       |-- report.html.j2             [planned]
|   |       `-- report.css                 [planned]
|   |-- training/
|   |   |-- train_detector.py              [planned]
|   |   `-- evaluate_detector.py           [planned]
|   |-- utils/
|   |   |-- config.py                      [planned]
|   |   |-- paths.py                       [planned]
|   |   `-- logging.py                     [planned]
|   `-- pipeline.py                        [planned]
|
|-- scripts/
|   |-- extract_frames_ffmpeg.py           [current]
|   `-- pose_action_demo.py                [current local prototype]
|
|-- tests/
|   |-- unit/                              [scaffold]
|   |-- integration/                       [scaffold]
|   `-- fixtures/                          [scaffold]
|
|-- notebooks/                             [current placeholder]
|-- docs/                                  [current]
|-- outputs/
|   |-- events/                            [scaffold, generated]
|   |-- reports/                           [scaffold, generated]
|   |-- visualizations/                    [scaffold, generated]
|   `-- debug/                             [scaffold, generated]
|-- runs/                                  [current, generated]
|
|-- Instructor Recordings/                 [current legacy local folder]
|-- Student Recordings/                    [current legacy local folder]
|-- yolo11n.pt                             [current local pretrained weight]
`-- yolo11n-pose.pt                        [current local pretrained weight]
```

## 4. Root files

| File | Format | Responsibility |
| --- | --- | --- |
| `README.md` | Markdown | Gives the short project description and links to this detailed map. Keep it brief so a new contributor can orient quickly. |
| `requirements.txt` | pip requirements | Lists Python runtime dependencies. Pin tested versions before a reproducible release; add development-only tools to a future `requirements-dev.txt`. |
| `data.yaml` | Ultralytics YAML | Tells Ultralytics where the currently selected training images and labels live and maps numeric class IDs to names. It currently describes `dataset_v2` as a one-class `person` dataset. |
| `.gitignore` | Git pattern file | Prevents recordings, raw frames, model weights, generated outputs, caches, IDE files, and run artifacts from being committed while retaining directory placeholders. |
| `yolo11n.pt` | PyTorch weight | Local Ultralytics object-detection checkpoint used by experiments. It is intentionally ignored by Git. New downloaded weights should preferably be placed under `models/detection/`. |
| `yolo11n-pose.pt` | PyTorch weight | Local Ultralytics pose checkpoint used by the pose demo. It is intentionally ignored by Git. New pose weights should preferably be placed under `models/pose/`. |

## 5. Configuration files

| File | Format | Responsibility |
| --- | --- | --- |
| `configs/default.yaml` | YAML | Central default paths, pretrained/trained model locations, inference thresholds, event timing, report options, and privacy-safe behavior. Code should load this before applying command-line overrides. |
| `configs/indicators.yaml` | YAML | Enables indicators independently and owns indicator-specific thresholds. Phase 1 visual indicators are enabled; advanced visual indicators are disabled. |
| `configs/zones.example.yaml` | YAML | Example normalized polygon calibration for a fixed classroom camera. Copy it to a camera-specific file and edit the points; never treat these example polygons as real calibration. |
| `configs/*.local.yaml` | YAML, future and ignored if added | Machine- or experiment-specific overrides such as local absolute paths. These should not contain credentials or be committed. |

Configuration precedence should be: command-line value > local override > `configs/default.yaml`.

## 6. Data directories and file contracts

| Path | Format | Responsibility |
| --- | --- | --- |
| `data/raw_videos/` | MP4/MOV/MKV, local only | Canonical location for source recordings. New recordings should be copied here under clear, anonymous names. |
| `data/raw_frames/` | JPG/PNG, local only | Frames extracted from raw recordings. Filenames should contain a video ID and zero-padded frame number. |
| `data/calibration/` | YAML/JSON | One camera-specific zone file per fixed camera. Store geometry only, not recordings. |
| `data/manifests/` | CSV/JSON | Records stable video IDs, source provenance, allowed use, split assignment, and optional calibration ID without putting identity into filenames. |
| `data/samples/` | Small images and labels | Tiny, safe fixtures used in documentation, smoke tests, and examples. Do not turn this into a second full dataset. |
| `data/dataset_v1/` | YOLO images and labels | First versioned dataset. Treat its class mapping as immutable; fix labels in a new version if the meaning changes. |
| `data/dataset_v2/` | YOLO images and labels | Current dataset selected by root `data.yaml`. Train/validation/test image folders must have corresponding label folders. |
| `data/Instructor_Student/` | YOLO images and labels | Imported or separately prepared instructor/student dataset. Record its origin and class mapping before combining it with another dataset. |
| `data/**/images/{train,val,test}/` | JPG/PNG | Model inputs split before training. Frames from the same source recording must not be split across train and validation/test sets. |
| `data/**/labels/{train,val,test}/` | YOLO TXT | One annotation file per image with the same base filename. `classes.txt` is informative; the dataset YAML remains the training source of truth. |
| `*.cache` inside datasets | Ultralytics cache | Generated acceleration files. They can be deleted and regenerated; they are not authoritative labels. |

The top-level `Instructor Recordings/` and `Student Recordings/` directories are preserved because they already contain local source media. They are legacy import locations. New code should read from `data/raw_videos/`, and the large media files should remain uncommitted.

## 7. Source-code responsibilities

Create a Python file below only when its first behavior and tests are implemented. Every source directory should become a Python package by adding `__init__.py` alongside the first module.

### `src/preprocessing/`

| Planned file | Responsibility |
| --- | --- |
| `video.py` | Probe video metadata, validate codecs/FPS/duration, sample frames, and expose timestamp-preserving frame iteration. This becomes the reusable home for logic currently demonstrated in `scripts/extract_frames_ffmpeg.py`.
| `dataset.py` | Validate image/label pairs, enforce class mappings, detect train/validation leakage by source video, and build immutable versioned splits. |

### `src/inference/`

| Planned file | Responsibility |
| --- | --- |
| `detection.py` | Load the chosen YOLO detector and return person boxes, class/role predictions, and confidences in a model-independent record shape. |
| `tracking.py` | Associate detections over time with anonymous temporary IDs using ByteTrack or BoT-SORT and report track-quality information. IDs must reset between recordings. |
| `pose.py` | Run pose estimation and convert keypoints into normalized, confidence-aware observations. It must not directly claim attention, engagement, or emotion. |


### `src/analysis/`

| Planned file | Responsibility |
| --- | --- |
| `events.py` | Define and validate the canonical timestamped event schema shared by inference, analysis, and reporting. Serialize events to JSONL. |
| `visual_indicators.py` | Convert tracks and pose observations into defensible visual events such as presence, counts, hand raises, position, zones, or movement. |
| `interaction_indicators.py` | Link teacher prompts, student responses, hand raises, feedback, orientation, and proximity into cross-modal interaction events. |
| `quality.py` | Measure blur, brightness, coverage, audio quality, ASR confidence, occlusion, and tracking reliability; provide the validity denominator for every metric. |
| `aggregates.py` | Produce lesson-level counts, rates, percentages, distributions, and timelines from events without changing the underlying evidence. |

### `src/reporting/`

| Planned file | Responsibility |
| --- | --- |
| `charts.py` | Build deterministic timelines, heatmaps, and summary charts from aggregate tables. Save plots under `outputs/visualizations/`. |
| `renderer.py` | Validate report inputs, render the Jinja2 template, attach evidence/limitations, and write HTML under `outputs/reports/`. PDF export may be added behind the same interface. |
| `templates/report.html.j2` | Own the semantic HTML report layout and Jinja2 placeholders. It should display observation, denominator, confidence, and limitation together. |
| `templates/report.css` | Own print-friendly report styling. No analysis logic or hard-coded metrics belongs here. |

### `src/training/`

| Planned file | Responsibility |
| --- | --- |
| `train_detector.py` | Validate the dataset contract, set seeds, launch Ultralytics training, and save the exact configuration and metrics with each run. |
| `evaluate_detector.py` | Evaluate a checkpoint on a held-out set, save class-level metrics and error examples, and compare against a declared baseline. |

Pose action rules should first be evaluated using the pretrained pose model. Add a separate action-training module only when the team has a labeled temporal-action dataset.

### `src/utils/`

| Planned file | Responsibility |
| --- | --- |
| `config.py` | Load YAML, merge overrides, validate required keys and ranges, and return a typed configuration object. |
| `paths.py` | Resolve repository-relative paths consistently and create only authorized output directories. |
| `logging.py` | Configure console/file logs with timestamps, stage names, video IDs, and run IDs while excluding personal data. |

### `src/pipeline.py`

This planned module will orchestrate one analysis run: load configuration, validate input, call preprocessing and inference, save events, aggregate indicators, and render the report. It should coordinate modules rather than contain detector, indicator, or plotting logic itself.

## 8. Scripts

Scripts are thin developer-facing entry points. Reusable behavior belongs under `src/`; a script should parse arguments and call that behavior.

| File | Language | Responsibility |
| --- | --- | --- |
| `scripts/extract_frames_ffmpeg.py` | Python | Current standalone FFmpeg wrapper that extracts one image every N seconds. Later, keep the CLI here but move reusable media logic into `src/preprocessing/video.py`. |
| `scripts/pose_action_demo.py` | Python | Current experimental YOLO pose + ByteTrack hand-raise demo. It is a proof of concept, not the production pipeline. |
| `scripts/train.py` | Python, future | Thin CLI that loads config and calls `src/training/train_detector.py`. |
| `scripts/analyze.py` | Python, future | Thin CLI that analyzes one recording through `src/pipeline.py`. |
| `scripts/report.py` | Python, future | Re-renders a report from saved events/aggregates without rerunning expensive models. |

## 9. Tests

| Path | Language/data | Responsibility |
| --- | --- | --- |
| `tests/unit/` | Python/pytest | Fast tests for one rule or function: coordinate normalization, hand-raise logic, event merging, configuration validation, and aggregation. |
| `tests/integration/` | Python/pytest | Small end-to-end checks across media loading, model output adapters, event storage, and report rendering. Heavy full-video benchmarks do not belong here. |
| `tests/fixtures/` | Tiny media/config/event samples | Small, non-sensitive, redistributable inputs with known expected outcomes. Never place real classroom recordings here. |

Test filenames should mirror source modules, for example `src/analysis/quality.py` -> `tests/unit/test_quality.py`.

## 10. Model and generated-output directories

| Path | Typical format | Responsibility |
| --- | --- | --- |
| `models/detection/` | `.pt`, `.onnx` | Downloaded or trained person/role detector checkpoints. |
| `models/pose/` | `.pt`, `.onnx` | Pose checkpoints. |
| `runs/` | generated folders | Raw framework experiment output. It is disposable and ignored; final promoted metrics should be summarized elsewhere. |
| `outputs/events/` | `.jsonl` | Canonical per-video timestamped evidence records. |
| `outputs/reports/` | `.html`, later `.pdf` | Human-readable final reports. |
| `outputs/visualizations/` | `.png`, `.svg`, `.csv` | Charts, heatmaps, and their source tables. |
| `outputs/debug/` | annotated media/logs | Temporary overlays and diagnostics used to inspect incorrect predictions. |

Model files and generated outputs are local artifacts and remain ignored by Git. The `.gitkeep` files preserve the intended empty directory layout.

## 11. Documentation files

| File | Responsibility |
| --- | --- |
| `docs/project_structure.md` | This architecture map and the source of truth for file ownership. Update it when a boundary or major file changes. |
| `docs/video_indicator_catalog.md` | Full catalog of defensible visual, audio, transcript, interaction, and quality indicators plus claims TeachLens must avoid. |
| `docs/annotation_guidelines.md` | Label names and bounding-box rules for the reviewed two-class annotation workflow. |
| `docs/annotation_review.md` | Review checklist and audit table for labeled batches. |
| `docs/dataset_preparation_notes.md` | Current notes for placing images/labels and splitting a dataset. |
| `docs/model_training_notes.md` | Detailed manual Ultralytics training, validation, prediction, and troubleshooting commands. |
| `docs/pose_training_rules.md` | Reserved for reviewed pose/action rules and their validation evidence. It is currently empty. |
| `docs/selected_videos.md` | Approval register for recordings selected for dataset preparation. Do not put sensitive identity data in it. |
| `docs/team_workflow.md` | Team Git workflow, commit guidance, and the rule against committing large artifacts. |
| `docs/video_sources_instructor.md` | Provenance and suitability notes for candidate instructor-focused source videos. |
| `docs/video_sources_students.md` | Provenance and suitability notes for candidate student-focused source videos. |
| `docs/diagrams/` | Exported diagrams only when Markdown/Mermaid cannot express them. Include an editable source when possible. |

## 12. Notebook policy

`notebooks/` is for short-lived exploration, data checks, and plots. A notebook may call reusable code from `src/`, but production logic must not exist only in a notebook. Name notebooks with a sequence and purpose, such as `01_dataset_audit.ipynb`, then move proven logic into tested Python modules.

## 13. Naming and ownership rules

- Use `snake_case.py` for Python modules and `test_<module>.py` for tests.
- Use stable anonymous video IDs such as `V001`; do not place student or teacher names in filenames.
- Treat dataset versions and their class-ID mappings as immutable contracts.
- Keep paths relative to the repository in committed YAML; machine-specific absolute paths belong in local overrides.
- Keep scripts thin, reusable computation in `src/`, settings in `configs/`, and generated artifacts in `outputs/` or `runs/`.
- Do not report internal states such as attention, engagement, emotion, intent, or understanding as facts. Report observable behavior, evidence coverage, confidence, and limitations.
- Never commit full classroom recordings, extracted frames, trained weights, or generated reports containing sensitive evidence.

## 14. Recommended implementation order

1. Reconcile the one-class versus two-class dataset contract and create the correct dataset YAML without changing old label meanings.
2. Move reusable frame/video logic into `src/preprocessing/video.py` and add unit tests.
3. Implement the canonical event model in `src/analysis/events.py` and write JSONL under `outputs/events/`.
4. Split the pose demo into detection, tracking, pose, and `visual_indicators.py` modules while preserving a thin demo script.
5. Add evidence coverage and recording-quality metrics before generating teaching conclusions.
6. Implement aggregation, charts, and an HTML report with explicit confidence and limitations.
7. Add the audio pipeline only after visual Phase 1 is testable on held-out recordings.

This order produces a reviewable vertical slice before introducing advanced semantic indicators or a web interface.
