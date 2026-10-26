# TeachLens: instructions and project handoff for AI agents

Last verified: **2026-10-26**. Applies repository-wide. Portable handoff from code, docs, local outputs, and earlier TeachLens conversations. Recheck changeable details; distinguish implementation, experiments, and plans.

## 1. Read this first

TeachLens is a **Senior Graduation Project (SGP)** for classroom teaching analytics. Its eventual purpose is to turn recorded classroom observations into useful, human-reviewable teaching feedback, including strengths, areas for improvement, and practical suggestions. The current implementation is much narrower: a **local, video-only physical-indicator pipeline** that saves timestamped JSONL evidence, a JSON summary, and an annotated review MP4, with an optional live preview window.

The current executable path is:

```text
recorded video
  -> sampled OpenCV frames with source timestamps
  -> local YOLO26 Pose checkpoint + ByteTrack
  -> model-independent person boxes, anonymous IDs, COCO-17 keypoints
  -> conservative geometry and temporal rules
  -> outputs/events/<video_id>.list_a.jsonl
  -> outputs/reports/<video_id>.list_a.json
  -> outputs/debug/<video_id>.list_a.mp4   (same-run overlays; optional live window)
```

The most important decisions are:

- Use **one model class, `0: person`**, with **two camera contexts**, one aimed at students and one at the instructor. Do not restart the project around a teacher-versus-student detector.
- Use **YOLO26 Pose** for the implemented pipeline. Old YOLO11 files and the hand-raise demo are experiments, not the selected production path.
- A pose model already returns person boxes. A separate person detector is **optional**, to be justified by missed-person measurements, not a prerequisite.
- Camera context supplies an **assumed role**, not verified identity. Ambiguous roles must stay `unknown`.
- **Audio is future work and outside the senior project's agreed physical-indicator scope.** A broad catalog entry is not authorization to implement it.
- The latest implementation request was **List A only**. Hand raises are List C and remain outside this pipeline, despite a separate experimental demo.
- Preserve uncertainty. Unknown posture is not sitting; unknown motion is not still; no detection is not proof of absence; tracker IDs are not unique people.
- Keep the implementation **Python-first, modular, configurable, documented, and easy to debug**, following `docs/project_structure.md`.

## 2. Working with the user and existing work

The user wants practical progress and plain explanations of files, commands, logs, and results. Preserve uncertainty and avoid overstating prototype progress. This handoff supports switching agents when usage runs low: use focused reads, saved evidence, and small checks instead of unnecessary downloads/training/full-video runs.

- Inspect Git status first. Preserve unrelated edits, untracked implementation, intentional deletions, and existing outputs; do not reset/clean/restore to tidy the tree.
- Current user instructions override historical plans. Do requested work, not every roadmap item.
- Notify before downloads; use TeachLens for package installs. Do not silently create another environment or upgrade a working ML stack.
- Use fresh anonymous IDs/output paths. Overwrite only intentionally.
- Update relevant docs and run proportionate checks. Distinguish synthetic tests, smoke runs, and accuracy evaluation.
- Commit/push when currently authorized. If asked to commit everything, account for all changes in scope rather than silently omit files. Historical push requests are not permanent authorization.

Team workflow: update from remote, respect assigned files/folders, clear commit messages, pull requests. Inspect dirty trees before pulling. weewali is the documented workflow/storage contact. No detailed ownership map or deadlines were established.

## 3. Sources of truth and reading order

Use current user instructions and actual code/configuration to resolve conflicts with historical prose. Keep documentation current when resolving a discrepancy.

| Read | Purpose and caution |
| --- | --- |
| `README.md` | Short project purpose and links. Its long-term product description is broader than the prototype. |
| `docs/list_a_pose.md` | Current run guide, implemented indicators, camera assumptions, calibration, limitations. |
| `docs/project_structure.md` | Agreed architecture and file responsibilities. Some historical paths no longer exist; some files are planned. |
| `configs/default.yaml` | Current runtime defaults. A key's presence alone does not prove code consumes it. |
| `configs/indicators.yaml` | Current List A switches plus deferred feature plans. |
| `src/` and `scripts/analyze.py` | Actual behavior and command-line contract. |
| `tests/unit/` | Existing executable behavior checks. |
| `docs/video_indicator_catalog.md` | Broad indicator catalog and limits on defensible claims, not an implementation checklist. |
| `data.yaml`, `data/person_detection/combined.yaml` | Current one-class detection dataset contract. These configure detection training, not the running pose pipeline. |
| `docs/annotation_guidelines.md` | Historical imported role mapping and box-label rules. Do not transfer its role IDs to the one-class set. |
| `docs/dataset_preparation_notes.md` | Historical dataset problems; its statement that v2 is current is stale. |
| `docs/model_training_notes.md` | Historical manual commands. Its opening warning overrides the old reversed role mapping/examples below it. |
| `docs/video_sources_instructor.md`, `docs/video_sources_students.md` | Candidate sources, suitability, provenance, and usage restrictions. |
| `docs/selected_videos.md`, `docs/annotation_review.md` | Approval/review registers; currently mostly placeholders, not proof of completed approval or review. |
| `docs/pose_training_rules.md` | Empty reserved document as of verification. Do not claim it contains reviewed rules. |

## 4. Architecture and files

Python 3.10+ handles executable work; YAML settings/calibration; YOLO TXT annotations; JSONL evidence; JSON current summaries; CSV future interchange; Markdown/Mermaid docs. Future reports use Jinja2 HTML/CSS, plots use Matplotlib, extraction uses external FFmpeg. Actual tests use unittest despite older pytest prose.

The event layer is the boundary: observations -> rules/events -> aggregates -> presentation. Models must not directly write evaluation prose. Reports/charts should regenerate from saved evidence.

| Implemented file | Responsibility |
| --- | --- |
| scripts/analyze.py | Thin argparse/logging CLI calling pipeline.analyze_video(). |
| src/pipeline.py | Input/ID/output validation, settings/zones/switches, lifecycle and output orchestration; AnalysisStopped for a live-preview stop. |
| src/preprocessing/video.py | VideoInfo/frames(): metadata, sampling, timestamps, optional start/end segment (seek + full-run grid), capture cleanup. |
| src/preprocessing/camera_motion.py | CameraMotionEstimator: ORB background matching (people masked), similarity fit, per-shot chaining, new shot on cut/zoom jump/few inliers; CameraMotion, Affine helpers (apply_affine/compose/invert, normalized coords). |
| src/inference/pose.py | Keypoint/PosePerson, adapt_result(), YoloPoseTracker; local weights and Ultralytics/ByteTrack adapter. |
| src/analysis/visual_indicators.py | Model-independent geometry, visibility, posture/motion, tracks/segments, FoundationAnalyzer; last_sample SampleSnapshot/TrackSnapshot per-sample rule state for overlays (None = not evaluated, "unknown" = evaluated without evidence). |
| src/analysis/events.py | Frozen validated Event, rounding, atomic JSONL. |
| src/analysis/aggregates.py | summarize_list_a() reads saved evidence. |
| src/reporting/annotated_video.py | ReviewOverlay (boxes/joints/IDs/roles/states/HUD), AnnotatedVideoWriter (atomic MP4, avc1 via MSMF on Windows, mp4v fallback), LivePreview (space/n/q keys). |
| src/utils/config.py, paths.py | Safe YAML, recursive overrides/validation, zones/switches, repository paths. |
| scripts/extract_frames_ffmpeg.py | Dataset frame extraction. |
| scripts/pose_action_demo.py | Separate YOLO11 Pose/ByteTrack hand-raise experiment with annotated MP4. |
| src/preprocessing/recording_time.py, scripts/recording_time.py | Recording start/end in HH:MM:SS, duration, frame rate, and clock time at positions (`--at`) from container tags via external ffprobe; start differences for several videos. GPS tags are detected but never printed. |

Implemented packages have __init__.py. Planned responsibilities: preprocessing/dataset.py for audits/splits; inference/detection.py and tracking.py only if needed; analysis/interaction_indicators.py and quality.py for later rules/quality; reporting/charts.py, renderer.py, templates/report.html.j2 and report.css; training/train_detector.py and evaluate_detector.py; utils/logging.py; thin scripts/train.py and report.py. Reporting holds only annotated_video.py; training/integration/fixture directories are scaffolds. Speech modules/models/audio remain future-only.

Create modules with real behavior/tests, not empty files to fill a tree. Keep any later JS/TS dashboard in a separate web/ app.

## 5. Environment and commands

Unpinned requirements: ultralytics, opencv-python, numpy, pandas, matplotlib, pyyaml, scipy, statsmodels, jinja2. Torch is installed but not explicitly listed. FFmpeg is available externally, not required by the OpenCV loop.

From repository root:

```powershell
conda activate TeachLens
python scripts/analyze.py --help
python -m unittest discover -s tests/unit -v
python scripts/analyze.py --video data/raw_videos/RaisingHands.mp4 --video-id students_review_01 --camera student_view --live
python scripts/analyze.py --video data/raw_videos/HowToSpeak.mp4 --video-id instructor_review_01 --camera instructor_view --local-config configs/web_sample_cameras.yaml --debug --no-save-video
python scripts/analyze.py --video data/raw_videos/HowToSpeak.mp4 --video-id instructor_V001_50m --camera instructor_view --local-config configs/web_sample_cameras.yaml --start 50:00 --end 55:00 --live
```

The downloaded web samples (HowToSpeak.mp4 especially) were filmed by operated cameras; TeachLens's own recordings use static cameras, which is the default config. Pass `--local-config configs/web_sample_cameras.yaml` for web samples that pan/zoom.

If Conda is not on PATH:

```powershell
& 'C:\Users\weewa\anaconda3\envs\TeachLens\python.exe' -s -m unittest discover -s tests/unit -v
```

Discover paths on other machines. -s excludes user-site packages. Earlier missing PyYAML was installed inside TeachLens; another default Python may still fail. If installation is necessary, notify first, select TeachLens, use python -m pip install -r requirements.txt. Do not reinstall reflexively.

Local models/pose/yolo26n-pose.pt exists and is ignored. Existence is checked before Ultralytics import/load; analysis must never silently download weights. --weight accepts another local pose checkpoint, but code does not fully validate family/task.

CLI: required --video, --video-id (letters/digits/hyphen/underscore), --camera (student_view/instructor_view). Optional --config (default configs/default.yaml), --local-config (explicit; no auto-discovery), --weight, --output-dir, --device (e.g. cpu/0), --overwrite, --start/--end (seconds or [HH:]MM:SS, end exclusive; parse_seconds in the script), --live/--no-live (default review.live_preview=false), --save-video/--no-save-video (default review.save_video=true), --path off/teacher/all (default review.path="off"; quoted in YAML because bare off parses as false), --debug. Debug logs counts; overlays come from the review outputs.

YAML-internal relative paths resolve from repository; relative CLI paths use caller's working directory. Use root. Output root contains events/<id>.list_a.jsonl, reports/<id>.list_a.json (not HTML), and debug/<id>.list_a.mp4 unless video saving is off.

Review config (`review:` in default.yaml) is consumed: save_video, video_codec (4-char, validated; default avc1), live_preview, show_rejected_keypoints. On this Windows machine, pip OpenCV encodes avc1 through MSMF (its FFmpeg lacks OpenH264), so the writer asks for MSMF first; elsewhere it tries the default backend, then mp4v with a warning. The live window needs a desktop session.

## 6. Camera settings and calibration

Precedence: supported CLI override > recursively merged local YAML > base YAML. Use ignored configs/*.local.yaml for machine/experiment settings, not credentials.

| Camera | expected_role | role_policy | fixed_camera | compensate_camera_motion | pose_joints | posture_method |
| --- | --- | --- | --- | --- | --- | --- |
| student_view | student | all_tracks | true | false | head, arms, hips | shoulder_rise |
| instructor_view | teacher | single_visible | true | false | all five groups | legs |

User decision 2026-10-26: TeachLens recordings use **static cameras** (no pan, zoom, or cuts), so both views default to fixed_camera=true. The web samples are different: measured 2026-10-26 on HowToSpeak.mp4 (ORB + RANSAC, 0.2 s steps, 16 spots across the lecture), that camera is operated, with background shifts of 10-25 px (at 960 px width) in most minutes and zoom/cut jumps (scale 0.4-0.7 or 1.4-1.7). The committed override configs/web_sample_cameras.yaml sets instructor_view fixed_camera=false + compensate_camera_motion=true for those clips; keep compensation code for them, not for own recordings.

Joints (user decision 2026-10-26): desks hide students' legs and the model still guesses hidden joints, so student rules do not use knees/ankles; the instructor's legs leave the frame when near the camera. cameras.<id>.pose_joints lists allowed groups (JOINT_GROUPS in src/inference/pose.py: head, arms, hips, knees, ankles; arms required). The pipeline applies restrict_joints() before both analyzer and overlay: confidence is zeroed for joints outside pose_joints or outside the image ([0,1]), positions kept. Masked joints are absent from position keypoints, and visibility parts that need them are null (not evaluated), not false. Validation: posture_method legs needs knees; shoulder_rise needs fixed_camera=true.

Student policy requires a student-only view. If teacher enters it, use role_policy: unknown until a defensible reviewed method exists. Instructor policy requires exactly one confirmed tracked person with no additional provisional track. Multiple people stay unknown. Event role teacher differs from historical annotation word instructor.

One model/tracker/analyzer per recording; IDs reset. No synchronized multi-camera fusion or cross-camera identity. Future joint analysis requires explicit offsets/calibration.

Defaults: sampling 0.2 s (~5 analyzed frames/s); box confidence 0.25, IoU 0.45, keypoint confidence 0.35; min_track_samples 3 (~0.4 s confirmation delay); max_track_gap_seconds 0.6; edge_margin 0.05; sitting/standing thresholds 135/160 degrees; motion boundaries 0.08/0.35/0.8 body-heights/s (still/low/medium/high); smoothing 3 samples. Actual sampling period is rounded stride/FPS. Classroom thresholds remain unvalidated.

Validator requires positive sampling <= maximum track gap, valid camera policies/confidence ranges/counts/gaps/edge margin and ordered angle/motion thresholds, but is not exhaustive.

Some settings are scaffolding: person_detector/dataset paths, random_seed, reporting/privacy keys, require_evidence_coverage, quality.evidence_coverage. Current segmentation does not enforce minimum_event_duration_seconds=0.5 or separate merge_gap_seconds=0.25; it uses track gap. Trace consumers before claiming a setting works.

load_indicator_flags() reads only ten visual switches: teacher_presence, visible_student_count, anonymous_tracking, role_classification, position, classroom_zone, entry_exit, visibility_occlusion, sitting_standing, motion_level. Missing defaults off; enabling other catalog keys does not implement them.

Zones: configs/zones.example.yaml contains placeholder board/teacher_front/student_seating polygons and a deliberately different camera ID. Copy under data/calibration, replace all points from actual camera, use normalized [0,1] polygons with >=3 points, matching camera_id, and cameras.<camera>.zones in an explicitly loaded local override.

Mean valid ankles are preferred; box-bottom midpoint is a limited proxy (students: always, since ankles are not used). A box reaching the image bottom (y2 >= 0.99) gives foot_point_source frame_bottom_cut_off and zone "unknown", since it does not show where the person stands. First matching polygon wins; unmatched -> outside_calibrated_zones. No calibration -> no zone events, not proof nobody visited a region. Static polygons are invalidated if a camera is moved (recalibrate after moving a tripod). No floor distance/speed or merged camera heatmaps without geometric calibration.

Camera compensation (web samples only; compensate_camera_motion with fixed_camera=false): per sample, CameraMotionEstimator (config camera_motion.min_inliers=40, max_scale_step=0.1) yields shot + to_shot_reference; the analyzer maps each person into shot coordinates (person_in_shot) for motion and path fields. An unmeasured step starts a new shot, so that sample's motion is unknown. fixed_camera=true (the default for both views) means image = shot 0 with no estimation; fixed_camera=false without compensation keeps motion/in-shot fields unknown/null. Shot coordinates are image-plane, relative to the shot's first frame, may leave [0,1], drift slowly, and cannot be joined across shots.

## 7. List A observations, rules, and JSONL

PosePerson: track ID, normalized xyxy box, detection confidence, keypoints, width/height. Keypoint: normalized x/y/confidence. Geometry restores pixel dimensions to preserve aspect ratio.

COCO-17: 0 nose; 1/2 left/right eye; 3/4 ear; 5/6 shoulder; 7/8 elbow; 9/10 wrist; 11/12 hip; 13/14 knee; 15/16 ankle. Adapter skips untracked detections/insufficient joints; absent confidence arrays become zero-confidence joints. model.track() uses persist=True, ByteTrack, class [0], not guaranteed re-identification.

OpenCV decodes sequentially, analyzes stride=max(1,round(interval*fps)), timestamps=index/fps. Deterministic for ordinary constant-frame-rate footage, not exact variable-frame-rate timestamps. Sampling saves inference, not all intervening decoding. Capture is released on exit.

Segments (--start/--end): frames() seeks with CAP_PROP_POS_FRAMES to the first full-run grid index at or after start (falls back to grab() if seeking is refused) and stops before end. Seeking was verified frame-exact (zero pixel difference vs sequential decode) on RaisingHands.mp4 (H.264) and HowToSpeak.mp4 (AV1, 1080p, 29.97 fps) up to frame 5400; recheck on new codecs. Timestamps stay source-relative; because the grid is kept, a 50:00 start begins at 3000.197 s on HowToSpeak. FoundationAnalyzer(start_time=...) suppresses entries for tracks present at segment start; finish() closes at min(end, source end). Tracker state starts fresh, so IDs/confirmation differ from a full run.

| indicator_name | Meaning |
| --- | --- |
| teacher_presence_sample | Instructor-context visible/not_detected/unknown; provisional/ambiguous -> unknown. |
| teacher_presence | Camera-assumed visible interval; may bridge short gaps, unlike sample fractions. |
| visible_student_count | Confirmed-track count; provisional/unknown samples limited. Not attendance. |
| persistent_anonymous_identity | Confirmed episode interval/samples/temporary_id, emitted on retirement/finish; stored role is not verified throughout. |
| role_classification | label, source=camera_context, verified_by_model=false assignment/change. |
| position | box_xyxy, box_center, foot_point/source, foot_point_in_shot, hip_center_in_shot (path anchor; both hips needed), camera_shot (0 fixed camera, null uncompensated moving camera), confident keypoints (index,x,y,confidence). |
| camera_motion | Compensated moving cameras only, role recording, per sample: shot, step_reliable (null first/false unmeasured), step_shift_fraction_of_width, step_scale, background_inliers, to_shot_reference (6 numbers, normalized). |
| classroom_zone | Calibrated per-track interval only. |
| entry_exit | Zero-duration apparent_entry_at_frame_edge/apparent_exit_at_frame_edge, not room transitions. "Near edge" = box touching the left/top/right edge or centre within 2 x edge_margin of any edge; touching only the bottom edge does not count (people near the camera / behind front desks are cut off there). |
| visibility_occlusion | Face/upper/lower/full-body availability (null when the camera's pose_joints exclude needed joints), truncation, overlap, possible-occlusion proxy (overlap >= 0.25 and an evaluated part not fully visible)/limitation. |
| sitting_standing | sitting/standing/unknown interval. |
| shoulder_rise | shoulder_rise cameras only: unsmoothed shoulder_widths_above_seated_level sample behind the posture label, for threshold tuning; limited coverage. |
| motion_level | Smoothed still/low/medium/high/unknown interval. |
| motion_speed | Unsmoothed body_heights_per_second sample; assumes_fixed_camera / camera_motion_compensated say which basis applied. |

Posture (posture_with_basis) prefers shoulder/hip/knee/ankle geometry on >=1 side. Median valid-side angles: standing knee>=standing threshold and hip>=threshold-10; sitting knee<=sitting threshold and hip<=standing threshold-5; otherwise unknown (never overridden). Only if no side has an ankle chain and analysis.upper_leg_posture=true: per side with shoulder/hip/knee, thigh >= thigh_torso_min_ratio (0.5) x torso, standing = hip>=standing-10 and thigh within thigh_axis_tolerance_degrees (30) of vertical, sitting = hip<=sitting threshold and thigh within tolerance of horizontal; all usable sides must agree. Fallback labels have basis upper_legs and limited coverage (full_legs -> adequate). Added 2026-10-26 because the instructor camera cuts off feet; waist-up views (no knees), walking strides, and desks remain unknown. The thresholds are unvalidated.

posture_method shoulder_rise (student_view default; fixed cameras only; basis shoulder_rise, limited coverage): per confirmed track, a deque of shoulder samples (midpoint x/y and width in pixels; both shoulders needed; ~300 s history). Seated level = SEATED_LEVEL_QUANTILE (0.8) of image y, i.e. the low end of the track's own shoulder heights; seat = median x; scale = median shoulder width (>= 8 px). Rise in widths: standing >= shoulder_rise_standing_min (0.7), sitting <= shoulder_rise_sitting_max (0.35, including below the level, e.g. leaning on the desk), between -> unknown; fewer than shoulder_rise_history_seconds (3 s) of history or > shoulder_rise_max_seat_shift (1.0) widths from the seat -> unknown. Assumes the student sits most of the time and stands up in place: a track that starts standing still (late arrival) is labelled sitting until seated samples exist, and long standing (> ~80% of the history) moves the level. Unvalidated starting values; the 2026-10-26 recording scenarios include calibration clips for tuning.

Motion uses median displacement of >=3 shared confident upper-body joints, normalized by average box height/time, sufficient scale, in image coordinates (fixed camera) or shot coordinates (compensated camera, same shot only). Missing/invalid/long-gap/new-shot/uncompensated-moving-camera evidence -> unknown. Categories use median smoothing; speed samples do not. This is apparent image motion, not ground speed/engagement.

Segments close on label/role/zone change, gap, retirement, finish. JSONL is not guaranteed event-start chronological order: intervals close after later observations. Sort timestamps for timelines. Instant observations have zero duration.

Illustrative schema (not a measured result):

```json
{"video_id":"example_01","track_id":7,"role":"teacher","indicator_name":"sitting_standing","start_time":58.2,"end_time":62.8,"duration":4.6,"value":"standing","confidence":null,"coverage":"adequate","zone":null}
```

Roles: teacher/student/unknown/both/classroom/recording. Coverage: adequate/limited/unknown means availability, not accuracy. Times are source-relative seconds, make_event() rounds milliseconds, duration matches end-start. Heuristics/camera roles/transitions have null confidence; no calibrated probability. Positions/joints retain model confidence.

Events lack camera_id (summary has it); retain association through unique IDs/manifests. JSONL stores observations/coordinates, not pixels. It is already inference output, reusable for summaries/charts/some new rules, not ground truth/training labels. It cannot recover missed people/discarded joints; effective config/model/source metadata are incomplete for exact replay.

Writing: evidence streams to a same-directory temporary file and publishes via os.replace on completion; the annotated MP4 is published right after it from its own hidden temporary file; summary independently does the same. Refuse collisions without explicit overwrite (all three paths are pre-checked). The set is not transactional: summary failure may leave complete JSONL/MP4; regenerate summary from saved events.

Live preview q/Esc raises AnalysisStopped: nothing is published (JSONL and MP4 temporaries are discarded) and the CLI exits 1 with a warning. There is still no supported partial result, checkpoint, or resume. Ctrl+C does not create a completed partial run. The MP4 writer discards its temporary file on any exception including KeyboardInterrupt, but the JSONL writer's except Exception cleanup does not catch KeyboardInterrupt and may leave a hidden .tmp file; inspect before using/deleting.

## 8. Reading the JSON summary correctly

| Field | Interpretation |
| --- | --- |
| `video_duration_seconds` | Source metadata duration, or final sampled time plus sample period when unavailable. Always the whole source, even for a segment. |
| `analyzed_start_seconds`, `analyzed_end_seconds`, `analyzed_duration_seconds` | Requested start (default 0) and min(end, source end). Every other duration covers only this range; a segment adds a limitation line. |
| `teacher_visible_seconds` | Sum of teacher-presence intervals, nullable when there are no teacher samples. |
| `teacher_visible_fraction_of_valid_samples` | `visible / (visible + not_detected)`; unknown samples excluded. Not fraction of the entire lecture. |
| `teacher_presence_sample_coverage` | Counts of visible/not-detected/unknown samples; always inspect beside the fraction. |
| `visible_student_count` | Min/max/median across **adequate** count samples, or null if none. |
| `student_count_valid_sampled_seconds` | Adequate count sample count times sample period, capped at the analyzed duration. |
| `distinct_tracker_id_values_by_role` | Distinct temporary ID values by episode role; not unique people or identities. |
| `track_episodes_by_role` | Episode counts; an ID can participate in more than one episode. |
| `zone_track_seconds` | Summed per-track zone durations; concurrent people can exceed video length. |
| `posture_track_seconds`, `motion_track_seconds` | Durations summed across tracks and roles, including explicit unknown buckets. Not necessarily teacher-only time. |
| `apparent_frame_edge_transitions` | Heuristic edge-transition counts; fragmentation can inflate them substantially. |
| `camera_motion` | Compensated cameras only (else null): sample_steps first/measured/unmeasured counts and number of shots. Many unmeasured steps mean motion/path data are fragmented. |
| `limitations` | Fixed explanatory cautions added by code. Not a validation score or automatically diagnosed error list. |

The reducer does not separate posture/motion/zones by role or filter every coverage flag. Preserve unknown buckets and define valid denominators before teacher-only/rate claims.

### Existing local runs: diagnostic examples, not benchmarks

These ignored artifacts were present when this handoff was verified. A fresh clone may not contain them, and subsequent runs may differ.

- `example_instructor.list_a.json`: duration **3822.185 s** (about 63:42), visible interval sum **2316.312 s**, valid-sample visibility fraction **0.8707**. Samples: **11567 visible, 1718 not_detected, 5807 unknown**. It reports **113 teacher-role ID values, 1442 unknown-role ID values**, **938 apparent entries and 1019 apparent exits**. These large numbers motivate tracking review; they do not prove that many people entered a lecture. Motion is entirely unknown because that run predates camera compensation (the instructor camera was then treated as moving and unmeasured); zones are empty because no calibration is set.
- `raisinghands_smoke_v2.list_a.json`: duration **21.521 s**, adequate student-count coverage **14.214 sampled seconds**, visible count **min 6 / median 8 / max 9**, **12 student tracker ID values**. A clip named RaisingHands does not mean this pipeline detects hand raises.
- Earlier `raisinghands_smoke.list_a.json` is also present; prefer identifying the exact run/config rather than mixing results.
- `raisinghands_review_smoke` (2026-10-26, current code, CPU) has JSONL, summary, and `outputs/debug/raisinghands_review_smoke.list_a.mp4` (H.264 via MSMF, 108 frames at 4.995 fps, 21.6 s, about 10 MB at 1280x720). Its summary numbers match `raisinghands_smoke_v2`; v2 lacks only the later `track_episodes_by_role` key, and its JSONL differs because events.py changed after that run. A same-config run with `--live` produced identical JSONL. Visible in the MP4: the teacher at the board is not detected, desks make many postures unknown, and some students stay provisional mid-clip, which limits count coverage. It predates the analyzed_* summary keys.
- `instructor_segment_smoke` (2026-10-26, HowToSpeak.mp4 `--start 50:00 --end 50:20`, CPU, ~38 s wall time): 99 samples from 3000.197 s, one teacher track ID, presence samples 96 visible / 1 not_detected / 2 unknown, posture standing 11.0 s / unknown 8.4 s, no entry/exit. MP4 1920x1080 H.264, 99 frames, about 19.5 MB. At 50:08 the teacher is clearly standing but posture is unknown because the frame cuts off below the knees (`cut off`); this motivated the upper-leg fallback below.
- `instructor_path_smoke` (same 50:00-50:20 range, `--path teacher`, after the upper-leg fallback and camera compensation, ~40 s): posture standing 13.4 s (one upper-leg segment) / unknown 5.8 s (walking strides with bent knees, plus knee-less frames); motion low 13.2 / still 3.6 / medium 1.6 / unknown 0.8 s; camera 95 measured / 3 unmeasured steps (real zoom jumps x1.67, x0.59, x1.69) -> 4 shots; median background inliers ~940. 50:08 now reads `standing (partial) | medium 0.45 bh/s`; at 50:05 (wide shot) the yellow path shows the walk to the board.
- `instructor_pan_smoke` (24:00-24:30, `--path teacher`, ~52 s): camera pans up to 10% of width per step, 147 measured / 2 unmeasured -> 3 shots; motion medium 11.2 / low 6.6 / still 5.6 s; posture mostly unknown (20.6 s; waist-up close-ups without knees); 2 teacher IDs; 27 not_detected samples after a cut to a full-screen slide. The path stays attached to the board while the camera pans and resets at the cut. Visual inspection only, not an accuracy measurement.
- The instructor smokes above ran with the then-default compensated instructor camera; reproduce them with `--local-config configs/web_sample_cameras.yaml`. Their entry/exit and posture numbers predate restrict_joints and the bottom-edge rule.
- `raisinghands_static_legs_smoke` (2026-10-26, `--end 10 --no-save-video`, student_view with pose_joints head/arms/hips and shoulder_rise): no knee/ankle joints in position events; 196 shoulder_rise samples between -0.20 and 0.21 widths (seated students, some raising hands), posture sitting 39.8 / unknown 33.8 track-seconds (unknown = first 3 s history per track). A smoke run, not an accuracy check; nobody stands in that clip.
- Two manually generated review images exist under `outputs/debug/example_instructor_snapshots/`: `event_45.045s.png` (presence with unknown posture) and `event_60.06s.png` (standing example). They are ad hoc evidence examples, not an implemented reusable snapshot-export feature.

Academic reports should describe a **working prototype with incomplete validation**. Final accuracy comparisons, engagement metrics, fairness/retention assessments, and teaching-evaluation reports are not complete.

## 9. Requested visual evidence and future reporting

Implemented 2026-10-26 at the user's request: the main analysis draws each **analyzed (sampled) frame** with model boxes, confident joints, anonymous IDs, box confidence, camera-assumed roles (blue student, orange teacher, magenta unknown, thin grey provisional), posture/motion (`?` = evaluated unknown; motion shows smoothed body-heights/s), visibility flags, calibrated zones/foot points, and a HUD (time, tracked/confirmed/provisional, count or presence with coverage, camera notes). Frames go to outputs/debug/<id>.list_a.mp4 at 1/sample_period fps (source timing kept; intermediate frames are not shown) and optionally to a live window (`--live`; space pause, n step, q/Esc stop with no outputs; closing the window continues headless). Overlays come from the same run's PosePerson observations and analyzer.last_sample, so they match the JSONL and are not a separate inference. Guide: docs/list_a_pose.md "Live review and annotated MP4".

Paths (user-requested 2026-10-26 for later teaching-effectiveness work): `--path teacher|all` draws hip-centre trails in shot coordinates (yellow for teacher), re-projected into each frame so they stay attached to the scene while the camera pans; trails clear at a new shot; review.path_seconds (null = whole range) and path_max_gap_seconds (1.0) control length/joins. The hip centre was chosen over ankles because the instructor camera often cuts off feet (both hips in 100% vs ankles 60% of samples at 50:00). The path data live in position events regardless of drawing; docs/list_a_pose.md shows how to extract them. Panel adds a camera line (shot, pan %, zoom, compensated / cut or unmeasured).

Still not implemented: snapshot/MP4 rendering **from saved JSONL** without rerunning pose (seek original frames by evidence timestamps, match intervals/instant positions deliberately; line numbers are not time), entry/exit markers (emitted only at track retirement, after the frame), and a thin scripts/report.py. Put such rendering in reporting with a thin CLI. The old demo's MP4 is separate. Review media does not prove whole-video accuracy.

Future reports show observation + denominator + confidence/validation evidence + limitation with reviewable source moments. Charts/heatmaps/HTML/PDF/feedback generation remain unimplemented; Jinja2 dependency does not prove rendering works.

## 10. Datasets, annotation, and training

| Path | Contract/status |
| --- | --- |
| data.yaml; data/person_detection/combined.yaml | Current 0=person, people_01 only; keep aligned. |
| data/person_detection/people_01 | Roboflow train/images, valid/images, test/images and labels; audit before trusting metrics. |
| people_02 | Appears duplicate; do not pool as independent data. |
| person_detection_v1 | Historical 0=person, many missing labels including test; absent/deleted at verification. |
| person_detection_v2 | Historical 0=person, train/val source overlap; absent at verification. |
| student_instructor_imported | Historical 0=student, 1=instructor, malformed annotation/split overlap; unnecessary for current camera-role design. |
| person_v1_data.yaml; role_data.yaml | Historical configs deleted in working tree despite doc references. Do not restore automatically. |

Earlier renames: dataset_v1 -> person_detection_v1; dataset_v2 -> person_detection_v2; Instructor_Student -> student_instructor_imported.

Prior audit: people exports had same Roboflow project/version, identical relative paths for 48,829 files, matching sampled hashes; 584 source filename stems shared across train/validation in people_01. Re-audit new exports; augmented variants leak despite different hashes.

Current YAML:

```yaml
path: data/person_detection
train:
  - people_01/train/images
val:
  - people_01/valid/images
test:
  - people_01/test/images
names:
  0: person
```

Combine distinct compatible datasets by adding image directories to split lists, retaining matching labels, verifying numeric mappings, auditing the whole collection. Duplicate exports/class renaming/multiple loosely passed YAMLs are not a repaired combined contract.

Rules: YOLO rows are class_id x_center y_center width height, normalized [0,1], numeric IDs, matching stems. Review tight boxes for every visible person, malformed rows, missing pairs, unclear/corrupt frames. Empty images can be useful negatives with empty labels; a horse is acceptable as a reviewed negative, never a horse labeled person. Unlabeled visible people are errors.

Split by source recording before extraction/augmentation; related frames/variants stay in one split. Check provenance/stems/hashes/near-duplicates; hold out both camera views without tuning on final tests. Preserve version/class contracts; repair derived versions. Record source/license/permission/review/split in manifests/docs. Placeholder registers prove no approval.

Shared pose/track outputs support many rules; no model per indicator. Test pretrained YOLO26 first, add detector only for measured recall/count failures. Box-only labels cannot train pose: need COCO-17 joints/visibility plus boxes, separate pose YAML (e.g. kpt_shape: [17,3]) and correct left/right flip mapping. Hand-raise event labels evaluate rules, keypoint labels train pose; action models need labeled sequences.

Phones/books/laptops/markers require object detection and evidence-based person association; phone-on-desk does not prove use/distraction. Segmentation requires masks/polygons, cannot recover hidden joints, and is not default without measured benefit. MediaPipe/MMPose were discussed alternatives, not adopted dependencies.

Train only when requested: audit, check TeachLens/hardware, tiny smoke case, local weights or notify download, record exact dataset/model/settings/seeds/metrics under runs with reproducible doc summary.

```powershell
python scripts/extract_frames_ffmpeg.py --input data/raw_videos/HowToSpeak.mp4 --output data/raw_frames --prefix instructor_V001 --every 2
python scripts/recording_time.py data/raw_videos/clock_test.MOV --at 0 8.125
```

Extraction makes <prefix>_frame%04d.jpg, FFmpeg fps=1/N/JPEG quality 2, integer --every. Its 2-second dataset default differs from 0.2-second analysis; sparse images cannot replace tracking input. Wrapper lacks explicit overwrite/positive-interval guard; use fresh prefix/folder.

## 11. Storage, privacy, and provenance

Git stores code/docs/configs/placeholders/tiny redistributable samples. Keep full media/frames/datasets/weights/generated evidence local. Check staging: ignores may miss new paths and do not untrack existing files. Ignored artifacts/checkpoints can still be valuable; do not delete casually.

Directory roles: data/raw_videos (present ClassroomManagement.mp4, HowToSpeak.mp4, RaisingHands.mp4, RaisingHands2.mp4 web samples; own clock_test.MOV); raw_frames (annotation images); calibration (camera geometry); manifests (source/permission/split); samples and tests/fixtures (tiny safe examples/placeholders); models/detection and pose (weights); outputs/events, reports, visualizations, debug (generated artifacts); runs (experiments). Root yolo11n.pt/yolo11n-pose.pt are historical. Legacy Instructor Recordings/Student Recordings are absent, not required new directories.

Use anonymous IDs. Do not incidentally add face recognition/crops, identity matching, names in filenames, or cloud uploads. By default the main analysis now saves an annotated MP4 of original classroom footage under ignored outputs/debug/ (user-requested); keep it local, never commit/share without permission, and use --no-save-video when review media is not needed. It saves no crops or still images; privacy YAML is not full consent/retention enforcement.

Camera sync (user goal 2026-10-26: automatically align instructor_view and student_view recordings). Clock test data/raw_videos/clock_test.MOV (iPhone 16 Pro, 4K HEVC, 8.333 s, 200 frames) filmed an atomic-synced web clock: first frame shows 18:31:18.26, frame at ~8.125 s shows 18:31:26.43, while com.apple.quicktime.creationdate = 18:31:19+04:00 (~0.7 s late) and creation_time = 14:31:20Z (~1.7 s late). Both mark the start (not the stop), at whole-second precision, so metadata is only coarse alignment; frame-level sync needs a shared signal. Audio cross-correlation was proposed (timing only) but needs the user's/supervisor's scope decision because audio is otherwise out of scope; recording both cameras on one machine's clock is the hardware alternative. The downloaded web samples have no recording-time tags. iPhone files also carry GPS location tags: never print/share them; strip metadata before sharing media.

Source registers: instructor V001-V003, students V001-V009 including datasets; IDs overlap, qualify context. Candidates are not approvals. Student V008 Whiffe/SCB-dataset explicitly requires copyright approval before use; preserve until updated approval.

Report observable proxies with valid denominators, not internal states/protected attributes/misconduct/official attendance. Even orientation percentages need implemented validated logic. Feedback supports human review, not autonomous verdicts. Permissions/retention/fairness/cost/sustainability remain incomplete assessments; local processing/sampling/model reuse/anonymous IDs are design choices, not a full governance program.

## 12. Tests and validation

Verified 2026-10-26: 52 unittest tests pass in TeachLens (test_recording_time.py adds tag parsing, iPhone/UTC/missing-tag cases, and formatting; ffprobe-shaped JSON, no video needed).

```powershell
python -m unittest discover -s tests/unit -v
python -m compileall -q src scripts/analyze.py
git diff --check
```

test_config.py covers defaults/switches, calibration ID, sparse sampling, missing weight/no download. test_events.py covers atomic writes/no overwrite/failed stream. test_visual_indicators.py covers synthetic posture/unknown, fixed-camera motion, zones/visibility, confirmation/count/roles, instructor ambiguity/presence, switches, boundary/edge transitions, last_sample provisional/confirmed state. test_annotated_video.py covers label text (provisional vs unknown vs not evaluated), HUD lines, draw-on-copy, live-window keys with a fake cv2 (q/Esc stop, pause/step, closed window continues), and MP4 publish/no-overwrite/discard-on-failure, plus the segment HUD line. test_video.py builds a tiny mp4v clip whose brightness encodes the frame index and checks segment times, seek accuracy, full-run grid alignment, and start-after-end. test_aggregates.py checks analyzed_* keys, valid-seconds capping, and the segment limitation. test_analyze.py loads scripts/analyze.py via importlib and checks parse_seconds forms/rejections. test_visual_indicators.py also covers no entry at segment start, the upper-leg posture fallback (standing, sitting, disabled, foreshortened thigh, waist-up, ambiguous full legs not overridden), and camera compensation (pan -> still with in-shot foot point; uncompensated or new shot -> unknown). test_camera_motion.py checks affine compose/invert, a known synthetic pan mapped back to the shot reference, still-camera identity (deadband), and a cut starting a new shot. test_annotated_video.py adds the partial-posture label, camera HUD lines, and the hip-centre teacher path (role filter, reset at new shot). test_aggregates.py adds camera_motion summary counts; test_config.py validates review.path, thigh tolerance, and camera_motion.max_scale_step, plus static-camera defaults, the web-sample override merge, student pose_joints without legs, and pose_joints/posture_method guards. test_visual_indicators.py adds restrict_joints (unused and out-of-image joints dropped, positions kept, null visibility parts), shoulder_rise_posture (sitting, standing, between thresholds, leaning, away from seat, short history, brief bending), a student stand-up timeline without legs, and a waist-up teacher at the frame bottom (no entry/exit, zone unknown). test_annotated_video.py adds the rise label and the "joints not used" HUD note. The live window itself was smoke-checked on RaisingHands.mp4 without key presses; nobody has visually reviewed it on screen yet.

These do not prove real-video/model accuracy or validated thresholds. Integration/fixtures are scaffolds; aggregation/video/adapter/end-to-end checks incomplete. Use targeted deterministic tests for changed rules, unknowns, gaps, boundaries, normalization, roles/denominators without GPU/downloads/private media.

For accuracy, label held-out clips from both cameras: missed/duplicate people, keypoint coverage, ID switches/fragmentation, posture unknown rate, event precision/recall with declared temporal matching. Compare models on identical clean held-out data. Reuse saved evidence for aggregation/report work; do not rerun a 63-minute lecture for a small edit.

## 13. Continuation priorities and gaps

Context, not an automatic queue: validate List A/camera assumptions/tracks/thresholds (the review MP4/live window now supports this visually); snapshots/MP4 rendered from saved JSONL without rerunning pose; role-separated summaries and valid/unknown coverage; run provenance (source/video metadata, effective config, model hash/version, calibration/schema IDs); graceful stop/partial semantics/cleanup/resume/progress (--start/--end now covers reviewing part of a long video, but a q-stop still saves nothing); dataset repair before training; charts/HTML from saved evidence.

Expand physical indicators deliberately: hand raises, orientation, paths/heatmaps, stationary/out-of-seat/proximity/grouping/object observations. Audio/transcript/semantic interaction stays deferred.

Absent/unvalidated: synchronized camera fusion, default real calibration, ground-plane speed and floor paths (paths are per-shot image-plane), validation of camera compensation/upper-leg posture/shoulder-rise posture/path accuracy against labelled clips (docs/test_recording_scenarios.pdf plans own static-camera test recordings for this), calibrated heuristic probabilities, official attendance, full recording quality, lockfile, complete run manifests, overlays rebuilt from saved evidence, final accuracy evaluation.

## 14. Coding and maintenance

Use snake_case, test_<module>.py, type hints, docstrings for units/assumptions/unknowns. Keep model tensors in adapters, rules in analysis, settings in YAML, orchestration in pipeline, parsing in scripts. Use pathlib/project_path, portable committed paths, local machine overrides, real package __init__.py. Move proven notebook logic into source.

Keep null/empty/zero/unknown distinct; no invented rule confidence. Preserve event contracts or version/migrate consumers/tests. Update architecture/run docs with changes. Log stage/video/time without personal data. Fresh IDs/explicit overwrite; inspect resolved paths before recursive Windows/OneDrive operations. Avoid unsolicited frameworks/models/services. Report concrete results, actual checks, remaining limits; smoke/synthetic checks are not accuracy benchmarks.

## 15. Repository snapshot

Observed remote: https://github.com/weewali/TeachLens-SGP.git; branch scripting-frame-extraction. Follow current branch/user conventions (Codex new-branch default codex/), not this snapshot blindly.

Last inspected commits: ea24b3e (audio/recording future work), dea3be3 (one-class camera roles), 8d4d4ba (architecture guide).

Before this file there were 412 status entries: modified docs/configs/data.yaml; untracked List A source/tests/run guide/combined dataset YAML; historical dataset/config deletions. Working List A code was local, not represented by these commits. Preserve it; a fresh clone at that commit lacks the full implementation.

OneDrive sync conflict (found 2026-10-26): the repository lives in OneDrive, and an older cloud copy of 12 files (Oct 14-22 versions of pipeline, visual_indicators, config, video, aggregates, analyze.py, default.yaml, two tests, list_a_pose.md, project_structure.md, AGENTS.md) took the normal names while this machine's newer edits were renamed `*-MrWolf.*`. The newer copies were restored to the normal names; the stale ones are backed up under ignored `outputs/onedrive_conflict_backup_2026-10-26/`. If `*-<computer name>.*` files reappear, compare before trusting either copy, and avoid editing the folder from two machines at once.

Writing this handoff changed no existing implementation/data. This snapshot predates its commit; inspect current status. Maintain the handoff after meaningful scope/architecture/environment/contracts/validated behavior changes; retire stale details.
