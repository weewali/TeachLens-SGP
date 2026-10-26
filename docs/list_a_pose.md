# List A physical indicators with YOLO26 Pose

This is the first executable TeachLens analysis slice. It processes **video
only** with a local YOLO26 pose checkpoint and ByteTrack, then writes
timestamped observations to JSONL, a small JSON summary, and an annotated
review MP4. An optional live window shows the same overlays while the run
progresses. It does not train an action model, process audio, identify
anyone, or score engagement.

## Setup and run

Use the existing `TeachLens` Conda environment. It needs `ultralytics`,
`opencv-python`, `torch`, and `PyYAML`. The tested environment has Python
3.10, Ultralytics 8.4.75, OpenCV 4.13.0, and PyYAML 6.0.3. Put the official
`yolo26n-pose.pt` file at `models/pose/yolo26n-pose.pt` (ignored by Git).
The script checks that the file exists **before** loading Ultralytics, so
running analysis cannot silently download weights. A different local YOLO26
pose checkpoint can be supplied with `--weight`.

From the repository root in PowerShell:

```powershell
conda activate TeachLens
python scripts/analyze.py --video data/raw_videos/RaisingHands.mp4 --video-id example_students --camera student_view --live
python scripts/analyze.py --video data/raw_videos/HowToSpeak.mp4 --video-id example_instructor --camera instructor_view --local-config configs/web_sample_cameras.yaml --path teacher
python -m unittest discover -s tests/unit -v
```

The defaults assume TeachLens's own **static cameras**. The downloaded web
samples in `data/raw_videos/` were filmed by operated cameras that pan, zoom,
and cut, so analyze those with `--local-config configs/web_sample_cameras.yaml`
(see [Static cameras and web samples](#static-cameras-web-samples-motion-and-paths)).
`docs/test_recording_scenarios.pdf` describes how to film test lectures that
match these assumptions.

If `conda` is not on PATH, invoke that environment's `python.exe` directly.
Use a **different anonymous `--video-id` for each camera recording**; output
names are based on this ID. One run writes three files:

| File | Contents |
| --- | --- |
| `outputs/events/<id>.list_a.jsonl` | Timestamped evidence, one event per line. |
| `outputs/reports/<id>.list_a.json` | Summary reduced from that evidence. |
| `outputs/debug/<id>.list_a.mp4` | Annotated review video (skip with `--no-save-video`). |

Outputs and weights are local and ignored by Git. `--debug` logs the detected
track count per sampled frame. Existing outputs are preserved by default; use
`--overwrite` only when you intentionally want to replace them.

## Analyzing part of a long video

`--start` and `--end` limit a run to one segment. Both accept seconds
(`3000`, `3000.5`) or clock time (`50:00`, `1:02:03.5`); `--end` is exclusive.
Either can be omitted to run from the beginning or to the end.

```powershell
python scripts/analyze.py --video data/raw_videos/HowToSpeak.mp4 --video-id instructor_V001_50m --camera instructor_view --local-config configs/web_sample_cameras.yaml --start 50:00 --end 55:00 --live
```

- The reader seeks directly to the segment, so a segment late in a long video
  starts in about a second instead of decoding everything before it. Seeking
  was checked to be frame-exact on the project's H.264 and AV1 files.
- Event times stay **relative to the start of the original video**, so
  `50:08.2` in the panel and `3008.2` in the JSONL both point at the same
  moment of the source recording.
- The segment analyzes exactly the frames a full run would analyze there. At
  29.97 fps the 0.2-second grid falls at 2999.997 s and 3000.197 s, so a
  segment starting at `50:00` begins at 3000.197 s.
- The summary adds `analyzed_start_seconds`, `analyzed_end_seconds`, and
  `analyzed_duration_seconds`; all durations in it cover only that range, and a
  limitation line says so. `video_duration_seconds` is still the whole source.
- Tracking starts fresh at `--start`: people already visible there are not
  reported as entries, and tracks still open at `--end` are closed there.
  Tracker IDs and confirmation delays therefore differ from a full run, so
  compare segment and full-run results only after allowing for that.
- Give each segment its own `--video-id` (for example with the start minute in
  it); output names come from the ID, not the range.

## Joints each camera uses

Students sit behind desks, which hide their knees and ankles, and the pose
model still guesses positions for hidden joints. The instructor's legs leave
the frame when they stand near the camera. So each camera lists the joint
groups its rules may use in `cameras.<camera>.pose_joints` (`head`, `arms`,
`hips`, `knees`, `ankles`; `arms` is required because every posture and motion
rule needs the shoulders):

| Camera | `pose_joints` | `posture_method` |
| --- | --- | --- |
| `student_view` | `head, arms, hips` (legs never used) | `shoulder_rise` |
| `instructor_view` | all five groups (legs used when visible) | `legs` |

Before any rule runs, joints outside that list, and joints the model placed
outside the image, get zero confidence. They are left out of `position`
keypoints and the overlay, and visibility parts that need them are `null`
(not evaluated) rather than `false`. The HUD notes `joints not used: knees,
ankles` for the student camera.

**`legs` posture** (instructor) uses hip and knee angles from full legs; when
no ankle is visible it falls back to shoulder-hip-knee (`standing (partial)`),
and with no knees it stays unknown.

**`shoulder_rise` posture** (students; fixed cameras only) compares the
shoulder midpoint with the track's own **seated level**: the low end (80th
percentile of image y) of that student's shoulder heights so far. The rise is
measured in the student's median shoulder width, because standing up raises
the shoulders by about one shoulder width:

- `standing` when the rise is at least `analysis.shoulder_rise_standing_min` (0.7);
- `sitting` when it is at most `shoulder_rise_sitting_max` (0.35), including
  below the level (leaning on the desk, head down);
- `unknown` in between, during the first `shoulder_rise_history_seconds` (3 s)
  of a track, without both shoulders, or when the shoulders are more than
  `shoulder_rise_max_seat_shift` (1.0) widths sideways from the seat (left the seat).

It assumes students sit most of the time and stand up in place. A student who
is tracked while already standing still (a late arrival) is labelled
`sitting` until they actually sit, and standing for most of the recent history
raises the seated level. Each evaluated sample also writes a `shoulder_rise`
event with the raw value, so the thresholds can be tuned on labelled clips. In
a 10-second smoke run on `RaisingHands.mp4`, seated students (some raising
hands) stayed between -0.20 and 0.21. The thresholds are **unvalidated
starting values**.

People cut off by the **bottom** of the image (near the camera, or behind
front desks) are not treated as leaving: a box touching only the bottom edge
no longer counts as "near the frame edge" for apparent entries and exits.
Their box bottom is not the floor either, so the foot point source is
`frame_bottom_cut_off` and a calibrated zone is `unknown` for those samples.

## Static cameras, web samples, motion, and paths

TeachLens recordings use static cameras on tripods, so both views default to
`fixed_camera: true`: the image is one shot for the whole run, motion is
measured in image coordinates, paths need no compensation, and calibrated
zones stay valid until a camera is moved.

The downloaded instructor sample (`HowToSpeak.mp4`) is different: an operated
camera. Measured on 0.2-second steps, its background moves about 10-25 px in
most minutes as the camera pans after the lecturer, with zooms and cuts
between shots. Image coordinates there describe the camera as much as the
person, so `configs/web_sample_cameras.yaml` sets the `instructor_view` to
`fixed_camera: false` and `compensate_camera_motion: true` for such clips:

1. `src/preprocessing/camera_motion.py` matches background features (people
   masked out) between consecutive samples and fits a shift/zoom/rotation.
2. Steps are chained into the coordinates of the current **shot's** first
   frame. A cut, a zoom jump over `camera_motion.max_scale_step`, or fewer
   than `camera_motion.min_inliers` agreeing features starts a new shot
   instead of guessing. In the test segments real cuts were caught (zoom
   x1.7 or x0.6 between samples) while normal steps agreed on ~900 features.
3. **Motion level** compares joints in shot coordinates, so a camera pan is
   not counted as the person moving. The sample right after a new shot is
   `unknown`. `motion_speed` events say `camera_motion_compensated: true`.
4. Each sample adds a `camera_motion` event (role `recording`) with the shot,
   whether the step was measured, pan, zoom, background inliers, and the
   `to_shot_reference` matrix. The summary's `camera_motion` counts measured
   and unmeasured steps and shots.

A camera set to `fixed_camera: true` (both views by default) needs none of
this: its image is shot 0 for the whole run. A moving camera with
compensation switched off keeps motion and in-shot positions unknown.

**Paths.** Add `--path teacher` (or `review.path: teacher`) to draw the
camera-assumed teacher's path in yellow; `--path all` draws every confirmed
track in its role colour. The path follows the **hip centre** (midpoint of
both hips), because hips are visible far more often than ankles: in the 50:00
segment both hips were found in every sample but ankles in only 60%. The trail
is drawn in shot coordinates, so it stays fixed to the board while the camera
pans, and it restarts at each cut. Gaps longer than
`review.path_max_gap_seconds` are not joined; `review.path_seconds` limits the
trail to the most recent seconds.

The path is **already in the JSONL** for later analysis, whether or not it is
drawn. Each `position` event has `foot_point`/`foot_point_source` (image),
`foot_point_in_shot`, `hip_center_in_shot`, and `camera_shot`. The teacher's
path is the time-sorted `hip_center_in_shot` values of `position` events with
`role == "teacher"`, compared only within the same `camera_shot`:

```python
import json
path = [(e["start_time"], e["value"]["camera_shot"], e["value"]["hip_center_in_shot"])
        for e in map(json.loads, open("outputs/events/<id>.list_a.jsonl", encoding="utf-8"))
        if e["indicator_name"] == "position" and e["role"] == "teacher"
        and e["value"]["hip_center_in_shot"] is not None]
path.sort()
```

Limitations: shot coordinates are **image-plane** positions relative to the
shot's first frame, not metres on the floor, and they can leave the 0-1 range
as the camera pans. Drift accumulates slowly within a long shot. Featureless
views (a bare wall, a full-screen slide) or fast blurred pans become
unmeasured steps. Positions from different shots cannot be joined without a
floor calibration. Hip height also changes with posture and distance, so use
the horizontal component for movement along the board.

## Live review and annotated MP4

Add `--live` to watch the analysis in a window as each sampled frame is
processed (set `review.live_preview: true` in a local override to make it the
default). The MP4 contains the same overlays, so a run can be reviewed later
without rerunning the model. Both are drawn from the observations and rule
state that produced the JSONL in the same run; they add no rules and do not
change the evidence.

Only **analyzed (sampled) frames** are drawn. With the default 0.2-second
sampling the MP4 plays at about 5 frames per second and keeps the source's
timing, so every frame in it is one the rules actually saw. Frames in between
were never analyzed and are not shown.

| On screen | Meaning |
| --- | --- |
| Blue box | Confirmed track, camera-assumed `student` |
| Orange box | Confirmed track, camera-assumed `teacher` |
| Magenta box | Confirmed track whose role stayed `unknown` (e.g. several people in `instructor_view`) |
| Thin grey box, `provisional` | Tracked for fewer than `min_track_samples`; not counted by any rule yet |
| No box | The model returned nobody there: a missed detection or an untracked person |
| `#7 student 0.82` | Anonymous tracker ID, assumed role, and model box confidence |
| `sitting` / `standing` / `posture ?` | Posture rule result; `?` means evaluated but unknown |
| `standing (partial)` | Judged from shoulder-hip-knee because no ankles were visible (limited coverage) |
| `sitting rise 0.12` / `posture ? rise 0.52` | `shoulder_rise` posture and the shoulder widths above the seated level behind it |
| `still 0.03 bh/s` / `motion ?` | Motion label and the smoothed body-heights/second behind it |
| Yellow line and dot | Teacher's hip-centre path in the current shot (`--path teacher`) |
| `camera: shot 1, pan 2.9% zoom x1.000` | Measured camera step (compensated); `cut or unmeasured` starts a new shot |
| `occluded?`, `cut off` | Visibility proxies; `zone <name>` appears only with calibration |
| Skeleton and white joints | COCO-17 joints at or above `inference.keypoint_confidence` |
| Grey rings (optional) | Joints below that threshold when `review.show_rejected_keypoints: true` |
| Green polygons and crosses | Calibrated zones and the foot point used for them (ankles: `+`, box-bottom proxy: `x`) |
| Top-left panel | Time, tracked/confirmed/provisional counts, student count or teacher presence with coverage, and reminders about camera assumptions |

Live window keys: **space** pauses or resumes, **n** advances one sampled frame
while paused, **q** or **Esc** stops the run. Stopping publishes **nothing**:
the JSONL, summary, and MP4 are all discarded, as for any failed run. Closing
the window hides the preview, but the analysis continues and writes all
outputs. Frames appear as fast as the computer analyzes them, not in real
time. The window needs a desktop session; use `--no-live` on a headless
machine.

The MP4 is H.264 (`review.video_codec: avc1`) when OpenCV can encode it, and
falls back to `mp4v` with a logged warning otherwise. It is written to a
hidden temporary file and published only after the JSONL completes, so an
interrupted run never leaves a half-written video under the final name. The
video is large: about 0.5 MB per second at 720p and 1 MB per second at 1080p
in the test runs, so a full one-hour 1080p lecture may need several GB. Use
`--start/--end` or `--no-save-video` for long recordings.

**Privacy:** the MP4 shows the original classroom footage with overlays. It
stays under the ignored `outputs/` directory; do not commit or share it
without the recording's permission. No face crops or identity matching are
produced.

Typical tuning loop: run a short segment with `--start/--end --live`, pause
where something looks wrong, note the time shown in the panel, then change one
setting in a local override (`configs/*.local.yaml`) and rerun the same range
with a new `--video-id`.

- Many `posture ?` labels: for the instructor, waist-up views without knees
  (cannot be decided from 2D pose), walking strides where the knee is bent
  mid-step, or `keypoint_confidence` too strict; for students, the first 3 s
  of each track, a hidden shoulder, a student away from their seat, or a rise
  between the two `shoulder_rise` thresholds (shown on the label). Enable
  `show_rejected_keypoints` to see which joints fell below the threshold
  (joints a camera does not use are not drawn at all).
- People without boxes: detections below `confidence_threshold`, or the model
  missing small or partly hidden people.
- `#` numbers changing for the same person: tracker ID switches that inflate
  track counts and apparent entries and exits.
- Grey `provisional` boxes that never turn blue: tracks too short to be
  confirmed under `min_track_samples`.

Overlays explain **one run's** decisions. They are not accuracy measurements;
validation still needs labelled clips (see below).

## What List A currently records

| Indicator | Evidence and limitation |
| --- | --- |
| Teacher presence | Visible intervals plus per-sample `visible`, `not_detected`, or `unknown`; only meaningful for the instructor camera. |
| Visible student count | Count at each sampled time, based on confirmed pose tracks; provisional tracks make coverage limited. Not attendance. |
| Persistent anonymous identity | Temporary tracker ID and observed interval. Reacquisition/ID switches can split one person into several tracks. |
| Role classification | Camera-context assignment, **not a learned role classifier**. `student_view` assumes all tracks are students; `instructor_view` assigns teacher only when exactly one confirmed person is visible. Otherwise role is `unknown`. |
| Position | Normalized person box, box center, the camera's usable COCO-17 keypoints, ankle-based foot point or box-bottom proxy (`frame_bottom_cut_off` when the box reaches the image bottom), and path data in camera-shot coordinates (`foot_point_in_shot`, `hip_center_in_shot`, `camera_shot`). |
| Classroom zone | Optional per-track intervals in calibrated camera polygons; `unknown` while the person is cut off at the image bottom. No zone is emitted until a real camera-specific YAML is set. |
| Entry and exit | Only **apparent** appearances/disappearances near the left, top, or right frame edge (or with the box centre near any edge), away from recording boundaries. A door or true room entry cannot be proven from a disappearing track. |
| Visibility/occlusion | Keypoint availability, frame-edge truncation, and overlap proxies; body parts outside the camera's `pose_joints` are `null`. These cannot distinguish occlusion from blur, distance, or model failure. |
| Sitting/standing | Instructor (`legs`): conservative 2D hip/knee-angle rule. When no side shows an ankle, a fallback uses shoulder-hip-knee: standing needs a straight hip (>= standing angle - 10) and a thigh within `thigh_axis_tolerance_degrees` of vertical; sitting needs a bent hip (<= sitting angle) and a near-horizontal thigh; thighs shorter than `thigh_torso_min_ratio` x torso are foreshortened and stay unknown. Fallback results have `limited` coverage; ambiguous full-leg geometry is never overridden. Waist-up views and walking strides still cause unknown. Students (`shoulder_rise`): shoulder rise above the track's own seated level, `limited` coverage (see [Joints each camera uses](#joints-each-camera-uses)). |
| Shoulder rise | `shoulder_rise` cameras only: per-sample shoulder widths above the seated level behind the student posture label, for tuning. |
| Motion level | Smoothed apparent joint displacement in body-heights/second: still/low/medium/high/unknown. Measured in image coordinates on a fixed camera, or in camera-shot coordinates when a moving camera is compensated; unmeasured camera steps are unknown. It is not ground-plane speed or engagement. |
| Camera motion | Moving, compensated cameras only: per-sample shot, measured pan/zoom, background inliers, and the matrix into shot coordinates. |

All events include `video_id`, anonymous `track_id` when applicable,
`role`, indicator name, start/end/duration, value, nullable confidence,
coverage, and optional zone. `confidence` is null for camera-assigned roles
and heuristic states/transitions because no calibrated probability exists;
individual pose keypoints retain their model confidence. JSONL
records are evidence; the JSON summary aggregates valid samples and includes
limitations. `unknown` time is not silently counted as sitting, standing,
still, absent, or zero students.

## Calibration and tuning

Copy `configs/zones.example.yaml` to a camera-specific YAML and replace **all**
example polygons with points measured from that fixed camera's frames. Set
`camera_id: student_view` or `camera_id: instructor_view` in that file, then
set `cameras.<camera>.zones` in an ignored `configs/*.local.yaml` file. If
polygons overlap, the first named zone in the YAML wins. The example file is
not a valid calibration.

Tune `video.sample_every_seconds`, `inference.keypoint_confidence`, and the
`analysis` thresholds in a local override. At 5 sampled frames per second,
the default three-sample track confirmation delays counts by about 0.4 s.
The 0.6-second gap tolerates brief missed tracks. Motion thresholds are
starting values, **not validated classroom measurements**. Both cameras are
static by default; set a camera's `fixed_camera: false` only if it pans or
shakes (as in the web samples); then `compensate_camera_motion` decides
whether motion is measured relative to the scene or left unknown, and
`shoulder_rise` posture is not allowed. Set `analysis.upper_leg_posture:
false` to require full legs for the instructor's posture. If the instructor
works behind a lectern or desk, remove `knees` and `ankles` from that
camera's `pose_joints` (posture then needs another method and stays out of
scope) so guessed legs are not used.
If the student camera includes the teacher, change `role_policy` to `unknown`
until you have a defensible role assignment method; otherwise they will be
counted as a student. The instructor camera's `single_visible` policy is
deliberately cautious when multiple people appear.

List A switches in `configs/indicators.yaml` control emitted indicators.
Hand raises and all other List B/C, interaction, object, and audio indicators
remain out of scope. `scripts/pose_action_demo.py` is still a separate
experimental YOLO11 demo, not the production List A pipeline.

## Code map

- `src/preprocessing/video.py`: decodes video and keeps source timestamps.
- `src/preprocessing/camera_motion.py`: camera pan/zoom/cut estimate and shot coordinates.
- `src/inference/pose.py`: local YOLO26 model, ByteTrack, normalized pose adapter.
- `src/analysis/visual_indicators.py`: pure List A geometry and temporal rules.
- `src/analysis/events.py`: validated event schema and JSONL output.
- `src/analysis/aggregates.py`: evidence-to-summary reduction.
- `src/reporting/annotated_video.py`: overlay drawing, atomic MP4 writer, live window.
- `src/utils/config.py` / `src/utils/paths.py`: YAML validation and paths.
- `src/pipeline.py`: run orchestration; `scripts/analyze.py`: thin CLI.

Run the unit tests after changing a threshold or rule. For any accuracy claim,
manually label representative frames/clips from **both** cameras and measure
person recall, track switches, posture unknown rate, and event precision.
