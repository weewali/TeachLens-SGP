# Two-camera pipeline: JSONL events

`main code/Main Code.py` writes one JSONL file per camera for each lecture, in the team's event format
(`src/analysis/events.py`, the same file as in the `scripting-frame-extraction` pipeline). These files are the hand-off
to the fusion layer.

```
outputs/events/<session>_instructor_view.jsonl
outputs/events/<session>_student_view.jsonl
```

`<session>` is the `--session` name (or the lecture folder name). `outputs/` is local and ignored by Git. A run stops
before any work if the files already exist; pass `--overwrite` to replace them.

## Fields on every line

`video_id`, `track_id`, `role`, `indicator_name`, `start_time`, `end_time`, `duration`, `value`, `confidence`,
`coverage`, `zone`, `start_real_time`, `end_real_time`, as defined in `src/analysis/events.py`.

- **Times** are seconds into each **original** video. The pipeline cuts both videos to the time both cameras were
  recording, then adds the cut back, so a time can be checked against the original file.
- **Real times** are wall-clock times: the recording start in that video's metadata plus the video time, about
  +/-1 s. They line the two files up. They are null when the video has no recording-time tag (downloads, edited or
  re-encoded files). If only one video has a tag and `--offset` is given, the other camera's start is worked out from it.
- **confidence** is null: these are rule-based states, not calibrated probabilities.
- **coverage** is `adequate`, `limited` or `unknown` (availability of evidence, not accuracy). `unknown` values are
  written as their own value, never as "no".
- Instant observations (one sampled frame) have `start_time == end_time`. Intervals end one sample after their last
  frame. Frames are sampled every 2 s.

## Instructor camera (`role: "teacher"`)

| indicator_name | kind | value | notes |
|---|---|---|---|
| `teacher_presence_sample` | every frame | `visible` / `not_detected` | `coverage: limited` when several people were in the teacher zone and the most confident one was taken |
| `teacher_presence` | interval | `visible` | |
| `teaching_state` | interval | `explaining` (facing the class), `interacting` (facing the class with a raised hand), `board_work` (in the board region, facing it), `other` (visible, not facing the class) | `coverage: limited` when the calibration says the camera is not on the students' side |
| `sitting_standing` | interval | `sitting` / `standing` / `unknown` | needs hips and knees in view |
| `position` | every visible frame | `box_center`, `box_xyxy`, `hip_center_in_shot` (fractions of the frame; null without both hips), `camera_shot` (always 0: fixed camera), `moved_since_previous_sample` | movement and path for the fusion layer |
| `stationary_episode` | interval | `stationary` | moved less than 1% of the frame between samples for at least 10 s |
| `raised_hand` | interval | `none` / `one` / `both` | wrist above the shoulder; covers all visible time (was `raised` intervals only before 2026-11-19) |
| `facing_direction` | interval | `toward_class` / `sideways_left` / `sideways_right` / `away_from_class` / `unknown` | the team's labels: face toward the camera = toward the class; shoulders but no face = back to the class; else the side the nose points to in the image. All `unknown` when `camera_faces_class` is false |
| `board_interaction` | interval | `board` / `none` / `unknown` | `board` = the `board_work` state; all `unknown` without a board region |
| `gesture_reach` | every visible frame | number | how far the wrists reach from the shoulders, in shoulder-widths |

`zone` is `board` or `away_from_board` when a board region is calibrated, otherwise null. `track_id` is null.

## Student camera (`role: "student"`)

| indicator_name | kind | value | notes |
|---|---|---|---|
| `visible_student_count` | every frame | number | students seen in a seat (detections that fit no seat are left out); not attendance |
| `head_posture` | interval per seat | `up` / `down` / `face_hidden` / `unknown` | raw posture; writing and reading look like `down` |
| `head_down_still` | interval per seat | `still` | head down or face hidden with almost no movement for 30 s or more; can be resting or quiet reading |
| `hand_raise` | interval per seat | `raised` | wrist above the shoulder for 2 samples or more |
| `hand_raise_episode` | one per raise, per seat | `{"samples": n}` | the same raises, one event each (the fusion layer counts them) |
| `device_use` | interval per seat | `phone` / `none` | a phone detected in the student's box; `none` = no phone seen (one under the desk is missed). May be course work, so it is not labelled distraction. Covers all time the seat was seen (only `phone` before 2026-11-19) |
| `head_down` | interval per seat | `down` / `up` / `unknown` | the team's labels for `head_posture` (`face_hidden` is `unknown`) |
| `orientation_proxy` | interval per seat | `on_target` / `off_target` / `looking_down` / `unknown` | head up with the face toward the student camera = `on_target`; head up, face turned away = `off_target`. Assumes the student camera is at the front of the room. Unlike the team's rule, facing a classmate who is answering is not recognised |
| `alertness` | interval per seat | `possible_sleep` / `awake` / `unknown` | `possible_sleep` = head down and still (above); `awake` = head up; closed eyes cannot be seen |
| `learning_material_use` | interval per seat | `book` | a book detected in the student's box |
| `sitting_standing` | interval per seat | `sitting` / `standing` / `unknown` | often `unknown`: desks hide legs |

`zone` is the anonymous seat (`seat_01`, `seat_02`, ..., numbered row by row from the top left of the student camera's
view). By default the seats are found from where students sat: shoulder midpoints are grouped over the whole lecture,
and a place counts as a seat when it is occupied in at least 10% of the samples. A person is matched to a seat within
one shoulder width (moving up counts half, so a student standing up stays in their seat), and each seat holds at most
one person per sample. Detections that fit no seat, such as legs under a desk, a chair or a bag seen as a person, are
left out. `"seats": "grid"` in the student calibration switches back to equal cells of `seat_grid`. `track_id` is null
because there is no tracker.

"Off-task" is not written. It is an interpretation, and the fusion layer can derive it from these observations.

## The team's fusion layer on these files (stage 6, since 2026-11-19)

Main Code.py runs the team's fusion layer (`src/analysis/fusion.py`, `lag_analysis.py`; `docs/fusion_lag_analysis.md`)
on the two files right after writing them. It replaced this pipeline's own lag analysis (user decision 2026-11-19).
Only the features these rules measure are analysed; the rest are left out, not counted as zero:

| Fusion feature | From | Notes |
|---|---|---|
| T_facing_class | `facing_direction` `toward_class` | only when `camera_faces_class` is true |
| T_board_or_screen | `board_interaction` `board` | only with a board region |
| T_stationary, T_path | `stationary_episode`, `position.hip_center_in_shot` | |
| S_on_target, S_off_target | `orientation_proxy` | |
| S_phone | `device_use` `phone` | |
| S_head_down | `head_down` `down` | S_looking_down is left out: here it is the same number |
| S_sleep | `alertness` `possible_sleep` | |
| S_hand_raises | `hand_raise_episode` | |
| events: turn_to_class, board_or_screen_start, raised_hand_start | `facing_direction`, `board_interaction`, `raised_hand` | |

Not measured, so not tested: the instructor's gestures, walking, board writing, pointing and head target; students
talking, smiling, writing, reading, using a laptop, moving and answering.

The team's `fuse_lesson.py` lines the cameras up by each event's real time. Here both cameras are already on one clock
(stage 1), so the files are read on that clock (`start_time` minus the cut): this also works for videos without a
recording time. The settings are the team's (`fusion:` in `configs/default.yaml`: 5-second windows, lags of 5-30 s).

Outputs, in the lecture's session folder: `reports/fusion.json` (the team's format plus `measured_by: main_code`,
`not_measured`, `clock`, `warnings`, or `{"error": ...}` when the lecture cannot be analysed), `tables/fusion_windows.csv`,
`tables/fusion_links.csv`, and the plain-language points `reports/links/report_points.json` / `.txt`
(`src/reporting/lag_facts.py` with the `measured_by: main_code` wording in `configs/report_vocabulary.yaml`). The web
app shows these points in the "Links over time" tab; when its AI step is enabled it rewrites them with
`scripts/report.py --you`. The statistics are in `reports/analysis_details.txt` section 3.

## Limits

All rules use hand-set thresholds that have not yet been checked against human labels. The instructor is the most
confident person in the instructor camera's teacher zone, and anyone in the student region of the student camera is a
student, so the instructor walking into the student camera's view is counted as a student there.
