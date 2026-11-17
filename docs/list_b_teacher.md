# List B: teacher visual behaviour

List B (catalog section B, `docs/video_indicator_catalog.md`) describes what
the teacher does: where they stand and move, which way they face, and what
they do with their hands. Implemented 2026-11-17 for the instructor camera.

## Which indicators need detection, and which are derived

Every pose-based List B indicator is a geometry rule over the keypoints the
pose model already produces for List A, so none needs a new model and they
add almost no run time. Several need no new rule at all: they are summaries
of events List A already saves.

| Indicator | How it is produced | Where |
| --- | --- | --- |
| Position heatmap | Derived: accumulate the saved teacher `position` events (hip centre or foot point) in a grid at report time | report stage (not built yet) |
| Movement path and distance | Derived: summary `teacher_path` sums hip-centre steps between samples (image plane, in frame heights) | `src/analysis/aggregates.py` |
| Classroom coverage | Derived: List A `classroom_zone` intervals per zone (`zone_track_seconds`), now that the room has zones | List A + zone file |
| Stationary episodes | New rule: hip speed (median of 3 samples) <= 0.08 body heights/s for >= 5 s; start, end, duration, location, zone | `teacher_indicators.py` |
| Facing the class | New rule from which face keypoints are visible (below) | `teacher_indicators.py` |
| Head-orientation target | New rule: facing + head pitch + a ray toward calibrated targets | `teacher_indicators.py` |
| Board/screen interaction | New rule: in front of a target and looking at it, pointing at it, or back to the class with a hand on it | `teacher_indicators.py` |
| Writing on board | New rule: board interaction + back or side to the class + wrist on the board + feet still, majority of 3 samples; board third as zone | `teacher_indicators.py` |
| Pointing | New rule: straight, extended, lifted arm; shoulder-to-wrist ray picks the target | `teacher_indicators.py` |
| Gesture rate and expansiveness | New rule: wrist speed relative to the torso while not walking or writing; rate per minute in the summary | `teacher_indicators.py` + summary |
| Raised hand | New rule: wrist above the head (or well above the shoulder) with the elbow up | `teacher_indicators.py` |
| Posture/torso lean | New rule: sideways torso tilt from vertical, plus a per-sample angle for tuning | `teacher_indicators.py` |
| Visual coverage of student areas | Not done (user decision 2026-11-17: needs both cameras together) | - |
| Proximity to students | Not done: needs the two cameras combined | - |
| Teaching-material use | Needs an object detector (datasets below); not trained yet | - |
| Observable facial actions | Needs face crops and an expression model (datasets below); not trained yet | - |

## Running

The switches are in `configs/indicators.yaml` under `teacher:`; thresholds
are in `configs/default.yaml` under `teacher:`. They run only on a camera
whose expected role is teacher, and only for the track holding the teacher
role. Board, screen, and desk *targets* come from the camera's zone file; the
scenario room's is `data/calibration/sgp_room_instructor_view.yaml`, loaded
by `configs/sgp_room.yaml` (the default local config of `run_scenarios.py`).

```powershell
python scripts/run_scenarios.py --scenario A1 --camera instructor_view --indicators facing_direction head_orientation
python scripts/run_scenarios.py --scenario B1 --camera instructor_view --indicators pointing gestures
python scripts/show_indicator.py sgp_A1_instructor_facing_direction_head_orientation head_orientation
python scripts/analyze.py --video data/raw_instructor_videos/sgp_A3_instructor.MOV --video-id a3_writing --camera instructor_view --local-config configs/sgp_room.yaml --indicators writing_on_board board_interaction
```

Indicator names for `--indicators`: `stationary_episodes`,
`facing_direction`, `head_orientation`, `board_interaction`,
`writing_on_board`, `pointing`, `gestures`, `raised_hand`, `torso_lean`.
The review video draws the targets (orange outlines), adds the labels under
the teacher's box (for example `side-on (right) | head: screen | points:
board`), and an arrow along a pointing arm.

## Events

| `indicator_name` | Kind | Values |
| --- | --- | --- |
| `facing_direction` | interval | `toward_class`, `sideways_left`, `sideways_right`, `away_from_class`, `unknown` (left/right are image directions) |
| `head_orientation` | interval | `students`, `notes`, `board`, `screen`, `desk`, `front_wall`, `side_left`, `side_right`, `unknown` |
| `board_interaction` | interval | `board`, `screen`, `none`, `unknown` |
| `writing_on_board` | interval | `writing`, `not_writing`; `zone` = `board_left/center/right` |
| `pointing` | interval | `board`, `screen`, `desk`, `students`, `side`, `none`, `unknown` |
| `gesture` | interval | `gesturing`, `hands_still`, `walking`, `writing`, `unknown` |
| `gesture_expansiveness` | sample | `wrist_reach_shoulder_widths` while gesturing |
| `raised_hand` | interval | `none`, `one`, `both`, `unknown` |
| `torso_lean` | interval | `upright`, `lean_left/right`, `strong_lean_left/right`, `unknown` |
| `torso_tilt` | sample | `degrees_from_vertical` (+ = toward the image right) |
| `stationary_episode` | interval | `hip_center` (normalized), `samples`; `zone` = floor zone |

The summary adds `teacher_behavior_seconds` and `teacher_behavior_episodes`
(per indicator and label), `teacher_gestures_per_minute`,
`teacher_stationary`, and `teacher_path`.

## How the rules decide

- **Facing** uses which face keypoints the model sees, checked on the
  scenario videos: facing the class shows the nose and both eyes (frontal
  and three-quarter views); a profile along the wall shows one eye and one
  ear, and that side gives the direction; a back view shows shoulders but no
  nose or eyes. Shoulder order is not used: pose models do not reliably swap
  left and right for people seen from behind (at 18 s in A1 the shoulders
  still looked wide in a clear profile).
- **Head orientation**: facing the class means `students`, or `notes` when
  the nose sits at least one eye-gap below the eyes (a clearly bowed head).
  Side-on, a ray at head height toward that side finds the first target,
  else `side_left`/`side_right`. With the back to the class, the target
  whose x range holds the head, else `front_wall`.
- **Pointing**: elbow at least 150 degrees, shoulder-to-wrist at least 0.6
  torso lengths, the arm at least 40 degrees up from hanging but not within
  35 degrees of vertical (that is a raised hand). The ray from the wrist
  along the arm names the target. With no target hit: `students` when facing
  the class (calling on someone, or addressing the room), `side` when
  side-on (a door, a window, a group at that side). Other reasons a teacher
  points that the targets cover: at slide content (`screen`), at board work
  (`board`), at their own laptop or notes (`desk`).
- **Writing**: the wrist must be inside the board polygon with the forearm
  level or raised (writing low on a board keeps the forearm level), the
  teacher side-on or back to the class, feet still (hip speed below 0.25
  body heights/s), for the majority of the last three samples. Without a
  calibrated board it falls back to back-to-class with a lifted forearm,
  marked `limited`.
- **Gestures**: wrist movement relative to the shoulder midpoint, in torso
  lengths per second (>= 0.3), so walking does not count as gesturing.

## Checks so far (2026-11-17; not accuracy measurements)

- **A1 scenario** (keypoints replayed): facing the class 254 of 271 s,
  turned toward the screen 17 s (checked in frames at 4, 18, 47 s), no false
  board interaction while gesturing in front of the screen (59 s, fixed),
  stationary episodes such as 176-235 s in the front-left zone, no pointing
  (the instructor used open-hand gestures).
- **HowToSpeak.mp4**, three cut-free shots with a board outline per shot:
  writing 45:01-45:13 and 45:31-45:34 found (17 of 37 samples); pointing
  along the board 47:19-47:40 found (8 samples board, 1 students) with no
  writing; side-on writing 47:49-48:07 found in 16 of 19 samples after the
  forearm rule was relaxed (the wrist was level with the elbow).
- **Seated reading (A4)**: glancing down at a laptop with the eyes, without
  bowing the head, is not visible to body keypoints (nose-below-eyes 0.52-0.59
  versus 0.37-0.50 while talking), so it reads `students`. A head-pose or
  face model on face crops would be needed.
- **People on the screen**: at 1280-pixel input the model found the people
  in the slides' photos (A2: three, A3: one) and the instructor's role was
  `unknown` for whole lessons. `cameras.instructor_view.ignore_people_inside_targets:
  [screen]` now drops boxes lying at least 85% inside the screen.

## Teaching material and facial expressions: datasets (not trained yet)

Both need new detectors. Training on this CPU-only machine would be slow,
and downloads need the team's go-ahead, so nothing is downloaded or trained.

**Objects (teacher and students).** The COCO-pretrained YOLO26 detector
(`yolo26n.pt`, a separate download) already knows `cell phone`, `laptop`,
`book`, `bottle`, `cup`, `keyboard`, `mouse`, `remote`, `scissors`,
`backpack`, so phones and laptops need no training at all. Pens, notebooks,
calculators, and markers need fine-tuning:
- Roboflow Universe "Study desk items" (CC BY 4.0, about 1,200 images):
  calculator, laptop, mobile phone, pen, lecture notes, and more.
- Ultralytics' Open Images V7 detectors (601 classes, YOLOv8 only) may
  cover pen, calculator, whiteboard; check `model.names` before relying on it.
- SCB-Dataset (Whiffe/SCB-dataset; up to 20 classes including phone use,
  computer use, reading, writing, board writing, blackboard, screen) is the
  closest classroom match, but its license is unstated and the student
  source register already notes it needs copyright approval.

**Facial expressions.**
- Chinese university classroom emotion images (Data in Brief 2024, Mendeley
  Data, CC BY 4.0, 5,527 images): surprise, happiness, neutral, confusion,
  boredom, from real classrooms.
- Roboflow Universe face-expression detection sets in YOLO format (for
  example "YOLO Facial Exp Detection", CC BY 4.0, seven classes): small;
  check size and quality.
- FER2013, RAF-DB, AffectNet, ExpW: large general sets; RAF-DB and AffectNet
  are non-commercial academic licenses, FER2013 has no stated license.
- DAiSEE (engagement, boredom, confusion, frustration in e-learning video):
  research use only via a request form.
Faces in the scenario 4K videos are 120-170 px wide, enough for crops from
the full frame (see the earlier face-size measurement). Record expressions
as observed facial actions, not emotions or engagement, and keep face crops
local.
