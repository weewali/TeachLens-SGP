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
| Gesture rate and expansiveness | New rule: hand movement relative to the torso while not walking or writing, ignoring hands at the face or on the desk; rate per minute in the summary | `teacher_indicators.py` + summary |
| Raised hand | New rule: wrist above the head (or well above the shoulder) with the elbow up | `teacher_indicators.py` |
| Posture/torso lean | New rule: sideways torso tilt from vertical, plus a per-sample angle for tuning | `teacher_indicators.py` |
| Visual coverage of student areas | Not done (user decision 2026-11-17: needs both cameras together) | - |
| Proximity to students | Not done: needs the two cameras combined | - |
| Teaching-material use | COCO objects (laptop, phone, book...) are detected on their own since 2026-11-17 (`docs/object_detection.md`); linking an object to the teacher, and pens/markers, are not done | `scripts/detect_objects.py` |
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
  lengths per second, so walking does not count. A hand gestures when it
  moves at least 0.15, or at least 0.08 while raised to chest height or above
  (holding a hand up toward the screen or talking with small hand movements
  changes little between samples a second apart). Lowered from 0.3 on
  2026-11-18 at the user's request: real gestures in A2 measured 0.12-0.16.
  Ignored: wrists below 0.5 confidence (hands behind the back get jumpy
  guesses), hands within 0.6 shoulder widths of the nose (chin or cheek
  rests), and hands on the desk target (typing, turning pages). Gesturing
  seconds: A1 57 -> 136 of 271, A2 54 -> 123 of 320, A3 (seated reader)
  11 -> 51 of 242; some laptop typing just above the desk outline still counts.

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

## Teaching material and facial expressions: datasets

**Objects (teacher and students).** The COCO-pretrained YOLO26 detector
(`models/detection/yolo26n.pt`, downloaded 2026-11-17) already knows
`cell phone`, `laptop`, `book`, `bottle`, `cup`, `keyboard`, `mouse`,
`remote`, `scissors`, `backpack`, so phones and laptops need no training;
it runs on its own with `scripts/detect_objects.py` (see
`docs/object_detection.md`). Pens, notebooks, calculators, and markers need
fine-tuning:
- Roboflow Universe "Study desk items" (CC BY 4.0, about 1,200 images):
  calculator, laptop, mobile phone, pen, lecture notes, and more.
- Ultralytics' Open Images V7 detectors (601 classes, YOLOv8 only) may
  cover pen, calculator, whiteboard; check `model.names` before relying on it.
- SCB-Dataset (Whiffe/SCB-dataset; up to 20 classes including phone use,
  computer use, reading, writing, board writing, blackboard, screen) is the
  closest classroom match, but its license is unstated and the student
  source register already notes it needs copyright approval.

**Facial expressions (checked 2026-11-17; nothing downloaded or trained).**
- Chinese university classroom emotion images (Data in Brief 2024, Mendeley
  Data 6dbdkb8g3d, 5,527 images; surprise, happiness, neutral, confusion,
  boredom): the record says CC BY 4.0, but the only downloadable file is a
  Data Use Agreement. The images are shared only after an institution signs
  it with East China Normal University (use limited to the named study,
  publication coordination, audit rights, data returned or destroyed at the
  end). A university/supervisor decision, not something to download.
- Roboflow Universe, CC BY 4.0 as declared by the uploader, YOLO detection
  labels, free Roboflow account needed to export:
  - "Affective states" (Malaviya National Institute of Technology Jaipur):
    2,008 images, Boredom, Confusion, Engaged, Frustration, Sleepy, Yawning;
    webcam close-ups; 2 exportable versions. The best fit found.
  - "Emotion in Classroom" (sonyausaktif): 2,272 images, Bored, Confused,
    Focused, happy; webcam images of students; no published version, so it
    must be forked into one's own workspace before export.
  - "student_emotion": 989 images, Neutral, Angry, Happy, Sad; look like
    internet photos, so the uploader's license is doubtful.
  - "Emotion Recognition" (35 images) and "Student Face Emotion" (194):
    too small.
- Kaggle "Facial Emotion Recognition Dataset" (fahadullaha, labelled CC BY
  4.0, 49.8k images, 7 emotions) is built from FER2013 and RAF-DB, whose own
  terms (RAF-DB: academic agreement) still apply; Kaggle "Natural Human Face
  Images" (5.5k) has license "Unknown" (images scraped from the web).
- FER2013, RAF-DB, AffectNet, ExpW, CK+, KDEF, JAFFE: large or standard
  sets, but each has an agreement, request form, non-commercial terms, or no
  stated license.
- LAION EmoNet-Face (CC BY 4.0, synthetic faces) forbids emotion recognition
  in educational institutions in its card: not usable here.
- DAiSEE (engagement, boredom, confusion, frustration in e-learning video):
  research use only via a request form.
**Decision 2026-11-17 (user): no expression classifier is trained.** Facial
expressions will be rules over face landmarks from MediaPipe Face Landmarker
(478 3D points, 52 ARKit-style blendshape scores such as `mouthSmileLeft`,
`browDownLeft`, `eyeBlinkLeft`, `jawOpen`, and a head-pose matrix; package
Apache 2.0). Faces are too small for its own short-range face detector in
the 4K frames, so it runs on square crops around the pose model's head
keypoints (nose and an eye visible), enlarged to 256 px. Feasibility test on
A2 (3 frames per camera, scratch script, not in the repository):
- Student view: 15 of 18 faces toward the camera got a mesh, about 20 ms
  each on CPU. Smiles matched the frames (0.70 and 0.71 smiling, 0.00-0.08
  neutral). No mesh, correctly: a cap brim with the head down, a hand over
  the face. One wrong mesh: a three-quarter face resting on a hand, points
  collapsed onto the ear, so a mesh needs a sanity check against the pose
  keypoints (nose position, mesh width versus eye/ear span).
- Instructor view: the teacher's face meshed in every frame (yaw about
  +20 degrees while facing the class from the side); people in the slide
  photo also meshed, which the screen filter already removes.
- The camera looks down on everyone (pitch +10 to +26 degrees), and neutral
  faces read `eyeSquint` 0.4-0.8, `eyeLookUp` up to 0.66, and blink up to
  0.46 (glasses, lowered eyes). Rules therefore need a per-person baseline
  (change from that track's own typical scores), sustained durations, and
  `unknown` when no mesh is found; report observed facial actions (smile,
  brows lowered/raised, eyes closed, yawn), not emotions.

All the open classroom sets are webcam close-ups, not faces 120-170 px wide
in a room camera, so a model trained on them would run on face crops
(upscaled), and its accuracy on the scenario videos must be checked.
Faces in the scenario 4K videos are 120-170 px wide, enough for crops from
the full frame (see the earlier face-size measurement). Record expressions
as observed facial actions, not emotions or engagement, and keep face crops
local.
