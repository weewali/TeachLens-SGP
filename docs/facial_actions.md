# Student facial actions (face landmarks, rule-based)

Added 2026-11-17. User decisions: facial expressions are detected with
rules over face-landmark points, not a trained expression classifier, and
only for **students** on the student camera, never the teacher.

## How it works

1. The pose model (as for every indicator) finds each student and their
   head keypoints. A face turned toward the camera (nose and an eye
   visible) gets a square crop around the head, enlarged to 256 px.
2. MediaPipe Face Landmarker (`models/face/face_landmarker.task`) puts 478
   points on the crop and returns 52 blendshape scores (facial movements in
   [0, 1], such as `mouthSmileLeft`, `browDownLeft`, `eyeBlinkLeft`,
   `jawOpen`) and a head-pose matrix (yaw, pitch, roll).
3. A sanity check drops meshes that do not sit on the pose model's face
   (mesh nose more than 0.15 crop widths from the pose nose, or mesh
   narrower than 0.30 of the crop; good A2 faces measured 0.03-0.07 and
   0.42-0.49), and faces narrower than 60 px.
4. Rules (`src/analysis/facial_actions.py`) compare each student with their
   **own resting face** over the last 300 s. The cameras look down on the
   class, so neutral faces already read high squint, look-up, and blink
   scores; a fixed threshold would mislabel them. Two A2 findings set how
   "resting" is measured:
   - Smile: the 25th percentile of the student's smile scores. With the
     median, a 20 s run that started while a student was smiling read his
     0.65 smile as normal.
   - Brows, blink, jaw, head angles: the median. With a low percentile, a
     student looking down past a classmate read browUp 0.45-0.54 for 40 s
     against an early 0.06-0.24 and was labelled "brows raised" throughout,
     though his brows were not raised (the head angle had changed). With
     the median, a lasting shift becomes the new normal within seconds.
   The first 10 faces of each track only build this baseline (label
   `unknown`, shown as "learning usual look").

| Indicator (`--indicators`) | Labels | Rule (defaults in `configs/default.yaml` `faces:`) |
| --- | --- | --- |
| `smile` | `smiling`, `not_smiling` | mean mouthSmile >= 0.35 and >= 0.20 above usual |
| `brows` | `lowered`, `raised`, `neutral` | browDown, or browUp (inner + outer), >= 0.30 and >= 0.25 above usual |
| `eyes_closed` | `closed`, `open` | mean eyeBlink >= 0.60 and >= 0.30 above usual, for >= 2 s |
| `yawn` | `yawning`, `not_yawning` | jawOpen >= 0.50 and >= 0.35 above usual, for >= 2 s |
| `head_direction` | `toward_camera`, `turned_left`, `turned_right`, `down`, `up` | yaw beyond +/-25 degrees from the camera axis (image directions), or pitch 15 degrees below/above the student's usual pitch |

Every label is `unknown` when there is no usable face that second: head
down on the desk, cap brim, hand over the face, turned away, too small, or
the mesh failed the check. Unknown is never counted as "not smiling".

## Running

Switches are in `configs/indicators.yaml` under `faces:` (all on). On an
instructor camera they are skipped with a warning.

```powershell
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators smile --tag smile
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators smile brows eyes_closed yawn head_direction --tag faces
python scripts/show_indicator.py sgp_A2_student_faces smile --value smiling
python scripts/show_indicator.py sgp_A2_student_faces facial_scores --track 2 --limit 20
```

The review video draws the face points (light green) and adds a line under
each student, for example `SMILING 0.71 | brows neutral | eyes open` and
`no yawn | head turned left (yaw -32)`; `face ? (none toward camera)` when
there is no usable face.

## Events and summary

Each indicator is an interval event per student track. `facial_scores` is a
per-sample event with the raw values behind the labels (smile, brow_down,
brow_up, eye_blink, jaw_open, yaw, pitch, roll, face_width_px) and the
student's current baseline, for threshold tuning. Face points and face
images are never saved; only these scores and angles are. The summary adds
`student_face_seconds` and `student_face_episodes` (per label, summed over
students; unknown included) and `student_face_samples_with_mesh`.

## Limits

- Observed facial actions, not emotions, engagement, or attention.
- One sample per second: blinks are not seen, and eyes-closed and yawn
  episodes start at the sample where their 2 s minimum is reached.
- A student who smiles more than about three quarters of the time raises
  their own resting level, so some of their smiles are missed; a brow, eye,
  or jaw state held for more than half of the recent history becomes that
  student's normal. The first 10 faces of each track are unknown, and a new
  tracker ID or a --start segment starts a new baseline (prefer whole
  synced windows).
- Yaw is relative to the camera, not to the teacher or board; the student
  camera sits at the front, so "toward camera" is roughly "toward the front".
- Thresholds are starting values from one scenario (A2), not validated.

## Checks so far (2026-11-17; not accuracy measurements)

A2 student view, synced window (326 samples, 6 students), all five
indicators, CPU: 382 s (the pose-only run of the same window is about
300 s; each face adds about 20 ms).

- Faces: 1,956 crops (faces toward the camera), 1,391 meshes (71%), 546
  with no mesh (cap brim with the head down, hand over the face, turned
  away), 19 rejected by the sanity check.
- Seconds per label, summed over students (known 1,326, unknown 623
  including each track's 10-face learning period): smiling 233, brows
  lowered 17 and raised 19, head turned right 29 / left 9 / down 5 / up 5,
  no closed eyes and no yawns (none happen in A2, so those two rules are
  untested on real video).
- Checked in frames: smiles at 2:50 and 3:53 matched visible smiles; a
  back-row student looking down at a phone at 1:25 read smiling 0.68,
  plausible but hard to confirm at that size; the brow artifact described
  above (head angle, not an expression) dropped from 40 s to 10 s with the
  median resting level. Labels in the review video avoid covering faces.
- Next: B1 (a sleeping student) and other scenarios for eyes closed and
  head down; a sleeping head on the desk usually gives no mesh, so
  `unknown`, which needs the pose-based head-down rule of List C instead.
