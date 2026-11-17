# List C: student behaviour (student camera)

Added 2026-11-17 at the user's request (catalog section C,
`docs/video_indicator_catalog.md`). Everything runs on the student camera
only; on an instructor camera the switches are skipped with a warning.
Labels are observable proxies: a head down may be reading, writing, or
resting; a phone may be course work; facing the front is not attention.

## What each request became

| # | Request | Indicator(s) | How |
| --- | --- | --- | --- |
| 1 | Seat occupancy (for heat maps of active rows/students) | `seat_occupancy` | A home seat is learnt from where each student sits (steady shoulder position); a student who comes back with a new tracker ID gets the same seat ID. The summary's `seats` table puts seats into rows (row 1 = nearest the camera) with per-seat activity: hand raises, answers, talking, smiling, phone, laptop, writing, reading, head down, sleep, on/off target, out of seat, movement. Heat-map input; the plot is report work (not built). |
| 2 | Hand raised, including slight and near-the-desk raises | `hand_raise` | high (wrist well above the head), medium (above the nose), low (forearm up from the desk below that, held 2 s or with an open palm). Heights are measured against the head because the camera looks down. A hand in the box around the face (chin, cheek, hair, phone call) counts only with an open palm/pointing finger/victory from the gesture model. `hand_raise_episode` events give start, end, highest level, side. |
| 3 | Hand-raise response latency | - | Skipped (user decision): needs the teacher's question from audio. |
| 4 | Hand-raise selection outcome | `hand_raise_outcome` | answered_likely if the student talks for >= 2 s or stands up from the raise until 12 s after it; no_answer_seen otherwise; unknown when the face was mostly not visible. Talking comes from the face model (below). |
| 5 | Head-orientation target, incl. facing a talking classmate | `head_target` | front (toward the camera = teacher/board/screen side), down, up, peer, talking_peer, side, behind. Direction from the face mesh (yaw/pitch), else from which face keypoints show; the nearest classmate in that direction is the peer. |
| 6 | Visual-orientation proxy | `orientation_proxy` | on_target when facing the current focus: a classmate who is answering (talking after a raise, or standing), otherwise the front. Facing a chatting classmate is off_target; looking down (notes, phone, reading) is kept apart; the answering student is `speaking`. Summary: on-target share of known time. |
| 7 | Note-taking/writing | `note_taking` | looking down, a wrist on the desk area moving a little (0.03-0.6 shoulder widths per second relative to the shoulders), no phone/laptop in use; majority of the last 3 samples (students look up every 10-20 s). |
| 8 | Reading | `reading` | the face model sees the eyes looking down (eyeLookDown), hands still, eyes not closed, no phone/laptop in use; a COCO book in front makes it `adequate`. |
| 9 | Device use, phone on the desk or under it, laptop | `device_use` | phone: looking down at a detected phone (COCO) at a hand, or holding it up. Phones are often too small or hidden to detect, so two inferred labels use head orientation and hands (user suggestion 2026-11-18): phone_held_possible = looking down at both hands held together in front of the body for 3 s, not turned to a neighbour; phone_hidden_possible = head down with both hands low (in the lap, under the desk) for 3 s. laptop: looking down with hands or eyes on an open laptop; a closed laptop (a flat box, height/width below 0.7; A2: closed 0.34-0.54, open about 1.07) is not in use and does not rule out a phone. |
| 10 | Learning material use | - | Skipped: no openly downloadable classroom-items dataset found. Roboflow "Study desk items" (CC BY 4.0: pen, lecture notes, calculator...) needs the user's own Roboflow account; the COCO `book` class is used as reading evidence only. |
| 11 | Standing/sitting | `sitting_standing` (List A) | Already implemented (shoulder rise on the student camera). |
| 12 | Out-of-seat movement; walkers lost at 1 frame/s | `out_of_seat` + track stitching | Out of seat: shoulders >= 1.5 widths sideways from the home seat (or beyond standing height) for 2 s; standing up in place stays in the seat. Stitching (`src/inference/tracking.py`, `tracking.stitch`): ByteTrack links boxes by overlap and cannot link a walker whose boxes a second apart do not touch; the stitcher gives a new tracker ID the ID of a person lost within 2.5 s who could have walked there (1.5 box heights/s) and has a similar size, only when exactly one fits. |
| 13 | Entry/exit | `entry_exit` (List A) + stitching | Fewer false entries/exits from fragmented walkers. |
| 14 | Nod/head shake | `head_gesture` | Face model on 5 frames per second (extra frames between samples, face crops only): nod = pitch down and back up (>= 5 degrees) within 1.2 s; shake = yaw reverses twice (>= 8 degrees) within 2 s. |
| 15 | Response gesture | `response_gesture` | MediaPipe Gesture Recognizer (`models/hands/gesture_recognizer.task`) on wrist crops: thumbs_up, thumbs_down, open_palm, pointing_up, victory, i_love_you, none. Applause is not detected. |
| 16 | Head-down episode and sleeping | `head_down`, `alertness` | Head down, from head orientation: the face mesh pitch drops; the head nears or drops below the shoulder line; the ears rise above the eyes or the nose drops well below them (a bowed head seen from above, works without a face mesh, e.g. under a cap); the nose shows but both eyes are hidden; or no face and no head above the shoulders (head on the desk). possible_sleep: the head rests at or below the shoulder line (or the face is hidden) and stays still for 20 s, or the eyes are closed for 10 s. A head bent over a phone stays above the shoulder line, so it is head down but awake (user correction 2026-11-18). |
| 17 | Body movement level | `motion_level` (List A) | Already implemented; the seat table adds moving seconds per seat. A later attention model could use restlessness (frequent medium/high motion) next to orientation, but movement alone is not attention. |
| 18 | Writing/presenting at the board across cameras; teacher entering the student view | `scripts/link_cameras.py` | Reads both saved runs and lines them up by real time: a student leaving the student view at an edge, then an extra non-teacher person appearing in the instructor view within 20 s (with time at the board), and the teacher leaving the instructor view, then a new person entering the student view (counted as a student by mistake). Timing links, not identity; ambiguous cases list every candidate. |
| 19 | Eyes closed as dozing vs attentive | `alertness` | awake (eyes open, or moving with the head up), tired (yawning), dozing (eyes closed >= 2 s), possible_sleep. "Awake" is not "attentive". |

Face-model indicators added for this list (`faces:` switches): `talking`
(inner-lip gap swings, measured on A2: silent <= 0.02 of face height,
talking 0.03-0.12 within a second) and `head_gesture`.

## Models and what turns them on

| Model | File | Used when |
| --- | --- | --- |
| YOLO26 pose + ByteTrack | `models/pose/yolo26n-pose.pt` | always |
| MediaPipe Face Landmarker | `models/face/face_landmarker.task` | any face switch, or alertness, reading, note_taking, device_use, head_down, head_target, orientation_proxy, hand_raise_outcome |
| same, 4 extra frames per second | - | talking, head_gesture, hand_raise_outcome, head_target, orientation_proxy (`faces.frames_per_second: 5`) |
| COCO YOLO26 detector | `models/detection/yolo26n.pt` | device_use, note_taking, reading (phone, laptop, book) |
| MediaPipe Gesture Recognizer | `models/hands/gesture_recognizer.task` | response_gesture, hand_raise, hand_raise_outcome |

With everything on, the student camera runs at about 2.2 s per analyzed
second on this CPU (pose alone: about 1 s). Testing one indicator loads
only the models it needs.

## Testing one indicator at a time

Each run writes `outputs/events/<id>.list_a.jsonl`, the summary, and the
review video `outputs/debug/<id>.list_a.mp4`; the runner names the run
`<file>_<indicators>`. Pick the scenario that shows the behaviour
(`docs/test_recording_scenarios.pdf`; takes do not follow the script times
exactly).

```powershell
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators hand_raise
python scripts/show_indicator.py sgp_A1_student_hand_raise hand_raise_episode
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators hand_raise_outcome sitting_standing
python scripts/show_indicator.py sgp_A1_student_hand_raise_outcome_sitting_standing hand_raise_outcome
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators seat_occupancy
python scripts/show_indicator.py sgp_A1_student_seat_occupancy seat_occupancy
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators out_of_seat
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators response_gesture
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators head_gesture
python scripts/run_scenarios.py --scenario A1 --camera student_view --indicators orientation_proxy
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators talking
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators head_target
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators device_use
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators head_down
python scripts/run_scenarios.py --scenario A2 --camera student_view --indicators alertness
python scripts/run_scenarios.py --scenario A3 --camera student_view --indicators reading
python scripts/run_scenarios.py --scenario A3 --camera student_view --indicators note_taking
python scripts/run_scenarios.py --scenario A3 --camera student_view --indicators yawn alertness
python scripts/show_indicator.py <run id> <indicator> --value <label>
```

Where each behaviour is in the scenarios (script roles): hand raises A1,
A3 (an ignored raise), A5, A6; standing answers A1, A6; chatting A2, A4,
A5, A6; phone A1, A2, A4, A5, A6, B1; laptop A2, A4, A5; notes A1, A3, B1;
reading A3; sleeping A2, A4, A5; yawning A3; looking away A3, A5. No
recorded scenario has a student leaving, arriving, walking to a classmate,
or going to the board, or the teacher entering the student view, so
`out_of_seat`, stitching, and the cross-camera links are checked with
synthetic tests only.

All together, and the cross-camera link (needs both cameras with List A's
position, entry_exit, and teacher_presence, which are on by default):

```powershell
python scripts/run_scenarios.py --scenario A1 --camera student_view --tag all
python scripts/run_scenarios.py --scenario A1 --tag base
python scripts/link_cameras.py --student sgp_A1_student_base --instructor sgp_A1_instructor_base --name sgp_A1
```

## Events and summary

| `indicator_name` | Kind | Values |
| --- | --- | --- |
| `hand_raise` | interval | high, medium, low, none, unknown |
| `hand_raise_episode` | per raise | max_level, side, open_palm_seen, samples |
| `hand_raise_outcome` | per raise | outcome (answered_likely, no_answer_seen, unknown), raise_end, talk_seconds, stood_up |
| `seat_occupancy` | interval | seat ID, away, unknown; zone = seat ID |
| `seat_map` | end of run | learnt seats: normalized position, shoulder width, tracker IDs |
| `out_of_seat` | interval | in_seat, out_of_seat, unknown |
| `head_down` | interval | down, up, unknown |
| `alertness` | interval | awake, tired, dozing, possible_sleep, unknown |
| `device_use` | interval | phone, phone_held_possible, phone_hidden_possible, laptop, none, unknown |
| `note_taking` | interval | writing, not_writing, unknown |
| `reading` | interval | reading, not_reading, unknown |
| `response_gesture` | interval | thumbs_up, thumbs_down, open_palm, pointing_up, victory, i_love_you, none, unknown |
| `head_target` | interval | front, down, up, peer, talking_peer, side, behind, unknown |
| `orientation_proxy` | interval | on_target, off_target, looking_down, speaking, unknown |
| `talking`, `head_gesture` | interval | talking/silent; nod/shake/none (`faces:` switches) |

Summary: `student_behavior_seconds` (per indicator and label, summed over
students), `student_hand_raises` (episodes by highest level, outcomes),
`student_orientation_proxy` (on-target share of known time, with and
without looking down), `seats` (seat and row tables with activity).

## Limits

- One camera, image plane: "front" cannot separate teacher, board, and
  screen; rows are image rows, not a floor plan; a moved camera or a
  reframed shot (A2 to A3) starts new seats.
- Small faces and hands: the gesture model found hands in about 1 in 6
  wrist crops on A1; faces turned away give no mesh, so talking, nods,
  reading, dozing, and outcomes are `unknown` for them.
- Talking is mouth movement: chewing (S-EAT) and laughing look like
  talking; a student who answers turned away from the camera is missed.
- Hand-raise outcomes: only talking or standing within 12 s of the raise;
  without the teacher's speech a raise answered later, or a student who
  talks to a neighbour right after lowering a hand, is misread.
- phone_hidden_possible is inferred from posture, never seen.
- All thresholds are starting values from A1/A2 frames, not validated.

## Checks so far (2026-11-17/18; not accuracy measurements)

Full student-view runs with everything on (A1, A2, A3): 10, 20, and 12
minutes on CPU. Seats: 6 seats in 2 rows in all three, matching the class.

- **Hand raises (A1).** At 0:23 five students raise open palms at head
  height. The first rule (heights against the shoulders) called them all
  `low` or missed them: the camera looks down, so a hand at head height is
  only a little above the shoulders in the image. Measuring against the
  head and adding the face box found four of five (one medium, three low;
  the fifth is beside his face and needs the open palm from the gesture
  model, which the full pipeline has). Chin, cheek, and hair touches at
  1:55, 2:40, and 4:02 no longer count. One-second "raises" (hands moving
  while talking) are gone since low raises must be held 2 s.
- **Hand raises after the fixes (A1 full run).** 0:23: all five raises
  (one medium, four low, open palms seen); 0:42-0:46: the back-right raise
  the teacher points at. Still wrong: a two-handed shrug at 0:43, a hand on
  the cheek at 1:26 and 1:49 (wrist just outside the face box), and a talking
  gesture at 3:45.
- **Hand-raise outcomes (A3).** With "any 2 s of talk within 12 s", all
  six A3 outcomes were answered_likely in a chatty class; answers now need
  3 s of talking while facing the front (talking to a neighbour is a chat).
  Without the teacher's audio, who was really picked cannot be confirmed.
- **Phone users (A2, user corrections 2026-11-18).** The back-right student
  bent over his phone was labelled possible_sleep (head down and still);
  sleep now needs the head resting at or below the shoulder line or a
  hidden face. The front-middle student in a cap used his phone the whole
  time but read `laptop` (a closed laptop in front of him) or nothing: his
  bowed head is now seen from the ear and nose positions, closed laptops do
  not count, and he reads phone_held_possible for 83 of 90 seconds
  (A2 1:30-3:00). The back-right student: head down throughout,
  phone_held_possible 9 s, no sleep. Phones themselves were detected for
  only a few seconds: in hands they are too small or covered, and running
  the detector on an enlarged crop of the hands found laptops on the desk,
  not the phones. Clasped hands look like a phone held in both hands, so
  the B1 camera (moved, steeper) at first gave 30-85 s of
  phone_held_possible to several attentive students; with the ear/nose
  thresholds set from B1's upright students (ears up to -0.52, nose up to
  0.69 eye distances) and hands within 0.6 shoulder widths, B1 6:00-8:00
  has 2 s left while the A2 phone users stay detected.
- **Sleeping (B1).** The back-left student sleeps with his head on his
  arm (about 4:45-10:00 with breaks). The pose model still guesses a nose
  and an eye on the hidden face (confidence 0.4-0.5), so "face hidden" alone
  missed him; his head point lies below the shoulder line (-0.19 shoulder
  widths). B1 4:30-6:30: head down 84 s, possible_sleep 27 s; whole B1:
  possible_sleep 30 s (5:43-6:10 and 9:41-9:44), because each still period
  needs 20 s before the label and he lifts his head now and then.
- **Writing.** Rarely found (A3 1:00-2:30: 11 s): in these takes the
  "note-takers" have no paper or notebook on the desk, so this is mostly
  correct; it has not been checked on real note-taking.
- **Talking (A2).** Mostly two students (121 s and 61 s), matching the
  scripted chatting pair.
