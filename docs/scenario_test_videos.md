# Scenario test videos (A1-A6, B1)

From 2026-11-16 the List A indicators are tested on TeachLens's own scripted
recordings (`docs/test_recording_scenarios.pdf`), one indicator at a time.
They were filmed on 2026-11-03 (about 19:15-20:35, UTC+04:00) with one camera
on the students and one on the instructor. They show identifiable people, so
the media stay local and git-ignored. Only this note and the manifest are
committed.

| Camera | Files | Format |
| --- | --- | --- |
| student_view | `data/raw_student_videos/sgp_<scenario>_student.MOV` | iPhone 16 Pro Max, 3840x2160 H.264, ~30 fps |
| instructor_view | `data/raw_instructor_videos/sgp_<scenario>_instructor.MOV` | iPhone 17 Pro Max, 3840x2160 HEVC, ~30 fps, rotation flag 180 degrees (OpenCV turns it upright) |

Both sets were cut to the teaching. A4 has `_t1` files (the student recording
cut out early) and an instructor-only `sgp_A4_t2_instructor.MOV`.
`20261103_162859000_iOS.MOV` is an instructor clip filmed after B1 with no
student counterpart (manifest scenario `unknown`). The earlier 720p
re-exports are kept in `data/raw_instructor_videos/unedited/`.

## Real times and lining up the cameras

There is no clap; the two cameras of a scenario are lined up by real
recording time before processing. `data/manifests/sgp_scenario_videos.csv`
holds, per file, the real time of its first frame (`recording_start`) and the
part both cameras recorded (`sync_start_seconds`/`sync_end_seconds`, in
seconds of that file). `scripts/run_scenarios.py` uses both.

- **Trimming on an iPhone keeps the uncut start time.** The instructor clips
  A2, A3, A4_t1, A6, and `20261103_162859000_iOS` were cut at the start, but
  their `com.apple.quicktime.creationdate` is still the uncut start.
  Matching their first frames with the uncut copies in `unedited/` gave the
  seconds cut: 14.067, 21.134, 34.934, 6.400, and 53.668. The manifest's
  `recording_start` adds them (`recording_start_source` says so). A1, A4_t2,
  A5, and B1 start at the uncut first frame, so their tags are right.
- **Check (2026-11-16, one-off, not part of the pipeline):** lining up each
  pair's sound tracks gave the same offset between the cameras as the
  corrected times, within 0.8 s for all seven scenarios (the precision of
  whole-second tags). Student times are their own tags, unchanged.
- Video editors (Windows re-export, FilmCraft) drop or replace the recording
  time. `scripts/trim_video.py` cuts losslessly and stamps the cut's real
  start; prefer it for future cuts.

| Scenario | Student window (s) | Instructor window (s) | Both recorded |
| --- | --- | --- | --- |
| A1 | 8.00-284.43 | whole file | 19:16:01.0-19:20:37.4 |
| A2 | 16.07-342.76 | whole file | 19:23:43.1-19:29:09.8 |
| A3 | 29.14-271.79 | whole file | 19:38:10.1-19:42:12.8 |
| A4 (`_t1`) | 6.94-64.61 | whole file | 19:45:23.9-19:46:21.6 |
| A5 | 1.00-end | 0-272.95 | 19:55:39.0-20:00:11.0 |
| A6 | 7.41-end | 0-368.61 | 20:02:27.4-20:08:28.6 |
| B1 | 2.00-627.74 | whole file | 20:10:46.0-20:21:11.7 |

## Testing one indicator at a time

`--indicators` turns on only the named indicators, so the JSONL, the summary,
and the review video `outputs/debug/<id>.list_a.mp4` show just them (the
video's panel says `showing: ...`). Boxes, tracker IDs, and roles are always
drawn because every indicator builds on them. The runner names each run
`<file>_<indicator>`, for example `sgp_A1_student_sitting_standing`.

```powershell
python scripts/run_scenarios.py --scenario A1 --indicators sitting_standing
python scripts/run_scenarios.py --scenario A1 --camera instructor_view --indicators teacher_presence
python scripts/show_indicator.py sgp_A1_student_sitting_standing sitting_standing --value standing
```

`scripts/analyze.py` has the same `--indicators` option, plus
`--recording-start` for a file whose tag is wrong. Each run decodes the 4K
video again: about 1.2x real time for a student file and 1.35x for an
instructor file on this CPU (A1, both cameras: about 12 minutes). Several
indicators can share one run (`--indicators position entry_exit`).

| Indicator | Camera | Viewer command (after the run) | Values |
| --- | --- | --- | --- |
| teacher_presence | instructor | `show_indicator.py <id> teacher_presence_sample` and `... teacher_presence` | visible, not_detected, unknown |
| visible_student_count | student | `show_indicator.py <id> visible_student_count --coverage adequate` | 0-6 |
| anonymous_tracking | both | `show_indicator.py <id> persistent_anonymous_identity` | one episode per ID; many short ones = ID switches |
| role_classification | both | `show_indicator.py <id> role_classification` | teacher, student, unknown |
| position | both | `show_indicator.py <id> position --track <TRACK_ID>` | box centre, foot point and its source |
| classroom_zone | both | `show_indicator.py <id> classroom_zone` | empty until zones are calibrated |
| entry_exit | both | `show_indicator.py <id> entry_exit` | apparent_entry/exit_at_frame_edge |
| visibility_occlusion | both | `show_indicator.py <id> visibility_occlusion --track <TRACK_ID>` | visible parts, cut-off, occluded?, overlap |
| sitting_standing | both | `show_indicator.py <id> sitting_standing`; students also `... shoulder_rise --track <TRACK_ID>` | sitting, standing, unknown |
| motion_level | both | `show_indicator.py <id> motion_level` and `... motion_speed --track <TRACK_ID>` | still, low, medium, high, unknown |

## Real-room obstacles the pipeline must cope with

- **Student heads at the bottom of the instructor view.** The pose model
  detects them, and any second person makes the instructor's role `unknown`.
  `cameras.instructor_view.ignore_people_starting_below: 0.75` drops boxes
  whose top is in the lowest quarter of the image (head boxes start at
  0.80-0.89; the instructor's boxes never started below 0.60). On 30 sampled
  frames, frames with more than one person fell from 26 to 5 (A3), 12 to 4
  (A5), and 7 to 0 (B1). Measured on the 720p copies; same framing as the 4K
  files.
- **Duplicate boxes on one seated instructor.** A second track appears on
  the same person and presence becomes `unknown`: in the new A4_t1 file
  (only `teacher_presence` on) 26 of 58 samples were `unknown` for this
  reason (visible 29, not detected 3). Not fixed yet.
- **The student camera was reframed between A2 and A3.** No zones are
  calibrated, so nothing depends on the exact view yet.
