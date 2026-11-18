# One command per lesson

Added 2026-11-18 at the user's request: `scripts/analyze_lesson.py` takes a
lesson's two recordings (student camera and instructor camera) and produces
every output in one go. Before that, each step was its own command
(`run_scenarios.py` or `analyze.py` per camera, `link_cameras.py`,
`fuse_lesson.py`). Those still work for redoing one step.

## What it does

```text
student video ----\                                   /-> <student run>.list_a.jsonl/.json/.mp4
                   >- 1 line up by real time -> 2 analyze each camera
instructor video -/   (keep the shared stretch)       \-> <instructor run>.list_a.jsonl/.json/.mp4
                                                         |
   3 (optional, --objects) COCO objects on both cameras -> <run>_objects.objects.jsonl/.json/.mp4
   4 link the cameras by timing ------------------------> <lesson>.cross_camera.json
   5 fuse into 5 s windows + lag analysis --------------> <lesson>.fusion.json, .fusion_windows.csv,
                                                          .fusion_links.csv, visualizations/<lesson>.fusion.png
   6 record what ran -----------------------------------> <lesson>.lesson.json
```

1. **Line up.** From each file's recording-start tag and duration (or a
   manifest row's corrected start and sync window), the stretch both cameras
   recorded is computed, and each camera is analyzed only there. On A1 this
   gives the same 8 s student offset as the manifest.
2. **Analyze** each camera with every switched-on indicator
   (`configs/indicators.yaml`; Lists A, B, C, facial actions), with the
   teacher's path drawn in the instructor review video.
3. **Objects** (only with `--objects`): the isolated COCO detector, for
   review. The student analysis already runs its own phone/laptop/book
   detection for device use.
4. **Link** the cameras (students who walk to the front, the teacher walking
   into the student view): docs/list_c_students.md.
5. **Fuse and test** whether student changes follow teacher actions:
   docs/fusion_lag_analysis.md.
6. **Record** the lesson: run IDs, analyzed ranges, whether each camera run
   was analyzed or reused, timings, findings, and every output path.

Existing lesson outputs are checked before step 2, so a name clash fails at
once, not after an hour of analysis. Code: `src/lesson.py` (planning,
alignment, steps), `src/reporting/charts.py` (timeline chart); the CLI only
parses arguments.

## Commands

Repository root, TeachLens environment:

```powershell
conda activate TeachLens
python scripts/analyze_lesson.py --student-video data/raw_student_videos/sgp_A1_student.MOV --instructor-video data/raw_instructor_videos/sgp_A1_instructor.MOV --name lesson_01
python scripts/analyze_lesson.py --scenario A2 --tag lesson
python scripts/analyze_lesson.py --scenario A3 --scenario A5 --tag lesson
python scripts/analyze_lesson.py --all-scenarios --tag lesson --dry-run
python scripts/analyze_lesson.py --scenario B1 --tag try --start 2:00 --end 4:00 --no-save-video
python scripts/analyze_lesson.py --scenario A1 --tag full --reuse
```

- Two files: `--student-video`, `--instructor-video`, `--name` (runs become
  `<name>_student` and `<name>_instructor`). A clip trimmed on an iPhone keeps
  the uncut start in its tag; pass the real start with `--instructor-start`
  / `--student-start` (ISO 8601), or cut clips with `scripts/trim_video.py`.
- Manifest scenarios: `--scenario` (repeatable) or `--all-scenarios`; runs are
  the manifest IDs plus `_<tag>`, the lesson is e.g. `sgp_A2_<tag>` or
  `sgp_A4_t1_<tag>`. A scenario with several takes uses the one take that
  has both cameras (A4: take 1), else `--take`.
- `--start/--end`: seconds (or `[HH:]MM:SS`) after the shared start, for a
  quick look at part of a lesson. `--full` analyzes whole files instead of
  the shared stretch.
- `--reuse` keeps camera runs that already exist and redoes only linking and
  fusion (seconds instead of hours); `--overwrite` replaces outputs.
- `--local-config` defaults to the scenario room (`configs/sgp_room.yaml`,
  board/screen/desk targets); give another room's file, or `none`.
- Also: `--save-video/--no-save-video`, `--objects`, `--no-plot`,
  `--device`, `--weight`, `--output-dir`, `--dry-run`.

At the end it prints one line per lesson: windows, camera links, findings,
and the path of `<lesson>.lesson.json`. A failed lesson is reported and the
next one still runs; the exit code is 1 if any failed.

## Run time (CPU, 2026-11-18)

This computer has no CUDA GPU, so the models run on the CPU. Decoding the 4K
HEVC files was the first bottleneck: 10 s of video took 37 s to decode and
convert in software. Two changes (2026-11-18): frames that are not analyzed
are only grabbed, not converted (23 s), and `video.hardware_decoding: true`
lets OpenCV decode on the graphics chip (1.7 s; falls back to software where
there is none). Hardware frames differ from software ones by about 1 grey
level on average; on A4 every pose-based result was identical, and face-based
labels moved by a few seconds (e.g. talking 7 s vs 6 s, dozing 5 s vs 1 s).

A4 (58 s shared), whole command:

| Decoding | Student camera | Instructor camera | Total |
| --- | --- | --- | --- |
| software, every frame converted | 8.0 min | 6.1 min | 14.3 min |
| hardware, analyzed frames only | 6.8 min | 1.9 min | 8.9 min |

The student camera's models now dominate (pose at 1280 pixels, the phone
detector at 1920, 5 face meshes per student per second, hand gestures): 6 to
9.5 minutes per minute of lesson, plus under 2 for the instructor camera
(scenario runs below).
Linking and fusion take seconds.

## Scenario lessons (2026-11-18)

Every scenario with both cameras, one command each, all indicators on,
hardware decoding, default settings. A1 reuses its earlier full runs
(`--scenario A1 --tag full --reuse`); A4 also has a software-decoded run
(`sgp_A4_t1_lesson`) and a hardware-decoded one (`sgp_A4_t1_hw`).

```powershell
python scripts/analyze_lesson.py --scenario A1 --tag full --reuse
python scripts/analyze_lesson.py --scenario A4 --tag lesson
python scripts/analyze_lesson.py --scenario A3 --scenario A5 --scenario A2 --scenario A6 --scenario B1 --tag lesson
```

| Lesson | Shared | Compute | Teacher faces class / gestures | Hand raises (answered likely) | On target (excluding looking down) | Phone share | Camera links | Strongest teacher -> student link | Findings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgp_A1_full | 275 s | reused | 0.94 / 0.51 | 11 (1) | 0.93 (0.97) | 0.11 | 0 | gesturing -> off target +25 s, r -0.43, q 0.29 | 0 |
| sgp_A2_lesson | 320 s | 44 min | 0.93 / 0.42 | 8 (5) | 0.65 (0.98) | 0.23 | 0 | facing class -> on target +20 s, r -0.58, q 0.05 | 0 |
| sgp_A3_lesson | 240 s | 31 min | 0.99 / 0.21 | 4 (1) | 0.81 (0.93) | 0.14 | 1 | facing class -> on target +20 s, r +0.35, q 0.13 | 0 |
| sgp_A4_t1_lesson | 55 s | 14 min | 1.00 / 0.48 | 1 (0) | 0.32 (0.97) | 0.46 | 0 | none (11 windows: too few to correlate) | 0 |
| sgp_A5_lesson | 270 s | 36 min | 0.90 / 0.43 | 12 (4) | 0.58 (0.85) | 0.27 | 0 | facing class -> on target +5 s, r -0.53, q 0.08 | 0 |
| sgp_A6_lesson | 355 s | 42 min | 0.91 / 0.49 | 14 (8) | 0.58 (0.91) | 0.37 | 0 | facing class -> talking +5 s, r +0.42, q 0.22 | 0 |
| sgp_B1_lesson | 620 s | 104 min | 0.95 / 0.49 | 34 (10) | 0.66 (0.81) | 0.15 | 2 | walking -> talking +25 s, r -0.29, q 0.55 | 0 |

Shares are of known time; six students (median) in every lesson; the
teacher was visible in 98-100% of valid samples. A4 is software-decoded
(14 min; 9 min with hardware decoding). Each lesson's files are listed in
`outputs/reports/<lesson>.lesson.json` (12 files per lesson, 34-369 MB,
mostly the review MP4s).

What the runs show:

- **The pipeline ran start to finish on every lesson** with no failures.
  The student camera dominates the time (6-9.5 minutes per minute of
  lesson; B1's 10 minutes took 97 minutes), the instructor camera under 2
  minutes per minute.
- **No lesson gives a finding.** Board/screen, writing, and pointing
  happened in at most 4 windows of any lesson, so they were never tested;
  the teacher faced the class 90-100% of the time, which leaves little to
  correlate. The near-misses (A2 q 0.05, A5 q 0.08) are on 54-64 windows.
- **Note-taking reads as "not on target".** In A2 and A5 the strongest links
  say that facing the class is followed by fewer students on target and more
  looking down or writing 20-25 s later, which looks like students taking
  notes after an explanation. `S_on_target` counts looking down as not on
  target; excluding looking down, students were on target 81-98% of the time
  in every lesson. A fused "attentive" outcome that treats writing as
  engaged would separate the two.
- **B1 (the longest, least scripted lesson)**: 124 windows, 34 hand raises.
  After the teacher turns to the class (13 onsets), answers rise by 0.17 per
  window (p 0.014), but after correcting for the number of tests q is
  0.34, so it is not a finding. The
  camera links caught the teacher walking into the student view twice
  (20:18:15 and 20:18:34), visible as the gaps in the teacher timeline.

These are scripted, 1-10 minute recordings with the same six students, so
the absence of findings says more about the data than about teaching. The
method needs real lectures (or several lessons pooled) and a check against
a human-coded lesson before any link is reported as a teaching strength or
weak point.
