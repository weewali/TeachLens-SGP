# Lesson fusion and lag analysis

Built 2026-11-18 from the SGP1 proposal (FR 6.1-6.4, section 4.1, Figure 4.2):
put the instructor camera's and the student camera's indicators on one lesson
timeline, then test whether changes in the students follow particular teacher
behaviours a few seconds later. This is the stage that turns the two cameras'
evidence into "teacher action X tends to be followed by student change Y".

It reads saved evidence only (the two runs' JSONL); no video, no model. It
finds **associations within a recording, not causes**, and nothing here has
been validated against a human-coded lesson.

## Where it sits

```text
student camera  -> analyze.py / run_scenarios.py -> outputs/events/<student run>.list_a.jsonl  \
                                                                                                  -> fuse_lesson.py
instructor cam. -> analyze.py / run_scenarios.py -> outputs/events/<instructor run>.list_a.jsonl /
   -> src/analysis/fusion.py        window table on real time (FR 6.1-6.2)
   -> src/analysis/lag_analysis.py  lag correlation, lagged model, peri-event (FR 6.3-6.4)
   -> outputs/reports/<name>.fusion.json, .fusion_windows.csv, .fusion_links.csv
   -> outputs/visualizations/<name>.fusion.png
```

The later report stage (HTML strengths / areas to improve) should read
`<name>.fusion.json` rather than rerun anything.

## Running it

`scripts/analyze_lesson.py` runs this step automatically after analyzing both
cameras (docs/lesson_pipeline.md). To redo only the fusion on two saved runs
(the default indicator switches are all on), from the repository root:

```powershell
conda activate TeachLens
python scripts/fuse_lesson.py --scenario A1 --tag full
python scripts/fuse_lesson.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1
```

`--scenario` reads the two run IDs from `data/manifests/sgp_scenario_videos.csv`
(`<video_id>_<tag>`; the report is named `sgp_<scenario>_<tag>`, e.g.
`sgp_A1_full`; `--take` for scenarios with several complete takes). Other options:
`--window` (seconds), `--config`, `--output-dir`, `--no-plot`, `--overwrite`.
A1 (55 windows) takes about 20 s on CPU, nearly all of it the shift tests.

Requires statsmodels (installed into TeachLens 2026-11-18: statsmodels 0.14.5
and patsy; nothing else changed).

## Step 1: the window table

Both runs must have real (wall-clock) times; the cameras line up to about
+/-1 s, so windows are 5 s (`fusion.window_seconds`). The table covers the time
both cameras analyzed (their position samples), one row per window.

- Teacher share features: the share of the window's **known** time with the
  label. Student share features: the share of known student-seconds, i.e. the
  class average. Unknown time is left out, not counted as "no"; a window with
  less than half a window of known evidence (`min_known_fraction`) is empty
  (NaN), and empty windows are skipped by every test.
- Counts: events starting in the window. Path: the teacher's hip-centre path
  length (frame heights; steps across a camera shot or longer than 2.5 s are
  not joined). Stationary: share of the window inside a stationary episode,
  only while the teacher is visible.

| Column | From | Proposal idea |
| --- | --- | --- |
| T_facing_class | facing_direction toward_class | explanation (no audio: facing the class is the proxy) |
| T_looking_at_students | head_orientation students | interaction |
| T_board_or_screen | board_interaction board/screen | board and slide use |
| T_writing | writing_on_board writing | board writing |
| T_pointing, T_pointing_at_students | pointing (any target / students) | pointing, interaction |
| T_gesturing | gesture gesturing | gesturing |
| T_walking, T_path, T_stationary | gesture walking, position, stationary_episode | movement in the room |
| S_on_target (+) | orientation_proxy on_target | attention |
| S_off_target (-) | orientation_proxy off_target | off-task |
| S_looking_down | orientation_proxy looking_down | (notes, phone, or reading) |
| S_phone (-) | device_use phone/phone_likely | phone use |
| S_laptop | device_use laptop | device use |
| S_writing (+), S_reading | note_taking, reading | note-taking, reading |
| S_talking, S_smiling | talking, smile | (facial actions) |
| S_head_down (-), S_sleep (-) | head_down down, alertness dozing/possible_sleep | disengagement |
| S_moving | motion_level medium/high | (movement) |
| S_hand_raises (+), S_answers (+) | hand_raise_episode, hand_raise_outcome answered_likely (counts) | participation |

(+) / (-) is the direction the proposal calls better or worse; it only names
a finding a strength or a weak point. The others are reported as
observations. The list lives in `FEATURES` in `src/analysis/fusion.py`.

## Step 2: the lag analysis

1. **Lag correlation.** For every tested teacher feature and every student
   feature, the Pearson r of student[t] with teacher[t - k] for k = 1..6
   windows (5-30 s later; `max_lag_windows`); the strongest lag is kept.
   r at lag 0 and the strongest reverse lag (students first) are stored too:
   a link that is as strong in reverse is probably a shared trend.
2. **Circular-shift test.** The teacher series is rotated against the
   student series (by more than the largest lag), which keeps each series'
   own slow drift but breaks their alignment; p = share of rotations whose
   strongest forward lag is at least as strong. If a lesson has fewer
   rotations than `permutations` (A1: 42), every rotation is used once, so a
   short lesson's smallest p is 1 / (rotations + 1).
3. **Many tests.** Benjamini-Hochberg q-values over all pairs; only links with
   q <= `alpha` (0.05) become findings.
4. **Lagged model** (the proposal's Figure 4.2):
   `Y_t = b0 + b1 X1_(t-k) + ... + a Y_(t-1) + e`, ordinary least squares with
   Newey-West (HAC) standard errors because windows are not independent; for
   each outcome in `fusion.outcomes`, k in `model_lags`, predictors from
   `model_predictors` that passed the activity guardrail. Y_(t-1) absorbs the
   class's slow drift, so b1 is the change beyond what the previous window
   predicts.
5. **Peri-event averages** for teacher actions that are short and rare
   (board/screen, writing, pointing, turning to the class, walking, raising a
   hand): mean student value in the 4 windows after each onset minus the 2
   before (`peri_event_*_windows`), compared with the same number of random
   onsets; onsets within 10 s of the previous one are merged. Needs at least
   `min_events` (3) onsets.

**Guardrail:** a teacher feature present in fewer than `min_active_windows`
(5) windows is not correlated or modelled. In the first A1 trial a single
window at the screen produced r = 0.82 and absurd model t-values; such
actions are listed under `untested_teacher_features` and only get
peri-event averages once they occur often enough.

## Outputs

`outputs/reports/<name>.fusion.json`:

| Key | Meaning |
| --- | --- |
| student_run, instructor_run, window_seconds, windows, start/end_real_time | what was fused |
| features | each column's indicator, labels, kind, desirable direction |
| feature_activity_windows | windows where each feature was present |
| tested_teacher_features, untested_teacher_features | after the guardrail (untested: windows present) |
| links | per pair: lag_windows/lag_seconds, r, r_lag0, r_reverse_max, p_shift, null_shifts, q (sorted by q) |
| lagged_models | per outcome and lag: terms (coef, se, p, ci95), r_squared, windows_used, hac_maxlags |
| teacher_onsets, peri_event | onset windows per teacher action; per action and outcome: events, mean_change, p_random_onsets, q, curve |
| findings | q <= alpha only: kind (strength / weak_point / observation), method, teacher and student feature, lag or window span, effect, q |
| settings, limitations | the settings used and fixed cautions |

`<name>.fusion_windows.csv` is the window table (also the table view of the
plot), `<name>.fusion_links.csv` the links, and
`outputs/visualizations/<name>.fusion.png` the teacher and class timelines
with hand raises marked. All stay local (ignored by Git).

## Settings (`fusion:` in configs/default.yaml)

window_seconds 5, min_known_fraction 0.5, max_lag_windows 6,
min_active_windows 5, permutations 500, alpha 0.05, seed 0, model_lags
[1, 2, 3], outcomes, model_predictors, min_events 3, onset_min_gap_seconds 10,
peri_event_before_windows 2, peri_event_after_windows 4. Unvalidated starting
values; the same seed gives the same p-values.

## First run: A1 (2026-11-18)

Runs `sgp_A1_student_full` and `sgp_A1_instructor_full` (all indicators):
55 windows of 5 s, 19:16:02-19:20:37.

- Tested teacher features: facing the class, looking at students, gesturing,
  stationary, path. Too rare: board or screen (1 window), walking (2),
  writing and pointing (0).
- Strongest links: gesturing -> fewer students off target 25 s later
  (r -0.43), stationary -> more talking 5 s later (r +0.49), stationary ->
  less smiling and fewer hand raises 20 s later. Each has p 0.023, the
  smallest possible with 42 rotations; after 50 tests q = 0.29. **No
  findings.**
- Lagged model, on-target share: the previous window carries it
  (coefficient 0.45, p 0.0002); gesturing 10 s earlier +0.079 (p 0.055);
  facing the class adds nothing (p 0.85).
- Turning to the class (5 onsets): no change in the share features
  (mean changes 0.01 or less, p 0.51-0.96); hand raises 0.4 fewer per
  window after it (p 0.10), driven by the burst of raises at the start.
  Board/screen had 1 onset, too few.

That is the expected result for A1: a 4.5-minute scripted clip with a teacher
who mostly faces the class and gestures, and students who are almost always
on target, has too little variation and too few windows for a test to
separate a real link from chance. The correlations of about 0.4 at p 0.023
would be worth following up in a full lecture, not reported as findings.

## Limits

- Associations, not causes: a teacher action and a student change may share
  a cause (the lesson plan, the topic, a script). Only consistent results
  across many lessons would support saying a behaviour "works".
- One lesson at a time; pooling several lessons (one model with a lesson
  term) is not built yet.
- Student values are class averages; individual students are not followed
  across the cameras (no identity matching, by design).
- Counts such as hand raises are sparse; one window with four raises can
  dominate a correlation.
- Every indicator's own limits carry over (docs/list_a_pose.md,
  docs/list_b_teacher.md, docs/list_c_students.md, docs/facial_actions.md).
  Audio (teacher questions, explanations) is out of scope, so "explanation"
  is approximated by facing the class.

## Checks

`tests/unit/test_fusion.py` (12 tests, synthetic events, no video): shares
lined up on real time with an offset camera start, unknown evidence left
out, counts and teacher path (no step across a shot), missing real times and
non-overlapping runs rejected, onset merging, a planted 2-window lag found
with a small p, unrelated series not significant, a short lesson unable to
reach a tiny p, a planted lagged-model coefficient recovered, a peri-event
step detected, Benjamini-Hochberg values, and the activity guardrail with
the finding labels. `tests/unit/test_config.py` checks the `fusion:` settings
validation.
