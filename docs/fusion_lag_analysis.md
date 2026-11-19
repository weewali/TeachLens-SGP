# Lesson fusion and lag analysis

Built 2026-11-18 from the SGP1 proposal (FR 6.1-6.4, section 4.1, Figure 4.2):
put the instructor camera's and the student camera's indicators on one lesson
timeline, then test whether changes in the students follow particular teacher
behaviours a few seconds later. This is the stage that turns the two cameras'
evidence into "teacher action X tends to be followed by student change Y".

It reads saved evidence only (the two runs' JSONL); no video, no model. It
finds **associations, not causes**, and nothing here has been validated
against a human-coded lesson.

Revised 2026-11-19 after the first scenario runs gave no findings (user
request, from the scenario booklet docs/test_recording_scenarios.pdf): new
teacher features for the booklet's ineffective-teaching codes (seated at the
desk, reading notes, back to the class), per-student on-task / off-task states
(note-taking now counts as on task), 9 hypotheses fixed in advance and tested
first, plain comparisons, a variation guardrail, pooled and between-lesson
analyses, and facts.json for the LLM stage. Every output and symbol:
docs/outputs_guide.md.

## Where it sits

```text
lessons/<stamp>_<lesson>/student/events.jsonl    \
                                                   -> src/analysis/fusion.py        window table on real time (FR 6.1-6.2)
lessons/<stamp>_<lesson>/instructor/events.jsonl /  -> src/analysis/lag_analysis.py  hypotheses, lag correlation, comparisons,
                                                                                    lagged model, peri-event (FR 6.3-6.4)
   -> lessons/<stamp>_<lesson>/fusion/fusion.json, windows.csv, links.csv, timeline.png
   -> lessons/<stamp>_<lesson>/facts.json   (src/reporting/facts.py) -> LLM report stage (docs/llm_integration.md)
several lessons -> src/analysis/lag_analysis.pooled_analyze -> pooled/<stamp>_<name>/pooled.json, facts.json
```

The LLM report stage reads facts.json rather than rerunning anything.

## Running it

`scripts/analyze_lesson.py` runs this step automatically after analyzing both
cameras (docs/lesson_pipeline.md). To redo only the fusion on two saved runs
(the default indicator switches are all on), from the repository root:

```powershell
conda activate TeachLens
python scripts/analyze_lesson.py --scenario A2 --tag lesson --reuse
python scripts/fuse_lesson.py --scenario A1 --tag full --overwrite
python scripts/fuse_lesson.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1_full --overwrite
python scripts/pool_lessons.py --name scenarios --all
```

`analyze_lesson.py --reuse` also redoes the camera links and the side-by-side
video. `fuse_lesson.py --scenario` reads the two run IDs from
`data/manifests/sgp_scenario_videos.csv` (`<video_id>_<tag>`; the lesson is
`sgp_<scenario>_<tag>`; `--take` for scenarios with several complete takes);
runs are found by ID (outputs/runs/ or a lesson's student/ and instructor/).
Other options: `--window` (seconds), `--config`, `--output-dir`,
`--no-plot`, `--overwrite`. A lesson takes 10-20 s, nearly all of it the
shift tests.

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
| T_back_to_class | facing_direction away_from_class | board writing with the back turned (T-WRITE) |
| T_reading_notes | head_orientation notes | reading notes or slides aloud (T-READ) |
| T_seated | sitting_standing sitting (legs, or behind the desk from the shoulders' height) | seated instructor (T-SIT) |
| T_at_desk | classroom_zone desk_area | staying at the desk |
| S_on_task (+) | per student and second: facing the focus, writing, or reading, and not off task | on task |
| S_off_task (-) | per student and second: phone (seen or likely), dozing, talking toward a neighbour, or facing away | off task |
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

S_on_task / S_off_task combine six student indicators second by second
(`student_state()`): off task wins when both apply; everything else is
unknown and left out. S_on_target counted a student writing notes as not on
target, which made the A2 and A5 near-misses ("facing the class is followed
by fewer students on target") mostly note-taking; S_on_task fixes that.

## Step 2: the lag analysis

0. **Hypotheses first** (`fusion.hypotheses`, 2026-11-19). Nine teacher ->
   student pairs fixed in advance from the booklet's codes, each with the
   expected direction:

   | Code | Teacher feature | Student feature | Expected |
   | --- | --- | --- | --- |
   | T-STAND | T_facing_class | S_on_task | rises |
   | T-STAND | T_looking_at_students | S_on_task | rises |
   | T-GEST | T_gesturing | S_on_task | rises |
   | T-WALK | T_walking | S_on_task | rises |
   | T-ASK | T_pointing_at_students | S_hand_raises | rises |
   | T-SIT | T_seated | S_off_task | rises |
   | T-SIT | T_at_desk | S_off_task | rises |
   | T-READ | T_reading_notes | S_off_task | rises |
   | T-WRITE | T_back_to_class | S_phone | rises |

   They are tested like the links below but corrected only among themselves
   (9 tests instead of 50), so a real effect can pass; `matches_expectation`
   says whether the sign agrees with the booklet. Every other teacher x
   student pair is exploratory and corrected separately. Each hypothesis also
   gets a **comparison**: the student value in the window after the teacher
   behaviour filled at least half a window versus after the other windows,
   with a Newey-West p-value (`comparison_lag_windows`, default 1 = 5-10 s
   later).
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

**Guardrail:** a teacher feature needs at least `min_active_windows` (5)
windows with it AND 5 without it; otherwise it is not correlated or
modelled (`teacher_feature_windows_with_without`, `not_tested`). In the first
A1 trial a single window at the screen produced r = 0.82 and absurd model
t-values; and a teacher seated for a whole lesson (A3) cannot be compared
with themselves standing. Rare actions get peri-event averages once they
occur often enough; whole-lesson behaviours need step 3.

## Step 3: several lessons (pooled)

`scripts/pool_lessons.py` (or `analyze_lesson.py --pool NAME`) tests the
hypotheses over several fused lessons (their fusion/windows.csv):

- **Within lessons:** each lesson keeps its own mean, lags never cross from
  one lesson into the next, the null rotates every lesson's teacher series
  by its own shift, and comparisons add one intercept per lesson. Lessons
  where the teacher feature does not vary are listed and left out.
- **Between lessons:** each lesson's average teacher behaviour against its
  average student value, Spearman's rho over the lessons with an exact
  permutation p, and the student value in lessons where the behaviour filled
  most of the time versus the others. This is how the booklet's design
  (effective A1/A2 vs a seated, reading teacher in A3/A4) shows up, but with
  a handful of lessons it is weak evidence: lessons differ in many ways.

## Outputs

In the lesson folder (docs/outputs_guide.md explains every key):
`fusion/fusion.json` (below), `fusion/windows.csv`, `fusion/links.csv` (one row
per hypothesis or exploratory pair), `fusion/timeline.png`, and `facts.json`.

`fusion/fusion.json`:

| Key | Meaning |
| --- | --- |
| student_run, instructor_run, window_seconds, windows, start/end_real_time | what was fused |
| features | each column's indicator, labels, kind, desirable direction |
| feature_activity_windows | windows where each feature was present |
| tested_teacher_features, untested_teacher_features | after the guardrail (untested: windows present) |
| teacher_feature_windows_with_without | windows with and without each teacher feature |
| hypotheses | per hypothesis: code, expected_sign, lag, r, p_shift, q, matches_expectation, comparison, or not_tested |
| links | exploratory pairs: lag_windows/lag_seconds, r, r_lag0, r_reverse_max, p_shift, null_shifts, q (sorted by q) |
| lagged_models | per outcome and lag: terms (coef, se, p, ci95), r_squared, windows_used, hac_maxlags |
| teacher_onsets, peri_event | onset windows per teacher action; per action and outcome: events, mean_change, p_random_onsets, q, curve |
| findings | q <= alpha only: kind (strength / weak_point / observation), method, teacher and student feature, lag or window span, effect, q |
| settings, limitations | the settings used and fixed cautions |

`fusion/windows.csv` is the window table (also the table view of the plot),
`fusion/links.csv` the tests, and `fusion/timeline.png` the teacher and class
timelines with hand raises marked. All stay local (ignored by Git).

## Settings (`fusion:` in configs/default.yaml)

window_seconds 5, min_known_fraction 0.5, max_lag_windows 6,
min_active_windows 5, permutations 500, alpha 0.05, seed 0, model_lags
[1, 2, 3], outcomes, model_predictors, comparison_lag_windows 1, hypotheses
(the nine above), min_events 3, onset_min_gap_seconds 10,
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

## Revised analysis on the scenarios (2026-11-19)

A3 run again in full and all scenario lessons pooled; results, numbers, and
what they mean: docs/lesson_pipeline.md, "Revised analysis on the scenarios".
In short: the teacher at the desk is followed by more students off task (A5:
58% vs 27%; pooled within lessons r +0.61, between lessons 42% vs 30%), and
gesturing by more students on task, both as the booklet expects, but none
passes q <= 0.05 with these short scripted lessons.

## Limits

- Associations, not causes: a teacher action and a student change may share
  a cause (the lesson plan, the topic, a script). Only consistent results
  across many lessons would support saying a behaviour "works".
- Pooling over lessons (2026-11-19) compares within lessons and between
  lessons; with a handful of scripted lessons both are weak evidence.
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
the finding labels. `tests/unit/test_lesson_outputs.py` (2026-11-19): the
on/off-task rules (note-taking is on task), a planted seated -> off-task
effect found as expected with its comparison, a lesson without variation
not tested, pooling within lessons (a lesson without variation left out),
facts.json keys, ids, directions, and no internal-state words.
`tests/unit/test_config.py` checks the `fusion:` settings validation.
