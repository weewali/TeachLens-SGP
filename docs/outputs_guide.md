# What every output file is, and how to read it

Written 2026-11-19 (user request). All outputs stay local under `outputs/`
(ignored by Git): they hold classroom footage and evidence about people.

## Folder layout

```text
outputs/
  lessons/<YYYY-MM-DD_HHMM>_<lesson>/      one lesson, everything about it (scripts/analyze_lesson.py)
      lesson.json                            the index: what ran, ranges, timings, hypotheses, every path
      student/                               the student-camera run
          events.jsonl  summary.json  review.mp4  run.json
      instructor/                            the instructor-camera run (same four files)
      cross_camera.json                      timing links between the two cameras
      fusion/
          windows.csv                        the lesson cut into 5 s windows, one column per feature
          fusion.json                        the lag analysis: hypotheses, links, models, findings
          links.csv                          the same tests as a table (one row per teacher -> student pair)
          timeline.png                       teacher and class timelines
      facts.json                             the lesson as facts for the LLM report stage
      side_by_side.mp4                       both review videos side by side, lined up by real time
      (LLM stage, later) final_feedback_report.txt, llm_output.json
  runs/<YYYY-MM-DD_HHMM>_<run>/            a camera run made on its own (analyze.py, run_scenarios.py,
                                           detect_objects.py: objects.jsonl, objects_summary.json, ...)
  pooled/<YYYY-MM-DD_HHMM>_<name>/         several lessons tested together: pooled.json, facts.json
  logs/                                    console logs of long batches
  archive/                                 old files kept for reference (never read by the code)
```

The date and time in a folder name are when it was created. Re-running with
the same name and `--reuse` or `--overwrite` updates that folder; a new
`--tag` or `--name` makes a new folder. `scripts/organize_outputs.py` moved
the older flat files (events/, reports/, debug/, visualizations/) into this
layout on 2026-11-19.

## Each file: what it holds, and what uses it later

| File | What it holds | Used later by |
| --- | --- | --- |
| `events.jsonl` | The evidence: one JSON object per line, one per observation or interval (indicator, track, role, value, video and real start/end times, coverage). Every other file is computed from it. | Fusion, links, summaries; re-analysis without the video; validation against human labels. |
| `summary.json` | Totals for one camera run: durations per label, counts, coverage, hand raises, seats, limitations (AGENTS.md section 8). | facts.json (instructor visibility, hand-raise counts); a quick look at a run. |
| `review.mp4` | The camera's analyzed frames (1 per second) with boxes, joints, IDs, roles, and every indicator's state. Small grey HUD at the top left. | Checking by eye; side_by_side.mp4; labelling for validation. |
| `run.json` | What was analyzed: run ID, camera, source video, analyzed range, recording start, indicators on, and the review video's frame timing (first_sample_seconds, sample_period, samples). | Finding a run again by ID; lining up review videos (side_by_side.mp4). |
| `cross_camera.json` | Timing links: a student who left the student view and appeared at the front (time at the board), and the teacher walking into the student view. | The UI (moments to jump to); later interaction indicators. |
| `fusion/windows.csv` | The lesson on one real-time line in 5 s windows. One row per window: `T_*` teacher features (share of the window, or path length), `S_*` student features (share of the known student-seconds, or counts), coverage columns. Empty cells = too little evidence (not zero). | All lag tests; pooling over lessons; charts. |
| `fusion/fusion.json` | The lag analysis of the lesson (symbols below). | facts.json; a human reviewer; the UI. |
| `fusion/links.csv` | Every tested pair in one table (`type` hypothesis or exploratory). | Spreadsheet review. |
| `fusion/timeline.png` | Teacher shares (top) and class shares (bottom) per window; dotted lines = hand raises; gaps = too little evidence. | Reports and slides. |
| `facts.json` | The lesson as numbered facts (F1, F2, ...) with a direction (favourable / unfavourable), an evidence strength, a glossary, cautions, and alternative explanations. Schema version 1 of the LLM stage. | The LLM report stage: `llm_report.py --facts <this file>` writes the positive / negative points and checks them (docs/llm_integration.md). |
| `side_by_side.mp4` | One frame per lesson second: student camera left, instructor camera right, both lined up by real time; header with the clock, lesson time, what the teacher is doing, and the class's on-task / off-task / phone shares in that window. | Reviewing a lesson as a whole; showing an instructor a moment. |
| `lesson.json` | The lesson's index: runs (and whether each was analyzed or reused), ranges, timings, links count, hypotheses (one line each), findings, every output path, and the next step (the LLM command). | The UI's entry point; anyone opening the folder. |
| `pooled/.../pooled.json` | The hypotheses tested over several lessons (within each lesson, and between lessons). | Conclusions that need more than one lesson; facts.json for a multi-lesson report. |

## The lag analysis in plain words

The question (SGP1 proposal FR 6): when the teacher does something, does the
class change a few seconds later?

1. **Windows.** Both cameras' evidence is cut into the same 5 s windows of
   real time (the two cameras line up to about +/-1 s). Window t has a
   teacher value X_t (e.g. the share of the window the teacher was seated)
   and a class value Y_t (e.g. the share of student-seconds off task).
2. **Lag correlation.** For a lag k of 1..6 windows (5..30 s), pair each
   class value with the teacher value k windows earlier and compute Pearson's
   correlation:

   r(k) = corr( Y_t , X_(t-k) )   over all windows t

   r ranges from -1 to +1: +1 means the class value rises exactly when the
   teacher value rose k windows earlier, 0 means no straight-line relation.
   The lag with the largest |r| is kept.
3. **Is it more than chance? (circular-shift test).** The teacher series is
   rotated in time (shifted by more than the largest lag, wrapping around),
   which keeps its own rhythm but breaks its timing against the class. For
   every rotation the best |r| is computed again. p_shift is the share of
   rotations that do as well as the real timing: p = (1 + rotations at
   least as strong) / (1 + rotations). Small p = the real timing matters.
4. **Many tests (q-values).** Testing many pairs makes some small p-values
   appear by chance. Benjamini-Hochberg turns each p into a q-value, the
   expected share of false findings among results at least that strong. A
   result is a finding only when q <= 0.05. The 9 hypotheses fixed in
   advance (from the scenario booklet's teaching codes, `fusion.hypotheses`)
   are corrected among themselves; the other, exploratory pairs separately.
5. **The plain comparison.** For each hypothesis: the class value in the
   window right after windows where the teacher behaviour filled at least
   half the window, versus after the other windows ("72% of students on task
   after the teacher faced the class, 61% otherwise"). Its p-value (p_hac)
   comes from a regression with Newey-West standard errors, which allow for
   neighbouring windows being alike.
6. **The lagged model** (the proposal's Figure 4.2):

   Y_t = b0 + b1 X1_(t-k) + b2 X2_(t-k) + ... + a Y_(t-1) + e_t

   Y_t: the class value now; X_(t-k): teacher features k windows ago;
   Y_(t-1): the class value one window ago (classes change slowly, so this
   soaks up the drift); b: how much the class value changes per unit of the
   teacher feature, beyond what the previous window predicts; e_t: what is
   left unexplained. Fitted by least squares with Newey-West errors.
7. **Rare actions (peri-event).** Actions that happen in a few short bursts
   (board, pointing, turning to the class) are compared as before/after:
   the class value in the 4 windows after each start minus the 2 windows
   before, against the same number of random start times.
8. **Several lessons (pooled).** A teacher who sits all lesson cannot be
   compared with themselves standing in that lesson. Pooling (a) tests the
   hypotheses within each lesson and adds the lessons up, and (b) compares
   lessons: each lesson's average teacher behaviour against its average
   class value (Spearman's rho over lessons, exact permutation p). (b) is
   weak evidence with few lessons, because lessons differ in many ways.

None of this shows cause: a behaviour and a class change can both follow
from the lesson plan.

## Symbols in fusion.json, pooled.json, and links.csv

| Key | Meaning |
| --- | --- |
| `window_seconds`, `windows` | Window length (5 s) and number of windows in the shared stretch. |
| `start_real_time`, `end_real_time` | Wall-clock start and end of the shared stretch (ISO 8601 with the UTC offset). |
| `min_known_fraction` | A share needs at least this fraction of a window of real evidence, else the cell is empty. |
| `features` | Every `T_`/`S_` column: source indicator and labels, `kind` (share, count, path, episode, on_task, off_task), `desirable` (+1 more is better, -1 more is worse, 0 neutral), description. |
| `feature_activity_windows` | Windows where each feature was present (non-zero). |
| `teacher_feature_windows_with_without` | [windows with the teacher feature, windows without]; both must be at least `min_active_windows` (5) to test it. |
| `tested_teacher_features`, `untested_teacher_features` | Which teacher features varied enough to test; untested ones with their windows present. |
| `outcome_features` | Student features with enough evidence and some variation. |
| `hypotheses[]` | The fixed-in-advance tests. `code`: the booklet code (T-SIT ...); `expected_sign`: +1 the class value should rise, -1 fall; `not_tested`: why it could not be tested. |
| `lag_windows`, `lag_seconds` | The lag k with the strongest correlation (windows, seconds). |
| `r` | Pearson correlation at that lag (-1..+1). |
| `r_lag0` | The correlation in the same window (no lag). |
| `r_reverse_max` | The strongest correlation with the class first and the teacher later; if it is as strong as `r`, the two probably just drift together. |
| `p_shift` | Circular-shift p-value (step 3). |
| `null_shifts` | How many rotations were used; with a short lesson every rotation is used once, so the smallest possible p is 1/(null_shifts + 1). |
| `q` | Benjamini-Hochberg q-value (step 4); a finding needs q <= `alpha` (0.05). |
| `matches_expectation` | true when the sign of `r` matches `expected_sign`. |
| `comparison` | Step 5: `lag_windows` (1 = the next window), `windows_with` / `windows_without`, `outcome_with` / `outcome_without` (the class value as a fraction, or counts per window for hand raises), `difference`, `difference_within_lessons` (pooled: the same with one intercept per lesson), `p_hac`. |
| `links[]` | Exploratory pairs (not hypotheses): same keys, corrected separately; never a finding unless q <= alpha. |
| `lagged_models[]` | Step 6 per outcome and lag: `terms` -> each coefficient (`coef`), its standard error (`se`), `p`, and 95% interval (`ci95`); `r_squared` (share of the class value's variation the model explains, 0..1); `windows_used`; `hac_maxlags` (how many neighbouring windows the Newey-West errors allow for). `X[t-1]` means X one window before. |
| `peri_event[]` | Step 7: `events` (starts used), `mean_change` (after minus before), `p_random_onsets`, `q`, `curve` (the class value from 2 windows before to 4 after; `curve_relative_windows` gives the positions). |
| `findings[]` | Results with q <= alpha: `kind` strength (the class moved the desirable way) / weak_point / observation (neutral feature), `method`, the pair, `effect` (r or mean change), `q`. |
| `teacher_onsets` | Window numbers where short teacher actions started (for step 7). |
| `settings` | The settings used (seed, permutations, alpha, ...); the same seed gives the same p-values. |
| `between_lessons` (pooled) | `lessons`, `spearman_rho` (rank correlation of lesson averages), `p_permutation` (over every ordering of the lessons), `per_lesson` averages, `lessons_with` (lessons where the teacher feature filled at least half the time), `outcome_with` / `outcome_without`. |
| `lessons_used`, `lessons_without_variation`, `r_by_lesson` (pooled) | Which lessons entered the within-lesson test, which were left out, and each lesson's own r at the pooled lag. |

## Features (the windows.csv columns)

Teacher (instructor camera): `T_facing_class`, `T_looking_at_students`,
`T_board_or_screen`, `T_writing`, `T_pointing`, `T_pointing_at_students`,
`T_back_to_class` (T-WRITE), `T_reading_notes` (head down at notes, T-READ),
`T_seated` (from the legs, or behind the desk from the shoulders' height,
T-SIT), `T_at_desk` (desk zone), `T_gesturing`, `T_walking`, `T_stationary`,
`T_path` (hip path length in frame heights).

Students (student camera, class level): `S_on_task` and `S_off_task` (each
student's state second by second: off task = phone seen or likely, dozing,
talking toward a neighbour, or facing away; on task = facing the focus,
writing, or reading; note-taking therefore counts as on task),
`S_on_target`, `S_off_target`, `S_looking_down`, `S_phone`, `S_laptop`,
`S_writing`, `S_reading`, `S_talking`, `S_smiling`, `S_head_down`,
`S_sleep`, `S_moving`, `S_hand_raises`, `S_answers` (counts per window).

## facts.json in short

`schema_version` 1; `lecture` (lesson, videos, duration mm:ss, windows);
`data_quality` (what was measured, instructor visible %, limitations);
`indicator_validation` (every detection: `checked_against_human_labels`
false until someone labels clips); `glossary`; `alternative_explanations`
(E1..E5); `facts` (F1..: `direction`, `kind` = comparison,
instructor_behaviour_share, highest/lowest_on_task_stretch, hand_raises,
no_participation_stretch, phone_use, lesson_comparison; numbers; `evidence`
with `strength` suggestive / weak / descriptive, `test_p`, `notes`);
`report_note`. Times are mm:ss from the start of the shared stretch.
