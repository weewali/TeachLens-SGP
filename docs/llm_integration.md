# Working with the LLM and UI stage (pipeline_creation branch)

Looked at 2026-11-19, branch `origin/pipeline_creation`, last commit 1bf2c49
("Add LLM report, validation, sessions and pooling", 2026-11-11). Recheck the
branch before relying on this; it may have moved on.

The step-by-step plan for building the LLM feedback and report stages on
this branch (which files the LLM reads, the facts.json fields, suggestions,
the report layout, tests, cost and key rules) is AGENTS.md section 17.

## What the branch contains

- `Downloaded from prototype/llm_report.py`: the LLM stage. Reads one
  `facts.json`, sends it (and only it) to Claude with a fixed system prompt,
  and checks the answer in code before keeping it: every number must come
  from the cited facts, a positive point may cite only favourable facts and a
  negative point only unfavourable ones, no causal wording ("caused", "led
  to"), no internal-state words ("attention", "engaged", "bored"), no
  judgement of the instructor, hedges when the evidence is weak, and a
  "not verified" warning when a detection has not been checked
  (`indicator_validation`). One retry with the failures listed; if it still
  fails, a rule-based report stays. Answers are cached by hash. Offline modes:
  `--dry-run` (prompt and cost only) and `--fake-response` (checks a saved
  answer). Writes `final_feedback_report.txt` and `llm_output.json` next to
  the facts file.
- `Downloaded from prototype/classroom_analytics copy.py`: a separate,
  earlier end-to-end pipeline (one camera, frames every 2 s, YOLO11 pose and
  YOLOv8 objects, roles from calibration zones, 10 s windows, lag analysis)
  that produces its own facts.json. `validate_indicators.py`: review sheets
  for human labels and scoring.
- `CLAUDE.md` (his handoff notes), `classroom_project/` (sample outputs,
  including extracted classroom frames and tables, committed), and the SGP1
  report.
- No user interface yet in that commit.

The branch starts from an old commit (aa0d107, before the List A pipeline),
so it does not contain this branch's code, and this branch does not contain
his.

## The contract: facts.json

The LLM stage needs one file, `facts.json` (schema_version 1). Since
2026-11-19 every lesson analyzed here writes it:
`outputs/lessons/<stamp>_<lesson>/facts.json` (src/reporting/facts.py), and a
pooled analysis writes one too. Same keys as his: `lecture`, `data_quality`
(with `limitations`), `indicator_validation`, `glossary`,
`alternative_explanations` (E1..), `facts` (F1.. with `direction`, `kind`,
numbers, `evidence.strength` suggestive / weak / descriptive, `test_p`,
`notes`), `report_note`.

Checked offline on 2026-11-19 with his unchanged `llm_report.py` and the A2
lesson's facts.json (8 facts): `--dry-run` built the prompt (about 2,500
input tokens, about $0.04 per call with Opus at list price; no call was
made); a correct answer citing our facts passed every check; a deliberately
bad answer was rejected for all five problems (wrong direction, invented
number, causal word, "attention", missing "not verified").

Run it on a lesson (his script, from the repository root of a checkout that
has it; a real call needs the `anthropic` package and an API key exported by
the user, and costs money, so ask first):

```powershell
python "Downloaded from prototype/llm_report.py" --facts outputs/lessons/<stamp>_<lesson>/facts.json --dry-run
python "Downloaded from prototype/llm_report.py" --facts outputs/lessons/<stamp>_<lesson>/facts.json --model haiku
```

`lesson.json` holds this command as `next_step`.

### Differences to know about

- Wording: his prompt's rule 6 talks about "focused" (head up). Our facts use
  "on task" / "off task", defined in the glossary; the code checks still
  apply. A one-line prompt change ("use the measures only as the glossary
  defines them") would make the rule general.
- Fact kinds: `comparison`, `hand_raises`, `no_participation_stretch`, and
  `phone_use` are his kinds; his `KIND_INDICATORS` then demands "not
  verified" wording for `head_up_posture`, `instructor_role`, `hand_raise`,
  and `phone_use`, which our facts.json lists (all unverified). Our extra
  kinds (`instructor_behaviour_share`, `highest_on_task_stretch`,
  `lowest_on_task_stretch`, `lesson_comparison`) pass his checks but get no
  forced warning; adding them to his `KIND_INDICATORS` (for example
  `instructor_behaviour_share: ("instructor_seated", "instructor_facing")`,
  the stretches: `("student_on_task_state",)`) would close that gap.
- Times: mm:ss from the start of the stretch both cameras recorded (his:
  from the start of the video).

## For the user interface

One lesson = one folder. A UI can list `outputs/lessons/*/lesson.json` and,
for each lesson, show `side_by_side.mp4` (both cameras lined up),
`fusion/timeline.png`, the hypotheses and findings from `lesson.json`, the
facts from `facts.json`, and the LLM's `final_feedback_report.txt` once
written. Every path is in `lesson.json["outputs"]`. All of it is local and
holds classroom footage; the UI should read it locally, not upload it.

## Suggested next steps (not done; a team decision)

1. Do not merge `pipeline_creation` as it is: it starts before this branch's
   code, and it commits extracted classroom frames and outputs, which the
   team docs say stay out of Git.
2. Bring over only the LLM stage (detailed in AGENTS.md section 17.3, with
   suggestions added as a third checked list): `llm_report.py` (and `validate_indicators.py`
   if wanted) as `src/reporting/llm_report.py` with a thin
   `scripts/llm_report.py`, logic unchanged, reading a lesson folder's
   facts.json and writing the report into the same folder. Then
   `analyze_lesson.py --llm` could run it as the last step after asking.
3. Decide which detection pipeline the project keeps. This one has two fixed
   cameras synchronized by real time, YOLO26 pose at 1 sample/s, face, hand,
   and object models, and per-student rules; his has one camera and
   zone-based roles. His facts.json builder and this one write the same
   schema, so the LLM stage works with either.
