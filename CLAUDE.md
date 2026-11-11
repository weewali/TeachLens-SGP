# TeachLens: handoff notes for Claude

Written 2026-11-11 by the Claude that built the pipeline in the original Codespace, so a new Claude session (another
profile, laptop or Codespace) can continue without re-deriving anything. **Verify against the actual files before
relying on any claim here; this file can go stale.** Update it at the end of every work session.

## What this project is
TeachLens (senior graduation project, UAEU, Project ID 11): analyse a recorded lecture and give the instructor an
evidence-based report on observable instructor and student behaviour. It is advisory. It reports observable behaviour,
never inner states (attention, engagement, understanding), and it describes associations, not causes.

## The user's objectives (in their words, roughly)
1. Better **association** analysis (instructor behaviour vs student behaviour over time). Cause comes later.
2. More **accuracy** (nothing has been validated against human labels yet).
3. An **LLM that turns the numbers into plain English** positive/negative points, with good reasoning.
Report format they want: `positive points:` then bullets, `negative points:` then bullets (about 3 each, fewer if the
evidence is thin; never pad).

## Working agreements with this user (follow these)
- **Ask before downloading anything** (pip packages, model weights) and **before anything that spends money or sends data
  out** (Claude API calls). Say what, where from, and the size/cost, then wait for a yes.
- Approved so far: pip `scipy statsmodels anthropic`; weights `yolo11n-pose.pt`; **one** real Claude call (about $0.04).
  Whether that call has been made yet is unknown to me: check `classroom_project/sessions/*/reports/llm_output.json`.
- **Never ask for or accept an API key in chat.** The user exports `ANTHROPIC_API_KEY` in the terminal themselves
  (a key exported in their terminal is not visible to Claude's own shell; they can instead save it to a private file
  and Claude passes it to the single call).
- The user likes **short, plain answers**, and wants limitations stated honestly. Do not overclaim. Say plainly when
  something is unverified or when an earlier statement turned out to be too generous.
- Do not commit or push unless asked. Do not delete the user's files.

## Code map (all under `Downloaded from prototype/`)
- `classroom_analytics.py`: the team's ORIGINAL prototype. Leave untouched.
- `classroom_analytics copy.py`: the real pipeline (the name has a space; always quote it).
- `llm_report.py`: facts.json -> Claude -> checked report. `validate_indicators.py`: accuracy review sheets + scoring.
- Root `requirements.txt`; `CLAUDE.md` (this file). Team docs in `docs/` (annotation, training notes, video sources).
- The user also shared two design docs in chat (not saved in `docs/` here): `project_structure.md` (planned `src/` layout,
  JSONL event schema, "models never write report prose; events -> analysis -> reporting") and
  `video_indicator_catalog.md` (indicator list; **rule: never report attention/engagement/emotion as fact, report narrow
  observable proxies with denominator + confidence + limitation**; speech/audio and recording-quality indicators are
  deliberately deferred). Ask the user for them again if needed.

## How to run (from the repo root)
```
python3 "Downloaded from prototype/classroom_analytics copy.py"                      # every video in classroom_project/videos/
python3 "Downloaded from prototype/classroom_analytics copy.py" --video PATH         # one video
python3 "Downloaded from prototype/llm_report.py" --session Video10min --dry-run     # shows prompt + cost, calls nothing
python3 "Downloaded from prototype/llm_report.py" --session Video10min --model haiku # real call: haiku ~0.1c, sonnet ~2c, opus ~4c
python3 "Downloaded from prototype/validate_indicators.py" make-sheets --session Video10min
python3 "Downloaded from prototype/validate_indicators.py" score --session Video10min   # after a person fills the sheets
```
Not in git, so recreate on a new machine (each download needs the user's OK): the video(s) in
`classroom_project/videos/` (`*.mp4` is gitignored), weights `yolov8n.pt` and `yolo11n-pose.pt` in the repo root (`*.pt`
is gitignored; the script never auto-downloads and tells you when a file is missing), `ffmpeg` (apt), and
`pip install -r requirements.txt`.

## Pipeline in one paragraph
Validate video -> FFmpeg standardise (25 fps, max 720p) and extract one frame every **2 s** with timestamps -> calibration
(`classroom_project/config/calibration.json`: teacher zone, board region, student region, 3x4 seat grid; optional
`calibration_<session>.json` per camera) -> detection (`yolo11n-pose` people + 17 keypoints; `yolov8n` phones/books) ->
rule-based behaviour states and timestamped events -> 10 s windows of indicators -> lag analysis -> rule-based report +
`facts.json` -> optional Claude rewrite with automatic checks. Outputs go to `classroom_project/sessions/<id>/`
(`tables/`, `reports/`, `frames/`, `yolo_outputs/` previews); pooled multi-lecture results go to `classroom_project/pooled/`;
review sheets to `classroom_project/validation/<id>/`.

## Decisions and why (do not undo without a reason)
- **`PREDICT_CHUNK = 20`**: one long YOLO call grew past 3 GB on this 2-core / 8 GB Codespace with no GPU, and a host
  watchdog killed it (SIGTERM from outside the container). Predicting 20 frames at a time keeps memory flat.
- **"focused" means head up** (nose near/above the shoulder line), a posture proxy. Writing looks like head-down, so it can
  count as off-task. The report note says so. The catalog rule above says to prefer names like "head-up".
- **Roles come from calibration zones** (anyone in the teacher zone = instructor). Cheap but fragile; see findings.
- **Lag analysis**: neighbouring windows are alike (autocorrelation about 0.67), so plain correlation p-values are far too
  optimistic. The code reports p-values controlled for the outcome's previous window (+ time in lecture), judges them
  against a Bonferroni limit, and labels hypotheses fixed in advance (`PRESPECIFIED`) separately from exploratory pairs.
  Gap comparisons in the report use a robust (HAC) test. Lags never cross from one lecture into the next when pooling.
- **`facts.json`** is the only thing the LLM sees: anonymised, pre-computed facts with ids, a direction
  (favourable/unfavourable) and an evidence strength. `llm_report.py` then **checks the answer in code**: every number
  must come from the cited facts, no causal wording, no attention/engagement words, correct direction, hedges where the
  evidence is weak or the detection is unverified (`indicator_validation`). One retry with the failures listed; if it still
  fails, the rule-based report stays. Answers are cached by hash so reruns are reproducible.
- **Models**: `opus` (claude-opus-5-5, default), `sonnet` (claude-sonnet-5-5), `haiku` (claude-haiku-5-5). Opus/Sonnet send
  the server-side fallback option; **Haiku must not** (it has no fallback). Opus 5.5 rejects temperature and cannot disable
  thinking; effort is set to `medium`. Prices are in `MODELS` in `llm_report.py`.

## Findings on the one test video (`Video10min.mp4`, 10:01, 854x480)
- It is **edited footage with a moving camera** (cuts to close-ups of the instructor, pans, zooms), not a fixed camera. Only
  about 21% of frames match the calibration framing, about 49% are pan/zoom but matchable, about 30% are cuts/close-ups.
  Zone-based roles are wrong in those segments: the "hand raises" at seat_08 (01:14, 01:44) are the instructor gesturing.
- The instructor stands behind a lectern almost the whole time, so **sitting vs standing cannot be tested** on this video
  (posture readable in about 26% of frames, 0% seated). The seated rule has never seen a seated example.
- Posture readability depends strongly on apparent person size (r about -0.65): the camera still affects measurement.
- Strongest exploratory lead (instructor distance to nearest student -> off-task rate): r -0.51 naive, about -0.33 after
  controlling for the previous window (p about 0.02). It does **not** survive correction for 42 tests. Earlier naive
  p-values quoted to the user (for example p = 0.007) were overstated; the user was told.
- Interaction (facing + gesture) and board-writing rules never fired; zero counts may mean missed behaviours.

## Status and what is next
Done: all 7 stages, multi-lecture sessions + pooling (mechanics tested on two clips only), `facts.json`, `llm_report.py`
(tested offline with simulated responses and a fake server, including 18+ deliberate failures), validation review sheets
and scoring (tested with fake labels), the executive summary text for the report.
Open, in rough priority order:
1. **Human labelling**: the user fills `classroom_project/validation/Video10min/review_events.csv` and `review_frames.csv`
   (about 20-30 min), then `score`. Until then every indicator is reported as unverified.
2. **Camera handling** (the user never answered options A/B/C): A = register each frame to a reference frame with ORB +
   RANSAC, map zones through the transform, exclude unmatchable frames; B = flag and exclude only; C = use a fixed-camera
   video instead. The team docs defer recording-quality indicators, so ask first.
3. The real Claude call (see agreements). Compare `haiku`/`sonnet`/`opus` on the same facts if the user wants.
4. More lectures (ideally fixed camera, including a seated instructor, with topic/activity metadata) for association;
   a causal layer is SGP2-optional in the report and needs confounders.
5. Wording: internal names still say `attention_level` / `off_task_rate` (the catalog prefers head-up / head-down).
6. Frame interval is 2 s. If changed to 1 s, also double `HAND_RAISE_MIN_FRAMES` (2->4) and `OFFTASK_MIN_FRAMES` (3->6), and
   `STATIONARY_MOVE` changes meaning.
7. Git hygiene: generated outputs under `classroom_project/` (previews with classroom frames, tables, reports) were
   committed. The team docs say not to commit extracted frames or sensitive evidence; suggest `.gitignore` + `git rm --cached`.
8. Tests and the `src/` restructure described in the team's structure doc do not exist yet.
