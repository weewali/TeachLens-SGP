# Models used in TeachLens

This list contains only the models that the project's code actually runs. TeachLens has two implementations:

- **System A**, the team's evidence pipeline (branch `scripting-frame-extraction`). It processes one camera recording
  per run.
- **System B**, the two-camera analysis, reporting and web application (branch `pipeline_creation`, `main code/`).

TeachLens is a hybrid system: pretrained deep-learning models detect people, body keypoints and objects; transparent rules turn them into observable behaviours; statistical tests link instructor and student behaviour; and an optional language model rewrites the feedback in plain English. No model was trained by the team.

![TeachLens pipeline](pipeline.png)

## 1. Computer vision models (pretrained, used without retraining)

All three are Ultralytics YOLO26 models with their published pretrained weights. The team did not train or fine-tune
any of them. Behaviours such as "heads up" or "hand raised" are not predicted by a model. They come from transparent,
rule-based checks on the keypoints and boxes these models return.

| Model | Type | Used in | Camera | Input size | What it provides |
|---|---|---|---|---|---|
| **YOLO26n-pose** (`yolo26n-pose.pt`) | Pose estimation: person box + 17 body keypoints (COCO format) | System B | Instructor camera | 640 px | The instructor's position, face direction (facing the class), raised hands, gestures, movement and sitting/standing |
| **YOLO26n-pose** (`yolo26n-pose.pt`) | Same model | System A | Both cameras, one recording at a time | Default | Presence, position, sitting/standing, motion level and visibility for each tracked person |
| **YOLO26m-pose** (`yolo26m-pose.pt`) | Pose estimation: person box + 17 body keypoints | System B | Student camera | 960 px | Each student's head and shoulders, used for heads up, head down and still, hand raised and the anonymous seats |
| **YOLO26s** (`yolo26s.pt`) | Object detection (COCO classes); only "cell phone" and "book" are used | System B | Student camera | 960 px | Phone in view and book in view, linked to the student whose box contains the object |

Why these sizes:
- **Nano pose model for the instructor:** the instructor is one large person close to the camera, so the smallest
  and fastest model is enough.
- **Medium pose model at a larger input size for the students:** students are many, small and partly hidden by desks.
- **Small detection model for phones and books:** these are small objects. Its accuracy on classroom footage has not
  been measured yet.

Common settings:
- Detection confidence threshold 0.25.
- A keypoint is used only when its confidence is at least 0.5 (System B) or 0.35 (System A).
- Everything runs on the CPU.
- System B samples one frame every 2 seconds; System A samples one frame per second.
- The weight files must be present locally. The code never downloads a model by itself.

Software versions on the team laptop: Ultralytics 8.4.215 on PyTorch 2.15 (CPU build).

## 2. Tracking model (System A)

| Model | Type | What it provides |
|---|---|---|
| **ByteTrack** [4] (Ultralytics `bytetrack.yaml`) | Multi-object tracker: a motion model plus detection-to-track association; no learned weights of its own | Links detections across sampled frames into temporary anonymous IDs |

A track is confirmed after 2 samples and tolerates one missed sample. Tracker IDs can switch, so an ID is not a unique
person.

System B uses no tracker. It finds anonymous seats from where students sat during the lecture.

## 3. Language model (report writing, System B)

| Model | Provider | Role |
|---|---|---|
| **Claude Opus 5.5** (`claude-opus-5-5`) | Anthropic | Rewrites the measured findings as at most three positive and three negative points in plain English |

How the language model is used:
- **Input:** the model sees only `facts.json`, a list of pre-computed, anonymised numbers. It never receives video,
  images, file names or names of people.
- **Settings:** medium effort, with the answer returned in a fixed JSON format.
- **Checks in code:**
  - Every number and time in the answer must appear in the facts it cites.
  - Positive points may cite only favourable facts.
  - Causal words ("caused", "led to"), inner-state words ("attention", "asleep", "bored") and judgements of the
    instructor as a person ("poor teacher") are rejected.
  - Hedges are required where the evidence is weak or a detection is unverified.
- **Fallback:** if the answer fails the checks twice, or the service is unavailable, the rule-based feedback is kept.
  The model can never block the report.
- **Deployment setting:** in the web application the rewrite is chosen by whoever runs the server
  (`TEACHLENS_LLM`); instructors do not choose it.
- **Estimated cost:** about USD 0.04 per report.

Status: implemented and tested offline with simulated answers, including deliberately wrong answers that the checks
rejected. **[TODO: no real API call had been made as of 17 November 2026; add the result of the first real call.]**

## 4. Statistical models (association analysis, System B)

These models link what the instructor was doing to what the students were doing. They describe associations, not
causes. Since 19 November 2026 the lag models are the team's fusion layer (`src/analysis/lag_analysis.py`), run on
System B's two event files with 5-second windows; they replaced System B's own lag analysis (Pearson correlations
and OLS on 10-second windows judged against a Bonferroni limit, with three pairs fixed in advance). The comparisons
still use System B's 10-second windows.

| Model | Purpose | Details |
|---|---|---|
| **Lag correlation with a circular-shift test** (NumPy; team's fusion layer) | Whether an instructor feature in one window goes with a student feature 1 to 6 windows (5 to 30 s) later; the strongest lag is kept | p = how often a rotated instructor timeline gives as strong a link (rotation keeps how alike neighbouring windows are, but breaks the alignment). Instructor features present in fewer than 5 windows are not tested. |
| **Benjamini–Hochberg q-values** | Many pairs are tested, so a few links appear by chance | A link is "unlikely to be chance" only if q ≤ 0.05. With few windows the smallest possible p limits this (a short lecture cannot reach it with one link). |
| **Lagged model, OLS with HAC (Newey–West) standard errors** (statsmodels; team's fusion layer) | Y(t) = b0 + b1·X1(t−k) + … + a·Y(t−1), for k = 1 to 3 windows | Y(t−1) absorbs the class's slow drift. Recorded in the analysis details, not used for the feedback points. |
| **Averages around the moments an action started** (team's fusion layer) | How a student feature changed in the 20 s after the instructor started an action (turned to the class, went to the board, raised a hand), compared with the 10 s before | Needs at least 3 starts; compared with the same number of random moments. |
| **Comparison with HAC (Newey–West) standard errors** (statsmodels) | Student behaviour while the instructor did one activity vs the rest of the lecture, and the 95% range of the averages in the report chart | Up to 3 lags, so it allows for neighbouring windows being alike. A difference is called "unlikely to be chance" only if p < 0.05 divided by the number of comparisons. |

## References

[4] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. European
Conference on Computer Vision (ECCV)*, 2022.

[5] Ultralytics, "Ultralytics YOLO (YOLO26)," https://github.com/ultralytics/ultralytics.
**[TODO: version 8.4.215 and access date.]**

**[TODO: add references for Claude Opus 5.5 (Anthropic model documentation), HAC / Newey–West (Newey and West,
1987), Benjamini–Hochberg (1995) and the Bonferroni correction (still used for the comparisons), numbered after
[5].]**
