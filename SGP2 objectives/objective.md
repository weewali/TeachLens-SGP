# Project objective

## Aim

TeachLens aims to give university instructors timely, evidence-based and private feedback on their teaching. It
analyses a recorded lecture and reports observable behaviour. It links what the instructor was doing to what the
students were visibly doing at the same moment or shortly after. The feedback is a short list of positive and negative
points in plain English, each tied to a time in the lecture. TeachLens is advisory: it describes associations, not
causes, and never claims to measure inner states such as attention, engagement, emotion or understanding.

## Specific objectives

1. **Capture both sides of the classroom.** Accept two recordings of the same lecture: one static camera facing the
   instructor and one facing the students. Line them up on one timeline from the recording-start time in each video's
   metadata, to within about one second.
2. **Detect people, faces, hands and objects with proven models.** Use pretrained models on a laptop CPU, one frame per
   second: YOLO26n-pose with ByteTrack for people and anonymous track IDs on both cameras; MediaPipe face landmarks and
   hand gestures, and a YOLO26n detector for phones, laptops and books, on the student camera.
3. **Turn detections into observable indicators with transparent rules.**
   - Instructor: visible, facing the class, at the board or screen, pointing, gesturing, walking, seated, at the desk,
     reading notes.
   - Students: on task / off task, facing the front, phone, laptop, writing, reading, talking, dozing posture, hand
     raises and answers, on anonymous seats.

   Unknown is never counted as zero.
4. **Record the evidence in a shared, reusable format.** One event file per camera (JSONL), summaries, 5-second window
   tables and `facts.json`, so results can be checked and reused without reprocessing the video.
5. **Analyse instructor–student associations honestly.**
   - Test whether students change 5–30 seconds after instructor actions (circular-shift test, Benjamini–Hochberg
     q-values over 9 hypotheses fixed in advance and over exploratory pairs).
   - Compare students during each instructor activity with the rest of the lecture (HAC errors, Bonferroni).
6. **Write feedback an instructor can act on.** Up to three positive and three negative points in plain English,
   without statistical jargon or caveats, in the voice of a supportive mentor (what went well and what to keep
   doing; for each negative point one idea to try next time), written by Claude Opus 5.5 from the measured facts and checked automatically against them
   (a point that fails the checks is left out); when no point passes or Claude cannot write it, the report shows the reason (an API key problem only as "Error, Code=01", a billing problem only as "Error, Code=02") and a "Try again" button.
7. **Deliver it through a simple web application.** Instructors give the lecture a title, upload the two videos,
   follow the analysis stage by stage, and read a report in tabs: feedback, charts, timestamped moments and key terms, which can be downloaded as a PDF.
   Context is optional: the topic, the slides, activities in class that were not teaching (with their duration) and
   additional information.
8. **Protect people and state limits openly.**
   - Processing stays local.
   - Nobody is identified: temporary track IDs and anonymous seats; students' faces blurred in the report's pictures (not the
     instructor's: the report is theirs).
   - Only anonymised numbers ever reach the language model.
   - Inner states are never claimed; associations are never presented as causes.
   - Every saved report file states its limitations; the web page gives plain teaching advice without them (user
     decision, 21 November 2026).
9. **Verify and evaluate.**
   - Test the software automatically.
   - Run it end to end on real two-camera recordings.
   - Check each detection's accuracy with human review sheets or the team's blind labelling frames (`evaluation/`).

### Status at the time of writing (20 November 2026)

| Objective | Status | Evidence |
|---|---|---|
| 1 Two cameras, aligned | Done | Real two-camera lectures and scripted scenario lessons lined up from metadata |
| 2 Pretrained models, CPU | Done | YOLO26n-pose, YOLO26n and two MediaPipe models run locally; nothing is downloaded at run time |
| 3 Observable indicators | Done (rules); accuracy not yet measured | Rule outputs on scenario lessons and a real lecture; no human labels yet |
| 4 Shared evidence format | Done | Lesson folders with events, summaries, windows and `facts.json` |
| 5 Association analysis | Done | Lag analysis and "during each activity" comparisons; no lesson so far gives a confirmed link (all were short) |
| 6 Plain-English feedback | Done (Claude writes it, with code checks); first real call on 21 November 2026: 3 of 4 points passed and are shown | Offline tests of the checks with simulated answers |
| 7 Web application | Done (local) | Upload, progress by stage, tabbed report, PDF download, delete; 35 automated web tests. Activities are stored but not used **[TODO]** |
| 8 Privacy and honesty | Done in design and code | Local processing, anonymous IDs and seats, blurred faces, word checks |
| 9 Verification | Partly done | 219 automated tests pass (170 team pipeline, 49 report stage and web app); human-labelled accuracy **[TODO]** |

## The positive side of the project

### For instructors

- **Timely feedback.** A report on the same lecture, available as soon as the analysis finishes, instead of one survey
  at the end of the semester.
- **Specific and actionable.** Each point names a moment ("between 04:00 and 04:30") and what the instructor was doing
  then, so the instructor can watch that part of the video and decide what to change.
- **Balanced.** Positive points are reported alongside negative ones, and points are never padded when the evidence is
  thin.
- **Private.** The report is for the instructor's own reflection. It is not a score, a ranking or a judgement of the
  person.

### For students

- **No student is identified or graded.** Students are temporary track IDs and anonymous seats, no faces are
  stored or matched, their faces are blurred in the report's pictures, and nothing is reported about any named student.
- **Benefit from better-informed teaching.** For example, a long stretch of explanation with no hand raises is pointed
  out together with a simple suggestion: ask a short question.

### For the university

- **Supports teaching development.** It adds observable evidence to existing quality processes such as student
  surveys and peer observation.
- **Low cost and practical.** Two static cameras or phones on tripods and an ordinary laptop. Pretrained models only,
  so no costly training dataset is needed, and Claude's feedback costs about USD 0.04 per report.
- **Reproducible and reusable.** Every result is saved as plain files (events, tables, facts), so a report can be
  rebuilt or re-checked without processing the video again. Since 20 November 2026 the team's work forms one
  modular pipeline.

### Design strengths

- **Trustworthy by construction.** Every sentence in the feedback traces back to a measured number. The AI rewrite is
  rejected automatically if it adds a number, claims a cause, or mentions an inner state.
- **Honest statistics.** The analysis allows for neighbouring moments being alike and for the number of tests made, so
  chance patterns are not reported as findings.
- **Ethical by design.** It follows the ACM and IEEE codes of ethics and the UAE Personal Data Protection Law
  (Federal Decree-Law No. 45 of 2021): local processing, consent, anonymity, and no inner-state or causal claims.
- **Accessible report.** Plain language and charts with a table view and keyboard access. The page keeps a white
  background on every computer and works on a phone screen.
- **Little extra work for instructors.** Only a lecture title and the two videos are required; everything else is
  optional. A form with no context gives a normal report.

> Note for the report: these are intended benefits and design properties. Their effect on teaching has not been
> measured. The accuracy of the detections will be stated once the human-labelled evaluation is done **[TODO]**.
