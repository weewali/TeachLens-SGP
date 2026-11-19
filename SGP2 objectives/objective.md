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
2. **Detect people and objects with proven models.** Use pretrained Ultralytics YOLO26 models:
   - YOLO26n-pose for the instructor;
   - YOLO26m-pose for the students;
   - YOLO26s for phones and books.

   Process everything locally on an ordinary laptop CPU.
3. **Turn detections into observable indicators with transparent rules.**
   - Instructor: visible, facing the class, at the board, moving, seated.
   - Students: heads up, head down and still for 30 s or more, hand raised, phone in view, book in view.

   Students are recorded on anonymous seats, and unknown is never counted as zero.
4. **Record the evidence in a shared, reusable format.** Write timestamped events for each camera in the team's JSONL
   format, plus 10-second window tables, so results can be checked and reused without reprocessing the video.
5. **Analyse instructor–student associations honestly.**
   - Compare student behaviour during each instructor activity with the rest of the lecture.
   - Test whether student changes 5–30 seconds later go with what the instructor did (the team's fusion layer,
     5-second windows).
   - Use statistics that allow for neighbouring moments being alike (HAC standard errors, control for the previous
     window).
   - Correct for multiple testing (Benjamini–Hochberg q-values in the team's fusion layer; Bonferroni for the
     comparisons).
6. **Write feedback an instructor can act on.** Produce up to three positive and three negative points in plain
   English, without statistical jargon. An optional rewrite by Claude Opus 5.5 is checked automatically against the
   measured numbers, with the rule-based text as a fallback.
7. **Deliver it through a simple web application.** Instructors give the lecture a title, upload the two videos,
   follow the analysis stage by stage, and read a report in tabs: feedback, charts, timestamped
   moments and key terms. Context is optional: the topic, notes, slides, and activities in class that were not teaching
   (a quiz, group work, a video or a break, added with one click and its duration in minutes).
8. **Protect people and state limits openly.** The design rules:
   - Processing stays local.
   - Nobody is identified.
   - Only anonymised numbers ever reach the language model.
   - Inner states are never claimed.
   - Every report states its limitations.
9. **Verify and evaluate.**
   - Test the software automatically.
   - Run it end to end on a real two-camera recording.
   - Prepare human-labelled checks of each detection's accuracy.

### Status at the time of writing (19 November 2026)

| Objective | Status | Evidence |
|---|---|---|
| 1 Two cameras, aligned | Done | Two real test lectures aligned from metadata (the student camera started 8.0 s and 2.0 s earlier) |
| 2 Pretrained models, CPU | Done | Three YOLO26 models run locally; no downloads at run time |
| 3 Observable indicators | Done (rules); accuracy not yet measured | Rule outputs on the two real lectures; no human labels yet |
| 4 Shared evidence format | Done | Two JSONL files per lecture in the team's event format |
| 5 Association analysis | Done | Comparisons, and the team's fusion layer on System B's event files, run on the two test lectures; no link passed the correction (see the results chapter) |
| 6 Plain-English feedback | Done (rule-based); AI rewrite implemented, first real call pending | Feedback on the real lectures; offline tests of the checks |
| 7 Web application | Done (local) | Upload with optional context and activities, progress, tabbed report, delete; 29 automated web tests. Marked activities are stored but not yet used by the analysis **[TODO]** |
| 8 Privacy and honesty | Done in design and code | Local processing, anonymous seats, word checks, limitations in the analysis details |
| 9 Verification | Partly done | 249 automated tests pass (98 System B, 151 System A); human-labelled accuracy **[TODO]** |

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

- **No student is identified or graded.** Seats are anonymous numbers, no faces are stored or matched, and nothing is
  reported about any named student.
- **Benefit from better-informed teaching.** For example, a long stretch of explanation with no hand raises is pointed
  out together with a simple suggestion: ask a short question.

### For the university

- **Supports teaching development.** It adds observable evidence to existing quality processes such as student
  surveys and peer observation.
- **Low cost and practical.** Two static cameras or phones on tripods and an ordinary laptop. Pretrained models only,
  so no costly training dataset is needed, and the optional AI rewrite costs about USD 0.04 per report.
- **Reproducible and reusable.** Every result is saved as plain files (events, tables, facts), so a report can be
  rebuilt or re-checked without processing the video again. The modular design lets the team's two implementations
  share one event format.

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
