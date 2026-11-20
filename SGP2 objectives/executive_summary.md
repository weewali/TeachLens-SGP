# Executive summary

University instructors usually receive feedback on their teaching once per semester, through student surveys. These
can be rushed or biased and rarely point to specific moments in a lecture. TeachLens (UAEU Senior Graduation Project,
Project ID 11) addresses this. It analyses a recorded lecture and gives the instructor private, evidence-based feedback
on observable classroom behaviour: a short list of positive and negative points in plain English, with charts and
timestamped moments.

**How it works.** Two static cameras record the same lecture: one facing the instructor and one facing the students.
Since 20 November 2026 TeachLens is one pipeline: the team's lesson pipeline does the processing, and the report stage
and web application sit on it. The system:
1. Lines the two videos up from the recording-start time stored in each file, to about one second.
2. Analyses each camera at one frame per second with pretrained models on an ordinary laptop CPU: YOLO26n-pose with
   ByteTrack (anonymous track IDs, seats), MediaPipe face and hand models on the student camera, and a YOLO26n
   detector for phones, laptops and books.
3. Applies transparent rules: the instructor facing the class, at the board or desk, seated, gesturing, walking; each
   student on task (facing the focus, writing, reading) or off task (phone, dozing posture, chatting, facing away),
   hand raises and answers.
4. Tests whether students change 5–30 seconds after instructor actions (5-second windows, circular-shift test,
   Benjamini–Hochberg correction) and compares students during each instructor activity with the rest of the lecture
   (HAC test, Bonferroni correction).
5. Writes up to three positive and three negative points by rule, then always has Claude Opus 5.5 rewrite them. The
   model sees only anonymised numbers, and code checks every answer: numbers must match the measurements, and causal
   or inner-state wording is rejected. If the rewrite fails, the rule-based points are shown together with the reason.

TeachLens is a hybrid system: pretrained deep-learning models for perception, transparent rules for behaviour,
statistics for the links, and a language model for the wording. No model was trained by the team.

A local Flask web application lets the instructor upload the videos (with an optional topic, slides, activities in
class and their duration, and other information), follow the analysis stage by stage, and read the report in tabs:
feedback, charts, timestamped moments and key terms, and download it as a PDF. Faces are blurred in every picture
the report shows.

**Results so far.**
- The team's pipeline ran start to finish on seven scripted two-camera scenario lessons (1–10 minutes each) with no
  failures. None gave a statistically confirmed link, which is expected for lessons this short.
- On a real 10.4-minute two-camera lecture the complete system (analysis, report and web job) ran end to end in
  16.4 minutes with no errors; it found no statistically confirmed link, and some detections (30 hand raises, 8 seats
  for 6 students) still need a human check.
- Run time on this laptop's CPU is about 2.2 minutes of computing per lecture minute (more on battery power), so a
  one-hour lecture takes roughly two hours or more.
- 210 automated tests pass (170 for the team's pipeline, 40 for the report stage and the web application).

**Limitations.** The detections have not been checked against human labels, so no accuracy figure can be claimed.
All results so far come from short lessons; real lectures last 55–75 minutes. With many instructor–student pairs a
single link rarely passes the statistical correction, so most feedback comes from the simpler "during each activity"
comparisons. Activities instructors note (such as a quiz) are stored but not yet used. No real call to the language
model has been made yet. Board and desk areas must be marked once per classroom to be measured.

**Next steps:**
- Human labelling with the built-in review sheets, then tuning the behaviour-rule thresholds against those labels
  (checked on lectures not used for tuning); the statistical settings stay fixed.
- Full-length recorded lectures (55–75 minutes), with consent.
- The first real AI rewrite.
- Marking the board, screen and desk areas of the classroom.
- Deciding how to use the activities instructors note **[TODO]**.

**[TODO: confirm the final numbers on the day of submission and add the team's contribution statement.]**
