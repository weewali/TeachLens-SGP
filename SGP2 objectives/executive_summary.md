# Executive summary

University instructors usually receive feedback on their teaching once per semester, through student surveys. These
can be rushed or biased and rarely point to specific moments in a lecture. TeachLens (UAEU Senior Graduation Project,
Project ID 11) addresses this. It analyses a recorded lecture and gives the instructor private, evidence-based feedback
on observable classroom behaviour. The feedback is a short list of positive and negative points in plain English, each
tied to a time in the lecture.

**How it works.** Two static cameras record the same lecture: one facing the instructor and one facing the students.
The system:
1. Lines the two videos up from the recording-start time stored in each file, to about one second.
2. Samples one frame every 2 seconds from each camera.
3. Runs pretrained Ultralytics YOLO26 models on an ordinary laptop CPU:
   - YOLO26n-pose for the instructor;
   - YOLO26m-pose for the students;
   - YOLO26s for phones and books.
4. Applies transparent rules to the body keypoints:
   - Instructor: facing the class, at the board, moving or seated.
   - Students: heads up, head down and still, hand raised, phone in view.
5. Places students on anonymous seats found automatically from where they sat.

The evidence is saved as timestamped events in the team's shared JSONL format. The analysis then:
- splits the lecture into 10-second windows and compares student behaviour during each instructor activity with the
  rest of the lecture;
- runs the team's fusion layer on the two event files (5-second windows) to test whether student changes 5–30 seconds
  later go with what the instructor did.

The statistics allow for neighbouring moments being alike and for the number of tests made.

Rules turn the results into feedback, and an optional rewrite by Claude Opus 5.5 rephrases it. The rewrite sees only
anonymised numbers and is checked automatically: every number must match the measurements, and causal or inner-state
wording is rejected. A local Flask web application lets the instructor upload the videos, follow the analysis, and
read a report in tabs: feedback, charts, timestamped moments and key terms. Instructors may add
optional context, including activities that were not teaching and how long each lasted (for example a 15-minute
quiz); these are
stored with the report but not yet used by the analysis.

**Results so far.** The full pipeline ran on a real two-camera test lecture of 4 minutes 36 seconds with six students:
- Both videos were aligned from their metadata, and the instructor was visible in every sampled frame.
- All six students were found in every sample, and their posture was readable in 99% of seat samples.
- Main findings:
  - three hand raises, all while the instructor was explaining to the class;
  - one phone moment;
  - a two-minute stretch of explanation with no hand raise.
- An inspection of the output showed that the original fixed seat grid merged two students and turned a chair and a
  pair of legs into "seats". This was corrected by finding seats from where students actually sat. On the rerun there
  were six seats for six students, and the four false detections were excluded.
- A second test lecture of 10 minutes 26 seconds (about six students) also ran end to end.
- The team's fusion layer found no link between instructor actions and student behaviour that passed the correction
  for multiple testing in either lecture. Lectures this short cannot show such a link (see the results chapter).
- 249 automated tests pass (98 for System B, 151 for System A). They cover the event format, seat finding, feedback
  wording, the fusion step, the AI-text checks and the web application.

**Limitations.** The detections have not yet been checked against human labels, so no accuracy figure can be claimed.
The association results come from two short test lectures (4.6 and 10.4 minutes), while real lectures last 55–75
minutes. They are descriptive, not general findings. Planned activities such as a quiz can look like off-task
behaviour. Instructors can note such activities and their length, but not when they happened, so the analysis cannot
yet leave those minutes out. No real call to the language model has been made
yet, and no board region was calibrated. The team's evidence pipeline (System A, YOLO26n-pose with ByteTrack) and this two-camera analysis (System B)
share one event format but have not been merged.

**Next steps:**
- Human labelling with the built-in review sheets to measure accuracy.
- Full-length recorded lectures (55–75 minutes), with consent.
- Decide how to use the activities instructors note (quiz, group work, video, break, with their length) in the
  analysis **[TODO]**.
- Board calibration.
- The first real AI-rewrite call.
- Integration of the two implementations.
- PDF export of the report.

**[TODO: confirm the final numbers on the day of submission and add the team's contribution statement.]**
