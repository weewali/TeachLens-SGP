# Introduction

## 1.1 Background and motivation

Good teaching depends on feedback. In most universities an instructor's main source of feedback is the end-of-semester
student survey. Surveys arrive after the course has ended, are often completed in a hurry, and describe impressions
rather than specific moments. An instructor learns that "the lectures were hard to follow", but not when, or what was
happening in the room at the time. Peer observation gives richer feedback but is costly and therefore rare.

Recent work applies computer vision to classroom video to recognise behaviour automatically [3]. Some systems estimate
student "attention" from detections and tracking, for example with YOLO and DeepSORT [2], or combine several signals to
assess engagement [1]. TeachLens builds on this line of work but takes a deliberately narrower and more careful
position:
- It reports only what a camera can actually see, such as a raised hand, a head kept down, or a phone in view.
- It never claims to measure attention, engagement, emotion or understanding.
- It links student behaviour to what the instructor was doing at the time, so the feedback is about teaching, not
  about individual students.

## 1.2 Problem statement

Instructors lack timely, specific and objective feedback on how observable student behaviour changes during their
lectures, and on how those changes relate to what they were doing at the time. Existing automated approaches often
present inferred inner states as facts or imply causes from simple correlations. Many also process identifiable video
in ways that raise privacy concerns.

TeachLens addresses three needs:
1. Recording both the instructor and the students clearly.
2. Measuring observable behaviour transparently and analysing it with statistics that do not overstate chance
   patterns.
3. Presenting the result to the instructor in plain language while keeping people anonymous and the data local.

## 1.3 Aim and objectives

The aim is to give university instructors timely, evidence-based and private feedback on their teaching from a
recorded lecture. The feedback describes associations between instructor activity and observable student behaviour,
never causes or inner states. The specific objectives are:

1. Record each lecture with two static cameras (instructor and students) and align them to about one second.
2. Detect people, body keypoints, phones and books with pretrained YOLO26 models on a laptop CPU.
3. Derive observable instructor and student indicators with transparent rules and anonymous seats.
4. Store timestamped evidence in the team's shared event format.
5. Analyse instructor–student associations with statistics that allow for autocorrelation and multiple testing.
6. Produce up to three positive and three negative points in plain English, with an optional, automatically checked
   AI rewrite.
7. Deliver the analysis through a local web application.
8. Protect privacy and state the limits of every result.
9. Verify the software and evaluate it on real recordings.

Full wording and the benefits of the project: see `objective.md`.

## 1.4 Scope

**In scope:**
- Recorded (not live) lectures filmed by two static cameras.
- Observable behaviour:
  - instructor: presence, facing the class, board work when a board region is calibrated, movement, sitting or
    standing;
  - students: heads up, head down and still, hand raises, phones and books in view.
- Anonymous seats.
- Associations within a lecture (pooling across lectures is not built yet).
- Plain-English feedback and a local web report.

**Out of scope:**
- Recognising or naming people.
- Measuring attention, engagement, emotion or learning.
- Grading students or rating instructors.
- Live monitoring.
- Audio and speech.
- Causal claims.

## 1.5 Changes since SGP1

| SGP1 plan | SGP2 | Reason |
|---|---|---|
| One fixed camera covering the instructor and students | Two static cameras, aligned from recording metadata | One view rarely shows both sides clearly. In a single-camera test, roles assigned by image zone failed when the framing changed. |
| Train YOLO11 action models (instructor and student classes) | Pretrained YOLO26 pose and detection models with transparent rules; no training | No labelled classroom action dataset was available. Rules on keypoints can be explained and tested. |
| Seat regions for student identity | Seats found automatically from where students sat (System B); temporary tracker IDs (System A) | A fixed grid did not match real seating. |
| Emotion recognition | Dropped | It cannot be measured reliably and would claim inner states. |
| Cross-correlation and lagged regression | Kept, as the team's fusion layer: a circular-shift test, Benjamini–Hochberg q-values, lagged models with HAC errors, and averages around the moments an action started | Neighbouring windows are alike, so naive p-values overstated certainty. |
| Report template, with an LLM filling placeholders | Rule-based feedback plus an optional Claude Opus 5.5 rewrite checked in code; local web application | Keeps every sentence traceable to a measured number. |

## 1.6 Structure of this report

**[TODO: match the official SGP2 template.]** A possible structure:
- Executive summary
- Introduction
- Professional standards and practice constraints
- System design and integration
- Implementation, testing and results analysis
- Conclusion and future work
- Statement of contribution

## References

[1] Li et al., multimodal learning engagement assessment, *International Journal of Human–Computer Interaction*, 2025.
**[TODO: full citation as in the SGP2 draft.]**

[2] Trabelsi et al., real-time attention monitoring with YOLOv5 and DeepSORT, *Big Data and Cognitive Computing*, 2023.
**[TODO: full citation.]**

[3] Liu, Jiang and Jiang, classroom behaviour recognition with computer vision: a systematic review, *Sensors*, 2025.
**[TODO: full citation.]**
