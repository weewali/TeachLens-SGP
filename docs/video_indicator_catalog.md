# Classroom Video Indicator Catalog

This catalog describes the broadest **defensible and useful** set of indicators that TeachLens could extract from a recorded classroom video. It includes visual indicators, audio indicators, transcript-based indicators, and indicators derived by combining them.

It is not correct to claim that a camera directly measures attention, engagement, understanding, boredom, or emotion. The system should record **observable behavior** such as head orientation, speaking, hand raising, or writing, then describe any higher-level value as a limited proxy.

## Common record format

Every event should store:

- `video_id`
- anonymous `track_id` or `speaker_id`
- `role`: `teacher`, `student`, `both`, or `classroom`
- `indicator_name`
- `start_time`, `end_time`, and `duration`
- raw value or event label
- model confidence
- visibility/audio-quality flag
- optional classroom zone, such as `board`, `front`, `left`, `center`, `right`, or `student_seating`

The report can then aggregate events into counts, rates, percentages, timelines, and heatmaps. A student track should remain anonymous unless there is a separate, approved reason to identify people.

## A. Foundation: people, roles, tracking, and location

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Teacher | Teacher presence | Whether the teacher is visible; first/last appearance; visible duration and percentage of lesson | Detect the `instructor` class in each sampled frame, track it with ByteTrack/BoT-SORT, and merge adjacent detections into intervals |
| Student | Visible student count | Number of students visible at each time; minimum, maximum, median, and time series | Detect the `student` class, track detections, and count reliable tracks per frame; suppress duplicate and low-confidence tracks |
| Teacher or student | Persistent anonymous identity | A temporary ID linking the same visible person across frames | Multi-object tracking using bounding-box motion and appearance embeddings; reset IDs between recordings |
| Teacher or student | Role classification | Whether a tracked person is the teacher or a student, with confidence | Use the project's two-class detector and stabilize predictions across the track; optionally use location and speaking evidence as secondary cues |
| Teacher or student | Position | Bounding-box center, body keypoints, and normalized coordinates over time | Person detection plus pose estimation; normalize coordinates by frame width and height |
| Teacher or student | Classroom zone | Time spent in configured regions such as board, lectern, aisle, or seating area | Calibrate polygon zones once per camera, then test the person's foot point or box center against each polygon |
| Teacher or student | Entry and exit | Timestamp when a person enters or leaves the visible scene | Detect creation and termination of stable tracks near frame boundaries; ignore short tracking losses |
| Teacher or student | Visibility/occlusion | Whether the full body, upper body, face, or required joints are visible | Combine bounding-box truncation, pose-keypoint confidence, face detection, and percentage of the track hidden by other boxes |
| Teacher or student | Sitting/standing | State, transition timestamps, duration, and percentage of visible time | Classify temporally smoothed pose geometry using hip/knee angles and torso-to-box proportions; train a small action classifier for difficult camera angles |
| Teacher or student | Motion level | Still, low, medium, or high body movement and duration | Use tracked keypoint displacement or optical flow normalized by body size and camera motion |

## B. Teacher visual behavior

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Teacher | Position heatmap | Where the teacher spends time in the classroom | Accumulate the tracked teacher foot point in a normalized grid or calibrated floor plane |
| Teacher | Movement path and distance | Trajectory, estimated distance traveled, and movement rate | Smooth the teacher track; sum displacement in normalized image coordinates or map it to real distance after camera calibration |
| Teacher | Classroom coverage | Number/percentage of teaching zones visited and duration in each | Combine the teacher track with configured classroom zones |
| Teacher | Stationary episodes | Start, end, duration, and location of periods with little movement | Threshold smoothed track speed for a minimum duration |
| Teacher | Facing the class | Percentage of visible time oriented toward students rather than the board/screen | Estimate torso and head orientation from pose/face landmarks; classify direction relative to calibrated class and board zones |
| Teacher | Head-orientation target | Approximate target: students, board, screen, notes, or unknown | Use head-pose estimation and cast a coarse direction ray toward calibrated scene regions; report as orientation, not true eye gaze |
| Teacher | Visual coverage of student areas | Time and frequency the teacher orients toward left, center, and right seating zones | Combine head/torso direction with zone calibration and aggregate by seating region |
| Teacher | Board/screen interaction | Start, end, and duration of attending to the board or projected screen | Detect proximity to the board/screen zone plus body orientation and relevant arm motion |
| Teacher | Writing on board | Writing episodes, duration, and board region used | Detect teacher near board, facing it, with repeated wrist motion inside the board region; a temporal action model improves reliability |
| Teacher | Pointing | Pointing events, duration, and approximate target | Detect an extended elbow/wrist pose; follow the shoulder-to-wrist vector to a board, screen, object, or student zone |
| Teacher | Gesture rate | Number of meaningful hand/arm gestures per minute | Track wrist/elbow trajectories, segment movements temporally, and exclude locomotion and writing |
| Teacher | Gesture expansiveness | Typical and maximum hand spread relative to shoulder width | Measure wrist distance and distance of wrists from torso, normalized by shoulder width |
| Teacher | Raised hand | Count and duration of one or both hands raised | Compare wrist height to the corresponding shoulder with keypoint-confidence checks and temporal smoothing; this already matches the repository demo |
| Teacher | Posture/torso lean | Upright, forward lean, sideways lean, or unknown over time | Compute shoulder/hip alignment from pose keypoints and smooth the state; do not attach personality judgments |
| Teacher | Proximity to students | Duration and frequency of teacher presence near student seating zones or individual anonymous tracks | Use calibrated floor-plane distance where possible; otherwise use zone adjacency and perspective-aware thresholds |
| Teacher | Teaching-material use | Episodes involving laptop, book, paper, marker, remote, or phone | Object detection plus hand-object proximity and temporal action classification; distinguish a presentation remote from a phone only when image detail allows |
| Teacher | Observable facial actions | Smile-like facial action, visible laughter, or neutral/unknown state, only when the face is sufficiently clear | Face landmarks/action-unit or expression classifier with strict size and confidence gates; record the visible action, not an internal emotion |

## C. Student visual behavior

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Student | Seat occupancy proxy | Which visible seat regions appear occupied and for how long | Calibrate seat polygons and associate stable student tracks with seats; this is not official attendance when seats or students are hidden |
| Student | Hand-raise event | Anonymous student ID, start/end time, duration, hand side, and count | Compare wrist and shoulder pose keypoints, as in the existing demo, and require the state across multiple frames |
| Student | Hand-raise response latency | Time from a teacher question/prompt to each student's raised hand | Fuse transcript-detected question timestamps with hand-raise events |
| Student | Hand-raise selection outcome | Whether a raised-hand student was apparently selected and then spoke | Link teacher pointing/gaze or name/call event to the student track, then confirm associated speech; keep an `uncertain` state |
| Student | Head-orientation target | Approximate orientation toward teacher, board/screen, desk/materials, peer, or elsewhere | Head-pose estimation plus calibrated target zones; use torso orientation if the face is too small |
| Student | Visual-orientation proxy | Percentage of visible time oriented toward the current instructional target | Compare head/torso orientation with the active target inferred from teacher location, slide/board activity, and lesson phase; call this a proxy, not attention |
| Student | Note-taking/writing | Start/end time, duration, and repeated writing episodes | Detect a sustained downward orientation with small wrist motion near a notebook/tablet region; improve with desk/object detection and a temporal action model |
| Student | Reading | Apparent reading episodes and duration | Detect book/paper/screen use plus sustained head orientation and limited hand movement; distinguish from head-down posture when image resolution permits |
| Student | Device use | Visible use of a phone, tablet, or laptop and duration | Object detection plus hand-object proximity and gaze/orientation; context determines whether use is learning-related, so do not label it distraction automatically |
| Student | Learning-material use | Use of book, worksheet, calculator, lab equipment, or other visible material | Object detection followed by hand-object interaction classification |
| Student | Individual work | Duration of independent writing/reading/task behavior | Activity classifier using pose, objects, head direction, teacher instruction, and low peer-speech overlap |
| Student | Peer discussion | Who appears to interact, group membership, and duration | Cluster nearby tracks facing each other and combine with localized/diarized student speech |
| Student | Group-work participation proxy | Speaking/gesture/activity contribution within an identified group | Fuse anonymous speaker turns, body orientation, hand motion, and group membership; never infer contribution quality from movement alone |
| Student | Standing/sitting | State, changes, and duration | Pose-based state classifier with temporal smoothing |
| Student | Out-of-seat movement | Leaving a calibrated seat, walking duration, and destination zone | Associate a track with a home seat and detect sustained movement away from it |
| Student | Entry/exit | Arrival to or departure from the visible classroom area | Stable track appearance/disappearance near a door or frame edge |
| Student | Nod/head shake | Count and timestamp of repeated vertical or horizontal head motion | Track face/head landmarks over a short temporal window and classify oscillation direction |
| Student | Response gesture | Thumbs-up, raised response card, applause, or other configured classroom gesture | Hand-pose/object detection plus a temporal gesture classifier trained for the chosen gesture set |
| Student | Head-down episode | Sustained head-down posture, timestamp, and duration | Head/torso pose threshold over time; report only posture because it may mean reading, writing, fatigue, prayer, or occlusion |
| Student | Body-movement level | Per-track motion intensity and unusually large changes | Normalize keypoint displacement by body scale and remove global camera motion; do not equate stillness or movement with engagement |
| Student | Writing/presenting at board | Which anonymous track approaches and interacts with the board; duration | Track a student entering the board zone and detect pointing/writing motion |

## D. Speech, voice, and transcript indicators

These indicators require usable audio. The normal pipeline is voice-activity detection, speaker diarization, audio-video active-speaker association, speech-to-text, and then rule-based or language-model analysis of the transcript. If the microphone captures only the teacher, student-speaking measures will be incomplete.

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Teacher | Teacher speaking time | Total speech duration, percentage of lesson, and timeline | Voice-activity detection (VAD), diarization, and assignment of the dominant/visible active speaker to the teacher |
| Student | Student speaking time | Total and per-anonymous-speaker duration and percentage of lesson | VAD plus diarization; associate visible lip movement with tracks when possible and retain an `unseen student` class |
| Teacher or student | Speaking turns | Turn count, start/end time, duration, and speaker role | Segment diarized speech whenever the speaker changes or a silence exceeds a configured threshold |
| Teacher or student | Average turn length | Mean, median, distribution, and longest turn | Aggregate durations of role-attributed speech turns |
| Teacher or student | Speech rate | Words per minute overall and over time | Count time-aligned ASR words during speech intervals; exclude long silences |
| Teacher or student | Pause behavior | Within-turn pause count and duration distribution | Use word timestamps and VAD gaps, separating short rhetorical pauses from turn-ending silence |
| Teacher or student | Filler words | Count and rate of configured fillers such as “um” and “uh” | Match language-specific filler lexicons against the transcript; use the audio model for non-lexical fillers |
| Teacher or student | Repeated words/phrases | Immediate repetitions and recurring phrases | Compare adjacent transcript tokens and n-grams, normalized by total words |
| Teacher or student | Speech volume | Relative loudness, variation, and clipping over time | Compute calibrated RMS/LUFS on each speaker segment; report relative levels unless the microphone is calibrated |
| Teacher or student | Pitch variation | Median pitch and variation during voiced speech | Run fundamental-frequency tracking per speaker segment; do not interpret pitch as emotion or identity |
| Teacher or student | Audibility/intelligibility proxy | Signal-to-noise ratio, clipping, ASR confidence, and unintelligible-word rate | Estimate noise/SNR, detect clipping, and aggregate ASR confidence or `[inaudible]` spans |
| Teacher | Question count | Number and rate of teacher questions with timestamps | Combine punctuation/prosody with a transcript classifier; distinguish genuine questions from rhetorical ones where possible |
| Teacher | Open versus closed questions | Count and percentage of open, closed, rhetorical, and procedural questions | Classify question text with rules plus a reviewed NLP/LLM rubric |
| Teacher | Cognitive level of questions | Recall, comprehension, application, analysis, evaluation, or creation categories | Apply a Bloom-style rubric to each transcript question; validate samples because this is a semantic inference |
| Student | Student questions | Count, rate, and which anonymous speaker asked | Classify diarized student turns as questions using text and rising intonation |
| Student | Student answers | Count and duration of apparent responses to teacher prompts | Link student turns occurring inside a configurable window after teacher questions/invitations |
| Teacher | Instructions/directions | Instruction count, timestamps, and duration | Classify imperative/procedural transcript spans; group multi-sentence directions into one event |
| Teacher | Explanations and examples | Episodes labeled explanation, worked example, analogy, demonstration, or summary | Segment teacher transcript and classify discourse function using a rubric-based model |
| Teacher | Check-for-understanding prompts | Count of prompts asking students to demonstrate or report understanding | Detect configured phrases and semantic equivalents, then check for an observable response |
| Teacher | Feedback to students | Feedback episodes and type: acknowledgment, elaboration, corrective, prompting, or evaluative | Link teacher turns to prior student turns and classify the response function from transcript context |
| Teacher | Praise/positive acknowledgment | Count and target context of explicit positive acknowledgments | Transcript phrase/rubric classifier; avoid scoring generic positivity without context |
| Teacher | Corrective feedback | Count, wording type, and whether a retry/follow-up occurred | Classify feedback following a student answer, then inspect the next turns in the exchange |
| Teacher | Wait time after question | Silence from the end of a teacher question to the first student response or teacher continuation | Use aligned question boundary, diarized turns, and VAD timestamps |
| Teacher | Wait time after student answer | Time before teacher feedback/follow-up | Measure between a linked student-response end and teacher-turn start |
| Teacher | Follow-up/probing questions | Count of questions that build on a student response | Link conversational turns and classify whether the next teacher question references or extends the answer |
| Teacher | Revoicing/paraphrasing | Instances where the teacher restates a student's contribution | Compare semantic similarity between a student turn and the immediately following teacher turn, then classify its discourse purpose |
| Teacher | Learning-objective mention | Whether and when stated objectives or outcomes are mentioned | Compare transcript spans with configured lesson objectives or objective-like language |
| Teacher | Topic/content coverage | Topics mentioned, duration per topic, and sequence | Use timestamped transcript topic segmentation; compare with a syllabus/lesson plan only when one is provided |
| Teacher | Lesson signposting | Openings, transitions, recaps, and closing summaries | Classify discourse markers and semantically equivalent transcript spans |
| Teacher | Language complexity | Sentence length, lexical diversity, terminology density, and estimated readability | Compute transcript statistics after correcting major ASR errors; interpret relative to learner level |
| Teacher or student | Subject vocabulary use | Count, variety, and context of configured domain terms | Match a course glossary plus semantic variants against the transcript |
| Teacher or student | Language/code switching | Time and words spoken in each detected language | Run language identification on sufficiently long transcript/audio windows; code-switch detection needs multilingual ASR |
| Teacher or student | Overlap/interruption | Simultaneous speech duration and likely interrupted turns | Use overlap-aware diarization; infer interruption only when a new speaker starts before another finishes and continues the floor |
| Student | Choral response | Timestamp and duration of several students answering together | Detect overlapping/multi-speaker student speech after a teacher prompt, supported by room-audio energy patterns |
| Student | Response relevance | Whether an answer addresses the preceding question | Compare question and answer meaning with a rubric-based semantic model; report confidence and allow human review |
| Student | Answer correctness | Correct/partially correct/incorrect/unknown for a response | Requires a supplied answer key, rubric, or trusted subject model plus transcript context; never infer reliably from video alone |
| Student | Peer explanation | Student turns that explain reasoning or content to peers | Detect during group-work segments and classify discourse function from local conversation context |

## E. Teacher–student interaction and whole-class indicators

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Both | Teacher/student talk ratio | Percentage of detected speech time by teacher versus students | Divide role-attributed speech duration by total valid speech duration and report unassigned speech separately |
| Both | Turn-taking pattern | Sequence and transitions such as T→S, S→T, T→T, and S→S | Aggregate the ordered diarized role-attributed turns into a transition matrix |
| Both | Question-response cycle | Teacher question, wait time, respondent, answer, feedback, and follow-up as one linked event | Use temporal rules and conversational NLP over diarized transcript turns |
| Both | Question response rate | Percentage of teacher questions receiving a visible or audible student response | Look for student speech, hand raises, response cards, nods, or configured gestures within a response window |
| Both | Verbal participation breadth | Number and percentage of distinct anonymous student speakers | Count stable diarized speakers, using audio-video association when possible; do not present as enrollment coverage if diarization is unreliable |
| Both | Participation distribution | Turns/time per anonymous student, seating zone, or group; concentration/equality statistics | Aggregate student turns by speaker/track and compute distributions such as top-speaker share or Gini coefficient |
| Both | Hand-raise participation breadth | Number and percentage of visible anonymous students who raised a hand | Count unique stable tracks with hand-raise events and divide by students with sufficient pose visibility |
| Both | Selection equity proxy | Distribution of apparent teacher selections across tracks or seating zones | Link teacher prompts/pointing/names to subsequent student speech, then aggregate; label uncertain selections |
| Both | Teacher orientation/contact distribution | Teacher-facing time and interactions directed to left, center, right, or groups | Combine head/torso direction, pointing, proximity, and dialogue links with seating zones |
| Both | Interaction heatmap | Where question, response, proximity, and selection events occur | Plot linked events using student/teacher positions in calibrated zones |
| Both | Student-question handling | Whether student questions receive acknowledgment, answer, deferral, or no clear response | Link each student question to following teacher turns and classify the response outcome |
| Both | Feedback latency | Time between a student answer/action and teacher feedback | Link interaction events and subtract timestamps |
| Both | Interruption/overlap rate | Frequency and duration of overlapping turns by role pairing | Aggregate overlap-aware diarization by teacher–student and student–student combinations |
| Classroom | Activity phase | Lecture, questioning, whole-class discussion, individual work, group work, presentation, transition, or break | Fuse speaker balance, positions, movement, transcript cues, and object/action evidence in a temporal phase classifier |
| Classroom | Time allocation by phase | Duration and percentage of lesson spent in each activity phase | Merge consecutive phase predictions and aggregate durations |
| Classroom | Transition time | Duration between one established activity phase and the next | Detect phase-boundary intervals with instructions, movement, silence, or material changes |
| Classroom | Whole-class visual-orientation proxy | Percentage of sufficiently visible students oriented toward the current instructional target | Aggregate per-student orientation only across valid, visible tracks and show the denominator |
| Classroom | Response synchrony | Timing/spread of hands, cards, gestures, or verbal responses after a prompt | Align response events to the prompt and calculate latency distribution |
| Classroom | Peer-interaction network | Anonymous graph of who appears to speak/interact with whom and for how long | Combine proximity, mutual orientation, group clustering, diarization, and temporal co-occurrence |
| Classroom | Group formation | Number, membership, location, and duration of apparent working groups | Spatially cluster stable student tracks and require mutual orientation or group speech/activity evidence |
| Classroom | Classroom noise level | Relative ambient noise, SNR, and noisy intervals | Analyze non-speech audio energy and spectral features; absolute decibels require a calibrated microphone |
| Classroom | Silence | Total silent time and durations, separated into pauses, wait time, individual work, or unknown silence | Use VAD, then classify the surrounding activity context |
| Classroom | Laughter/applause event | Timestamp, duration, and whether one or many people participated | Audio event detection plus visible facial/hand-motion confirmation when available; do not infer why it occurred |
| Classroom | Attendance proxy | Maximum/median unique visible student tracks or occupied seats | Combine long-lived tracks and seat occupancy; explicitly label it incomplete when the camera does not show every seat |
| Classroom | Evidence coverage | Percentage of lesson for which each indicator had adequate visibility/audio | Aggregate the quality gates required by each indicator; show this beside every reported metric |

## F. Recording-quality indicators

These do not evaluate the teacher or students, but they determine whether the behavioral indicators are trustworthy.

| Who | Indicator | What it records | How to detect it |
|---|---|---|---|
| Recording | Frame quality | Resolution, frame rate, blur, darkness, glare, and compression artifacts | Read video metadata and calculate sharpness, brightness, saturation, and blockiness per frame |
| Recording | Camera stability | Shake, pan, tilt, zoom, and scene cuts | Estimate global optical flow/homography and detect abrupt histogram/feature changes |
| Recording | Scene coverage | Percentage of lesson in which teacher, student area, board, and required body parts are visible | Combine detections, calibrated regions, truncation, and keypoint-confidence coverage |
| Recording | Audio quality | Missing audio, clipping, SNR, echo/reverberation, and channel balance | Inspect audio metadata and compute signal-quality measures over time |
| Recording | ASR quality proxy | Average word confidence and low-confidence transcript spans | Aggregate ASR token confidence and mark unintelligible regions |
| Recording | Track quality | ID switches, fragmented tracks, and average detection confidence | Use tracker diagnostics and trajectory discontinuity checks |

## G. Indicators that should not be claimed from ordinary classroom video

TeachLens should not present the following as factual video measurements:

- A student's or teacher's internal emotion, including boredom, happiness, anxiety, anger, or confusion.
- True attention, engagement, motivation, interest, effort, learning, or understanding.
- Intelligence, personality, learning style, disability, medical condition, mental health, or intent.
- Race, ethnicity, religion, socioeconomic status, nationality, gender identity, or sexual orientation.
- Cheating, misconduct, dangerousness, honesty, or guilt based only on appearance or posture.
- Official attendance or identity unless there is an explicitly approved identity system and complete camera coverage.

Where the product needs a related measure, use a narrowly named observable proxy. For example, report **“oriented toward instructional target for 62% of valid visible time”**, not **“62% attentive.”**

## H. Recommended TeachLens implementation order

### Phase 1: reliable MVP

1. Teacher/student detection and anonymous tracking.
2. Presence, visible counts, position, zones, and evidence coverage.
3. Hand raises, sitting/standing, head/torso orientation, and teacher movement.
4. Teacher speaking time, student speaking time, turns, and talk ratio.
5. Teacher questions, student responses, and wait time.
6. A lesson timeline combining the above events.

### Phase 2: useful classroom analytics

1. Board writing, pointing, note-taking, device/material use, and peer discussion.
2. Participation breadth and distribution.
3. Feedback, instructions, checks for understanding, and activity phases.
4. Zone/orientation heatmaps and question-response cycles.

### Phase 3: advanced and reviewable analytics

1. Question cognitive level, explanation type, response relevance, and feedback type.
2. Group interaction networks and selection-equity proxies.
3. Answer correctness only when a course-specific rubric or answer key exists.

Semantic indicators in Phase 3 should expose the supporting transcript clip and be reviewable by a human rather than appearing as unquestionable scores.

## I. Suggested detection architecture

```text
Video ──> teacher/student detector ──> multi-object tracker ──> pose/head/object models
   │                                      │                         │
   │                                      └──── positions/zones ────┤
   │                                                                │
Audio ──> VAD ──> diarization ──> active-speaker association ──> ASR transcript
                                                                    │
                     temporal rules + action models + transcript NLP
                                                                    │
                       timestamped evidence events with confidence
                                                                    │
                         aggregates, timelines, heatmaps, report
```

The safest reporting pattern is: **observation + denominator + confidence + limitation**. For example: “12 hand raises were detected among 18 students whose upper-body pose was visible for at least half of the lesson; estimated event precision on the validation set: 0.87.”
