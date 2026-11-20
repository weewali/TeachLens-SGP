# Professional standards and practice constraints

## 1. Codes, laws and standards followed

| Standard or code | What it requires | How TeachLens applies it |
|---|---|---|
| ACM Code of Ethics and Professional Conduct | Avoid harm, be honest, respect privacy | Reports only observable behaviour. Every saved report file lists its limitations (the web page shows only the points and the key terms). Results are labelled as associations, not causes. |
| IEEE Code of Ethics | Honesty about claims and limitations; protect privacy | Accuracy is not claimed until it is measured. The AI text is checked so it cannot add numbers or claim causes. |
| UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021) | Lawful, limited processing of personal data, with consent | Recordings require the consent of the instructor and the students. Video is processed only on the local computer. Outputs use temporary track IDs and anonymous seats. No faces are stored or matched. Students' faces are pixelated in the report's pictures; the instructor's own report shows the instructor. |
| NIST Privacy Framework (principles) | Identify and limit privacy risks | Data minimisation: the language model receives only anonymised numbers, never video, images, file names or names. |
| ISO/IEC 27001 (principles; no formal certification) | Protect information assets | Recordings, frames, review images and model weights are kept out of Git. The web application listens only on the local computer. API keys are never written into code or shared in chat. |
| ISO/IEC 25010 (quality model, as a guide) | Functional suitability, reliability, usability, maintainability | Modular stages, automated tests, clear error messages, and a visible reason (an API key problem only as "Code=01", a billing problem only as "Code=02") with "Try again" when the AI step fails. |
| Accessibility practice (informed by WCAG 2.1; not formally audited) | Content usable by people with different abilities | Chart colours were checked for colour-vision deficiency and contrast. Every chart has a table view and keyboard access. The page keeps a white background on every computer and works at phone width. |

## 2. Engineering practice

- **Version control and teamwork:** Git and GitHub, with separate branches; since 20 November 2026 this branch runs the team's pipeline, with the report stage and web application on top. A shared event
  format (`src/analysis/events.py`, identical in both) lets their outputs be combined later.
- **Modular, layered design:** evidence → analysis → reporting. Models never write report text, and every report can
  be rebuilt from saved files without reprocessing the video.
- **Configuration instead of code changes:**
  - YAML settings for the pipeline (sampling, thresholds, cameras, indicators, fusion) and a room file per classroom
    (board, screen and desk areas);
  - deployment settings for the web application (`TEACHLENS_LLM_MODEL`, `TEACHLENS_ROOM_CONFIG`, `TEACHLENS_OUTPUTS`).
- **Automated testing:** 217 unit tests (Python `unittest`; 170 for the team's pipeline, 47 for the report stage and
  web application). They run without
  network access or video and cover the event format, seat finding, feedback wording, the fusion step, the AI-text
  checks and the web application.
- **Reproducibility:**
  - Fixed model files that are never downloaded at run time.
  - Saved tables, events and `facts.json`.
  - AI answers cached by a hash of their input, with the prompt version recorded.
- **Safe defaults:**
  - Existing results are not overwritten without `--overwrite`.
  - Unknown values stay "unknown" and are never counted as zero.
  - When Claude cannot write the feedback, the report shows why and can try again; charts and moments stay available.
- **Scientific practice:**
  - Three hypotheses were fixed on 16 November 2026, before any two-camera data existed. Since 19 November 2026 the
    team's fusion layer tests every instructor × student pair and corrects over all of them (Benjamini–Hochberg), so
    no pair is singled out after seeing the data.
  - The statistical settings are fixed in advance and are never tuned on the data; only the behaviour-rule
    thresholds may later be tuned, against human labels.
  - Weak results are labelled as leads.
  - p-values are corrected for the number of tests and for autocorrelation between neighbouring windows.
- **Documentation:** module documentation, a code map (`CLAUDE.md`), indicator rules (`docs/main_code_events.md`), a
  web application guide (`web/README.md`), and the report chapters in `SGP2 objectives/`, updated with every change.
  The pipeline figure is drawn by `scripts/make_pipeline_diagram.py`.

## 3. Practice constraints

| Constraint | Description | How it was handled |
|---|---|---|
| **Hardware** | CPU-only laptops; an 8 GB / 2-core cloud workspace during development | Pretrained nano models; one analysed frame per second; videos decoded on the laptop's graphics chip where possible. About 2.2 minutes of computing per lecture minute (more on battery power), so a one-hour lecture takes two hours or more. |
| **Data** | No labelled classroom action dataset; recordings of real people need consent | No model was trained. Transparent rules on pretrained keypoints were used instead. The team recorded its own test lecture **[TODO: confirm consent forms]**. Review sheets were built for human labelling. |
| **Privacy and legal** | Video shows identifiable people; phone videos carry GPS metadata | Local processing only. Temporary track IDs and anonymous seats. No face crops are stored and no identity matching. Students' faces are pixelated in the report's pictures (the instructor is not, since the report is theirs); the analysis itself reads the original videos, which stay on the local computer. Review images stay local. Metadata is never printed or shared. The language model sees only numbers. |
| **Ethical** | Risk of overclaiming (e.g. "students were bored") or of using the tool to judge people | Observable wording only, enforced by code checks. Associations, not causes. Advisory, private feedback for the instructor. No scores or rankings. |
| **Camera setup** | Static cameras; the instructor camera must sit on the students' side so a visible face means facing the class; desks hide students' legs | Setup rules documented. Legs are not used for students. Every report states the facing assumption. |
| **Time alignment** | The two cameras start at different times; metadata start times are whole seconds | Alignment from metadata (about ±1 s), with manual start-time options. 5-second windows keep a 1 s error small. |
| **Accuracy and validation** | Rule thresholds are starting values; no human labels yet | The saved report files label all results "unverified"; since 19 November 2026 the web page no longer shows that note **[TODO: decide whether the page should say it]**. The `validate_indicators.py` review sheets and scoring are ready **[TODO: labelling]**. |
| **External AI service** | Cost, availability, data leaving the computer, possible errors in generated text | Always on (user decision, 20 November 2026), with the key set by the deployment, never by the instructor. About USD 0.04 per report, charged to the account of the API key set by the deployment. The key is read from a private `.env` file at start-up. Anonymised numbers only. Every number is checked against the facts. One retry; without a key or on any error, no feedback is shown, only the reason; an API key problem is shown only as "Code=01" and a billing problem of the key's account only as "Code=02", so the page never reveals anything about the key or the account. |
| **Instructor time** | Instructors teach 55–75-minute lectures and will not fill in long forms after each one | Only a lecture title and the two videos are required. Context is optional. An activity in class that was not teaching is added with one button and one box for its duration in minutes. A form with no context gives a normal report. |
| **Schedule** | One semester for implementation, testing and the report | Prioritised the end-to-end pipeline and the report over a dashboard and audio analysis. PDF export deferred. |
| **Integration** | Two implementations were built in parallel | Merged on 20 November 2026: the team's pipeline does the processing and writes `facts.json`; the report stage and web application read that lesson folder. Duplicate processing was removed. |
| **Economic** | Student project budget | Uses existing phones or low-cost cameras and tripods (SGP1 estimate: 500–1,200 AED per camera, 100–250 AED per tripod), free open-source software, and a pay-per-use AI step. |
