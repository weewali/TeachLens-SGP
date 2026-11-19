# Professional standards and practice constraints

## 1. Codes, laws and standards followed

| Standard or code | What it requires | How TeachLens applies it |
|---|---|---|
| ACM Code of Ethics and Professional Conduct | Avoid harm, be honest, respect privacy | Reports only observable behaviour. Every report lists its limitations. Results are labelled as associations, not causes. |
| IEEE Code of Ethics | Honesty about claims and limitations; protect privacy | Accuracy is not claimed until it is measured. The AI text is checked so it cannot add numbers or claim causes. |
| UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021) | Lawful, limited processing of personal data, with consent | Recordings require the consent of the instructor and the students. Video is processed only on the local computer. Outputs use anonymous seat numbers. No faces are stored or matched. |
| NIST Privacy Framework (principles) | Identify and limit privacy risks | Data minimisation: the language model receives only anonymised numbers, never video, images, file names or names. |
| ISO/IEC 27001 (principles; no formal certification) | Protect information assets | Recordings, frames, review images and model weights are kept out of Git. The web application listens only on the local computer. API keys are never written into code or shared in chat. |
| ISO/IEC 25010 (quality model, as a guide) | Functional suitability, reliability, usability, maintainability | Modular stages, automated tests, clear error messages, and a fallback when the AI step fails. |
| Accessibility practice (informed by WCAG 2.1; not formally audited) | Content usable by people with different abilities | Chart colours were checked for colour-vision deficiency and contrast. Every chart has a table view and keyboard access. The page keeps a white background on every computer and works at phone width. |

## 2. Engineering practice

- **Version control and teamwork:** Git and GitHub, with separate branches for the two implementations. A shared event
  format (`src/analysis/events.py`, identical in both) lets their outputs be combined later.
- **Modular, layered design:** evidence → analysis → reporting. Models never write report text, and every report can
  be rebuilt from saved files without reprocessing the video.
- **Configuration instead of code changes:**
  - per-camera calibration files (teacher zone, board region, student region, seat mode);
  - deployment settings for the web application (`TEACHLENS_LLM`, `TEACHLENS_LLM_MODEL`, `TEACHLENS_FFMPEG_DIR`).
- **Automated testing:** 249 unit tests (Python `unittest`; 98 for System B, 151 for System A). They run without
  network access or video and cover the event format, seat finding, feedback wording, the fusion step, the AI-text
  checks and the web application.
- **Reproducibility:**
  - Fixed model files that are never downloaded at run time.
  - Saved tables, events and `facts.json`.
  - AI answers cached by a hash of their input, with the prompt version recorded.
- **Safe defaults:**
  - Existing results are not overwritten without `--overwrite`.
  - Unknown values stay "unknown" and are never counted as zero.
  - The AI rewrite falls back to the rule-based text.
- **Scientific practice:**
  - Three hypotheses were fixed on 16 November 2026, before any two-camera data existed. Since 19 November 2026 the
    team's fusion layer tests every instructor × student pair and corrects over all of them (Benjamini–Hochberg), so
    no pair is singled out after seeing the data.
  - Weak results are labelled as leads.
  - p-values are corrected for the number of tests and for autocorrelation between neighbouring windows.
- **Documentation:** module documentation, a code map (`CLAUDE.md`), indicator rules (`docs/main_code_events.md`), a
  web application guide (`web/README.md`), and the report chapters in `SGP2 objectives/`, updated with every change.
  The pipeline figure is drawn by `scripts/make_pipeline_diagram.py`.

## 3. Practice constraints

| Constraint | Description | How it was handled |
|---|---|---|
| **Hardware** | CPU-only laptops; an 8 GB / 2-core cloud workspace during development | Pretrained models sized per task (nano for the instructor, medium for students); one frame every 2 s. The model runs on 20 frames at a time, after a single long call grew past 3 GB and was stopped. |
| **Data** | No labelled classroom action dataset; recordings of real people need consent | No model was trained. Transparent rules on pretrained keypoints were used instead. The team recorded its own test lecture **[TODO: confirm consent forms]**. Review sheets were built for human labelling. |
| **Privacy and legal** | Video shows identifiable people; phone videos carry GPS metadata | Local processing only. Anonymous seats. No face crops or identity matching. Review images stay local. Metadata is never printed or shared. The language model sees only numbers. |
| **Ethical** | Risk of overclaiming (e.g. "students were bored") or of using the tool to judge people | Observable wording only, enforced by code checks. Associations, not causes. Advisory, private feedback for the instructor. No scores or rankings. |
| **Camera setup** | Static cameras; the instructor camera must sit on the students' side so a visible face means facing the class; desks hide students' legs | Setup rules documented. Legs are not used for students. Every report states the facing assumption. |
| **Time alignment** | The two cameras start at different times; metadata start times are whole seconds | Alignment from metadata (about ±1 s), with a manual offset option. 10-second windows make a 1 s error small. |
| **Accuracy and validation** | Rule thresholds are starting values; no human labels yet | All results are labelled "unverified". The `validate_indicators.py` review sheets and scoring are ready **[TODO: labelling]**. |
| **External AI service** | Cost, availability, data leaving the computer, possible errors in generated text | Optional and set by the deployment, not the instructor. About USD 0.04 per report. Anonymised numbers only. Every number is checked against the facts. One retry, then the rule-based fallback. |
| **Instructor time** | Instructors teach 55–75-minute lectures and will not fill in long forms after each one | Only a lecture title and the two videos are required. Context is optional. An activity in class that was not teaching is added with one button and one box for its duration in minutes. A form with no context gives a normal report. |
| **Schedule** | One semester for implementation, testing and the report | Prioritised the end-to-end pipeline and the report over a dashboard and audio analysis. PDF export deferred. |
| **Integration** | Two implementations built in parallel | Shared JSONL event format; System B also writes the indicators the team's fusion layer reads and runs that layer itself, so both systems' results reach the same reporting layer (`lag_facts.py`) in one format. |
| **Economic** | Student project budget | Uses existing phones or low-cost cameras and tripods (SGP1 estimate: 500–1,200 AED per camera, 100–250 AED per tripod), free open-source software, and a pay-per-use AI step. |
