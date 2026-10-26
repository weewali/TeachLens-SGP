# TeachLens-SGP

TeachLens is an AI-based classroom teaching analytics system developed for our Senior Graduation Project. The system analyzes recorded classroom lecture videos to detect instructor behaviors, student engagement patterns, and classroom interaction trends. It then converts these results into structured indicators and generates a human-readable teaching evaluation report with strengths, weak points, and practical suggestions for improvement.

## Project map

The agreed repository layout, language choices, file responsibilities, data flow, and implementation order are documented in [docs/project_structure.md](docs/project_structure.md).

The executable pipeline will remain Python-first. YAML files configure runs, Markdown files document decisions and datasets, and Jinja2/HTML/CSS templates will render reports.

The first implemented slice is [List A physical pose analysis](docs/list_a_pose.md):
local YOLO26 pose tracking, timestamped evidence, a JSON summary, and an
annotated review MP4 (with an optional live window via `--live`,
`--start/--end` to analyze one segment, and `--path teacher` to draw the
instructor's path, compensated for camera pans). It does not analyze audio or
infer attention/engagement.
