# Classroom objects (COCO detector, no poses)

Added 2026-11-17 at the user's request: an isolated test of which classroom
objects a COCO-pretrained YOLO26 detector finds in the instructor and student
views. It runs on its own, separate from the pose pipeline (no people, poses,
tracks, or roles), on the same sampled frames (1 per second) and the same
synced windows and real times as the indicator runs.

## Files

| File | Role |
| --- | --- |
| `models/detection/yolo26n.pt` | COCO-pretrained YOLO26n detector (80 classes), 5.3 MB from the Ultralytics v8.4.0 release; git-ignored, never downloaded by the code |
| `src/inference/objects.py` | `ObjectDetector` (keeps only the chosen classes), `DetectedObject`, `adapt_detections()` |
| `src/pipeline.py` | `detect_objects()`: frames, detection, events, summary, review video |
| `src/analysis/aggregates.py` | `summarize_objects()`: per-class presence from the saved events |
| `src/reporting/annotated_video.py` | `ObjectOverlay`: coloured boxes, `label confidence` tags, a count panel |
| `scripts/detect_objects.py` | One video; `--list-classes` prints the 80 COCO names |
| `scripts/run_scenarios.py --objects` | Both views of a scenario, limited to the synced window |
| `configs/default.yaml` `objects:` | Weights, confidence 0.25, IoU 0.45, input size 1280, default classes |

Default classes: `cell phone`, `laptop`, `book`, `bottle`, `cup`,
`keyboard`, `mouse`, `remote`, `scissors`, `backpack`, `handbag`. COCO has
no pen, pencil, marker, notebook, calculator, whiteboard, or projector
screen; those need a fine-tuned detector (dataset candidates in
`docs/list_b_teacher.md`).

## Running

```powershell
python scripts/run_scenarios.py --scenario A2 --objects
python scripts/run_scenarios.py --scenario A2 --objects --classes laptop "cell phone" --tag phones
python scripts/detect_objects.py --video data/raw_student_videos/sgp_A2_student.MOV --video-id a2_student_objects --camera student_view --start 1:00 --end 2:00
python scripts/show_indicator.py sgp_A2_student_objects object_count
python scripts/show_indicator.py sgp_A2_student_objects object --value "cell phone"
```

Class names with spaces need quotes. `--objects` cannot be combined with
`--indicators`. Outputs in `outputs/runs/<YYYY-MM-DD_HHMM>_<id>/`:
`objects.jsonl`, `objects_summary.json`, `objects_review.mp4`, `objects_run.json`.

## Events and summary

| `indicator_name` | Kind | Value |
| --- | --- | --- |
| `object` | one per detection per sample | `{"label", "box_xyxy"}` (normalized), detector confidence |
| `object_count` | one per sample | `{class: count}` for every chosen class, zeros included |

All events have role `classroom`, no track ID, and real times like the pose
runs. The summary gives, per class, `samples_seen`, `fraction_of_samples`,
`max_count_in_one_sample`, `first_seen_seconds`, `last_seen_seconds`, and
`mean_confidence`.

## First check: A2, both views (2026-11-17; not an accuracy measurement)

`run_scenarios.py --scenario A2 --objects`, CPU, input 1280: 306 s for the
student view (326 samples) and 340 s for the instructor view (327 samples).

| View | Class: share of samples (max per sample, mean confidence) |
| --- | --- |
| Student | bottle 100% (3, 0.76), laptop 100% (4, 0.73), cell phone 75% (2, 0.30), book 12% (1, 0.26), cup 2% |
| Instructor | bottle 100% (3, 0.62), laptop 100% (1, 0.71), book 99% (1, 0.31) |

Checked in frames (student 3:37, instructor 2:40): in the student view the
closed MacBook (0.96), a phone lying on a desk (0.29), and the water
bottles are right; a closed laptop on a back desk is missed. In the
instructor view the bottles are right, but the "laptop" is the desk's
Dell monitor (COCO `tv` is not in the class list, so the detector picks
the nearest chosen class), the "book" is a brown folder or bag on the desk,
and the two open MacBooks behind the monitor are missed. Phones have low
confidence (about 0.3) even when correct, so raising the threshold would
lose them. Fixed desk items are seen in every sample; a per-desk baseline or
person association is needed before counts mean anything.

## Limits

- An object seen is not an object in use: a phone on a desk does not show
  that anyone is using it, and objects are not linked to people yet (that
  needs the pose boxes and a rule, for example a phone box inside a
  student's hand region).
- No tracking: the same laptop is counted again in every sample, and counts
  are per frame, not numbers of distinct objects.
- Small, partly hidden, or edge-on items (a phone held flat, a closed laptop
  seen from the side) are missed; confidence 0.25 is a starting value, not
  tuned on the scenarios.
