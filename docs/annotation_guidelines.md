# Annotation Guidelines

## Allowed Labels

The imported two-class dataset in `data/student_instructor_imported/` uses:

```text
0: student
1: instructor
```

The separate `person_detection_v1` and `person_detection_v2` datasets use
`0: person`. Do not change a dataset's numeric class meaning in place.

Do not use behavior labels such as `writing`, `walking`, `focused`,
`distracted`, `bored`, or `engaged` in a role-detection dataset.

## Bounding Box Rules

- Draw a tight box around each visible person.
- Label the teacher or lecturer as `instructor`.
- Label each visible learner as `student`.
- Give every visible student a separate bounding box.
- Remove or mark blurry, unclear, or useless frames for review.
- Every image must have a matching `.txt` YOLO label file.
