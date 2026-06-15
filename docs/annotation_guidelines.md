# Annotation Guidelines

## Allowed Labels

Only these two labels are allowed in `dataset_v1`:

```text
0: instructor
1: student
```

Do not use behavior labels such as `writing`, `walking`, `focused`, `distracted`, `bored`, or `engaged` in the first dataset version.

## Bounding Box Rules

- Draw a tight box around each visible person.
- Label the teacher or lecturer as `instructor`.
- Label each visible learner as `student`.
- Give every visible student a separate bounding box.
- Remove or mark blurry, unclear, or useless frames for review.
- Every image must have a matching `.txt` YOLO label file.
