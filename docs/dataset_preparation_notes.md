# Dataset Preparation Notes

## Dataset Location

The current one-class person experiment is stored at:

```text
data/person_detection_v2/
```

## Expected Layout

```text
data/person_detection_v2/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image should have a matching `.txt` label file with the same base filename.

## Data Split

Use `images/train` and `labels/train` for training data. Use `images/val` and `labels/val` for validation data.
Split by source recording before training; frames from one recording must not
appear in both train and validation. The current `person_detection_v2` splits
contain the same source frames, so its validation results are not usable.

For role detection, use `role_data.yaml` with the imported mapping
`0: student`, `1: instructor` only after its malformed annotation and split
overlap have been repaired in a new dataset version.
