# Dataset Preparation Notes

## Dataset Location

The first official YOLO dataset version is stored locally at:

```text
data/dataset_v1/
```

## Expected Layout

```text
data/dataset_v1/
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
Normally, 80% of the images should be `images/train` and 20% in `images/val`