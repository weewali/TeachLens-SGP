# Model Training Notes

The first model will use a YOLO-based object detector to identify:

```text
0: instructor
1: student
```

Use the root `data.yaml` file for dataset paths and class names.

Training commands, metrics, and experiment notes can be added here once the dataset is ready.

Important note: the training photos we are using have students facing the camera, while the teacher is sitting and facing away from the camera. So when testing the model, start with a similar video first: students facing the camera and instructor sitting/facing away. After that, test on different classroom angles to see how well the model generalizes.

---

## 1. Create or activate the conda environment

Download Anaconda if you haven't.

Open Anaconda Prompt or PowerShell.

Create a new environment:

```bash
conda create -n TeachLens python=3.10 -y
```

Activate it:

```bash
conda activate TeachLens
```

Install YOLO:

```bash
pip install ultralytics opencv-python
```

Check that YOLO installed correctly:

```bash
yolo checks
```

---

## 2. Prepare the project folder

The dataset should be organized like this:

```text
TeachLens-SGP/
│
├── data/
│   └── dataset_v1/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       │
│       └── labels/
│           ├── train/
│           └── val/
│
├── data.yaml
│
└── runs/
```

The `images` folders contain the actual photos.

The `labels` folders contain the YOLO annotation `.txt` files.

Each image must have a matching label file with the exact same name.

Example:

```text
data/dataset_v1/images/train/frame001.jpg
data/dataset_v1/labels/train/frame001.txt
```

Another example:

```text
data/dataset_v1/images/val/frame150.jpg
data/dataset_v1/labels/val/frame150.txt
```

---

## 3. Split the images into train and val

Use most images for training and some for validation.

Example:

```text
150 images → train
30 images  → val
```

Do not put all images in train only. YOLO needs validation data to check how well the model is learning.

Both train and val images must be annotated.

---

## 4. Annotate the images

Use LabelImg or another annotation tool.

Make sure the annotation format is YOLO format, not PascalVOC/XML.

The classes must be:

```text
instructor
student
```

The class order matters:

```text
0 = instructor
1 = student
```

A YOLO label file looks like this:

```text
0 0.512 0.438 0.120 0.300
1 0.710 0.500 0.090 0.250
```

The format is:

```text
class_id x_center y_center width height
```

All box values must be normalized between 0 and 1.

Do not write class names inside the label files. Use class numbers only.

Correct:

```text
0 0.512 0.438 0.120 0.300
```

Wrong:

```text
instructor 0.512 0.438 0.120 0.300
```

Wrong:

```text
0 512 438 120 300
```

---

## 5. Create the `data.yaml` file

Create a file named:

```text
data.yaml
```

Put it in the main project folder:

```text
TeachLens-SGP/data.yaml
```

Paste this inside:

```yaml
path: data/dataset_v1

train: images/train
val: images/val

names:
  0: instructor
  1: student
```

---

## 6. What `data.yaml` means

The `data.yaml` file tells YOLO where the dataset is and what the class names are.

This part:

```yaml
path: data/dataset_v1
```

means the main dataset folder is:

```text
data/dataset_v1
```

This part:

```yaml
train: images/train
```

means the training images are inside:

```text
data/dataset_v1/images/train
```

This part:

```yaml
val: images/val
```

means the validation images are inside:

```text
data/dataset_v1/images/val
```

This part:

```yaml
names:
  0: instructor
  1: student
```

means class `0` is instructor and class `1` is student.

The class numbers here must match the numbers used in the label `.txt` files.

---

## 7. Check the dataset before training (Optional)

From inside the project folder, run:

```powershell
cd "C:\Users\weewa\OneDrive\Desktop\TeachLens-SGP"
```

Check training images:

```powershell
Get-ChildItem .\data\dataset_v1\images\train\*.jpg | Measure-Object
```

Check validation images:

```powershell
Get-ChildItem .\data\dataset_v1\images\val\*.jpg | Measure-Object
```

Check training labels:

```powershell
Get-ChildItem .\data\dataset_v1\labels\train\*.txt | Measure-Object
```

Check validation labels:

```powershell
Get-ChildItem .\data\dataset_v1\labels\val\*.txt | Measure-Object
```

The number of label files should match the number of image files.

Example:

```text
150 train images
150 train labels

30 val images
30 val labels
```

---

## 8. Train the YOLO model

From the main project folder (in the CLI, cd to the project folder), run:

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=8 project=runs/detect name=teachlens_v1
```

Explanation:

```text
model=yolo11n.pt
```

Uses the small YOLO model. It is fast and good for testing.

```text
data=data.yaml
```

Tells YOLO to use our dataset configuration file.

```text
epochs=80
```

Trains the model for 80 rounds.

```text
imgsz=640
```

Resizes images to 640x640 during training.

```text
batch=8
```

Processes 8 images at a time. If the computer is slow or gives memory errors, reduce it to 4.

```text
project=runs/detect
```

Saves the training results inside `runs/detect`.

```text
name=teachlens_v1
```

Names this training run `teachlens_v1`.

After training, the best model will be saved here:

```text
runs/detect/teachlens_v1/weights/best.pt
```

---

## 9. Validate the model

Run:

```bash
yolo val model=runs/detect/teachlens_v1/weights/best.pt data=data.yaml
```

This checks how well the model performs on the validation images.

Important metrics:

```text
precision
recall
mAP50
mAP50-95
```

For the first test model, do not expect perfect results. The goal is to check whether the pipeline works.

---

## 10. Test the model on unseen photos

Create a folder for unseen test images:

```text
data/test_images/
```

Put photos there that were not used in training or validation.

Then run:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/test_images conf=0.25 save=True
```

The output images will be saved in:

```text
runs/detect/predict
```

For debugging, use lower confidence:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/test_images conf=0.01 save=True
```

Use `conf=0.01` only for debugging. It shows even weak detections.

For normal testing, use:

```bash
conf=0.25
```

or:

```bash
conf=0.40
```

---

## 11. Test the model on a similar video

Because the training images show:

```text
students facing the camera
teacher sitting and facing away from the camera
```

Start testing on a similar classroom video.

Use a video where:

```text
- students are facing the camera
- instructor is sitting or facing away
- classroom angle is similar to the training images
- lighting is similar
```

Run:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/raw_videos/test_video.mp4 conf=0.25 save=True
```

The output video will be saved inside:

```text
runs/detect/predict
```

---

## 12. Test the model with tracking on video

Prediction detects objects frame by frame.

Tracking detects objects and tries to keep the same ID for each person across frames.

Run:

```bash
yolo track model=runs/detect/teachlens_v1/weights/best.pt source=data/raw_videos/test_video.mp4 conf=0.25 save=True tracker=bytetrack.yaml
```

The output will be saved inside:

```text
runs/track
```

Use this when you want to visualize students/instructors being followed across the video.

---

## 13. Common problems

### Problem: YOLO detects nothing

Try:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/dataset_v1/images/train conf=0.01 save=True
```

If it still detects nothing on training images, check:

```text
- Are the labels in YOLO .txt format?
- Are the label files empty?
- Do image names and label names match?
- Is data.yaml pointing to the correct dataset?
- Are the class IDs correct?
- Are you using best.pt and not the original yolo11n.pt?
```

---

### Problem: label files are missing

Each image needs a matching label file.

Example:

```text
frame001.jpg
frame001.txt
```

If the label file does not exist, YOLO treats that image as having no objects.

---

### Problem: using the wrong model

Correct:

```bash
model=runs/detect/teachlens_v1/weights/best.pt
```

Wrong after training:

```bash
model=yolo11n.pt
```

`yolo11n.pt` is the original pretrained model. `best.pt` is our trained model.

---

### Problem: model only works on the same video

That means the dataset is too limited.

To improve it, add more training images with:

```text
- different classrooms
- different lighting
- different camera angles
- standing instructors
- sitting instructors
- students facing different directions
- crowded and less crowded rooms
```

The current model is only a test model. It is not enough for the final senior project system.

---

## 14. Final useful commands

Train:

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=8 project=runs/detect name=teachlens_v1
```

Validate:

```bash
yolo val model=runs/detect/teachlens_v1/weights/best.pt data=data.yaml
```

Predict on images:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/test_images conf=0.25 save=True
```

Predict on video:

```bash
yolo predict model=runs/detect/teachlens_v1/weights/best.pt source=data/raw_videos/test_video.mp4 conf=0.25 save=True
```

Track on video:

```bash
yolo track model=runs/detect/teachlens_v1/weights/best.pt source=data/raw_videos/test_video.mp4 conf=0.25 save=True tracker=bytetrack.yaml
```
