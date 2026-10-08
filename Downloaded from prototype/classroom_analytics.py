import os
import cv2
import math
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Path Configuration
BASE_DIR = "classroom_project"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_frames")
YOLO_OUT_DIR = os.path.join(BASE_DIR, "yolo_outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

def setup_directories():
    """Creates directories for inputs, processing frames, and outputs."""
    for folder in [BASE_DIR, VIDEO_DIR, FRAME_DIR, SAMPLE_DIR, YOLO_OUT_DIR, REPORT_DIR]:
        os.makedirs(folder, exist_ok=True)
    print("Directories initialized.")

def extract_frames(video_path, output_dir, every_n_seconds=2):
    """Extracts one frame every N seconds from the target video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Could not read FPS from video at {video_path}")

    frame_interval = int(fps * every_n_seconds)
    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")
    return saved

def run_yolo_predictions(frame_dir, output_project_dir):
    """Runs YOLOv8 model inference over extracted frames and saves visualized output."""
    model = YOLO("yolov8n.pt")
    results = model.predict(
        source=frame_dir,
        save=True,
        project=output_project_dir,
        name="run1",
        conf=0.25,
        imgsz=640,
        device="cpu" # Change to "cuda" or 0 if GPU is available in your Codespace
    )
    return model

def count_people(model, frame_paths):
    """Calculates person counts per frame and saves to a CSV report."""
    rows = []
    for path in frame_paths:
        result = model(path, device="cpu", verbose=False)[0]
        person_count = 0
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for c in classes:
                if c == 0:
                    person_count += 1
        rows.append({"frame": os.path.basename(path), "person_count": person_count})
    
    df_people = pd.DataFrame(rows)
    people_csv = os.path.join(REPORT_DIR, "people_counts.csv")
    df_people.to_csv(people_csv, index=False)
    print(f"People counts saved to {people_csv}")
    return df_people

def generate_and_save_report(df_people):
    """Analyzes statistics and writes a summary text feedback report."""
    avg_people = df_people["person_count"].mean()
    max_people = df_people["person_count"].max()

    feedback_lines = [
        f"Average detected people per analyzed frame: {avg_people:.2f}.",
        f"Maximum detected people in a frame: {int(max_people)}."
    ]

    if avg_people >= 10:
        feedback_lines.append("The classroom appears to have a relatively high number of visible participants.")
    else:
        feedback_lines.append("The classroom appears to have a relatively small or partially visible group of participants.")

    feedback_lines.append("This prototype currently focuses on visual classroom analysis only.")
    feedback_lines.append("A future stage may include behavior classification and stronger teacher-student correlation logic.")

    report_path = os.path.join(REPORT_DIR, "final_feedback_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classroom Analytics Prototype Report\n")
        f.write("=" * 40 + "\n\n")
        for line in feedback_lines:
            f.write("- " + line + "\n")
    print(f"Final report successfully written to {report_path}")

if __name__ == "__main__":
    # Setup folders
    setup_directories()

    # Check for input video file
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    if not video_files:
        print(f"Please place your input video (e.g. Video10min.mp4) in the '{VIDEO_DIR}' directory and run again.")
    else:
        video_path = os.path.join(VIDEO_DIR, video_files[0])
        print(f"Using video: {video_path}")

        # Execute pipeline
        extract_frames(video_path, FRAME_DIR, every_n_seconds=2)
        frame_files = sorted([os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
        
        model = run_yolo_predictions(FRAME_DIR, YOLO_OUT_DIR)
        df_people = count_people(model, frame_files)
        generate_and_save_report(df_people)
