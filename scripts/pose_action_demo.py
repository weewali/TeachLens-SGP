import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO pose keypoint indexes
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10


def detect_hand_raised(keypoints, confidence, box, min_conf=0.25):
    """
    Determine whether either wrist is above its corresponding shoulder.

    Returns:
        True  -> hand raised
        False -> visible hand is not raised
        None  -> required keypoints are not sufficiently visible
    """
    x1, y1, x2, y2 = box

    person_height = max(y2 - y1, 1)
    vertical_margin = 0.05 * person_height

    body_sides = [
        (LEFT_SHOULDER, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_WRIST),
    ]

    found_visible_side = False

    for shoulder_index, wrist_index in body_sides:
        shoulder_conf = confidence[shoulder_index]
        wrist_conf = confidence[wrist_index]

        if shoulder_conf < min_conf or wrist_conf < min_conf:
            continue

        found_visible_side = True

        shoulder_y = keypoints[shoulder_index][1]
        wrist_y = keypoints[wrist_index][1]

        # Image y-coordinates get smaller when moving upward.
        if wrist_y < shoulder_y - vertical_margin:
            return True

    if found_visible_side:
        return False

    return None


class ActionSmoother:
    """
    Prevent a single incorrect frame from triggering an action.
    """

    def __init__(self, window_size=15, min_valid_frames=8):
        self.history = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
        self.min_valid_frames = min_valid_frames

    def update(self, track_id, action_name, frame_result):
        action_history = self.history[track_id][action_name]
        action_history.append(frame_result)

        # Ignore frames where the necessary keypoints were not visible.
        valid_results = [
            result for result in action_history
            if result is not None
        ]

        if len(valid_results) < self.min_valid_frames:
            return False

        positive_ratio = sum(valid_results) / len(valid_results)

        # The rule must be true in at least 70% of valid frames.
        return positive_ratio >= 0.70


def analyze_video(input_path, output_path, model_path):
    model = YOLO(model_path)

    video = cv2.VideoCapture(str(input_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Approximately half a second of history.
    window_size = max(10, round(fps * 0.5))
    min_valid_frames = max(5, round(window_size * 0.5))

    smoother = ActionSmoother(
        window_size=window_size,
        min_valid_frames=min_valid_frames,
    )

    while True:
        success, frame = video.read()

        if not success:
            break

        # This is where your program connects to YOLO pose and ByteTrack.
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]

        # Draw YOLO skeletons, boxes, and track IDs.
        annotated_frame = result.plot()

        has_tracking_results = (
            result.boxes is not None
            and result.boxes.id is not None
            and result.keypoints is not None
            and result.keypoints.conf is not None
        )

        if has_tracking_results:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()
            keypoint_confidences = result.keypoints.conf.cpu().numpy()

            # Boxes and keypoints use the same person ordering.
            for track_id, box, points, confidence in zip(
                track_ids,
                boxes,
                keypoints,
                keypoint_confidences,
            ):
                frame_result = detect_hand_raised(
                    keypoints=points,
                    confidence=confidence,
                    box=box,
                )

                hand_is_stably_raised = smoother.update(
                    track_id=track_id,
                    action_name="hand_raised",
                    frame_result=frame_result,
                )

                if hand_is_stably_raised:
                    x1, y1, _, _ = map(int, box)

                    cv2.putText(
                        annotated_frame,
                        f"ID {track_id}: HAND RAISED",
                        (x1, max(30, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

        writer.write(annotated_frame)

    video.release()
    writer.release()

    print(f"Finished. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO pose tracking and action rules."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        default="outputs/pose_actions.mp4",
        help="Output video path.",
    )

    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model.",
    )

    args = parser.parse_args()

    analyze_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=args.model,
    )


if __name__ == "__main__":
    main()