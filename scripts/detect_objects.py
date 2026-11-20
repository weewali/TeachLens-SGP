"""Detect classroom objects in one video with the COCO YOLO26 detector (no poses).

Runs on its own, separate from the pose pipeline (scripts/analyze.py), on the
same sampled frames (video.sample_every_seconds). Writes objects.jsonl,
objects_summary.json, objects_run.json, and the review video
objects_review.mp4 (boxes drawn) to outputs/runs/<YYYY-MM-DD_HHMM>_<id>/.

Examples (repository root, TeachLens environment):
    python scripts/detect_objects.py --video data/raw_student_videos/sgp_A2_student.MOV --video-id a2_student_objects --camera student_view
    python scripts/detect_objects.py --video data/raw_instructor_videos/sgp_A2_instructor.MOV --video-id a2_instructor_laptop --camera instructor_view --classes laptop "cell phone" --start 2:00 --end 3:00
    python scripts/detect_objects.py --list-classes
"""

import argparse
import logging
from pathlib import Path
import sys

# Allow direct `python scripts/detect_objects.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import parse_real_time, parse_seconds  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="Local classroom recording")
    parser.add_argument("--video-id", help="Anonymous recording ID")
    parser.add_argument("--camera", choices=("student_view", "instructor_view"))
    parser.add_argument("--classes", nargs="+", metavar="NAME",
                        help='COCO classes to keep (quote names with spaces: "cell phone"); '
                             "default: objects.classes in the config")
    parser.add_argument("--list-classes", action="store_true",
                        help="Print every class the detector knows and stop")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", type=Path, help="Uncommitted YAML override")
    parser.add_argument("--weight", type=Path,
                        help="Local YOLO26 detection .pt (default: objects.detector); never auto-downloaded")
    parser.add_argument("--output-dir", type=Path, help="Root for events/, reports/ and debug/")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--overwrite", action="store_true", help="Replace output for this video ID")
    parser.add_argument("--start", type=parse_seconds, default=0.0,
                        help="Analyze from this time (seconds or [HH:]MM:SS); default: video start")
    parser.add_argument("--end", type=parse_seconds,
                        help="Stop before this time (seconds or [HH:]MM:SS); default: video end")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None,
                        help="Write debug/<id>.objects.mp4; default: review.save_video")
    parser.add_argument("--recording-start", type=parse_real_time,
                        help="Real start of the first frame (ISO 8601) when the file's tag is wrong")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config, args.local_config)
        weight = (args.weight.resolve() if args.weight
                  else project_path(config.get("objects", {}).get("detector", "models/detection/yolo26n.pt")))
        if args.list_classes:
            from src.inference.objects import ObjectDetector
            detector = ObjectDetector(weight, [], 0.25)
            print("\n".join(sorted(detector.names.values())))
            return 0
        if not (args.video and args.video_id and args.camera):
            parser.error("--video, --video-id and --camera are required (or use --list-classes)")
        from src.pipeline import detect_objects
        events, summary, video = detect_objects(
            args.video.resolve(), args.video_id, args.camera, config, weight=weight,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            device=args.device, overwrite=args.overwrite, save_video=args.save_video,
            start=args.start, end=args.end, labels=args.classes, real_start=args.recording_start)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError, RuntimeError) as exc:
        logging.error("Object detection failed: %s", exc)
        return 1
    print(f"Events: {events}\nSummary: {summary}")
    if video is not None:
        print(f"Annotated video: {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
