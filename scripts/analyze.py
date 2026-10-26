"""CLI for YOLO26 pose-based List A physical classroom indicators."""

import argparse
import logging
import math
from pathlib import Path
import sys

# Allow direct `python scripts/analyze.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import AnalysisStopped, analyze_video  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_path  # noqa: E402


def parse_seconds(value: str) -> float:
    """Accept 90, 90.5, 1:30, or 1:02:03.5 as seconds from the video start."""
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError:
        parts = []
    if (not 1 <= len(parts) <= 3 or any(not math.isfinite(p) or p < 0 for p in parts)
            or any(p >= 60 for p in parts[1:])):
        raise argparse.ArgumentTypeError(f"expected seconds or [HH:]MM:SS, got {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Local classroom recording")
    parser.add_argument("--video-id", required=True, help="Anonymous recording ID")
    parser.add_argument("--camera", required=True, choices=("student_view", "instructor_view"))
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", type=Path, help="Uncommitted YAML override")
    parser.add_argument("--weight", type=Path, help="Local YOLO26 pose .pt; never auto-downloaded")
    parser.add_argument("--output-dir", type=Path, help="Root for events/, reports/ and debug/")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--overwrite", action="store_true", help="Replace output for this video ID")
    parser.add_argument("--start", type=parse_seconds, default=0.0,
                        help="Analyze from this time (seconds or [HH:]MM:SS); default: video start")
    parser.add_argument("--end", type=parse_seconds,
                        help="Stop before this time (seconds or [HH:]MM:SS); default: video end")
    parser.add_argument("--live", action=argparse.BooleanOptionalAction, default=None,
                        help="Show annotated frames while analyzing (space pause, n step, "
                             "q/Esc stop without outputs); default: review.live_preview")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None,
                        help="Write the annotated debug/<id>.list_a.mp4; default: review.save_video")
    parser.add_argument("--path", choices=("off", "teacher", "all"),
                        help="Draw foot-point trails in the review video; default: review.path")
    parser.add_argument("--debug", action="store_true", help="Log per-sampled-frame counts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config, args.local_config)
        events, summary, video = analyze_video(
            args.video.resolve(), args.video_id, args.camera, config,
            weight=args.weight.resolve() if args.weight else None,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            device=args.device, overwrite=args.overwrite,
            save_video=args.save_video, live_preview=args.live,
            start=args.start, end=args.end, path=args.path)
    except AnalysisStopped as exc:
        logging.warning("%s", exc)
        return 1
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        logging.error("Analysis failed: %s", exc)
        return 1
    print(f"Events: {events}\nSummary: {summary}")
    if video is not None:
        print(f"Annotated video: {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
