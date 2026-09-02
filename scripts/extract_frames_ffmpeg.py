# How to use
# python scripts/extract_frames_ffmpeg.py --input data/raw_videos/instructor_video01.mp4 --output data/raw_frames --prefix instructor_video01 --every 2


import argparse
import subprocess
from pathlib import Path

def extract_frames(input_video, output_folder, prefix, everyseconds=2):
    """
    Extracts frames from a video using FFmpeg

    Args:
        input_video (str): Path to the input video
        output_video (str): Folder where frames will be saved
        prefix (str): Naming prefix for extracted frames
        every_seconds: Extract one frame every N seconds 
    """

    input_video = Path(input_video)
    output_folder = Path(output_folder)

    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")
    
    output_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = output_folder / f"{prefix}_frame%04d.jpg"

    fps_value = f"1/{everyseconds}"

    command = [
        "ffmpeg",
        "-i", str(input_video),
        "-vf", f"fps={fps_value}",
        "-q:v", "2",
        str(output_pattern)
    ]

    print("Running FFmpeg command:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print(f"Frames extracted successfully to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video using FFmpeg")

    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/raw_frames", help="Output folder for frames")
    parser.add_argument("--prefix", required=True, help="Prefix for frame names")
    parser.add_argument("--every", type=int, default=2, help="Extract one frame every N seconds")

    args = parser.parse_args()

    extract_frames(
        input_video = args.input,
        output_folder = args.output,
        prefix = args.prefix,
        everyseconds = args.every
    )