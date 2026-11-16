"""Run List A analysis over the SGP scenario recordings listed in a manifest.

Each manifest row is one camera recording of one scripted scenario
(docs/test_recording_scenarios.pdf). Rows run one after another with the same
settings, so a whole test set can be re-run after a rule or threshold change.
By default only each row's synced window (sync_start_seconds and
sync_end_seconds: the stretch both cameras recorded, from their real
recording times) is analyzed; --full analyzes whole files. Each row's
recording_start (the real time of the file's first frame, corrected in the
manifest where a trimmed file's tag is wrong) is used for the events' real
times. Instructor recordings get the teacher hip path in their review video.
--indicators turns on only the named indicators, so the JSONL and the review
video show just them; the run IDs then end with those names.

Examples (repository root, TeachLens environment):
    python scripts/run_scenarios.py --dry-run
    python scripts/run_scenarios.py --scenario A1 --indicators sitting_standing
    python scripts/run_scenarios.py --scenario A1 --scenario A6 --tag base
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

# Allow direct `python scripts/run_scenarios.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.recording_time import parse_timestamp  # noqa: E402
from src.utils.config import LIST_A_INDICATORS  # noqa: E402
from src.utils.paths import project_path  # noqa: E402

LOGGER = logging.getLogger("run_scenarios")
DEFAULT_MANIFEST = project_path("data/manifests/sgp_scenario_videos.csv")
CAMERAS = ("student_view", "instructor_view")


@dataclass(frozen=True)
class Run:
    """One planned analysis: output video ID, source file, and camera."""

    video_id: str
    video: Path
    camera: str
    scenario: str
    start: float = 0.0
    end: float | None = None
    real_start: datetime | None = None  # manifest recording_start; None = read the file's tag

    @property
    def path(self) -> str:
        """Overlay trail for the review video: the teacher's path on the instructor camera."""
        return "teacher" if self.camera == "instructor_view" else "off"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for number, row in enumerate(rows, start=2):
        missing = [key for key in ("video_id", "scenario", "camera", "file") if not row.get(key)]
        if missing:
            raise ValueError(f"{path} line {number}: missing {', '.join(missing)}")
        if row["camera"] not in CAMERAS:
            raise ValueError(f"{path} line {number}: unknown camera {row['camera']!r}")
    return rows


def plan_runs(rows: list[dict[str, str]], data_root: Path, *,
              scenarios: list[str] | None = None, cameras: list[str] | None = None,
              tag: str | None = None, synced: bool = True) -> list[Run]:
    """Select manifest rows and resolve their files under data_root.

    Rows whose scenario is "unknown" (unlabelled clips) run only when asked
    for by name. A tag is appended to every video ID so runs with different
    settings never overwrite each other. With synced, a row's
    sync_start_seconds/sync_end_seconds limit the analyzed range (an empty
    value means the file's start or end).
    """
    wanted = {s.upper() for s in scenarios} if scenarios else None
    runs = []
    for row in rows:
        scenario = row["scenario"]
        if wanted is None:
            if scenario.lower() == "unknown":
                continue
        elif scenario.upper() not in wanted:
            continue
        if cameras and row["camera"] not in cameras:
            continue
        video_id = f"{row['video_id']}_{tag}" if tag else row["video_id"]
        file = Path(row["file"])
        start = float(row.get("sync_start_seconds") or 0) if synced else 0.0
        end = row.get("sync_end_seconds") if synced else None
        real_start = (parse_timestamp(row["recording_start"], naive_is_utc=False)
                      if row.get("recording_start") else None)
        runs.append(Run(video_id, file if file.is_absolute() else data_root / file,
                        row["camera"], scenario, start, float(end) if end else None, real_start))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=project_path("."),
                        help="Folder that the manifest's relative file paths start from "
                             "(default: repository root)")
    parser.add_argument("--scenario", action="append",
                        help="Scenario to run (repeatable, e.g. A1); default: all labelled rows")
    parser.add_argument("--camera", action="append", choices=CAMERAS,
                        help="Camera to run (repeatable); default: both")
    parser.add_argument("--tag", help="Suffix for every video ID, e.g. base or rise060")
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", type=Path, help="Uncommitted YAML override")
    parser.add_argument("--weight", type=Path, help="Local YOLO26 pose .pt; never auto-downloaded")
    parser.add_argument("--output-dir", type=Path, help="Root for events/, reports/ and debug/")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None,
                        help="Write annotated review MP4s; default: review.save_video")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    existing.add_argument("--skip-existing", action="store_true",
                          help="Skip runs whose summary already exists")
    parser.add_argument("--full", action="store_true",
                        help="Analyze whole files instead of each row's synced window")
    parser.add_argument("--indicators", nargs="+", choices=LIST_A_INDICATORS, metavar="NAME",
                        help="Turn on only these List A indicators (default: configs/indicators.yaml); "
                             f"choices: {', '.join(LIST_A_INDICATORS)}")
    parser.add_argument("--dry-run", action="store_true", help="List the planned runs and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Name an indicator test after its indicators unless a tag is given.
    tag = args.tag or ("_".join(args.indicators) if args.indicators else None)

    try:
        runs = plan_runs(read_manifest(args.manifest), args.data_root.resolve(),
                         scenarios=args.scenario, cameras=args.camera, tag=tag,
                         synced=not args.full)
    except (OSError, ValueError) as exc:
        LOGGER.error("Cannot read manifest: %s", exc)
        return 1
    if not runs:
        LOGGER.error("No manifest rows match the selection")
        return 1
    missing = [run.video for run in runs if not run.video.is_file()]
    for run in runs:
        window = f"{run.start:g}-{'end' if run.end is None else f'{run.end:g}'} s"
        begins = run.real_start.strftime("%H:%M:%S.%f")[:11] if run.real_start else "file tag"
        print(f"{run.scenario:<5} {run.camera:<16} {run.video_id:<44} {window:<16} "
              f"starts {begins}  {run.video.name}")
    if missing:
        LOGGER.error("Missing video files (check --data-root): %s",
                     ", ".join(str(path) for path in missing))
        return 1
    if args.dry_run:
        return 0

    # Heavy imports only for a real run, so --dry-run works without Ultralytics.
    from src.pipeline import AnalysisStopped, analyze_video
    from src.utils.config import load_config

    config = load_config(args.config, args.local_config)
    output_root = (args.output_dir.resolve() if args.output_dir
                   else project_path(config["paths"]["outputs"]))
    failed = []
    for number, run in enumerate(runs, start=1):
        summary = output_root / "reports" / f"{run.video_id}.list_a.json"
        if args.skip_existing and summary.exists():
            LOGGER.info("[%d/%d] %s: summary exists, skipped", number, len(runs), run.video_id)
            continue
        LOGGER.info("[%d/%d] %s (%s)", number, len(runs), run.video_id, run.camera)
        started = time.perf_counter()
        try:
            analyze_video(run.video, run.video_id, run.camera, config,
                          weight=args.weight.resolve() if args.weight else None,
                          output_dir=output_root, device=args.device,
                          overwrite=args.overwrite, save_video=args.save_video,
                          live_preview=False, start=run.start, end=run.end, path=run.path,
                          indicators=args.indicators, real_start=run.real_start)
        except (AnalysisStopped, FileNotFoundError, FileExistsError,
                ValueError, RuntimeError) as exc:
            LOGGER.error("%s failed: %s", run.video_id, exc)
            failed.append(run.video_id)
            continue
        LOGGER.info("%s done in %.0f s", run.video_id, time.perf_counter() - started)
    if failed:
        LOGGER.error("%d of %d runs failed: %s", len(failed), len(runs), ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
