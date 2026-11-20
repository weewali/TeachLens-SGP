"""Link a lesson's student-view and instructor-view runs by real time.

Reads two saved runs (found by run ID; no video or model) and writes
cross_camera.json in the lesson's folder, outputs/lessons/<stamp>_<name>/
(scripts/analyze_lesson.py runs this step automatically): students who left
the student view and appeared at the front (with time at the board), and the
teacher walking into the student view. Run both cameras with position,
entry_exit, and teacher_presence on (all three are List A defaults).

Examples (repository root, TeachLens environment):
    python scripts/link_cameras.py --student sgp_A1_student_full --instructor sgp_A1_instructor_full --name sgp_A1_full
    python scripts/link_cameras.py --student s_run --instructor i_run --name lesson --window 30
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import _publish_json, link_runs  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.output_layout import find_lesson, find_run, lesson_paths, new_folder, run_paths  # noqa: E402
from src.utils.paths import project_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True, help="Student-view run ID")
    parser.add_argument("--instructor", required=True, help="Instructor-view run ID")
    parser.add_argument("--name", required=True, help="Lesson name (its folder under outputs/lessons/)")
    parser.add_argument("--window", type=float, default=20.0,
                        help="Seconds after leaving one view to look for the person in the other")
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", type=Path, default=project_path("configs/sgp_room.yaml"),
                        help="Config with the instructor camera's zone file (board target)")
    args = parser.parse_args()
    folders = {view: find_run(args.output_dir, run)
               for view, run in (("student", args.student), ("instructor", args.instructor))}
    missing = [view for view, folder in folders.items() if folder is None]
    if missing:
        print(f"No saved {' or '.join(missing)} run under {args.output_dir}", file=sys.stderr)
        return 1
    config = load_config(args.config, args.local_config)
    report = {"lesson": args.name, "student_run": args.student, "instructor_run": args.instructor,
              **link_runs(run_paths(folders["student"])["events"], run_paths(folders["instructor"])["events"],
                          config, args.window)}
    folder = find_lesson(args.output_dir, args.name) or new_folder(args.output_dir, "lessons", args.name)
    out = lesson_paths(folder)["cross_camera"]
    _publish_json(out, report, sort_keys=False)
    for link in report["links"]:
        if link["link"] == "student_to_front":
            others = ", ".join(f"#{c['track_id']} +{c['seconds_after_leaving']}s board {c['at_board_seconds']}s"
                               for c in link["instructor_view_tracks"])
            print(f"student #{link['student_view_track']} left the student view at "
                  f"{link['left_student_view'][11:19]} -> instructor view {others}")
        else:
            others = ", ".join(f"#{c['track_id']} +{c['seconds_after_leaving']}s for {c['seconds_in_view']}s"
                               for c in link["student_view_tracks"])
            print(f"teacher left the instructor view at {link['left_instructor_view'][11:19]} -> "
                  f"student view {others}")
    print(f"{len(report['links'])} links; report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
