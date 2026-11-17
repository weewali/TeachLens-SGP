"""Link a lesson's student-view and instructor-view runs by real time.

Reads two saved runs (outputs/events/<id>.list_a.jsonl; no video or model)
and writes outputs/reports/<name>.cross_camera.json: students who left the
student view and appeared at the front (with time at the board), and the
teacher walking into the student view. Run both cameras with position,
entry_exit, and teacher_presence on (all three are List A defaults).

Examples (repository root, TeachLens environment):
    python scripts/link_cameras.py --student sgp_A1_student --instructor sgp_A1_instructor --name sgp_A1
    python scripts/link_cameras.py --student s_run --instructor i_run --name lesson --window 30
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.cross_camera import link_cameras, read_events  # noqa: E402
from src.utils.config import load_config, load_targets  # noqa: E402
from src.utils.paths import project_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True, help="Student-view run ID")
    parser.add_argument("--instructor", required=True, help="Instructor-view run ID")
    parser.add_argument("--name", required=True, help="Name for the report file")
    parser.add_argument("--window", type=float, default=20.0,
                        help="Seconds after leaving one view to look for the person in the other")
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    parser.add_argument("--config", type=Path, default=project_path("configs/default.yaml"))
    parser.add_argument("--local-config", type=Path, default=project_path("configs/sgp_room.yaml"),
                        help="Config with the instructor camera's zone file (board target)")
    args = parser.parse_args()
    paths = {view: args.output_dir / "events" / f"{run}.list_a.jsonl"
             for view, run in (("student", args.student), ("instructor", args.instructor))}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print(f"Missing runs: {', '.join(missing)}", file=sys.stderr)
        return 1
    config = load_config(args.config, args.local_config)
    board = load_targets(config["cameras"]["instructor_view"].get("zones"), "instructor_view").get("board")
    report = link_cameras(read_events(paths["student"]), read_events(paths["instructor"]),
                          window_seconds=args.window, board=board,
                          sample_period=float(config["video"]["sample_every_seconds"]))
    report.update({"student_run": args.student, "instructor_run": args.instructor})
    out = args.output_dir / "reports" / f"{args.name}.cross_camera.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
