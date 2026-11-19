"""Print one List A indicator from a saved run as a readable table.

Reads outputs/events/<video_id>.list_a.jsonl, or <video_id>.objects.jsonl
from scripts/detect_objects.py (no video or model needed), and
lists that indicator's events in time order, optionally filtered by track,
role, value, coverage, and a time range. Interval indicators also get a total
per value at the end.

Examples (repository root, TeachLens environment):
    python scripts/show_indicator.py sgp_A1_student --list
    python scripts/show_indicator.py sgp_A1_student sitting_standing --value standing
    python scripts/show_indicator.py sgp_A1_instructor teacher_presence_sample --from 1:00 --to 1:30
    python scripts/show_indicator.py sgp_A1_student position --track 3 --limit 20
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

# Allow direct `python scripts/show_indicator.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import project_path  # noqa: E402


def parse_time(value: str) -> float:
    """Seconds from 90, 90.5, 1:30, or 1:02:03."""
    seconds = 0.0
    try:
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds or [HH:]MM:SS, got {value!r}") from None
    return seconds


def clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"


def compact(name: str, value: Any) -> str:
    """Short text for an event value; dicts keep only what is useful on screen."""
    if not isinstance(value, dict):
        return str(value)
    if "label" in value:
        return str(value["label"])
    if name == "position":
        cx, cy = value["box_center"]
        fx, fy = value["foot_point"]
        return f"centre ({cx:.2f},{cy:.2f}) foot ({fx:.2f},{fy:.2f}) {value['foot_point_source']}"
    if name == "visibility_occlusion":
        parts = [part for part in ("face", "upper", "lower", "full")
                 if value.get(f"{part}_landmarks_visible" if part == "face"
                              else f"{part}_body_visible")]
        if value.get("box_truncated"):
            parts.append("cut-off")
        if value.get("possible_occlusion"):
            parts.append("occluded?")
        return f"visible: {','.join(parts) or 'none'} overlap {value.get('maximum_box_overlap_fraction')}"
    if name == "motion_speed":
        return f"{value['body_heights_per_second']} bh/s"
    if name == "shoulder_rise":
        return f"{value['shoulder_widths_above_seated_level']:+.2f} widths"
    if name == "persistent_anonymous_identity":
        return f"{value['samples']} samples"
    if name == "facial_scores":
        base = value.get("baseline")
        return (f"smile {value['smile']:.2f} brow down {value['brow_down']:.2f} up {value['brow_up']:.2f} "
                f"blink {value['eye_blink']:.2f} jaw {value['jaw_open']:.2f} "
                f"yaw {value['yaw']:+.0f} pitch {value['pitch']:+.0f} {value['face_width_px']:.0f}px"
                + (f" | usual smile {base['smile']:.2f} blink {base['eye_blink']:.2f} pitch {base['pitch']:+.0f}"
                   if base else " | learning"))
    return ", ".join(f"{key}={item}" for key, item in value.items()
                     if not isinstance(item, (dict, list)))


def select(events: list[dict[str, Any]], name: str, *, track: int | None = None,
           role: str | None = None, value: str | None = None, coverage: str | None = None,
           start: float | None = None, end: float | None = None) -> list[dict[str, Any]]:
    """Events of one indicator, filtered and sorted by start time.

    The time range keeps events that overlap it, so a long interval that
    began before --from still shows.
    """
    chosen = [e for e in events if e["indicator_name"] == name
              and (track is None or e["track_id"] == track)
              and (role is None or e["role"] == role)
              and (value is None or compact(name, e["value"]) == value)
              and (coverage is None or e["coverage"] == coverage)
              and (start is None or e["end_time"] >= start)
              and (end is None or e["start_time"] <= end)]
    return sorted(chosen, key=lambda e: (e["start_time"], e["track_id"] or 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_id")
    parser.add_argument("indicator", nargs="?", help="indicator_name, e.g. sitting_standing")
    parser.add_argument("--list", action="store_true", help="Count events per indicator and stop")
    parser.add_argument("--track", type=int, help="Only this tracker ID")
    parser.add_argument("--role", choices=("teacher", "student", "unknown", "classroom"))
    parser.add_argument("--value", help="Only this value, e.g. standing, visible, 6")
    parser.add_argument("--coverage", choices=("adequate", "limited", "unknown"))
    parser.add_argument("--from", dest="start", type=parse_time, help="Video time, e.g. 1:30")
    parser.add_argument("--to", dest="end", type=parse_time, help="Video time, e.g. 2:00")
    parser.add_argument("--limit", type=int, help="Show at most this many rows")
    parser.add_argument("--output-dir", type=Path, default=project_path("outputs"))
    args = parser.parse_args()

    path = args.output_dir / "events" / f"{args.video_id}.list_a.jsonl"
    if not path.is_file():  # an object-detection run (scripts/detect_objects.py)
        path = path.with_name(f"{args.video_id}.objects.jsonl")
    if not path.is_file():
        print(f"No events file: {path.with_name(f'{args.video_id}.list_a.jsonl')} "
              "or .objects.jsonl", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    if args.list or not args.indicator:
        for name, count in sorted(Counter(e["indicator_name"] for e in events).items()):
            print(f"{name:<32} {count:>6}")
        return 0

    rows = select(events, args.indicator, track=args.track, role=args.role, value=args.value,
                  coverage=args.coverage, start=args.start, end=args.end)
    print(f"{'video time':<17} {'real time':<9} {'track':>5} {'role':<9} {'coverage':<9} value")
    for event in rows[:args.limit]:
        span = clock(event["start_time"])
        if event["duration"]:
            span += f"-{clock(event['end_time'])}"
        real = (event.get("start_real_time") or "")[11:19]
        track = "" if event["track_id"] is None else event["track_id"]
        print(f"{span:<17} {real:<9} {track!s:>5} {event['role']:<9} {event['coverage']:<9} "
              f"{compact(args.indicator, event['value'])}")
    print(f"{len(rows)} events" + (f" (showing {args.limit})" if args.limit and len(rows) > args.limit else ""))
    totals: dict[str, float] = defaultdict(float)
    for event in rows:
        if event["duration"] and not isinstance(event["value"], dict):  # labels, not episodes
            totals[str(event["value"])] += event["duration"]
    if totals:
        print("seconds per value:", ", ".join(f"{key} {seconds:.1f}" for key, seconds in sorted(totals.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
