"""Review sheets: people check the pipeline's detections, and the report then says how far they can be trusted.

Moved from the pipeline_creation branch's `main code/validate_indicators.py` (2026-11-20) and rebuilt on a lesson
folder (outputs/lessons/<stamp>_<lesson>/):

make:  picks moments spread over the lesson, saves the matching frame of each camera's review video (boxes, IDs and
       labels drawn in) as an image, and writes sheet.csv: one row per detection to check (camera, time, track,
       indicator, what the pipeline said), with an empty "human" column.
score: reads the filled sheet and, per indicator, the precision of the pipeline's positive detections (the share a
       person confirmed) with a 95% Wilson interval and the overall agreement. scores.json is written next to the
       sheet, and facts.json's indicator_validation is updated (checked_against_human_labels, precision), so the AI
       report stops calling a checked detection "not verified" (or calls it unreliable below 0.7).

The images show real people: the folder stays local (outputs/ is ignored by Git).
"""
import csv
import json
import math
from pathlib import Path
from typing import Any

SHEET_COLUMNS = ["item", "image", "camera", "lesson_seconds", "video_seconds", "track", "seat", "indicator",
                 "pipeline_says", "positive", "human", "notes"]
# indicator -> (camera, values that count as a positive detection, facts.json indicator_validation key)
CHECKED = {
    "hand_raise": ("student", None, "hand_raise"),                       # None: anything but none/unknown/down
    "device_use": ("student", ("phone", "phone_likely"), "phone_use"),
    "orientation_proxy": ("student", ("on_target",), "student_on_task_state"),
    "alertness": ("student", ("dozing",), "student_on_task_state"),
    "facing_direction": ("instructor", ("toward_class",), "instructor_facing"),
    "sitting_standing": ("instructor", ("sitting",), "instructor_seated"),
    "gesture": ("instructor", ("gesturing",), "instructor_gestures"),
}
NOT_POSITIVE = ("none", "unknown", "down", None)
YES = ("y", "yes", "1", "true", "correct", "right")
NO = ("n", "no", "0", "false", "wrong", "incorrect")


def _events(lesson: Path, camera: str) -> list[dict[str, Any]]:
    path = lesson / camera / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_positive(indicator: str, value: Any) -> bool:
    values = CHECKED[indicator][1]
    return value not in NOT_POSITIVE if values is None else value in values


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def sheet_rows(lesson: Path, samples: int = 20) -> list[dict[str, Any]]:
    """The rows to check: at `samples` times spread over each camera's analysed stretch, every checked indicator's
    state for every track that had one then (positives and negatives, so both kinds of error can be seen)."""
    rows = []
    for camera in ("student", "instructor"):
        run_path = lesson / camera / "run.json"
        if not run_path.exists():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        first, period, count = float(run["first_sample_seconds"]), float(run["sample_period"]), int(run["samples"])
        picks = sorted({round(i * (count - 1) / max(1, samples - 1)) for i in range(min(samples, count))})
        events = [e for e in _events(lesson, camera) if e.get("indicator_name") in CHECKED
                  and CHECKED[e["indicator_name"]][0] == camera and e.get("value") != "unknown"]
        for index in picks:
            t = first + index * period
            for e in events:
                if float(e["start_time"]) - 1e-6 <= t <= float(e["end_time"]) + 1e-6:
                    rows.append({"camera": camera, "sample": index, "video_seconds": round(t, 2),
                                 "lesson_seconds": round(t - float(run.get("analyzed_start_seconds", first)), 2),
                                 "track": e.get("track_id"), "seat": e.get("zone") or "",
                                 "indicator": e["indicator_name"], "pipeline_says": e.get("value"),
                                 "positive": "yes" if is_positive(e["indicator_name"], e.get("value")) else "no"})
    for number, row in enumerate(rows, start=1):
        row["item"] = number
    return rows


def save_frames(lesson: Path, rows: list[dict[str, Any]], folder: Path) -> None:
    """One image per (camera, sample) from the camera's review video (frame index = sample index)."""
    import cv2
    wanted: dict[str, set[int]] = {}
    for row in rows:
        wanted.setdefault(row["camera"], set()).add(int(row["sample"]))
    names = {}
    for camera, samples in wanted.items():
        video = lesson / camera / "review.mp4"
        if not video.exists():
            continue
        capture = cv2.VideoCapture(str(video))
        try:
            for sample in sorted(samples):
                capture.set(cv2.CAP_PROP_POS_FRAMES, sample)
                ok, frame = capture.read()
                if ok:
                    name = f"{camera}_{sample:05d}.jpg"
                    cv2.imwrite(str(folder / name), frame)
                    names[(camera, sample)] = name
        finally:
            capture.release()
    for row in rows:
        row["image"] = names.get((row["camera"], int(row["sample"])), "")


def make_sheets(lesson: Path, folder: Path, samples: int = 20) -> Path:
    """Writes the images and sheet.csv into folder; returns the sheet's path. Refuses to overwrite a filled sheet."""
    folder.mkdir(parents=True, exist_ok=True)
    sheet = folder / "sheet.csv"
    if sheet.exists() and any(row.get("human") for row in read_sheet(sheet)):
        raise FileExistsError(f"{sheet} already has answers; move it away before making new sheets.")
    rows = sheet_rows(lesson, samples)
    save_frames(lesson, rows, folder)
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "human": "", "notes": ""})
    (folder / "README.txt").write_text(
        "For each row, open the image and look at the person with the given track number (shown on the box).\n"
        "In the 'human' column write yes if the pipeline is right about this indicator at that moment, no if it is\n"
        "wrong, and leave it empty if you cannot tell. Then run: python scripts/review_sheets.py score --lesson ...\n",
        encoding="utf-8")
    return sheet


def read_sheet(sheet: Path) -> list[dict[str, str]]:
    with open(sheet, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(sheet: Path) -> dict[str, Any]:
    """Per indicator: reviewed rows, agreement, and the precision of the pipeline's positive detections."""
    results: dict[str, Any] = {}
    for row in read_sheet(sheet):
        answer = (row.get("human") or "").strip().lower()
        if answer not in YES + NO:
            continue
        entry = results.setdefault(row["indicator"], {"reviewed": 0, "agreed": 0, "positives_reviewed": 0,
                                                      "positives_confirmed": 0})
        right = answer in YES
        entry["reviewed"] += 1
        entry["agreed"] += right
        if row.get("positive") == "yes":
            entry["positives_reviewed"] += 1
            entry["positives_confirmed"] += right
    for entry in results.values():
        entry["agreement"] = round(entry["agreed"] / entry["reviewed"], 3)
        n = entry["positives_reviewed"]
        entry["precision"] = round(entry["positives_confirmed"] / n, 3) if n else None
        entry["precision_interval_95"] = wilson(entry["positives_confirmed"], n)
    return results


def update_facts(facts_path: Path, results: dict[str, Any]) -> dict[str, Any]:
    """Marks the checked detections in facts.json's indicator_validation (the lowest precision wins when several
    indicators feed one entry)."""
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    validation = facts.setdefault("indicator_validation", {})
    for indicator, entry in results.items():
        key = CHECKED.get(indicator, (None, None, None))[2]
        if key is None or entry["reviewed"] == 0:
            continue
        current = validation.get(key, {})
        precision = entry["precision"] if entry["precision"] is not None else entry["agreement"]
        if current.get("checked_against_human_labels") and current.get("precision") is not None:
            precision = min(precision, current["precision"])
        validation[key] = {"checked_against_human_labels": True, "precision": precision,
                           "reviewed": entry["reviewed"] + int(current.get("reviewed", 0))}
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts
