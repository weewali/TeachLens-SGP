"""Measures how accurate the detections are, by comparing them with a person's labels.

Nothing in the pipeline has been checked against the video by a human yet, so every indicator is reported as
unverified. This tool fixes that in two steps:

  make-sheets   writes review sheets (CSV + annotated images) for a person to fill in
  score         reads the filled sheets, computes precision / recall with 95% intervals, and records the result in the
                lecture's facts.json so the report text knows which indicators have been checked

Usage (from the repository root):
    python3 "Downloaded from prototype/validate_indicators.py" make-sheets --session Video10min
    ... a person fills in classroom_project/validation/Video10min/review_events.csv and review_frames.csv ...
    python3 "Downloaded from prototype/validate_indicators.py" score --session Video10min
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

SESSIONS = os.path.join("classroom_project", "sessions")
VALIDATION = os.path.join("classroom_project", "validation")
EVENT_LABELS = ("hand_raise", "phone_use", "reading")
ROLE_COLOURS = {"instructor": (0, 0, 255), "student": (255, 100, 0)}

README = """\
How to fill in the sheets (about 20-30 minutes)

review_events.csv  - one row per detected event. Open the image named in the 'image' column.
  is_real : TRUE if the event really happened and was attributed to the right person (a hand raise by a STUDENT,
            a phone really in use), FALSE if not. Use 'note' for anything odd (e.g. "this is the instructor").

review_frames.csv  - a random sample of frames. Open the image named in the 'image' column (boxes show what the
  program thought). Fill in:
  instructor_visible    : TRUE / FALSE - is the instructor visible in the frame at all?
  instructor_box_correct: TRUE / FALSE - is the box labelled 'instructor' really the instructor? (leave blank if none)
  instructor_posture    : sitting / standing / cannot_tell
  hand_raise_seats      : the seat labels (as printed on the image) of students whose hand is raised,
                          separated by ';' (e.g. seat_03;seat_08), or 'none'

Leave a cell blank if you cannot tell. Do not edit the other columns. Label what you SEE, not what the boxes say.
"""


def wilson(successes, n, z=1.96):
    """95% Wilson interval for a proportion (works for small samples, unlike the usual +/- formula)."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 2), round(min(1.0, centre + half), 2)]


def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_session(session):
    root = os.path.join(SESSIONS, session)
    if not os.path.isdir(root):
        sys.exit(f"No analysed lecture called '{session}' under {SESSIONS}. Run the pipeline first.")
    tables = os.path.join(root, "tables")
    return (root, pd.read_csv(os.path.join(tables, "frame_detections.csv")), pd.read_csv(os.path.join(tables, "events.csv")),
            pd.read_csv(os.path.join(tables, "frames.csv")))


def annotate(image, people, highlight_seat=None):
    """Draws every boxed person with their role / seat, and a thicker yellow box around the one under review."""
    for r in people.itertuples():
        if r.role == "other":
            continue
        colour = ROLE_COLOURS[r.role]
        label = "instructor" if r.role == "instructor" else r.seat
        thick = 2
        if highlight_seat is not None and r.role == "student" and r.seat == highlight_seat:
            colour, thick = (0, 255, 255), 4
        cv2.rectangle(image, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, thick)
        cv2.putText(image, label, (int(r.x1), max(14, int(r.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return image


def posture_word(value):
    return "unknown" if pd.isna(value) else ("seated" if value == 1.0 else "standing")


def make_sheets(session, n_frames, seed):
    root, people, events, frames = load_session(session)
    out = os.path.join(VALIDATION, session)
    images = os.path.join(out, "images")
    os.makedirs(images, exist_ok=True)
    frame_of_time = lambda t: int((frames["time_s"] - t).abs().idxmin())

    rows = []
    for n, e in enumerate(events[events["label"].isin(EVENT_LABELS)].itertuples(), 1):
        index = frame_of_time(e.start_s)
        image = cv2.imread(frames.loc[index, "path"]) if os.path.exists(str(frames.loc[index, "path"])) else \
            cv2.imread(os.path.join(root, "frames", f"frame_{index:04d}.jpg"))
        name = f"event_{n:03d}_{e.label}_{e.entity_id}.jpg"
        cv2.imwrite(os.path.join(images, name), annotate(image, people[people["frame"] == index], highlight_seat=e.entity_id))
        rows.append({"event_id": n, "label": e.label, "seat": e.entity_id, "time": fmt_time(e.start_s), "start_s": e.start_s,
                     "end_s": e.end_s, "image": os.path.join("images", name), "is_real": "", "note": ""})
    pd.DataFrame(rows, columns=["event_id", "label", "seat", "time", "start_s", "end_s", "image", "is_real", "note"]) \
        .to_csv(os.path.join(out, "review_events.csv"), index=False)

    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(len(frames), size=min(n_frames, len(frames)), replace=False).tolist())
    hands = events[events["label"] == "hand_raise"]
    rows = []
    for index in sample:
        t = frames.loc[index, "time_s"]
        in_frame = people[people["frame"] == index]
        instructor = in_frame[in_frame["role"] == "instructor"]
        seats = sorted(hands[(hands["start_s"] <= t) & (t < hands["end_s"])]["entity_id"].unique())
        image = cv2.imread(os.path.join(root, "frames", f"frame_{index:04d}.jpg"))
        name = f"frame_{index:04d}.jpg"
        cv2.imwrite(os.path.join(images, name), annotate(image, in_frame))
        rows.append({"frame": index, "time": fmt_time(t), "image": os.path.join("images", name),
                     "predicted_instructor_present": len(instructor) > 0,
                     "predicted_instructor_posture": posture_word(instructor["seated"].iloc[0]) if len(instructor) else "none",
                     "predicted_hand_raise_seats": ";".join(seats) or "none",
                     "instructor_visible": "", "instructor_box_correct": "", "instructor_posture": "", "hand_raise_seats": "", "note": ""})
    pd.DataFrame(rows).to_csv(os.path.join(out, "review_frames.csv"), index=False)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README)
    print(f"Review sheets written to {out}/ ({len([r for r in rows])} frames, "
          f"{len(events[events['label'].isin(EVENT_LABELS)])} events). Read README.txt there, fill in the CSVs, then run `score`.")


def truthy(value):
    return {"true": True, "false": False}.get(str(value).strip().lower())


def proportion(successes, n):
    return {"n_reviewed": n, "precision": round(successes / n, 2) if n else None, "interval_95": wilson(successes, n)}


def score(session, patch_facts):
    out = os.path.join(VALIDATION, session)
    events = pd.read_csv(os.path.join(out, "review_events.csv"), dtype=str).fillna("")
    frames = pd.read_csv(os.path.join(out, "review_frames.csv"), dtype=str).fillna("")
    summary = {}

    for label in EVENT_LABELS:                                           # precision of each event type
        sub = events[events["label"] == label]
        verdicts = [truthy(v) for v in sub["is_real"] if truthy(v) is not None]
        if verdicts:
            summary[label] = proportion(sum(verdicts), len(verdicts))

    shown = frames[frames["predicted_instructor_present"].str.lower() == "true"]   # is the box called 'instructor' right?
    correct = [truthy(v) for v in shown["instructor_box_correct"] if truthy(v) is not None]
    visible = [(truthy(v), p.lower() == "true") for v, p in zip(frames["instructor_visible"], frames["predicted_instructor_present"]) if truthy(v) is not None]
    if correct or visible:
        entry = proportion(sum(correct), len(correct)) if correct else {}
        seen = [pred for human, pred in visible if human]                   # frames where the instructor really is visible
        if seen:
            entry["presence_recall"] = round(sum(seen) / len(seen), 2)
            entry["presence_recall_interval_95"] = wilson(sum(seen), len(seen))
            entry["n_frames_instructor_visible"] = len(seen)
        summary["instructor_role"] = entry

    posture = [(h.lower(), p) for h, p in zip(frames["instructor_posture"], frames["predicted_instructor_posture"])
               if h.lower() in ("sitting", "standing")]
    if posture:
        known = [(h, p) for h, p in posture if p in ("seated", "standing")]
        right = sum(1 for h, p in known if (h == "sitting") == (p == "seated"))
        entry = proportion(right, len(known)) if known else {}
        entry["accuracy"] = entry.pop("precision", None)
        entry["readable_share"] = round(len(known) / len(posture), 2)
        summary["instructor_seated"] = entry

    tp = fn = fp = 0                                                       # recall of hand raises, seat by seat
    for human, predicted in zip(frames["hand_raise_seats"], frames["predicted_hand_raise_seats"]):
        if human.strip() == "":
            continue
        truth = set() if human.strip().lower() == "none" else {s.strip() for s in human.split(";") if s.strip()}
        guess = set() if predicted.strip().lower() == "none" else {s.strip() for s in predicted.split(";") if s.strip()}
        tp, fn, fp = tp + len(truth & guess), fn + len(truth - guess), fp + len(guess - truth)
    if tp + fn:
        summary.setdefault("hand_raise", {})["recall"] = round(tp / (tp + fn), 2)
        summary["hand_raise"]["recall_interval_95"] = wilson(tp, tp + fn)
        summary["hand_raise"]["n_true_hand_raises_in_sample"] = tp + fn
        summary["hand_raise"]["note_recall"] = "Hand raises are rare, so a random sample holds few; read the interval."

    with open(os.path.join(out, "validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not summary:
        print("No filled-in labels found yet. Fill in the sheets first (see README.txt).")
        return
    for name, entry in summary.items():
        print(f"{name}: {entry}")

    facts_path = os.path.join(SESSIONS, session, "reports", "facts.json")
    if patch_facts and os.path.exists(facts_path):
        with open(facts_path, encoding="utf-8") as f:
            facts = json.load(f)
        for name, entry in summary.items():
            facts.setdefault("indicator_validation", {})[name] = {"checked_against_human_labels": True, **entry}
        with open(facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"Recorded in {facts_path}; the report text will now say which indicators were checked.")
        print("Re-running the pipeline overwrites facts.json; run `score` again afterwards.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-sheets")
    make.add_argument("--session", required=True)
    make.add_argument("--frames", type=int, default=60, help="how many random frames to review (default 60)")
    make.add_argument("--seed", type=int, default=1)
    scoring = sub.add_parser("score")
    scoring.add_argument("--session", required=True)
    scoring.add_argument("--no-patch", action="store_true", help="do not write the result into facts.json")
    args = parser.parse_args()
    if args.command == "make-sheets":
        make_sheets(args.session, args.frames, args.seed)
    else:
        score(args.session, patch_facts=not args.no_patch)


if __name__ == "__main__":
    main()
