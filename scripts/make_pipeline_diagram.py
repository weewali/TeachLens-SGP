"""Draws the TeachLens pipeline diagram for the SGP2 report: "SGP2 objectives/pipeline.png".

    python scripts/make_pipeline_diagram.py [--out PATH]

The eight cards follow the web app's seven stages (web/models/job.py STAGES; its stage 6 is cards 6 and 7). Edit the
STAGES text below whenever the pipeline changes, and run this again (the user asked for the SGP2 objectives to be
kept up to date with every change). Needs only matplotlib.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SGP2 objectives" / "pipeline.png"
UPDATED = "20 November 2026"

W, H = 2400, 1650                      # pixels; one axis unit = one pixel (saved at 100 dpi)
INK, MUTED, LINE = "#1f2430", "#5b6475", "#c9ccd6"
CARD, HEAD, HEAD_OPTIONAL = "#f6f7f9", "#142e4a", "#51627a"
KINDS = {                              # badge colours: fill, text, border
    "ai": ("#1f64b8", "#ffffff", "#1f64b8"),
    "stats": ("#6b4fbb", "#ffffff", "#6b4fbb"),
    "tool": ("#e6e9f0", INK, "#c9ccd6"),
}

# One card per pipeline stage: (number, title, body items, output files, optional?)
# Body items: ("bold", text) | ("text", text) | ("badges", [(kind, label), ...]) | ("gap", pixels)
STAGES = [
    (1, "Line up the cameras", [
        ("text", "Read each video's recording-start time (ffprobe) and keep the stretch both cameras recorded."),
        ("text", "Result: one lesson clock for both cameras, accurate to about 1 s."),
        ("badges", [("tool", "ffprobe")]),
    ], "lesson.json (ranges)", False),
    (2, "Student camera", [
        ("text", "One frame per second; every student tracked with an anonymous ID and given a seat."),
        ("badges", [("ai", "YOLO26n-pose · 1280 px")]),
        ("badges", [("ai", "YOLO26n · phones, books")]),
        ("badges", [("ai", "MediaPipe · faces, hands")]),
    ], "student/events.jsonl,\nsummary.json, review.mp4", False),
    (3, "Instructor camera", [
        ("text", "One frame per second; transparent rules: facing the class, board, gestures, pointing, seated, "
                 "walking, notes."),
        ("badges", [("ai", "YOLO26n-pose · 1280 px")]),
        ("badges", [("tool", "ByteTrack")]),
    ], "instructor/events.jsonl,\nsummary.json, review.mp4", False),
    (4, "Link the cameras", [
        ("text", "Students who walk to the front, and the instructor walking into the student view, are matched "
                 "by time."),
        ("badges", [("tool", "Timing rules")]),
    ], "cross_camera.json", False),
    (5, "Links over time", [
        ("text", "5-second windows: do students change 5–30 s after an instructor action?"),
        ("text", "9 hypotheses fixed in advance, plus exploratory pairs; rare actions are not tested."),
        ("badges", [("stats", "Circular-shift test")]),
        ("badges", [("stats", "Benjamini–Hochberg q")]),
    ], "fusion/fusion.json,\nwindows.csv, facts.json", False),
    (6, "During each activity", [
        ("text", "Students while you did each thing (facing the class, at the board, seated, ...) compared with "
                 "the rest of the lecture."),
        ("badges", [("stats", "HAC test · Bonferroni")]),
    ], "facts.json (added),\ntables/*.csv", False),
    (7, "Claude writes feedback", [
        ("text", "Reads the facts (numbers only) and writes up to 3 positive and 3 negative points as plain teaching "
                 "advice to the instructor (\"you\"): what to keep doing, and one idea to try next time."),
        ("badges", [("ai", "Claude Opus 5.5")]),
    ], "final_feedback_report.txt", False),
    (8, "Check the text", [
        ("text", "Code checks every number and word against the facts; one retry; points that still fail are left "
                 "out."),
        ("text", "No rule-based text: if Claude fails, the report shows why (key or billing problems only as "
                 "Code=01 / Code=02), with a \"Try again\" button."),
        ("badges", [("tool", "Automatic checks")]),
    ], "report_status.json,\nllm_output.json", False),
]

COL_X = [380, 890, 1400, 1910]         # left edge of the four stage columns
CARD_W, HEAD_H = 470, 78
ROWS = [(230, 470), (790, 470)]        # (top, height) of the two rows of stages


def text(ax, x, y, s, size=13, weight="normal", color=INK, family="DejaVu Sans", style="normal", ha="left"):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, family=family, style=style, ha=ha,
                   va="top", linespacing=1.3, multialignment=ha)


def box(ax, x, y, w, h, fill, edge, radius=16, dashed=False, lw=2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge, linewidth=lw, linestyle=(0, (6, 4)) if dashed else "-", zorder=z))


def badge(ax, renderer, x, y, kind, label):
    fill, ink, edge = KINDS[kind]
    t = text(ax, x + 16, y + 9, label, size=13, weight="bold", color=ink)
    t.set_zorder(4)
    width = t.get_window_extent(renderer).width
    box(ax, x, y, width + 32, 40, fill, edge, radius=9, lw=1.5, z=3)
    return width + 32


def arrow(ax, points, dashed=False):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        last = (x2, y2) == points[-1]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_length=12,head_width=7" if last else "-",
                                     color="#5b6475", linewidth=2.5, linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=0, zorder=2))


def stage_card(ax, renderer, x, y, h, number, title, items, output, optional):
    box(ax, x, y, CARD_W, h, CARD, LINE, dashed=optional)
    head = HEAD_OPTIONAL if optional else HEAD
    box(ax, x, y, CARD_W, HEAD_H, head, head, z=2)
    ax.add_patch(Rectangle((x, y + HEAD_H - 18), CARD_W, 18, facecolor=head, edgecolor=head, zorder=2))
    text(ax, x + 26, y + 20, f"{number}   {title}", size=19, weight="bold", color="#ffffff").set_zorder(3)
    cy = y + HEAD_H + 24
    for kind, value in items:
        if kind == "gap":
            cy += value
        elif kind == "bold":
            text(ax, x + 28, cy, value, size=13.5, weight="bold")
            cy += 30
        elif kind == "text":
            lines = textwrap.wrap(value, 40)
            text(ax, x + 28, cy, "\n".join(lines), size=13.5)
            cy += 24 * len(lines) + 14
        elif kind == "badges":
            bx = x + 28
            for badge_kind, label in value:
                bx += badge(ax, renderer, bx, cy, badge_kind, label) + 12
            cy += 52
    lines = output.split("\n")
    oy = y + h - 26 - 21 * len(lines)
    ax.plot([x + 28, x + CARD_W - 28], [oy - 14, oy - 14], color=LINE, linewidth=1.2, zorder=2)
    label = text(ax, x + 28, oy, "Output:", size=11.5, weight="bold", color=MUTED)
    text(ax, x + 28 + label.get_window_extent(renderer).width + 10, oy, output, size=11.5, color=MUTED,
         family="DejaVu Sans Mono")
    assert cy < oy - 20, f"stage {number}: the text runs into the output line; make the card taller or the text shorter"


def side_box(ax, x, y, w, h, title, lines, dashed=False):
    box(ax, x, y, w, h, CARD, LINE, dashed=dashed)
    text(ax, x + w / 2, y + 26, title, size=17, weight="bold", ha="center")
    cy = y + 26 + 34 * title.count("\n") + 46
    for line, muted in lines:
        wrapped = textwrap.wrap(line, 24)
        text(ax, x + w / 2, cy, "\n".join(wrapped), size=12.5, color=MUTED if muted else INK, ha="center",
             style="italic" if muted else "normal")
        cy += 22 * len(wrapped) + 14


def draw(out):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    renderer = fig.canvas.get_renderer()

    text(ax, 60, 50, "TeachLens pipeline", size=36, weight="bold")
    text(ax, 60, 128, "How two videos of one lecture become a feedback report. All steps run on an ordinary laptop "
                      "CPU; the web app shows them as 7 stages while it works.", size=15, color=MUTED)
    text(ax, W - 60, 60, f"updated {UPDATED}", size=12, color=MUTED, ha="right")

    # inputs
    (top1, h1), (top2, h2) = ROWS
    half = (h1 - 20) / 2
    side_box(ax, 40, top1, 290, half, "Instructor\nvideo", [("one static camera on the students' side", True)])
    side_box(ax, 40, top1 + half + 20, 290, half, "Student\nvideo", [("one static camera facing the students", True)])
    side_box(ax, 40, top2, 290, h2, "Optional\ncontext", [
        ("Topic, slides", False),
        ("Activities in class: quiz, group work, video, break (minutes)", False),
        ("Additional information", False),
        ("Kept with the report; not used by the analysis yet", True)], dashed=True)
    mid1 = top1 + h1 / 2
    arrow(ax, [(330, top1 + half / 2), (355, top1 + half / 2), (355, mid1 - 20), (COL_X[0], mid1 - 20)])
    arrow(ax, [(330, top1 + half + 20 + half / 2), (355, top1 + half + 20 + half / 2), (355, mid1 + 20),
               (COL_X[0], mid1 + 20)])

    # the eight stages in two rows, joined left to right
    for n, (number, title, items, output, optional) in enumerate(STAGES):
        top, h = ROWS[n // 4]
        x = COL_X[n % 4]
        stage_card(ax, renderer, x, top, h, number, title, items, output, optional)
        if n % 4:
            arrow(ax, [(x - 40, top + h / 2), (x, top + h / 2)], dashed=optional)
    turn_y = (top1 + h1 + top2) / 2
    arrow(ax, [(COL_X[3] + CARD_W / 2, top1 + h1), (COL_X[3] + CARD_W / 2, turn_y), (COL_X[0] + CARD_W / 2, turn_y),
               (COL_X[0] + CARD_W / 2, top2)])

    # the web application around it
    wy = top2 + h2 + 70
    box(ax, 380, wy, COL_X[3] + CARD_W - 380, 120, "#ffffff", HEAD, lw=2.5)
    text(ax, 410, wy + 22, "Web application", size=17, weight="bold")
    text(ax, 410, wy + 62, "Upload the two videos  →  follow stages 1–7  →  read the report in tabs:  Feedback  |  "
                           "Charts  |  Moments  |  Key terms  →  Download PDF", size=14)
    text(ax, COL_X[3] + CARD_W - 30, wy + 24, "Flask + Jinja2, runs on this computer only", size=12, color=MUTED,
         ha="right")
    arrow(ax, [(COL_X[2] + CARD_W / 2, top2 + h2), (COL_X[2] + CARD_W / 2, wy)])

    # legend and notes
    ly = wy + 170
    lx = 60
    for kind, label in (("ai", "AI model"), ("stats", "Statistics"), ("tool", "Tool or method")):
        fill, _, edge = KINDS[kind]
        box(ax, lx, ly, 44, 34, fill, edge, radius=8, lw=1.5)
        text(ax, lx + 58, ly + 6, label, size=14)
        lx += 290
    box(ax, lx, ly, 44, 34, "#ffffff", MUTED, radius=8, lw=1.5, dashed=True)
    text(ax, lx + 58, ly + 6, "Optional", size=14)
    notes = ("The YOLO26 and MediaPipe models are pretrained and used without retraining. Only anonymised numbers, never "
             "video, images or names, go to Claude. Students' faces are blurred in the report's pictures.\n"
             "Since 20 November 2026 one pipeline: the team's processing (steps 1-5) with the report stage and web app "
             "on top. Accuracy has not been checked against human labels yet.")
    text(ax, 60, ly + 70, notes, size=12, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="#ffffff")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    print("Pipeline diagram written to", draw(parser.parse_args().out))
