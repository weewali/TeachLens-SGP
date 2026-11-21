"""Draws the TeachLens pipeline diagram for the SGP2 report: "SGP2 objectives/pipeline.png".

    python scripts/make_pipeline_diagram.py [--out PATH]

The eight step cards follow the web app's seven stages (web/models/job.py STAGES; its stage 6 is steps 6 and 7),
grouped into three phases shown as horizontal bands. Edit the STAGES text below whenever the pipeline changes, and
run this again (the user asked for the SGP2 objectives to be kept up to date with every change). Every claim here
was checked against the code and a real lecture on 2026-11-21: say only what the code does.

UAEU look (user requests 2026-11-21): the web app's colours with a deeper red, navy and gold labels, a red brand line
on top and the UAEU logo whole beside the title, as on the web app's header; no date. Each phase is a tinted panel
under a tall red band with an icon, its name and a one-line summary, and the arrow between phases names what passes
on (user: "make the phase bigger or more visual"). Drawn at A4 width (one axis
unit = one point) so the text stays readable in print. Needs matplotlib and Pillow (installed with matplotlib); the
logo (an SVG) is drawn by the Edge or Chrome the PDF export already uses, and left out when neither is installed.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SGP2 objectives" / "pipeline.png"
LOGO = ROOT / "web" / "views" / "static" / "uaeu-logo.svg"

W, H, DPI = 842, 960, 220             # points (A4 landscape width); the empty space at the bottom is cut off
# UAEU colours, as in web/views/static/style.css; a deeper red than the web's #c01d2a (user: "too bright")
RED, RED_DARK, RED_SOFT = "#961721", "#6e1119", "#f7ecec"
INK, TEXT, MUTED, LINE, PANEL, NAVY = "#232528", "#4b4f54", "#5b6a7c", "#dee3e7", "#f7f7f7", "#142e4a"
FONT, MONO = "Segoe UI", "Consolas"    # the web app's fallback fonts (Inter and Roboto are not installed)
KINDS = {                              # label colours: fill, text, border (user choice 2026-11-21: navy and gold)
    "ai": (NAVY, "#ffffff", NAVY),
    "stats": ("#fbf4e4", "#7a5a12", "#c9a14a"),
    "tool": ("#f4f7f8", INK, "#b1bcc6"),
}

# One card per step: (number, title, body items, output files, unused?)
# Body items: ("text", text) | ("note", text, shown muted) | ("badges", [(kind, label), ...])
STAGES = [
    (1, "Line up the cameras", [
        ("text", "Read each video's recording-start time (ffprobe) and keep the stretch both cameras recorded."),
        ("text", "Result: one lesson clock for both cameras, accurate to about 1 s."),
        ("badges", [("tool", "ffprobe")]),
    ], "lesson.json (ranges)", False),
    (2, "Student camera", [
        ("text", "One frame per second (faces 5 per second); every student tracked with an anonymous ID and given "
                 "a seat."),
        ("badges", [("ai", "YOLO26n-pose · 1280 px"), ("tool", "ByteTrack"), ("ai", "YOLO26n · phones, laptops, books"),
                    ("ai", "MediaPipe · faces, hands")]),
    ], "student/events.jsonl, summary.json, review.mp4", False),
    (3, "Instructor camera", [
        ("text", "One frame per second; people shown on the screen are ignored; transparent rules: facing the "
                 "class, gestures, pointing, seated, walking, reading notes."),
        ("note", "Board, screen and desk only once the room's areas are marked."),
        ("badges", [("ai", "YOLO26n-pose · 1280 px"), ("tool", "ByteTrack"), ("ai", "YOLO26n · screens")]),
    ], "instructor/events.jsonl, summary.json, review.mp4", False),
    (4, "Link the cameras", [
        ("text", "Students who walk to the front, and the instructor walking into the student view, are matched "
                 "by time."),
        ("note", "Saved for the team; not used by the report yet."),
        ("badges", [("tool", "Timing rules")]),
    ], "cross_camera.json", True),
    (5, "Links over time", [
        ("text", "5-second windows: do students change 5–30 s after an instructor action? 9 hypotheses fixed in "
                 "advance, plus exploratory pairs; rare actions are not tested."),
        ("note", "Exploratory: in one lecture a link rarely passes the correction. Its 5-second table feeds the "
                 "charts and step 6."),
        ("badges", [("stats", "Circular-shift test"), ("stats", "Benjamini–Hochberg q")]),
    ], "fusion/fusion.json, windows.csv, facts.json", False),
    (6, "During each activity", [
        ("text", "Students while you did each thing (facing the class, using gestures, seated, ...) compared with "
                 "the rest of the lecture; at least 5 windows on each side."),
        ("badges", [("stats", "HAC test · Bonferroni")]),
    ], "facts.json (added), tables/*.csv", False),
    (7, "Claude writes feedback", [
        ("text", "Reads the anonymised facts and writes up to 3 positive and 3 negative points as plain teaching "
                 "advice to the instructor on what to keep doing and one idea to try next time."),
        ("badges", [("ai", "Claude Opus 5.5")]),
    ], "final_feedback_report.txt, llm_output.json", False),
    (8, "Check the text", [
        ("text", "Every number must come from the cited facts; the wording is checked against rules (no causes, "
                 "inner states, talk about the measurements, or promises). One retry; points that still fail are "
                 "left out."),
        ("text", "No rule-based text: if Claude fails, the report shows why (key or billing problems only as "
                 "Code=01 / Code=02), with a \"Try again\" button."),
        ("badges", [("tool", "Automatic checks")]),
    ], "report_status.json", False),
]
# (name, steps, icon, one-line summary); FLOWS = what passes from one phase to the next
PHASES = [
    ("Watch the lecture", (1, 4), "camera", "Both videos become timed, anonymous records of what people visibly do."),
    ("Find the patterns", (5, 6), "chart", "Statistics relate what students do to what the instructor is doing."),
    ("Write the advice", (7, 8), "bubble", "Claude turns the facts into teaching advice; code checks every point."),
]
FLOWS = ["events of both cameras", "facts.json (anonymised facts)"]
PANEL_TINT, PANEL_EDGE, BAND_INK = "#fbf4f4", "#ead2d5", "#f3d4d8"


def uaeu_logo(width, height):
    """The UAEU logo as an image (pixels), drawn twice as large by a hidden Edge or Chrome and shown at width x height;
    None when no browser is found or it fails (the diagram is then drawn without the logo)."""
    try:
        sys.path.insert(0, str(ROOT))
        from web.models.pdf import find_browser
        browser = find_browser()
        if not browser:
            return None
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            folder = Path(folder)
            page, shot = folder / "logo.html", folder / "logo.png"
            page.write_text(f'<!doctype html><html><body style="margin:0;background:#fff"><img src="{LOGO.as_uri()}" '
                            f'style="display:block;width:{2 * width}px;height:{2 * height}px"></body></html>',
                            encoding="utf-8")
            subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--user-data-dir={folder / 'profile'}", f"--window-size={2 * width},{2 * height}",
                            f"--screenshot={shot}", page.as_uri()], capture_output=True, timeout=60)
            return plt.imread(shot) if shot.exists() else None
    except Exception as error:   # the logo is a nicety: never stop the diagram for it
        print("UAEU logo not drawn:", error)
        return None


class Canvas:
    """A matplotlib figure where one axis unit is one point, with the few drawing helpers the diagram needs."""

    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w / 72, h / 72), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.limits()
        self.ax.axis("off")
        self.fig.patch.set_facecolor("#ffffff")
        self.renderer = self.fig.canvas.get_renderer()

    def limits(self):
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)

    def text(self, x, y, s, size=8, weight="normal", color=TEXT, family=FONT, ha="left", va="top", style="normal",
             z=5):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, family=family, ha=ha, va=va,
                            style=style, zorder=z)

    def width(self, t):
        return t.get_window_extent(self.renderer).width * 72 / DPI

    def measure(self, s, size, weight="normal", family=FONT, style="normal"):
        t = self.text(0, 0, s, size, weight, family=family, style=style)
        w = self.width(t)
        t.remove()
        return w

    def para(self, x, y, s, width, size=8, weight="normal", color=TEXT, style="normal", draw=True):
        """A paragraph wrapped to width (measured, not counted in characters); returns the height used."""
        lines, cur = [], ""
        for word in s.split():
            trial = (cur + " " + word).strip()
            if not cur or self.measure(trial, size, weight, style=style) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines += [cur] if cur else []
        if draw:
            for i, line in enumerate(lines):
                self.text(x, y + i * size * 1.32, line, size, weight, color, style=style)
        return len(lines) * size * 1.32

    def box(self, x, y, w, h, fill, edge, radius=6, lw=0.8, dashed=False, z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                         facecolor=fill, edgecolor=edge, linewidth=lw,
                                         linestyle=(0, (4, 3)) if dashed else "-", zorder=z))

    def rect(self, x, y, w, h, fill, z=1):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none", zorder=z))

    def badges(self, x, y, labels, width, size=6.8, draw=True):
        """Labels in rows inside width; returns the height used."""
        h, gap = size * 2.1, 4
        bx, by = x, y
        for kind, label in labels:
            w = self.measure(label, size, "bold") + 2 * size * 0.9
            if bx > x and bx + w > x + width:
                bx, by = x, by + h + gap
            if draw:
                fill, ink, edge = KINDS[kind]
                self.box(bx, by, w, h, fill, edge, radius=3, lw=0.7, z=3)
                self.text(bx + size * 0.9, by + h / 2, label, size, "bold", ink, va="center", z=4)
            bx += w + gap
        return (by - y) + h if labels else 0

    def output(self, x, y, s, width, size=6.6, draw=True):
        """'Output:' and the file names, wrapped at commas inside width; returns the height used."""
        indent = self.measure("Output:", size, "bold") + 4
        lines, cur = [], ""
        for name in (n.strip() for n in s.split(",")):
            trial = f"{cur}, {name}" if cur else name
            if cur and self.measure(trial + ",", size, family=MONO) > width - indent:
                lines.append(cur + ",")
                cur = name
            else:
                cur = trial
        lines.append(cur)
        if draw:
            self.text(x, y, "Output:", size, "bold", RED_DARK)
            for i, line in enumerate(lines):
                self.text(x + indent, y + i * size * 1.35, line, size, color=MUTED, family=MONO)
        return len(lines) * size * 1.35

    def arrow(self, points, lw=1.3, head=(6, 3.2)):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            last = (x2, y2) == points[-1]
            self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), color=RED_DARK, linewidth=lw, shrinkA=0, shrinkB=0,
                                              arrowstyle=f"-|>,head_length={head[0]},head_width={head[1]}" if last
                                              else "-", zorder=2))

    def number(self, x, y, n, fill=RED):
        self.ax.add_patch(Circle((x, y), 7.5, facecolor=fill, edgecolor="none", zorder=4))
        self.text(x, y, str(n), 8.5, "bold", "#ffffff", ha="center", va="center_baseline", z=5)


def icon(c, kind, cx, cy):
    """A small red icon centred on (cx, cy), about 16 pt across: a video camera, a bar chart or a speech bubble."""
    def add(patch):
        c.ax.add_patch(patch)

    if kind == "camera":
        add(FancyBboxPatch((cx - 8.5, cy - 4.8), 11, 9.6, boxstyle="round,pad=0,rounding_size=2", facecolor=RED,
                           edgecolor="none", zorder=4))
        add(Polygon([(cx + 3.2, cy - 1.4), (cx + 8.5, cy - 4.6), (cx + 8.5, cy + 4.6), (cx + 3.2, cy + 1.4)],
                    closed=True, facecolor=RED, edgecolor="none", zorder=4))
        add(Circle((cx - 5.6, cy - 1.9), 1.1, facecolor="#ffffff", edgecolor="none", zorder=5))
    elif kind == "chart":
        for bx, height in ((cx - 7.2, 5), (cx - 1.7, 8.8), (cx + 3.8, 12.4)):
            add(Rectangle((bx, cy + 6 - height), 3.6, height, facecolor=RED, edgecolor="none", zorder=4))
        c.ax.plot([cx - 8.6, cx + 8.6], [cy + 6.6] * 2, color=RED, linewidth=1.2, solid_capstyle="round", zorder=4)
    else:   # speech bubble with two lines of text
        add(FancyBboxPatch((cx - 8.5, cy - 7), 17, 11.5, boxstyle="round,pad=0,rounding_size=3", facecolor=RED,
                           edgecolor="none", zorder=4))
        add(Polygon([(cx - 4.5, cy + 4), (cx - 6, cy + 8.5), (cx + 0.5, cy + 4)], closed=True, facecolor=RED,
                    edgecolor="none", zorder=4))
        for y, right in ((cy - 3.6, cx + 5), (cy + 0.2, cx + 1.5)):
            c.ax.plot([cx - 5, right], [y, y], color="#ffffff", linewidth=1.2, solid_capstyle="round", zorder=5)


def phase_band(c, x, y, w, h, n, name, steps, kind, summary):
    """The header of a phase panel: an icon, "PHASE n", the name, a divider, the one-line summary, the steps."""
    c.box(x, y, w, h, RED, RED, radius=7, lw=0, z=2)
    c.rect(x, y + h / 2, w, h / 2, RED, z=2)                 # square bottom corners: the band sits on its panel
    c.ax.add_patch(Circle((x + 25, y + h / 2), 14.5, facecolor="#ffffff", edgecolor="none", zorder=3))
    icon(c, kind, x + 25, y + h / 2)
    c.text(x + 48, y + 8, f"PHASE {n}", 7, "bold", BAND_INK, z=4)
    c.text(x + 48, y + 17.5, name, 13.5, "bold", "#ffffff", z=4)
    c.ax.plot([x + 214] * 2, [y + 10, y + h - 10], color="#c25a63", linewidth=0.8, zorder=4)
    c.text(x + 228, y + h / 2, summary, 8.8, color="#fbe9eb", va="center_baseline", z=4)
    c.text(x + w - 14, y + h / 2, f"steps {steps[0]}–{steps[1]}", 7.4, color=BAND_INK, ha="right",
           va="center_baseline", z=4)


def card_body(c, x, y, width, items, output, draw=True):
    """The text, notes and labels of one step card from y; returns the height used, including the output line."""
    ty = y
    for item in items:
        if item[0] == "text":
            ty += c.para(x, ty, item[1], width, draw=draw) + 4
        elif item[0] == "note":
            ty += c.para(x, ty, item[1], width, size=7.4, color=MUTED, style="italic", draw=draw) + 4
    ty += c.badges(x, ty + 1, [b for item in items if item[0] == "badges" for b in item[1]], width, draw=draw)
    return ty - y + 18 + c.output(0, 0, output, width, draw=False)


def header(c):
    """A red brand line, the UAEU logo whole, a divider, then the title (as on the web app's header)."""
    c.rect(0, 0, c.W, 5, RED)
    x = 26
    logo = uaeu_logo(513, 90)                               # the logo's own proportions (631.9 x 110.9)
    if logo is not None:
        lw, lh = 154, 27
        c.ax.imshow(logo, extent=(x, x + lw, 17 + lh, 17), zorder=2, interpolation="antialiased")
        c.limits()                                          # imshow resets the limits
        c.ax.plot([x + lw + 14] * 2, [15, 17 + lh + 2], color=LINE, linewidth=0.8)
        x += lw + 28
    c.text(x, 13, "TeachLens pipeline", 17, "bold", INK)
    c.text(x, 36, "How two videos of one lecture become a feedback report, on an ordinary laptop.", 8.5, color=MUTED)
    c.ax.plot([26, c.W - 26], [58, 58], color=LINE, linewidth=0.6)


def inputs(c, y):
    c.text(26, y + 2, "Input", 8, "bold", RED_DARK)
    x = 64
    for label in ("Instructor video", "Student video"):
        w = c.measure(label, 8, "bold") + 20
        c.box(x, y - 2, w, 17, PANEL, LINE, radius=4, lw=0.8)
        c.rect(x + 4, y + 2, 2.5, 9, RED, z=2)
        c.text(x + 11, y + 6.5, label, 8, "bold", INK, va="center")
        x += w + 8
    c.text(x + 4, y + 6.5, "two static cameras: one facing the instructor, one facing the students", 7.5,
           color=MUTED, va="center")
    context = "Optional context: topic, slides, activities (kept with the report; not used by the analysis)"
    w = c.measure(context, 7.5) + 20
    c.box(c.W - 26 - w, y - 2, w, 17, "#ffffff", "#b1bcc6", radius=4, lw=0.8, dashed=True)
    c.text(c.W - 26 - w + 10, y + 6.5, context, 7.5, color=TEXT, va="center")


def legend(c, x, y):
    for kind, label in (("ai", "AI model"), ("stats", "Statistics"), ("tool", "Tool or method")):
        fill, _, edge = KINDS[kind]
        c.box(x, y, 15, 11, fill, edge, radius=2.5, lw=0.7)
        t = c.text(x + 20, y + 5.5, label, 7.5, color=INK, va="center")
        x += 20 + c.width(t) + 18
    c.box(x, y, 15, 11, "#ffffff", MUTED, radius=2.5, lw=0.7, dashed=True)
    c.text(x + 20, y + 5.5, "Optional, or not used by the report yet", 7.5, color=INK, va="center")


def draw(out):
    c = Canvas(W, H)
    header(c)
    inputs(c, 70)
    full, pad, gap, inset, band = c.W - 52, 9, 14, 10, 42
    stage = {s[0]: s for s in STAGES}
    y, last_mid = 100, None
    for p, (name, (a, b), kind, summary) in enumerate(PHASES):
        if p:                                                # what passes to the next phase
            y += 28
            c.arrow([(c.W / 2, y - 27), (c.W / 2, y - 1)], lw=1.8, head=(7.5, 4.2))
            c.text(c.W / 2 + 11, y - 14, FLOWS[p - 1], 7.6, color=RED_DARK, style="italic", va="center_baseline")
        steps = list(range(a, b + 1))
        cw = (full - 2 * inset - gap * (len(steps) - 1)) / len(steps)
        h = 26 + max(card_body(c, 0, 0, cw - 2 * pad, stage[n][2], stage[n][3], draw=False) for n in steps)
        c.box(26, y, full, band + inset + h + inset, PANEL_TINT, PANEL_EDGE, radius=7, lw=0.8, z=0)
        phase_band(c, 26, y, full, band, p + 1, name, (a, b), kind, summary)
        y += band + inset
        for i, n in enumerate(steps):
            _, title, items, output, unused = stage[n]
            x = 26 + inset + i * (cw + gap)
            c.box(x, y, cw, h, "#ffffff", "#b1bcc6" if unused else LINE, radius=5, lw=0.8, dashed=unused)
            c.number(x + pad + 7, y + 13, n, fill=MUTED if unused else RED)
            c.text(x + pad + 20, y + 13, title, 9.5, "bold", INK, va="center_baseline")
            card_body(c, x + pad, y + 26, cw - 2 * pad, items, output)
            out_h = c.output(0, 0, output, cw - 2 * pad, draw=False)
            c.ax.plot([x + pad, x + cw - pad], [y + h - out_h - 11] * 2, color=LINE, linewidth=0.6)
            c.output(x + pad, y + h - out_h - 6, output, cw - 2 * pad)
            if i:
                c.arrow([(x - gap + 2, y + h / 2), (x - 2, y + h / 2)], lw=1.1)
            last_mid = x + cw / 2
        y += h + inset                                       # the bottom of the phase panel

    # the optional accuracy check on the left, the web application on the right, under step 8
    y += 18
    aw = full * 0.34
    c.box(26, y, aw, 52, "#ffffff", "#b1bcc6", radius=6, lw=0.8, dashed=True)
    c.text(36, y + 9, "Accuracy check", 9, "bold", INK)
    c.text(26 + aw - 10, y + 10, "optional", 6.8, color=MUTED, ha="right")
    c.para(36, y + 24, "A person checks sampled detections (review sheets, evaluation/); checked precision goes into "
                       "facts.json.", aw - 20, size=7.2)
    wx = 26 + aw + 12
    web_w = full - aw - 12
    c.arrow([(last_mid, y - 18), (last_mid, y)], lw=1.5)
    c.box(wx, y, web_w, 52, RED_SOFT, RED, radius=6, lw=1.2)
    c.text(wx + 12, y + 9, "Web application", 10, "bold", RED_DARK)
    c.text(wx + web_w - 12, y + 10, "Flask + Jinja2, runs on this computer only", 6.8, color=MUTED, ha="right")
    c.text(wx + 12, y + 27, "Upload the two videos  →  follow stages 1–7  →  read the report in tabs:", 7.8, color=INK)
    c.text(wx + 12, y + 39, "Feedback  |  Charts  |  Moments  |  Key terms  →  Download PDF", 7.8, "bold", INK)
    legend(c, 26, y + 64)   # no note paragraph under it (user request 2026-11-21)

    out.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(out, dpi=DPI, facecolor="#ffffff")
    plt.close(c.fig)
    from PIL import Image, ImageOps   # cut the empty space under the last line, keeping a margin
    image = Image.open(out).convert("RGB")
    box = ImageOps.invert(image).getbbox()
    if box:
        image.crop((0, 0, image.width, min(image.height, box[3] + int(22 * DPI / 72)))).save(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    print("Pipeline diagram written to", draw(parser.parse_args().out))
