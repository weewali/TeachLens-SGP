"""Static, print-ready versions of the report's two statistical charts, for the PDF (user request 2026-11-20: on paper
nothing can be pointed at, so the PDF needs simple, self-explaining graphs).

- timeline_png:   students on task over the lecture (30-second average) as one line, what the instructor was doing
                  shaded behind it, hand raises as purple ticks along the bottom, the highest and lowest points labelled.
- comparison_png: a grouped bar chart: for each student measure (on task, phone, head down) one bar per instructor
                  activity, with its value and its 95% range.

Same data and colours as the web charts (web/models/stat_charts.py, style.css --viz-*). Returned as PNG bytes.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.analysis.activity_comparison import MEASURES  # noqa: E402

COLOURS = {"explaining": "#2a78d6", "board": "#eb6834", "not_facing": "#1baf7a", "out_of_view": "#9aa1ad"}
INK, MUTED, GRID = "#142e4a", "#5b6475", "#e3e6ec"
LINE, RAISE = "#c01d2a", "#7a1fa2"   # the on-task line in UAEU red; hand raises in purple (clear of every activity colour)
SMOOTH_WINDOWS = 6            # 6 x 5 s = a 30-second average
DPI = 200


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9bfca")
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _clock(minutes, _position=None):
    seconds = int(round(minutes * 60))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def timeline_png(data):
    """PNG bytes of the lecture timeline, or None without data."""
    parts = (data or {}).get("parts") or []
    if not parts:
        return None
    starts = np.array([p["start"] for p in parts], float) / 60
    ends = np.array([p["end"] for p in parts], float) / 60
    values = pd.Series([np.nan if p["on_task"] is None else p["on_task"] for p in parts], dtype=float)
    # the same 30-second stretches as the report's highest / lowest facts (src/reporting/facts.py): complete
    # stretches only, each drawn at its middle, so the chart's highest and lowest match the feedback's numbers
    smooth = values.rolling(SMOOTH_WINDOWS, min_periods=SMOOTH_WINDOWS - 1).mean()
    middle = ends - SMOOTH_WINDOWS * (ends - starts) / 2

    # a solid strip of what the instructor was doing above the chart, so the colours are clear on paper
    fig, (strip, ax) = plt.subplots(2, 1, figsize=(7.0, 2.9), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 9], "hspace": 0.08})
    _style(ax)
    strip.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        strip.spines[side].set_visible(False)
    strip.tick_params(bottom=False)
    strip.set_ylabel("You were", fontsize=7, color=MUTED, rotation=0, ha="right", va="center", labelpad=4)
    run_start, present = 0, []
    for i in range(1, len(parts) + 1):   # one shaded band per run of the same activity
        if i == len(parts) or parts[i]["activity"] != parts[run_start]["activity"]:
            key = parts[run_start]["activity"]
            colour = COLOURS.get(key, "#cccccc")
            strip.axvspan(starts[run_start], ends[i - 1], color=colour, linewidth=0)
            ax.axvspan(starts[run_start], ends[i - 1], color=colour, alpha=0.10, linewidth=0)
            if key not in present:
                present.append(key)
            run_start = i
    ax.fill_between(middle, smooth, color=LINE, alpha=0.18, linewidth=0)   # the area under the line
    ax.plot(middle, smooth, color=LINE, linewidth=2.0)
    if smooth.notna().any():
        for index, word, offset in ((smooth.idxmax(), "Highest", 6), (smooth.idxmin(), "Lowest", -11)):
            value = smooth[index]
            ax.plot(middle[index], value, "o", color=LINE, markersize=4)
            ax.annotate(f"{word} {value:.0f}%", (middle[index], value), textcoords="offset points",
                        xytext=(0, offset), ha="center", fontsize=7, color=INK)
    raises = [m["start"] / 60 for m in data.get("moments", []) if m["kind"] == "hand_raise"]
    if raises:
        ax.plot(raises, [2.5] * len(raises), "|", color=RAISE, markersize=10, markeredgewidth=1.6)
    ax.set_xlim(0, ends[-1])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    span = ends[-1]
    step = next(s for s in (1, 2, 5, 10, 15, 30) if span / s <= 8)
    ax.set_xticks(np.arange(0, span + 1e-9, step))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_clock))
    ax.set_xlabel("Time in the lecture (mm:ss)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Students on task", fontsize=7.5, color=MUTED)
    labels = {a["key"]: a["label"] for a in data.get("activities", [])}
    handles = [Line2D([], [], color=LINE, linewidth=2.0, label="On task (30-second average)")]
    handles += [Patch(color=COLOURS.get(key, "#cccccc"), label=labels.get(key, key))
                for key in COLOURS if key in present]   # the same order in every lecture
    if raises:
        handles.append(Line2D([], [], color=RAISE, marker="|", linestyle="none", markersize=9, markeredgewidth=1.6,
                              label="Hand raised"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(handles), frameon=False,
              fontsize=7, handlelength=1.6, columnspacing=1.2)
    return _png(fig)


def comparison_png(rows):
    """PNG bytes of the grouped bar chart (rows from stat_charts.comparison_values), or None without rows."""
    if not rows:
        return None
    labels = [label for _, label, _, _ in MEASURES]
    columns = [column for column, *_ in MEASURES]
    x = np.arange(len(columns))
    width = 0.78 / len(rows)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    _style(ax)
    top = 0
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        for j, column in enumerate(columns):
            cell = row["measures"][column]
            if np.isnan(cell["mean"]):
                continue
            value = 100 * cell["mean"]
            ax.bar(x[j] + offset, value, width * 0.92, color=COLOURS.get(row["key"], "#cccccc"))
            upper = value
            if cell["range"]:
                low, high = (100 * v for v in cell["range"])
                ax.errorbar(x[j] + offset, value, yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                            color="#3b4252", linewidth=0.8, capsize=2)   # max(): rounding at the range's edge
                upper = high
            ax.text(x[j] + offset, upper + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=6.8, color=INK)
            top = max(top, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(0, min(110, max(30, top + 12)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.set_ylabel("Share of the time", fontsize=7.5, color=MUTED)
    handles = [Patch(color=COLOURS.get(r["key"], "#cccccc"), label=f"{r['label']} ({r['share']}% of the lecture)")
               for r in rows]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(handles), frameon=False,
              fontsize=7, handlelength=1.2, columnspacing=1.4)
    return _png(fig)
