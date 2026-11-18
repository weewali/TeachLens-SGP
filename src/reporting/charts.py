"""Charts drawn from saved evidence (Matplotlib, written as PNG files)."""

from pathlib import Path
from typing import Any

# Categorical hues in fixed order (validated on the light surface and for
# colour-vision deficiencies; some are below 3:1 contrast, so every series is
# in the legend and the window CSV is the table view).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, MUTED, SURFACE, GRID, AXIS = "#1f1f1e", "#6b6a64", "#fcfcfb", "#e4e3df", "#c9c8c2"
TEACHER_SERIES = (("T_facing_class", "facing class"), ("T_board_or_screen", "board or screen"),
                  ("T_gesturing", "gesturing"), ("T_walking", "walking"), ("T_writing", "writing"))
STUDENT_SERIES = (("S_on_target", "on target"), ("S_phone", "phone"), ("S_talking", "talking"),
                  ("S_head_down", "head down"), ("S_sleep", "dozing or sleep"))


def plot_fusion_timelines(table: Any, title: str, path: Path) -> Path:
    """Teacher shares (top) and class shares (bottom) per fusion window; gaps = too little evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    half = table["lesson_seconds"].diff().fillna(0).max() / 2
    minutes = (table["lesson_seconds"] + half) / 60
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, facecolor=SURFACE)
    for axis, label, series in ((axes[0], "teacher share of the window", TEACHER_SERIES),
                                (axes[1], "class share of the window", STUDENT_SERIES)):
        axis.set_facecolor(SURFACE)
        for color, (column, name) in zip(SERIES_COLORS, series):
            if column in table:
                axis.plot(minutes, table[column], color=color, linewidth=1.5, marker="o", markersize=2.5,
                          label=name)
        axis.set_ylabel(label, color=INK, fontsize=9)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, labelcolor=INK)
    if "S_hand_raises" in table:
        for minute in (table.loc[table["S_hand_raises"] > 0, "lesson_seconds"] + half) / 60:
            axes[1].axvline(minute, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("lesson minutes (window centres; dotted lines: hand raises; gaps: too little evidence)",
                       color=INK, fontsize=9)
    fig.suptitle(f"{title}: teacher and student timelines", color=INK, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path
