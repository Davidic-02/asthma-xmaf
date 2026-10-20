"""Shared drawing helpers and palette for the framework figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, GREEN, DEEP = "#2F9CD8", "#57B85F", "#1F6FA8"
PALE, RED, GREY = "#E9EFF5", "#D64545", "#8A97A3"
PANEL, INK, SUB = "#6F7C89", "#12202B", "#3F5261"
AMBER = "#F2B544"


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def panel(ax, x, y, w, h, title=None, *, fs=12.5, dashed=True, edge=PANEL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=1.2, edgecolor=edge, facecolor="none",
        linestyle=(0, (6, 4)) if dashed else "-", zorder=1))
    if title:
        ax.text(x + 0.010, y + h - 0.014, title, ha="left", va="top",
                fontsize=fs, fontweight="bold", color=INK, zorder=3)


def box(ax, x, y, w, h, lines, face, *, fs=10.5, fg=None, edge=None, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.010",
        linewidth=1.2 if edge else 0, edgecolor=edge or "none", facecolor=face,
        linestyle=(0, (5, 3)) if dashed else "-", zorder=2))
    fg = fg or ("white" if face in (BLUE, DEEP, GREEN, RED, GREY) else SUB)
    n, step = len(lines), fs * 0.0021
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + ((n - 1) / 2 - i) * step * 1.55, ln,
                ha="center", va="center", fontsize=fs, zorder=3, color=fg,
                fontweight="bold" if (i == 0 and n > 1) else "normal")


def arrow(ax, p0, p1, *, color=INK, dashed=False, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color,
        linestyle=(0, (4, 3)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=2, shrinkB=2))


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(out / f"{name}.{ext}", format=ext, facecolor="white", **kw)
    plt.close(fig)
    print(f"wrote {out / name}.png / .pdf")
