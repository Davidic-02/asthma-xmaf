"""Generate Figure 1: system architecture, as publication-quality PDF/PNG/SVG.

Drawn with matplotlib rather than a diagramming application so the figure is
reproducible from the repository and can be regenerated when the architecture
changes. Journals generally require vector format (PDF/EPS) for line art and
>=300 dpi for raster; both are emitted.

Usage:  python src/figures/make_architecture.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"

# Greyscale-safe palette: many journals print figures in black and white, so
# boxes are distinguished by tone and border rather than hue alone.
DATA = "#E8EEF4"
AGENT = "#CFE0EE"
DECISION = "#B4CFE4"
EXPLAIN = "#E9E3F2"
OUTPUT = "#DCE9DC"
EDGE = "#3A4A58"

FIG_W, FIG_H = 9.2, 9.6

FS_TITLE = 10.0
FS_BODY = 8.0
FS_NOTE = 7.2

# Axes-fraction distances derived from the figure height so that text spacing
# matches the font size instead of being guessed. 1 axes unit == FIG_H inches.
PT = 1.0 / 72.0 / FIG_H          # one typographic point in axes units
LINE = 11.5 * PT                 # body line pitch
TITLE_DROP = 15.0 * PT           # top edge -> title baseline
BODY_START = 30.0 * PT           # top edge -> first body line


def box(ax, x, y, w, h, title, lines, face, *, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=1.0, edgecolor=EDGE, facecolor=face, zorder=2,
    ))
    ax.text(x + w / 2, y + h - TITLE_DROP, title,
            ha="center", va="top", fontsize=FS_TITLE,
            fontweight="bold" if bold else "normal", color="#12202B", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - BODY_START - i * LINE, ln,
                ha="center", va="top", fontsize=FS_BODY, color="#28414F", zorder=3)


def arrow(ax, xy_from, xy_to, *, style="-|>", dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=EDGE, zorder=1,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        shrinkA=1.5, shrinkB=1.5,
    ))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def bh(n_lines):
        """Height of a box holding a title plus n body lines."""
        return BODY_START + n_lines * LINE + 8 * PT

    # ---------------------------------------------------------------- inputs
    ax.text(0.5, 0.985, "Child presenting with possible asthma",
            ha="center", va="top", fontsize=11.5, fontweight="bold",
            color="#12202B")

    col = [0.045, 0.365, 0.685]
    w = 0.27
    h3 = bh(3)

    y_data = 0.955 - h3
    box(ax, col[0], y_data, w, h3, "Clinical history",
        ["NHANES 2007–2012", "n = 6,329 children", "14 symptom features"], DATA)
    box(ax, col[1], y_data, w, h3, "Spirometry",
        ["NHANES 2007–2012", "92.3% available", "7 physiological features"], DATA)
    box(ax, col[2], y_data, w, h3, "Lung sound",
        ["SPRSound (CC BY 4.0)", "747 children", "5,216 recordings"], DATA)

    # ---------------------------------------------------------------- agents
    y_agent = y_data - 0.055 - h3
    box(ax, col[0], y_agent, w, h3, "Symptom Agent",
        ["calibrated logistic reg.", "SHAP attribution", "ROC-AUC 0.863"], AGENT)
    box(ax, col[1], y_agent, w, h3, "Pulmonary Agent",
        ["calibrated logistic reg.", "SHAP attribution", "ROC-AUC 0.646"], AGENT)
    box(ax, col[2], y_agent, w, h3, "Acoustic Agent",
        ["AST embeddings (768-d)", "+ 58 named features", "ROC-AUC 0.732"], AGENT)

    for c in col:
        arrow(ax, (c + w / 2, y_data), (c + w / 2, y_agent + h3))

    # The decoupling is the paper's central design claim, so it is labelled.
    ax.text(0.5, y_agent - 0.016,
            "agents trained independently — no paired multimodal cohort required",
            ha="center", va="top", fontsize=FS_NOTE, style="italic",
            color="#54626E", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.6))

    # ------------------------------------------------------- evidence bundle
    ev_h = 0.030
    y_ev = y_agent - 0.062 - ev_h
    ax.add_patch(FancyBboxPatch(
        (0.045, y_ev), 0.91, ev_h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.0, edgecolor=EDGE, facecolor="#F2F5F8",
        linestyle=(0, (5, 3)), zorder=2,
    ))
    ax.text(0.5, y_ev + ev_h / 2,
            "Evidence bundle:   calibrated probability   ·   named contributing factors   ·   modality-present flag",
            ha="center", va="center", fontsize=FS_BODY, color="#28414F", zorder=3)

    for c in col:
        arrow(ax, (c + w / 2, y_agent), (c + w / 2, y_ev + ev_h))

    # ------------------------------------------------------- decision agent
    dec_h = bh(3)
    y_dec = y_ev - 0.055 - dec_h
    box(ax, 0.215, y_dec, 0.57, dec_h, "Decision Agent",
        ["logistic meta-learner over agent evidence",
         "abstains on low confidence   ·   flags inter-agent conflict",
         "fused ROC-AUC 0.870   ·   0.881 after 15% referral"], DECISION)
    arrow(ax, (0.5, y_ev), (0.5, y_dec + dec_h))

    # ------------------------------------------------------------- outputs
    out_h = bh(3)
    y_out = y_dec - 0.058 - out_h
    box(ax, col[0], y_out, w, out_h, "Asthma likelihood",
        ["calibrated probability", "graceful degradation:", "0.845 without spirometry"], OUTPUT)
    box(ax, col[1], y_out, w, out_h, "Referral / abstention",
        ["uncertain cases deferred", "to clinical review"], OUTPUT)
    box(ax, col[2], y_out, w, out_h, "Conflict flag",
        ["8.2% of children", "73.1% asthma prevalence", "vs 12.7% baseline"], OUTPUT)

    arrow(ax, (0.40, y_dec), (col[0] + w / 2, y_out + out_h))
    arrow(ax, (0.50, y_dec), (col[1] + w / 2, y_out + out_h))
    arrow(ax, (0.60, y_dec), (col[2] + w / 2, y_out + out_h))

    # -------------------------------------------------- explanation channel
    exp_h = bh(3)
    y_exp = y_out - 0.058 - exp_h
    box(ax, 0.045, y_exp, 0.91, exp_h, "Cross-Modal Explanation Fusion",
        ["unified rationale over SHAP (clinical) and named acoustic features",
         "faithfulness verified by deletion against a random-feature control:",
         "comprehensiveness 0.100 vs 0.036 random (2.8×)   ·   sufficiency 0.017 vs 0.084"],
        EXPLAIN)

    # Attribution flows from the agents and from the decision into the
    # explanation layer; dashed to distinguish from the inference path.
    arrow(ax, (col[0] + 0.03, y_agent), (0.10, y_exp + exp_h), dashed=True)
    arrow(ax, (0.5, y_out), (0.5, y_exp + exp_h), dashed=True)
    arrow(ax, (col[2] + w - 0.03, y_agent), (0.90, y_exp + exp_h), dashed=True)

    ax.set_ylim(y_exp - 0.015, 1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        path = OUT / f"figure1_architecture.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight",
                    facecolor="white", **kw)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
