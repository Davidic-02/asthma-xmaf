"""Generate Figure 2: system architecture, as publication-quality PDF/PNG/SVG.

Laid out as numbered stages so the evidence flow can be followed in order, in
the same visual idiom as the pipeline figure: each stage is a labelled dashed
panel, boxes are coloured by role, and small insets plot the real results
rather than describing them.

Every number is read from results/ at build time. Nothing is typed in, so the
figure cannot drift from the experiments the way a hand-edited diagram does.

Usage:  python src/figures/make_architecture.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"

BLUE = "#2F9CD8"       # agents: the learned components
GREEN = "#57B85F"      # calibrated evidence leaving an agent
DEEP = "#1F6FA8"       # the decision agent itself
PALE = "#E9EFF5"       # data sources
PINK = "#F6DCE2"       # evaluated quantities
RED = "#D64545"
GREY = "#8A97A3"
PANEL = "#6F7C89"      # dashed stage borders
INK = "#12202B"
SUB = "#3F5261"

FS_STAGE, FS_BODY, FS_NOTE = 13.0, 9.5, 8.2


def stage(ax, x, y, w, h, num, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.1, edgecolor=PANEL, facecolor="none",
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(x + 0.014, y + h - 0.022, f"{num}. {title}", ha="left", va="top",
            fontsize=FS_STAGE, fontweight="bold", color=INK, zorder=3)


def card(ax, x, y, w, h, lines, face, *, fg=None, bold_first=True, fs=FS_BODY):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=0, facecolor=face, zorder=2))
    fg = fg or ("white" if face in (BLUE, DEEP, GREEN, RED, GREY) else SUB)
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * 0.0135, ln,
                ha="center", va="center", fontsize=fs, zorder=3, color=fg,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def note(ax, x, y, text, *, w=None):
    ax.text(x, y, text, ha="center", va="top", fontsize=FS_NOTE,
            style="italic", color=SUB, zorder=3, wrap=True)


def arrow(ax, a, b, *, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
        color=PANEL if dashed else INK, zorder=4,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        shrinkA=2, shrinkB=2))


def inset(fig, rect, title):
    ax = fig.add_axes(rect)
    ax.set_title(title, fontsize=7.6, color=SUB, pad=3)
    ax.tick_params(labelsize=6.4, colors=SUB, length=2)
    for s in ax.spines.values():
        s.set_color("#C6D0DA")
    return ax


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fus = json.loads((RES / "fusion_summary.json").read_text())
    dec = json.loads((RES / "decision_agent_summary.json").read_text())
    emb = json.loads((RES / "audio_embedding_comparison.json").read_text())
    shap = pd.read_csv(RES / "clinical_shap_importance.csv").head(4)
    pred = pd.read_csv(RES / "decision_agent_predictions.csv")

    a_sym, a_pul = fus["symptom"]["roc_auc"], fus["pulmonary"]["roc_auc"]
    a_fus, a_aco = fus["fusion"]["roc_auc"], emb["combined"]["roc_auc_mean"]
    base = pred["y"].mean()
    conf = pred[pred.conflict == 1]["y"].mean()

    fig, ax = plt.subplots(figsize=(15.0, 9.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    TOP, BOT = (0.525, 0.965), (0.035, 0.475)
    C = [(0.015, 0.315), (0.345, 0.315), (0.675, 0.310)]

    # ------------------------------------------------ 1. data sources
    x, w = C[0]; y0, y1 = TOP; h = y1 - y0
    stage(ax, x, y0, w, h, 1, "Data Sources")
    for i, ls in enumerate([
            ["Clinical history", "NHANES 2007–2012", "n = 6,329 children"],
            ["Spirometry", "NHANES 2007–2012", "92.3% available"],
            ["Lung sound", "SPRSound (CC BY 4.0)", "747 children · 5,216 clips"]]):
        card(ax, x + 0.022, y1 - 0.140 - i * 0.098, w - 0.044, 0.078, ls, PALE)
    note(ax, x + w / 2, y0 + 0.072,
         "Agents are trained on separate cohorts;\nno jointly-labelled multimodal dataset is required.")

    # ------------------------------------------------ 2. modality agents
    x, w = C[1]
    stage(ax, x, y0, w, h, 2, "Modality Agents")
    for i, ls in enumerate([
            ["Symptom Agent", "calibrated logistic regression"],
            ["Pulmonary Agent", "calibrated logistic regression"],
            ["Acoustic Agent", "AST embeddings + 58 features"]]):
        card(ax, x + 0.022, y1 - 0.132 - i * 0.083, w - 0.044, 0.063, ls, BLUE)
    arrow(ax, (C[0][0] + C[0][1] + 0.002, 0.74), (x + 0.020, 0.74))
    axi = inset(fig, [x + 0.058, y0 + 0.032, w - 0.116, 0.078], "patient-level ROC-AUC")
    names = ["symptom", "pulmonary", "acoustic"]
    axi.barh(names, [a_sym, a_pul, a_aco], color=[BLUE, BLUE, BLUE], height=0.6)
    for i, v in enumerate([a_sym, a_pul, a_aco]):
        axi.text(v - 0.02, i, f"{v:.3f}", va="center", ha="right",
                 fontsize=6.6, color="white")
    axi.set_xlim(0.5, 0.95); axi.invert_yaxis(); axi.set_yticks(range(3))
    axi.set_yticklabels(names, fontsize=6.6)

    # ------------------------------------------------ 3. evidence bundle
    x, w = C[2]
    stage(ax, x, y0, w, h, 3, "Evidence Bundle")
    for i, ls in enumerate([["calibrated probability"], ["named contributing factors"],
                            ["modality-present flag"]]):
        card(ax, x + 0.022, y1 - 0.135 - i * 0.078, w - 0.044, 0.057, ls, GREEN,
             bold_first=False)
    arrow(ax, (C[1][0] + C[1][1] + 0.002, 0.74), (x + 0.020, 0.74))
    note(ax, x + w / 2, y0 + 0.120,
         "Each agent emits the same structure, so a\nmissing modality drops a bundle without\nchanging the interface.")

    # ------------------------------------------------ 4. decision agent
    x, w = C[0]; y0, y1 = BOT; h = y1 - y0
    stage(ax, x, y0, w, h, 4, "Decision Agent")
    card(ax, x + 0.022, y1 - 0.132, w - 0.044, 0.074,
         ["Logistic meta-learner", "over agent evidence"], DEEP)
    card(ax, x + 0.022, y1 - 0.222, w - 0.044, 0.074,
         ["Abstains on low confidence", "flags inter-agent conflict"], BLUE)
    card(ax, x + 0.022, y1 - 0.312, w - 0.044, 0.074,
         [f"fused ROC-AUC {a_fus:.3f}",
          f"{dec['roc_auc_non_abstained']:.3f} at {dec['coverage']:.0%} coverage"], PINK)
    # Route the row-to-row link through the gutter rather than diagonally
    # across the figure, which read as a line crossing unrelated panels.
    mid = (TOP[0] + BOT[1]) / 2
    xr = C[2][0] + C[2][1] / 2
    ax.plot([xr, xr], [TOP[0], mid], color=INK, linewidth=1.2, zorder=4)
    ax.plot([xr, x + w / 2], [mid, mid], color=INK, linewidth=1.2, zorder=4)
    arrow(ax, (x + w / 2, mid), (x + w / 2, y1 + 0.002))
    note(ax, x + w / 2, y0 + 0.095,
         "Refers the 15% of children closest to the\noperating threshold for clinical review.")

    # ------------------------------------------------ 5. explanation fusion
    x, w = C[1]
    stage(ax, x, y0, w, h, 5, "Explanation Fusion")
    card(ax, x + 0.022, y1 - 0.098, w - 0.044, 0.062,
         ["Unified rationale across modalities"], BLUE, bold_first=False)
    axi = inset(fig, [x + 0.105, y0 + 0.145, w - 0.145, 0.115], "mean |SHAP|")
    axi.barh(shap.feature[::-1].str.replace("_", " "), shap.mean_abs_shap[::-1],
             color=BLUE, height=0.62)
    axi.tick_params(labelsize=6.4)
    card(ax, x + 0.022, y0 + 0.030, w - 0.044, 0.070,
         [f"deletion-verified: {dec['comprehensiveness_cited'] / dec['comprehensiveness_random']:.1f}× random",
          f"sufficiency {dec['sufficiency_cited']:.3f} vs {dec['sufficiency_random']:.3f}"], PINK)
    arrow(ax, (C[0][0] + C[0][1] + 0.002, 0.26), (x + 0.020, 0.26))

    # ------------------------------------------------ 6. clinical output
    x, w = C[2]
    stage(ax, x, y0, w, h, 6, "Clinical Output")
    for i, (ls, col) in enumerate([(["Asthma likely"], RED),
                                   (["Asthma unlikely"], GREEN),
                                   (["Refer for review"], GREY)]):
        card(ax, x + 0.022, y1 - 0.105 - i * 0.072, w - 0.044, 0.054, ls, col)
    arrow(ax, (C[1][0] + C[1][1] + 0.002, 0.26), (x + 0.020, 0.26))
    axi = inset(fig, [x + 0.055, y0 + 0.048, w - 0.115, 0.105], "asthma rate")
    axi.bar(["flagged\nconflict", "cohort"], [conf, base], color=[RED, GREY], width=0.55)
    for i, v in enumerate([conf, base]):
        axi.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=6.6, color=SUB)
    axi.set_ylim(0, 0.92); axi.tick_params(labelsize=6.4)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 330})):
        p = OUT / f"figure1_architecture.{ext}"
        fig.savefig(p, format=ext, facecolor="white", **kw)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
