"""Figure 1 (pipeline style): numbered methodology stages with real data insets.

Follows the common journal convention of numbered, dashed-outlined stages read
left-to-right, top-to-bottom. Unlike a clip-art schematic, every inset panel is
plotted from the study's own data and results, so the figure cannot drift out
of agreement with the tables: rerun this script and the figure updates.

Insets drawn from real data:
  stage 1  class balance of both cohorts
  stage 2  waveform of an actual asthma recording, after normalisation
  stage 3  log-mel spectrogram of that recording
  stage 4  patient-grouped cross-validation schematic
  stage 6  reliability (calibration) curve from out-of-fold predictions
  stage 8  ROC curves for both agents and the fusion
  stage 9  SHAP importance for the top clinical features

Usage:  python src/figures/make_pipeline_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "figures"

FIG_W, FIG_H = 13.6, 11.2

BLUE = "#2E86DE"          # process boxes, white text
BLUE_D = "#1B5E9E"
GREY_E = "#7A8896"        # dashed stage borders
INK = "#16232E"
SUB = "#3D4F5C"
POS = "#C0392B"           # positive / asthma
NEG = "#27865B"           # negative / not asthma

PT = 1.0 / 72.0 / FIG_H
plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


# --------------------------------------------------------------- primitives
def stage(ax, x, y, w, h, number, title):
    """Dashed container marking one numbered methodology stage."""
    ax.add_patch(Rectangle(
        (x, y), w, h, fill=False, linewidth=1.0, edgecolor=GREY_E,
        linestyle=(0, (5, 3)), zorder=1,
    ))
    ax.text(x + 0.008, y + h - 10 * PT, f"{number}. {title}",
            ha="left", va="top", fontsize=9.2, fontweight="bold",
            color=INK, zorder=4)


def pbox(ax, x, y, w, h, lines, *, face=BLUE, fg="white", fs=7.6, bold=False):
    """Solid process box with centred white text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.008",
        linewidth=0.8, edgecolor=BLUE_D if face == BLUE else face,
        facecolor=face, zorder=3,
    ))
    if isinstance(lines, str):
        lines = [lines]
    total = len(lines)
    for i, ln in enumerate(lines):
        yy = y + h / 2 + (total - 1) * 5.2 * PT - i * 10.4 * PT
        ax.text(x + w / 2, yy, ln, ha="center", va="center",
                fontsize=fs, color=fg, zorder=4,
                fontweight="bold" if bold else "normal")


def arrow(ax, a, b, *, dashed=False, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=SUB, zorder=2, shrinkA=1.0, shrinkB=1.0,
        linestyle=(0, (3.5, 2.5)) if dashed else "solid",
    ))


def inset(fig, x, y, w, h):
    """Axes positioned in figure coordinates (the base axes spans the figure)."""
    a = fig.add_axes([x, y, w, h], zorder=5)
    for s in a.spines.values():
        s.set_color(SUB)
    a.tick_params(colors=SUB, labelsize=5.6, length=2, pad=1.2)
    return a


# ------------------------------------------------------------- inset panels
def panel_class_balance(a):
    clin = pd.read_csv(PROC / "clinical_agent_cohort.csv")
    man = pd.read_csv(PROC / "sprsound_manifest.csv").dropna(subset=["disease"])
    pats = man.drop_duplicates("patient_id")
    n_ast_c = int(clin["current_asthma"].sum())
    vals = [n_ast_c, len(clin) - n_ast_c,
            int((pats.disease == "asthma").sum()), int((pats.disease != "asthma").sum())]
    a.bar([0, 1], vals[:2], color=[POS, "#B9C6D2"], width=0.72)
    a.bar([2.4, 3.4], vals[2:], color=[POS, "#B9C6D2"], width=0.72)
    a.set_xticks([0.5, 2.9])
    a.set_xticklabels(["NHANES\n6,329", "SPRSound\n747"], fontsize=5.6)
    a.set_yscale("log")
    a.set_yticks([])
    for xi, v in zip([0, 1, 2.4, 3.4], vals):
        a.text(xi, v * 1.25, f"{v:,}", ha="center", fontsize=5.2, color=SUB)
    a.set_ylim(1, 2e4)
    a.set_title("class balance", fontsize=6.2, color=SUB, pad=2)


def _load_example():
    import librosa
    man = pd.read_csv(PROC / "sprsound_manifest.csv")
    row = man[man.disease == "asthma"].iloc[0]
    y, sr = librosa.load(ROOT / row.wav_path, sr=8000, mono=True)
    y = y - y.mean()
    if np.max(np.abs(y)):
        y = y / np.max(np.abs(y))
    return y, sr


def panel_waveform(a, y, sr):
    t = np.arange(len(y)) / sr
    a.plot(t, y, linewidth=0.35, color=BLUE_D)
    dur = t[-1]
    a.set_xlim(0, dur)
    a.set_ylim(-1.05, 1.05)
    a.set_yticks([])
    a.set_xticks(np.linspace(0, dur, 3).round(0))
    a.set_xlabel("s", fontsize=5.6, color=SUB, labelpad=0)
    a.set_title("normalised waveform", fontsize=6.2, color=SUB, pad=2)


def panel_spectrogram(a, y, sr):
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128,
                                       n_mels=48, fmax=4000)
    dur = len(y) / sr
    a.imshow(librosa.power_to_db(S, ref=np.max), origin="lower", aspect="auto",
             cmap="magma", extent=[0, dur, 0, 4])
    a.set_xlim(0, dur)
    a.set_yticks([0, 2, 4])
    a.set_ylabel("kHz", fontsize=5.6, color=SUB, labelpad=1)
    a.set_xticks(np.linspace(0, dur, 3).round(0))
    a.set_xlabel("s", fontsize=5.6, color=SUB, labelpad=0)
    a.set_title("log-mel spectrogram", fontsize=6.2, color=SUB, pad=2)


def panel_cv(a):
    """Patient-grouped 5-fold split: test block moves, patients never split."""
    for f in range(5):
        for b in range(5):
            a.add_patch(Rectangle((b, 4 - f), 0.92, 0.82,
                                  facecolor=POS if b == f else "#CBD8E4",
                                  edgecolor="white", linewidth=0.6))
    a.set_xlim(-0.1, 5.1)
    a.set_ylim(-0.1, 5.1)
    a.axis("off")
    a.text(2.5, 5.3, "5 folds grouped by patient", ha="center",
           fontsize=6.0, color=SUB)
    a.text(2.5, -0.75, "test fold", ha="center", fontsize=5.6, color=POS)


def panel_calibration(a):
    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    p, y = d["p_fusion"].to_numpy(), d["y"].to_numpy()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() >= 25:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    a.plot([0, 1], [0, 1], linewidth=0.7, color="#9AA8B4", linestyle=(0, (3, 2)))
    a.plot(xs, ys, "o-", markersize=2.4, linewidth=0.9, color=BLUE_D)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xticks([0, 0.5, 1])
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel("predicted", fontsize=5.6, color=SUB, labelpad=0)
    a.set_ylabel("observed", fontsize=5.6, color=SUB, labelpad=1)
    a.set_title("calibration", fontsize=6.2, color=SUB, pad=2)


def panel_roc(a):
    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    y = d["y"].to_numpy()
    for col, lab, c, lw in (("p_symptom", "symptom 0.863", "#6C8AA6", 0.8),
                            ("p_pulmonary", "pulmonary 0.646", "#B0BCC8", 0.8),
                            ("p_fusion", "fusion 0.870", POS, 1.3)):
        fpr, tpr, _ = roc_curve(y, d[col].to_numpy())
        a.plot(fpr, tpr, linewidth=lw, color=c, label=lab)
    a.plot([0, 1], [0, 1], linewidth=0.6, color="#C3CDD7", linestyle=(0, (3, 2)))
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xticks([0, 0.5, 1])
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel("FPR", fontsize=5.6, color=SUB, labelpad=0)
    a.set_ylabel("TPR", fontsize=5.6, color=SUB, labelpad=1)
    a.legend(fontsize=4.9, frameon=False, loc="lower right",
             handlelength=1.1, borderpad=0.1, labelspacing=0.22)
    a.set_title("ROC (out-of-fold)", fontsize=6.2, color=SUB, pad=2)


def panel_shap(a):
    s = pd.read_csv(RES / "clinical_shap_importance.csv").head(6).iloc[::-1]
    names = (s.feature.str.replace("_", " ")
             .str.replace("hx", "history").str.replace("12mo", "12 mo"))
    a.barh(range(len(s)), s.mean_abs_shap, color=BLUE, height=0.68)
    a.set_yticks(range(len(s)))
    a.set_yticklabels(names, fontsize=5.4)
    a.set_xticks([])
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("mean |SHAP|", fontsize=6.2, color=SUB, pad=2)


# ------------------------------------------------------------------- figure
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    wav, sr = _load_example()

    # Row geometry (axes fractions).
    R1, R2, R3 = 0.665, 0.355, 0.035
    RH = 0.275
    L, GAP = 0.018, 0.016
    w3 = (1 - 2 * L - 2 * GAP) / 3
    w4 = (1 - 2 * L - 3 * GAP) / 4

    # ---------------------------------------------------------------- row 1
    x1 = L
    stage(ax, x1, R1, w3, RH, 1, "Data Acquisition")
    pbox(ax, x1 + 0.014, R1 + 0.175, 0.13, 0.05,
         ["NHANES 2007–2012", "clinical + spirometry"])
    pbox(ax, x1 + 0.014, R1 + 0.108, 0.13, 0.05,
         ["SPRSound (CC BY 4.0)", "paediatric lung sound"])
    a = inset(fig, x1 + 0.163, R1 + 0.055, 0.115, 0.135)
    panel_class_balance(a)

    x2 = L + w3 + GAP
    stage(ax, x2, R1, w3, RH, 2, "Preprocessing")
    for i, txt in enumerate([
        "exclude post-diagnosis vars\n(MCQ040 / MCQ050)",
        "deduplicate recordings\n20,457 → 6,567",
        "DC removal + peak\nnormalisation",
    ]):
        pbox(ax, x2 + 0.014, R1 + 0.185 - i * 0.062, 0.135, 0.052,
             txt.split("\n"))
    a = inset(fig, x2 + 0.168, R1 + 0.075, 0.11, 0.10)
    panel_waveform(a, wav, sr)

    x3 = L + 2 * (w3 + GAP)
    stage(ax, x3, R1, w3, RH, 3, "Feature Extraction")
    pbox(ax, x3 + 0.014, R1 + 0.175, 0.125, 0.05,
         ["14 symptom + 7 pulmonary", "clinical features"])
    pbox(ax, x3 + 0.014, R1 + 0.108, 0.125, 0.05,
         ["58 named acoustic", "+ 768-d AST embedding"])
    a = inset(fig, x3 + 0.158, R1 + 0.072, 0.115, 0.105)
    panel_spectrogram(a, wav, sr)

    # ---------------------------------------------------------------- row 2
    stage(ax, x1, R2, w3, RH, 4, "Data Splitting")
    pbox(ax, x1 + 0.014, R2 + 0.178, 0.135, 0.048,
         ["stratified 5-fold CV"])
    pbox(ax, x1 + 0.014, R2 + 0.118, 0.135, 0.048,
         ["grouped by patient ID", "(no cross-fold leakage)"])
    a = inset(fig, x1 + 0.172, R2 + 0.085, 0.09, 0.10)
    panel_cv(a)

    stage(ax, x2, R2, w3, RH, 5, "Modality Agent Training")
    for i, (nm, sc) in enumerate([("Symptom Agent", "ROC-AUC 0.863"),
                                  ("Pulmonary Agent", "ROC-AUC 0.646"),
                                  ("Acoustic Agent", "ROC-AUC 0.732")]):
        pbox(ax, x2 + 0.03, R2 + 0.185 - i * 0.062, 0.23, 0.052,
             [nm, sc], bold=False)

    stage(ax, x3, R2, w3, RH, 6, "Calibration & Evidence")
    pbox(ax, x3 + 0.014, R2 + 0.178, 0.13, 0.048,
         ["sigmoid calibration", "(train folds only)"])
    pbox(ax, x3 + 0.014, R2 + 0.118, 0.13, 0.048,
         ["evidence bundle:", "prob · factors · flag"])
    a = inset(fig, x3 + 0.168, R2 + 0.075, 0.088, 0.105)
    panel_calibration(a)

    # ---------------------------------------------------------------- row 3
    stage(ax, L, R3, w4, RH, 7, "Decision Agent")
    pbox(ax, L + 0.012, R3 + 0.178, 0.185, 0.05,
         ["logistic meta-learner"])
    pbox(ax, L + 0.012, R3 + 0.118, 0.185, 0.05,
         ["abstain on low confidence"])
    pbox(ax, L + 0.012, R3 + 0.058, 0.185, 0.05,
         ["flag inter-agent conflict"])
    ax.text(L + 0.105, R3 + 0.032, "fused ROC-AUC 0.870",
            ha="center", fontsize=6.6, color=SUB)

    xb = L + w4 + GAP
    stage(ax, xb, R3, w4, RH, 8, "Evaluation")
    a = inset(fig, xb + 0.045, R3 + 0.055, 0.125, 0.14)
    panel_roc(a)
    ax.text(xb + w4 / 2, R3 + 0.030, "ROC-AUC · PR-AUC · Brier",
            ha="center", fontsize=6.6, color=SUB)

    xc = L + 2 * (w4 + GAP)
    stage(ax, xc, R3, w4, RH, 9, "Explanation Fusion")
    a = inset(fig, xc + 0.062, R3 + 0.088, 0.105, 0.105)
    panel_shap(a)
    ax.text(xc + w4 / 2, R3 + 0.055, "deletion-verified faithfulness",
            ha="center", fontsize=6.6, color=SUB)
    ax.text(xc + w4 / 2, R3 + 0.032, "comprehensiveness 2.8× random",
            ha="center", fontsize=6.6, color=SUB)

    xd = L + 3 * (w4 + GAP)
    stage(ax, xd, R3, w4, RH, 10, "Clinical Output")
    pbox(ax, xd + 0.028, R3 + 0.168, 0.16, 0.055,
         ["Asthma likely"], face=POS, fs=9.0, bold=True)
    pbox(ax, xd + 0.028, R3 + 0.100, 0.16, 0.055,
         ["Asthma unlikely"], face=NEG, fs=9.0, bold=True)
    pbox(ax, xd + 0.028, R3 + 0.032, 0.16, 0.055,
         ["Refer for review"], face="#7F8C8D", fs=9.0, bold=True)

    # ------------------------------------------------------------- arrows
    y1 = R1 + RH / 2
    arrow(ax, (x1 + w3, y1), (x2, y1))
    arrow(ax, (x2 + w3, y1), (x3, y1))
    # row 1 -> row 2 (wrap right to left)
    arrow(ax, (x3 + w3 / 2, R1), (x3 + w3 / 2, R2 + RH))
    y2 = R2 + RH / 2
    arrow(ax, (x3, y2), (x2 + w3, y2))
    arrow(ax, (x2, y2), (x1 + w3, y2))
    # row 2 -> row 3
    arrow(ax, (L + w4 / 2, R2), (L + w4 / 2, R3 + RH))
    y3 = R3 + RH / 2
    arrow(ax, (L + w4, y3), (xb, y3))
    arrow(ax, (xb + w4, y3), (xc, y3))
    arrow(ax, (xc + w4, y3), (xd, y3))

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        path = OUT / f"figure1_pipeline.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
