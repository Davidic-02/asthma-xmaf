"""Generate Figures 3-6 (results) from the stored result files.

Every value is read from results/ rather than typed in, so the figures cannot
disagree with the tables. Regenerate after any experiment changes.

  Figure 3  clinical agents and fusion: ROC, precision-recall, calibration
  Figure 4  clinical attribution: SHAP importance and odds ratios
  Figure 5  acoustic agent: representation comparison and two-stage ablation
  Figure 6  decision agent: risk-coverage, conflict subgroup, faithfulness

Usage:  python src/figures/make_results_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"

BLUE = "#2E86DE"
DARK = "#1B5E9E"
RED = "#C0392B"
GREEN = "#27865B"
GREY = "#8C9AA6"
SUB = "#3D4F5C"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": SUB,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "text.color": "#16232E",
    "axes.labelcolor": SUB,
    "legend.frameon": False,
    "figure.dpi": 110,
})


def tidy(ax, *, hide=("top", "right")):
    for s in hide:
        ax.spines[s].set_visible(False)
    return ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, format=ext, bbox_inches="tight", facecolor="white", **kw)
    print(f"wrote {OUT / name}.pdf / .png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def figure3():
    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    y = d["y"].to_numpy()
    series = [
        ("p_symptom", "Symptom Agent", GREY, 1.2, "-"),
        ("p_pulmonary", "Pulmonary Agent", "#B9C6D2", 1.2, "-"),
        ("p_flat", "Flat pooled features", "#6C8AA6", 1.2, (0, (4, 2))),
        ("p_fusion", "Fusion (Decision Agent)", RED, 1.9, "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    for col, lab, c, lw, ls in series:
        p = d[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, linewidth=lw, linestyle=ls,
                label=f"{lab} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#C9D3DC", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(a) ROC, out-of-fold")
    ax.legend(loc="lower right", fontsize=7.4)
    tidy(ax)

    ax = axes[1]
    prev = y.mean()
    for col, lab, c, lw, ls in series:
        p = d[col].to_numpy()
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=c, linewidth=lw, linestyle=ls,
                label=f"{lab} ({average_precision_score(y, p):.3f})")
    ax.axhline(prev, color="#C9D3DC", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(0.02, prev + 0.02, f"prevalence {prev:.3f}", fontsize=7, color=SUB)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision–recall")
    ax.legend(loc="upper right", fontsize=7.4)
    tidy(ax)

    # Reliability: equal-count bins, so each point rests on similar support.
    ax = axes[2]
    for col, lab, c in (("p_flat", "Flat pooled features", "#6C8AA6"),
                        ("p_fusion", "Fusion", RED)):
        p = d[col].to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() >= 25:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=c, markersize=3.4, linewidth=1.3, label=lab)
    ax.plot([0, 1], [0, 1], color="#C9D3DC", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("(c) Calibration")
    ax.legend(loc="upper left", fontsize=7.4)
    tidy(ax)

    fig.tight_layout()
    save(fig, "figure3_clinical_performance")


# ---------------------------------------------------------------- figure 4
def figure4():
    shap = pd.read_csv(RES / "clinical_shap_importance.csv").head(10).iloc[::-1]
    ors = pd.read_csv(RES / "clinical_odds_ratios.csv")

    # NHANES RIDRETH1 codes are meaningless to a reader; name the categories.
    ETHNICITY = {
        "ethnicity_1.0": "Mexican American",
        "ethnicity_2.0": "Other Hispanic",
        "ethnicity_3.0": "Non-Hispanic White",
        "ethnicity_4.0": "Non-Hispanic Black",
        "ethnicity_5.0": "Other/multiracial",
    }
    RENAME = {
        "wheeze_12mo": "wheeze, past 12 months",
        "wheeze_attacks": "wheeze attack count",
        "family_hx_asthma": "family history of asthma",
        "exercise_wheeze": "exercise-induced wheeze",
        "nocturnal_dry_cough": "nocturnal dry cough",
        "household_smoker": "any household smoker",
        "n_household_smokers": "number of household smokers",
        "income_poverty_ratio": "income-to-poverty ratio",
        "household_size": "household size",
        "bmi": "BMI",
        "age": "age",
        "sex": "sex",
    }

    def pretty(s):
        return s.map(lambda k: ETHNICITY.get(k, RENAME.get(k, k.replace("_", " "))))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.barh(range(len(shap)), shap.mean_abs_shap, color=BLUE, height=0.7)
    ax.set_yticks(range(len(shap)))
    ax.set_yticklabels(pretty(shap.feature), fontsize=8)
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("(a) Symptom Agent feature importance")
    tidy(ax)

    # Show the odds ratios that move the decision most, both directions.
    ax = axes[1]
    o = ors.reindex(ors.odds_ratio.sub(1).abs().sort_values(ascending=False).index)
    o = o.head(8).iloc[::-1]
    colors = [RED if v > 1 else GREEN for v in o.odds_ratio]
    ax.barh(range(len(o)), o.odds_ratio, color=colors, height=0.7)
    ax.axvline(1.0, color=SUB, linewidth=0.9)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels(pretty(o.feature), fontsize=8)
    ax.set_xlabel("Odds ratio (>1 increases asthma odds)")
    ax.set_title("(b) Adjusted odds ratios")
    for i, v in enumerate(o.odds_ratio):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=7, color=SUB)
    ax.set_xlim(0, max(2.9, o.odds_ratio.max() * 1.18))
    tidy(ax)

    fig.tight_layout()
    save(fig, "figure5_clinical_attribution")


# ---------------------------------------------------------------- figure 5
def figure5():
    emb = json.loads((RES / "audio_embedding_comparison.json").read_text())
    two = json.loads((RES / "audio_2stage_summary.json").read_text())

    labels = {
        "handcrafted": "Handcrafted (58)",
        "embeddings": "AST embeddings (768)",
        "embeddings_pca32": "AST + PCA-32",
        "combined": "Combined",
        "combined_pca64": "Combined + PCA-64",
    }
    keys = list(labels)
    means = [emb[k]["roc_auc_mean"] for k in keys]
    stds = [emb[k]["roc_auc_std"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    best = int(np.argmax(means))
    cols = [RED if i == best else BLUE for i in range(len(keys))]
    ax.bar(range(len(keys)), means, yerr=stds, capsize=3.5, color=cols,
           width=0.62, error_kw=dict(ecolor=SUB, elinewidth=0.9))
    ax.axhline(0.5, color="#C9D3DC", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(len(keys) - 0.55, 0.508, "chance", fontsize=7, color=SUB, ha="right")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([labels[k] for k in keys], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Patient-level ROC-AUC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("(a) Acoustic representation (mean ± SD over folds)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.012, f"{m:.3f}", ha="center", fontsize=7.4, color=SUB)
    tidy(ax)

    ax = axes[1]
    names = ["Stage 1:\nCAS detection", "Stage 1:\nDAS detection",
             "Stage 2:\nevents only", "Single stage:\nacoustic features"]
    vals = [two["stage1_cas_auc_mean"], two["stage1_das_auc_mean"],
            two["stage2_roc_auc_mean"], emb["combined"]["roc_auc_mean"]]
    cols = [BLUE, BLUE, GREY, RED]
    ax.bar(range(4), vals, color=cols, width=0.6)
    ax.axhline(0.5, color="#C9D3DC", linewidth=0.9, linestyle=(0, (3, 2)))
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 0.9)
    ax.set_title("(b) Two-stage decomposition (negative result)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.4, color=SUB)
    tidy(ax)

    fig.tight_layout()
    save(fig, "figure6_acoustic_agent")


# ---------------------------------------------------------------- figure 6
def figure6():
    dec = pd.read_csv(RES / "decision_agent_predictions.csv")
    s = json.loads((RES / "decision_agent_summary.json").read_text())
    fus = pd.read_csv(RES / "fusion_oof_predictions.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

    # (a) risk-coverage: discrimination as the least certain cases are referred
    ax = axes[0]
    p, y = dec["p_fused"].to_numpy(), dec["y"].to_numpy()
    unc = np.abs(p - 0.5)
    order = np.argsort(-unc)          # most certain first
    covs, aucs = [], []
    for cov in np.linspace(0.5, 1.0, 26):
        k = max(int(cov * len(p)), 50)
        idx = order[:k]
        if len(np.unique(y[idx])) < 2:
            continue
        covs.append(k / len(p))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    ax.plot(covs, aucs, "-", color=RED, linewidth=1.6)
    ax.axvline(s["coverage"], color=SUB, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(s["coverage"] - 0.01, min(aucs) + 0.004,
            f"operating point\ncoverage {s['coverage']:.2f}",
            fontsize=7, color=SUB, ha="right")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("ROC-AUC on retained cases")
    ax.set_title("(a) Risk–coverage")
    tidy(ax)

    # (b) graceful degradation by modality availability
    ax = axes[1]
    groups = [("Spirometry\npresent", fus[fus.has_pulmonary == 1]),
              ("Spirometry\nabsent", fus[fus.has_pulmonary == 0])]
    xs = np.arange(2)
    wf = 0.36
    fusv = [roc_auc_score(g.y, g.p_fusion) for _, g in groups]
    symv = [roc_auc_score(g.y, g.p_symptom) for _, g in groups]
    ax.bar(xs - wf / 2, symv, width=wf, color=GREY, label="Symptom only")
    ax.bar(xs + wf / 2, fusv, width=wf, color=RED, label="Fusion")
    for x, v in zip(xs - wf / 2, symv):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.4, color=SUB)
    for x, v in zip(xs + wf / 2, fusv):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.4, color=SUB)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(n={len(g):,})" for n, g in groups], fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.75, 0.92)
    ax.set_title("(b) Graceful degradation")
    ax.legend(fontsize=7.4, loc="lower left")
    tidy(ax)

    # (c) explanation faithfulness against a random-feature control
    ax = axes[2]
    metrics = ["Comprehensiveness\n(higher better)", "Sufficiency\n(lower better)"]
    cited = [s["comprehensiveness_cited"], s["sufficiency_cited"]]
    rand = [s["comprehensiveness_random"], s["sufficiency_random"]]
    xs = np.arange(2)
    ax.bar(xs - wf / 2, cited, width=wf, color=RED, label="Cited features")
    ax.bar(xs + wf / 2, rand, width=wf, color=GREY, label="Random control")
    for x, v in zip(xs - wf / 2, cited):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.4, color=SUB)
    for x, v in zip(xs + wf / 2, rand):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.4, color=SUB)
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Prediction change")
    ax.set_title("(c) Explanation faithfulness")
    ax.legend(fontsize=7.4)
    tidy(ax)

    fig.tight_layout()
    save(fig, "figure7_decision_agent")


# ---------------------------------------------------------------- figure 7
def figure7():
    """Confusion matrices at two operating points.

    A single matrix at the default 0.5 cut-off understates a screening tool:
    at 12.7% prevalence that threshold favours specificity. We therefore show
    the default alongside a screening threshold chosen for 90% sensitivity,
    which is the trade-off a case-finding application actually makes.
    """
    from sklearn.metrics import confusion_matrix

    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    y, p = d["y"].to_numpy(), d["p_fusion"].to_numpy()

    fpr, tpr, thr = roc_curve(y, p)
    t90 = thr[int(np.argmin(np.abs(tpr - 0.90)))]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, (cut, title) in zip(axes, [
            (0.5, "(a) Default threshold (0.50)"),
            (t90, f"(b) Screening threshold ({t90:.2f})")]):
        yh = (p >= cut).astype(int)
        cm = confusion_matrix(y, yh)
        tn, fp, fn, tp = cm.ravel()
        # Row-normalised: each row shows how that true class was assigned.
        norm = cm / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r - 0.10, f"{cm[r, c]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if norm[r, c] > 0.55 else "#16232E")
                ax.text(c, r + 0.16, f"{norm[r, c]:.1%}", ha="center", va="center",
                        fontsize=9,
                        color="white" if norm[r, c] > 0.55 else SUB)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No asthma", "Asthma"])
        ax.set_yticklabels(["No asthma", "Asthma"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        ppv, npv = tp / (tp + fp), tn / (tn + fn)
        ax.text(0.5, -0.30,
                f"sensitivity {sens:.3f}   specificity {spec:.3f}   "
                f"PPV {ppv:.3f}   NPV {npv:.3f}",
                transform=ax.transAxes, ha="center", fontsize=8, color=SUB)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "figure4_confusion_matrix")


def main():
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()


if __name__ == "__main__":
    main()
