"""Figure 2 (revised): evidence flow, showing exactly what was evaluated jointly.

Responds to the reviewer comment that the earlier figure implied the Symptom,
Pulmonary and Acoustic Agents were integrated in a single experiment. Here the
fused path (Symptom and Pulmonary Agents, same children in NHANES) and the
independent path (Acoustic Agent, different children in SPRSound) are drawn as
separate lanes, and the one connection between them is dashed and labelled as
not evaluated. Result insets are removed; results are given in Section 4.

Usage:  python src/figures/make_architecture_v2.py
"""
from pathlib import Path

from figlib import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"


def main():
    fig, ax = canvas(13, 7.4)

    # ---- lane A: evaluated jointly --------------------------------------
    panel(ax, 0.006, 0.445, 0.988, 0.520,
          "A   Evaluated jointly: NHANES, both modalities measured on the same children")
    box(ax, 0.030, 0.700, 0.190, 0.105, ["Symptom Agent", "calibrated logistic regression"], BLUE)
    box(ax, 0.030, 0.545, 0.190, 0.105, ["Pulmonary Agent", "calibrated logistic regression"], BLUE)
    box(ax, 0.290, 0.545, 0.175, 0.260,
        ["Evidence bundle", "from each agent:", "calibrated probability", "contributing factors",
         "modality-present flag"], GREEN, fs=10)
    arrow(ax, (0.220, 0.752), (0.290, 0.700))
    arrow(ax, (0.220, 0.597), (0.290, 0.650))

    panel(ax, 0.530, 0.480, 0.250, 0.405, None)
    ax.text(0.540, 0.870, "Decision Agent", fontsize=12.5, fontweight="bold",
            color=INK, ha="left", va="top")
    box(ax, 0.545, 0.745, 0.220, 0.082, ["Logistic meta-learner"], DEEP, fs=10)
    box(ax, 0.545, 0.640, 0.220, 0.082, ["Abstention band", "(15% nearest threshold)"], DEEP, fs=10)
    box(ax, 0.545, 0.535, 0.220, 0.082, ["Conflict flag", "(agents differ by > 0.35)"], DEEP, fs=10)
    arrow(ax, (0.465, 0.675), (0.530, 0.675))

    box(ax, 0.815, 0.690, 0.165, 0.150,
        ["Explanation", "top-3 Symptom Agent", "attributions + modality", "status + conflict note"],
        PALE, fs=9.5)
    box(ax, 0.815, 0.610, 0.165, 0.054, ["Asthma likely"], RED, fs=10)
    box(ax, 0.815, 0.545, 0.165, 0.054, ["Asthma unlikely"], GREEN, fs=10)
    box(ax, 0.815, 0.480, 0.165, 0.054, ["Refer for review"], GREY, fs=10)
    arrow(ax, (0.780, 0.700), (0.815, 0.760))
    arrow(ax, (0.780, 0.640), (0.815, 0.575))

    # ---- lane B: evaluated independently --------------------------------
    panel(ax, 0.006, 0.040, 0.988, 0.355,
          "B   Evaluated independently: SPRSound, different children")
    box(ax, 0.030, 0.095, 0.190, 0.150,
        ["Acoustic Agent", "58 descriptors +", "AST embedding"], BLUE, fs=10)
    box(ax, 0.290, 0.095, 0.175, 0.150,
        ["Evidence bundle", "calibrated probability", "named acoustic factors"], GREEN, fs=10)
    box(ax, 0.530, 0.095, 0.250, 0.150,
        ["Independent evaluation", "patient-grouped 5-fold CV", "patient-level ROC-AUC"],
        "#F6DCE2", fs=10)
    arrow(ax, (0.220, 0.170), (0.290, 0.170))
    arrow(ax, (0.465, 0.170), (0.530, 0.170))

    # the single, unevaluated connection
    arrow(ax, (0.450, 0.245), (0.600, 0.480), color=GREY, dashed=True, rad=-0.12)
    box(ax, 0.660, 0.262, 0.215, 0.092,
        ["Possible by design,", "not evaluated here:", "no paired cohort exists"],
        "white", fs=9.5, fg=SUB, edge=GREY, dashed=True)
    save(fig, OUT, "figure2_architecture_v2")


if __name__ == "__main__":
    main()
