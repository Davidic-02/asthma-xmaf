"""Generate the print-size evidence-flow architecture figure."""
from pathlib import Path

from figlib import *

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"


def main():
    fig, ax = canvas(6.5, 5.2)

    panel(ax, 0.035, 0.515, 0.93, 0.40,
          "A  Evaluated jointly: NHANES, same children", fs=8.5)
    box(ax, 0.065, 0.705, 0.22, 0.095,
        ["Symptom Agent", "calibrated logistic"], BLUE, fs=7.4)
    box(ax, 0.065, 0.575, 0.22, 0.095,
        ["Pulmonary Agent", "calibrated logistic"], BLUE, fs=7.4)
    box(ax, 0.345, 0.59, 0.22, 0.19,
        ["Evidence bundle", "probability + factors", "modality flag"], GREEN, fs=7.1)
    box(ax, 0.625, 0.59, 0.30, 0.19,
        ["Decision Agent", "meta-learner", "referral + conflict flag"], DEEP, fs=7.1)
    arrow(ax, (0.285, 0.752), (0.345, 0.69), lw=1.0)
    arrow(ax, (0.285, 0.622), (0.345, 0.66), lw=1.0)
    arrow(ax, (0.565, 0.685), (0.625, 0.685), lw=1.0)
    box(ax, 0.625, 0.535, 0.30, 0.035,
        ["Asthma likely / unlikely / refer"], RED, fs=6.7)

    panel(ax, 0.035, 0.075, 0.93, 0.32,
          "B  Evaluated independently: SPRSound, different children", fs=8.5)
    box(ax, 0.065, 0.17, 0.22, 0.115,
        ["Acoustic Agent", "58 descriptors +", "AST embedding"], BLUE, fs=7.2)
    box(ax, 0.345, 0.17, 0.22, 0.115,
        ["Evidence bundle", "calibrated probability", "named factors"], GREEN, fs=7.0)
    box(ax, 0.625, 0.17, 0.30, 0.115,
        ["Independent evaluation", "patient-grouped five-fold CV"], "#F6DCE2", fs=7.0)
    arrow(ax, (0.285, 0.227), (0.345, 0.227), lw=1.0)
    arrow(ax, (0.565, 0.227), (0.625, 0.227), lw=1.0)
    box(ax, 0.30, 0.015, 0.40, 0.035,
        ["Possible by design, not evaluated: no paired cohort"],
        "white", fs=6.8, fg=SUB, edge=GREY, dashed=True)
    save(fig, OUT, "figure2_architecture_v2")


if __name__ == "__main__":
    main()
