"""Figure 1 (revised): data-processing and modelling pipeline, two lanes.

Responds to the reviewer comment that the earlier figure combined the
methodological pipeline with result visualisations. This version carries stages
only; every result is shown in the Results section instead. The two lanes are
the two cohorts, and the figure makes explicit that only lane A is fused.

Counts are read from results/ at build time.

Usage:  python src/figures/make_pipeline_figure_v2.py
"""
import json
from pathlib import Path

import pandas as pd

from figlib import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "paper" / "figures"


def main():
    meta = json.loads((RES / "clinical_agent_meta.json").read_text())
    n_nh, prev = meta["n"], meta["prevalence"]
    rec = pd.read_csv(RES / "audio_agent_oof_predictions.csv")
    n_rec, n_pat = len(rec), rec.patient_id.nunique()

    fig, ax = canvas(13, 6.9)
    W, G, X0 = 0.1495, 0.0176, 0.006
    xs = [X0 + k * (W + G) for k in range(6)]

    def lane(y, h, head, cols):
        ax.text(0.006, y + h + 0.025, head, fontsize=13, fontweight="bold",
                color=INK, ha="left", va="center")
        for k, (title, boxes) in enumerate(cols):
            panel(ax, xs[k], y, W, h, f"{k + 1}. {title}", fs=11.5)
            m = len(boxes)
            bh, pad = 0.105, 0.016
            tot = m * bh + (m - 1) * pad
            top = y + (h - 0.045 + tot) / 2 - 0.004
            for j, (lines, face) in enumerate(boxes):
                by = top - (j + 1) * bh - j * pad
                box(ax, xs[k] + 0.010, by, W - 0.020, bh, lines, face, fs=10)
        for k in range(5):
            arrow(ax, (xs[k] + W, y + h / 2 - 0.02), (xs[k + 1], y + h / 2 - 0.02))

    lane(0.585, 0.325,
         "A   NHANES clinical cohort: Symptom and Pulmonary Agents are fused",
         [("Data", [(["NHANES 2007-2012", f"children 6-17, n = {n_nh:,}"], PALE),
                    (["Current asthma", f"prevalence {prev:.1%}"], PALE)]),
          ("Preprocessing", [(["Exclude post-", "diagnosis variables"], PALE),
                             (["Median imputation,", "standardisation"], PALE)]),
          ("Modalities", [(["Symptom", "14 reported items"], PALE),
                          (["Pulmonary", "7 lung-function items"], PALE)]),
          ("Agent training", [(["Symptom Agent", "calibrated logistic"], BLUE),
                              (["Pulmonary Agent", "calibrated logistic"], BLUE)]),
          ("Decision Agent", [(["Meta-learner over", "agent probabilities"], DEEP),
                              (["Abstention, conflict", "flag, explanation"], DEEP)]),
          ("Evaluation", [(["Stratified 5-fold CV", "paired bootstrap"], PINK_ := "#F6DCE2"),
                          (["External validation", "NHANES 2015-2018"], PINK_)])])

    lane(0.115, 0.325,
         "B   SPRSound acoustic cohort: Acoustic Agent is evaluated independently",
         [("Data", [(["SPRSound", "paediatric lung sound"], PALE),
                    ([f"{n_rec:,} recordings", f"{n_pat} children"], PALE)]),
          ("Preprocessing", [(["Deduplicate", "recordings"], PALE),
                             (["DC removal, peak", "normalisation"], PALE)]),
          ("Features", [(["58 handcrafted", "acoustic descriptors"], PALE),
                        (["768-d AST", "embedding (frozen)"], PALE)]),
          ("Agent training", [(["Acoustic Agent", "calibrated logistic"], BLUE)]),
          ("Evidence", [(["Calibrated", "probability"], GREEN),
                        (["Named acoustic", "factors"], GREEN)]),
          ("Evaluation", [(["Patient-grouped", "5-fold CV"], "#F6DCE2"),
                          (["Patient-level", "aggregation"], "#F6DCE2")])])

    box(ax, 0.006, 0.012, 0.988, 0.062,
        ["No participant appears in both cohorts, so the Acoustic Agent is not fused "
         "with the clinical agents in this study."],
        "white", fs=11, fg=SUB, edge=PANEL, dashed=True)
    save(fig, OUT, "figure1_pipeline_v2")


if __name__ == "__main__":
    main()
