"""Generate the print-size two-lane pipeline figure."""
import json
from pathlib import Path

import pandas as pd

from figlib import *

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "paper" / "figures"


def main():
    meta = json.loads((RES / "clinical_agent_meta.json").read_text())
    n_nh, prev = meta["n"], meta["prevalence"]
    rec = pd.read_csv(RES / "audio_agent_oof_predictions.csv")
    n_rec, n_pat = len(rec), rec.patient_id.nunique()

    fig, ax = canvas(6.5, 6.2)
    x0, gap, w = 0.035, 0.018, 0.294
    xs = [x0 + i * (w + gap) for i in range(3)]

    def lane(y, title, stages):
        ax.text(x0, y + 0.425, title, fontsize=8.5, fontweight="bold",
                color=INK, ha="left", va="center")
        for i, (heading, boxes) in enumerate(stages):
            panel(ax, xs[i], y, w, 0.35, heading, fs=7.6)
            bh = 0.09
            start = y + 0.235
            for j, (lines, face) in enumerate(boxes):
                box(ax, xs[i] + 0.012, start - j * 0.115, w - 0.024, bh,
                    lines, face, fs=6.7)
            if i < 2:
                arrow(ax, (xs[i] + w, y + 0.175),
                      (xs[i + 1], y + 0.175), lw=1.0)

    lane(0.535,
         "A  NHANES clinical cohort: Symptom and Pulmonary Agents are fused",
         [
             ("1. Data", [
                 (["NHANES 2007-2012", f"{n_nh:,} children"], PALE),
                 (["Current asthma", f"prevalence {prev:.1%}"], PALE),
             ]),
             ("2. Features", [
                 (["Symptom", "14 reported items"], PALE),
                 (["Pulmonary", "7 lung-function items"], PALE),
             ]),
             ("3. Decision", [
                 (["Calibrated agents", "logistic models"], BLUE),
                 (["Meta-learner", "paired bootstrap"], DEEP),
             ]),
         ])
    lane(0.105,
         "B  SPRSound acoustic cohort: Acoustic Agent is evaluated independently",
         [
             ("1. Data", [
                 (["SPRSound", "paediatric lung sound"], PALE),
                 ([f"{n_rec:,} recordings", f"{n_pat} children"], PALE),
             ]),
             ("2. Features", [
                 (["58 handcrafted", "acoustic descriptors"], PALE),
                 (["768-d AST", "frozen embedding"], PALE),
             ]),
             ("3. Evaluation", [
                 (["Acoustic Agent", "calibrated logistic"], BLUE),
                 (["Patient-grouped", "five-fold CV"], "#F6DCE2"),
             ]),
         ])

    box(ax, 0.035, 0.018, 0.93, 0.045,
        ["No participant appears in both cohorts; acoustic evidence is not fused with clinical agents."],
        "white", fs=7.0, fg=SUB, edge=PANEL, dashed=True)
    save(fig, OUT, "figure1_pipeline_v2")


if __name__ == "__main__":
    main()
