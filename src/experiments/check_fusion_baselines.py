"""Verify exact OOF fusion baselines used in the manuscript."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def main():
    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    y = d["y"].to_numpy()
    average = (d["p_symptom"] + d["p_pulmonary"]) / 2
    auc = {
        "symptom": float(roc_auc_score(y, d["p_symptom"])),
        "simple_average": float(roc_auc_score(y, average)),
        "fusion": float(roc_auc_score(y, d["p_fusion"])),
        "flat": float(roc_auc_score(y, d["p_flat"])),
    }
    out = {
        **auc,
        "simple_average_minus_symptom": auc["simple_average"] - auc["symptom"],
        "fusion_minus_simple_average": auc["fusion"] - auc["simple_average"],
        "rounding_note": (
            "The symptom and simple-average AUCs both round to 0.863 at "
            "three decimals but are not equal."
        ),
    }
    (RES / "fusion_baseline_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
