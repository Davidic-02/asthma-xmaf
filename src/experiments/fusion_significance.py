"""Is the fusion gain distinguishable from zero, or from the flat baseline?

Editor's point: fusion improves the Symptom Agent by 0.863 -> 0.871, and a flat
model over the pooled feature table reaches 0.867. If those differences are
inside sampling noise, the architecture cannot be defended on discrimination and
the paper must say so explicitly rather than imply otherwise.

Paired bootstrap over patients: every resample scores all models on the SAME
patients, so the comparison removes between-patient variance and is far tighter
than comparing two independent confidence intervals.

Usage:  python src/experiments/fusion_significance.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SEED = 42
N_BOOT = 10000


def main():
    d = pd.read_csv(RES / "fusion_oof_predictions.csv")
    y = d.y.to_numpy()
    models = {"symptom": d.p_symptom.to_numpy(), "pulmonary": d.p_pulmonary.to_numpy(),
              "fusion": d.p_fusion.to_numpy(), "flat": d.p_flat.to_numpy()}

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    idx = [i for i in idx if len(np.unique(y[i])) > 1]

    auc = {k: np.array([roc_auc_score(y[i], v[i]) for i in idx]) for k, v in models.items()}
    out = {"n_boot": len(idx), "n": int(len(y)), "positives": int(y.sum())}

    print(f"paired bootstrap, {len(idx):,} resamples of {len(y):,} patients\n")
    for k, v in models.items():
        lo, hi = np.percentile(auc[k], [2.5, 97.5])
        obs = roc_auc_score(y, v)
        out[k] = {"roc_auc": float(obs), "ci": [float(lo), float(hi)],
                  "pr_auc": float(average_precision_score(y, v))}
        print(f"  {k:10s} AUC {obs:.3f}  95% CI {lo:.3f}-{hi:.3f}")

    print("\npaired differences (positive favours the first model):")
    out["contrasts"] = {}
    for a, b in [("fusion", "symptom"), ("fusion", "flat"), ("flat", "symptom")]:
        diff = auc[a] - auc[b]
        obs = roc_auc_score(y, models[a]) - roc_auc_score(y, models[b])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        # two-sided bootstrap p: proportion of resamples on the far side of 0
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        p = float(min(p, 1.0))
        sig = "significant" if lo > 0 or hi < 0 else "NOT distinguishable from zero"
        out["contrasts"][f"{a}_minus_{b}"] = {
            "delta": float(obs), "ci": [float(lo), float(hi)], "p": p, "significant": bool(lo > 0 or hi < 0)}
        print(f"  {a:7s} - {b:8s} {obs:+.4f}  95% CI {lo:+.4f} to {hi:+.4f}  p = {p:.3f}  [{sig}]")

    (RES / "fusion_significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/fusion_significance.json")


if __name__ == "__main__":
    main()
