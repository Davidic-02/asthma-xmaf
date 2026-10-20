"""Temporal external validation of the Symptom Agent on NHANES 2015-2018.

Design
------
The development cohort (2007-2012, n=6,329) and the validation cohort
(2015-2018, n=4,111) share no participants and are separated by three to
eleven years of sampling. NHANES retired the RDQ respiratory module and SPX
spirometry after 2011-2012, so only the TRANSPORTABLE subset of variables
exists in both eras. To keep the comparison like-for-like, the development
model is RE-FITTED on that subset rather than the published 21-feature model
being partially applied.

Three numbers are therefore reported and must not be conflated:
  1. published internal  -- 21 features, 5-fold CV on 2007-2012
  2. transportable internal -- 11 features, same folds, same cohort
  3. transportable external -- 11 features, fitted on ALL of 2007-2012 and
                               applied once to the untouched 2015-2018 cohort

(1) vs (2) measures what the retired modules were worth. (2) vs (3) is the
generalisation gap, which is the quantity a reviewer is asking for.

Imputation medians and scaling parameters are learned on development data
only; the validation cohort is never seen during fitting. The operating
threshold is likewise fixed on development out-of-fold predictions at 90%
sensitivity and then applied unchanged.

Usage:  python src/experiments/external_validation.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "fusion"))
from fuse_clinical_modalities import base_model, encode  # noqa: E402

RESULTS = ROOT / "results"
DEV = ROOT / "data" / "processed" / "clinical_agent_cohort.csv"
EXT = ROOT / "data" / "processed" / "clinical_agent_external.csv"
TARGET = "current_asthma"
SEED = 42
N_SPLITS = 5
N_BOOT = 2000
TARGET_SENS = 0.90

TRANSPORTABLE = ["sex", "age", "ethnicity", "income_poverty_ratio",
                 "household_size", "family_hx_asthma", "household_smoker",
                 "n_household_smokers", "bmi", "height", "log_cotinine"]


def metrics(y, p):
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def boot_ci(y, p, fn, n=N_BOOT, seed=SEED):
    """Percentile bootstrap CI, resampling patients."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def calibration_fit(y, p):
    """Calibration intercept and slope from logistic recalibration on the
    linear predictor. Slope 1 / intercept 0 is perfect."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    slope = LogisticRegression(penalty=None, max_iter=5000).fit(lp.reshape(-1, 1), y)
    return {"slope": float(slope.coef_[0][0]),
            "intercept": float(slope.intercept_[0]),
            "mean_predicted": float(p.mean()),
            "observed": float(y.mean())}


def at_threshold(y, p, thr):
    tn, fp, fn_, tp = confusion_matrix(y, (p >= thr).astype(int)).ravel()
    return {"threshold": float(thr),
            "sensitivity": tp / (tp + fn_), "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
            "npv": tn / (tn + fn_) if (tn + fn_) else float("nan"),
            "accuracy": (tp + tn) / len(y),
            "tp": int(tp), "fp": int(fp), "fn": int(fn_), "tn": int(tn)}


def internal_cv(X, y, cols_label):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof, per_fold = np.zeros(len(y)), []
    for tr, te in cv.split(X, y):
        m = CalibratedClassifierCV(base_model(), method="sigmoid", cv=3).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        per_fold.append(roc_auc_score(y[te], p))
    out = metrics(y, oof)
    out["cv_auc_mean"] = float(np.mean(per_fold))
    out["cv_auc_std"] = float(np.std(per_fold))
    print(f"  {cols_label:26s} AUC {out['roc_auc']:.3f} "
          f"(fold {out['cv_auc_mean']:.3f}+-{out['cv_auc_std']:.3f})  "
          f"PR-AUC {out['pr_auc']:.3f}  Brier {out['brier']:.4f}")
    return out, oof


def main():
    warnings.filterwarnings("ignore")
    dev = pd.read_csv(DEV).dropna(subset=[TARGET]).reset_index(drop=True)
    ext = pd.read_csv(EXT).dropna(subset=[TARGET]).reset_index(drop=True)
    ydev = dev[TARGET].astype(int).to_numpy()
    yext = ext[TARGET].astype(int).to_numpy()

    print(f"development 2007-2012: n={len(dev):,}  positives={ydev.sum():,} "
          f"({ydev.mean():.1%})")
    print(f"validation  2015-2018: n={len(ext):,}  positives={yext.sum():,} "
          f"({yext.mean():.1%})\n")

    published = [c for c in dev.columns if c not in (TARGET, "SEQN", "cycle")]
    print("internal cross-validation on the development cohort:")
    full_res, _ = internal_cv(encode(dev, published), ydev,
                              f"published ({len(published)} feats)")
    Xd = encode(dev, TRANSPORTABLE)
    trans_res, oof = internal_cv(Xd, ydev,
                                 f"transportable ({len(TRANSPORTABLE)} feats)")

    # Operating threshold fixed on DEVELOPMENT out-of-fold predictions.
    order = np.argsort(-oof)
    cum_tp = np.cumsum(ydev[order])
    k = int(np.searchsorted(cum_tp, TARGET_SENS * ydev.sum()))
    thr = float(oof[order][min(k, len(oof) - 1)])
    dev_op = at_threshold(ydev, oof, thr)
    print(f"\ndevelopment threshold at {TARGET_SENS:.0%} sensitivity: {thr:.4f}")
    print(f"  dev  sens {dev_op['sensitivity']:.3f}  spec {dev_op['specificity']:.3f}"
          f"  PPV {dev_op['ppv']:.3f}  NPV {dev_op['npv']:.3f}")

    # --- the external test: fit once on ALL development data ---------------
    model = CalibratedClassifierCV(base_model(), method="sigmoid", cv=5).fit(Xd, ydev)
    pext = model.predict_proba(encode(ext, TRANSPORTABLE))[:, 1]

    ext_res = metrics(yext, pext)
    ext_res["roc_auc_ci"] = boot_ci(yext, pext, roc_auc_score)
    ext_res["pr_auc_ci"] = boot_ci(yext, pext, average_precision_score)
    ext_res["calibration"] = calibration_fit(yext, pext)
    ext_res["operating_point"] = at_threshold(yext, pext, thr)

    trans_res["roc_auc_ci"] = boot_ci(ydev, oof, roc_auc_score)

    print(f"\nEXTERNAL 2015-2018:")
    print(f"  AUC {ext_res['roc_auc']:.3f} "
          f"[{ext_res['roc_auc_ci'][0]:.3f}-{ext_res['roc_auc_ci'][1]:.3f}]  "
          f"PR-AUC {ext_res['pr_auc']:.3f}  Brier {ext_res['brier']:.4f}")
    c = ext_res["calibration"]
    print(f"  calibration slope {c['slope']:.3f}  intercept {c['intercept']:+.3f}")
    print(f"  predicted mean {c['mean_predicted']:.3f} vs observed {c['observed']:.3f}")
    o = ext_res["operating_point"]
    print(f"  at fixed threshold: sens {o['sensitivity']:.3f}  "
          f"spec {o['specificity']:.3f}  PPV {o['ppv']:.3f}  NPV {o['npv']:.3f}")

    gap = trans_res["roc_auc"] - ext_res["roc_auc"]
    print(f"\ngeneralisation gap (transportable internal - external): {gap:+.3f}")

    # Per-cycle, to show the result is not carried by one survey wave.
    per_cycle = {}
    for cyc, sub in ext.groupby("cycle"):
        m = sub.index.to_numpy()
        per_cycle[cyc] = {"n": int(len(m)), "prevalence": float(yext[m].mean()),
                          "roc_auc": float(roc_auc_score(yext[m], pext[m]))}
        print(f"  {cyc}: n={len(m):,}  prev {yext[m].mean():.1%}  "
              f"AUC {per_cycle[cyc]['roc_auc']:.3f}")

    out = {"development": {"n": int(len(dev)), "prevalence": float(ydev.mean()),
                           "cycles": sorted(dev.cycle.unique().tolist())},
           "validation": {"n": int(len(ext)), "prevalence": float(yext.mean()),
                          "cycles": sorted(ext.cycle.unique().tolist())},
           "published_internal": full_res,
           "transportable_internal": trans_res,
           "transportable_external": ext_res,
           "development_operating_point": dev_op,
           "per_cycle": per_cycle,
           "generalisation_gap": float(gap),
           "transportable_features": TRANSPORTABLE,
           "retired_after_2012": [c for c in published if c not in TRANSPORTABLE]}
    (RESULTS / "external_validation.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"SEQN": ext.SEQN, "cycle": ext.cycle,
                  "y": yext, "p": pext}).to_csv(
        RESULTS / "external_validation_predictions.csv", index=False)
    print(f"\nwrote results/external_validation.json")


if __name__ == "__main__":
    main()
