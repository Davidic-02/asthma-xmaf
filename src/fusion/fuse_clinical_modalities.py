"""Decision-level fusion of two genuinely paired modalities in NHANES.

The same child provides both:
  Symptom Agent      - reported history: wheeze, nocturnal cough, family
                       history, smoke exposure, demographics, anthropometry
  Pulmonary Agent    - objective physiology: FEV1, FVC, FEV1/FVC, PEF, and
                       serum cotinine as a measured exposure biomarker

Unlike the clinical/audio pair, these are measured on the *same* patients, so
a fused result here is real rather than simulated.

Compared here:
  1. Symptom Agent alone
  2. Pulmonary Agent alone
  3. Late fusion  - meta-learner over the two agents' calibrated probabilities
  4. Flat model   - one model over the pooled feature table (the usual
                    "multimodal" baseline that skips the agent structure)

A modality-dropout analysis answers RQ5: how the framework behaves when
spirometry is unavailable, which is the common case outside a lab.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "clinical_agent_cohort.csv"
RESULTS = ROOT / "results"
SEED = 42
N_SPLITS = 5

SYMPTOM = [
    "sex", "age", "ethnicity", "income_poverty_ratio", "household_size",
    "family_hx_asthma", "wheeze_12mo", "wheeze_attacks", "exercise_wheeze",
    "nocturnal_dry_cough", "household_smoker", "n_household_smokers",
    "bmi", "height",
]
PULMONARY = [
    "fev1", "fvc", "pef", "fev1_fvc_ratio", "fev1_per_ht", "pef_per_ht",
    "log_cotinine",
]
TARGET = "current_asthma"


def base_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")),
    ])


def encode(df, cols):
    """Numeric matrix; object columns become integer codes."""
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return X.to_numpy(float)


def scores(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def main() -> None:
    if not COHORT.exists():
        sys.exit(f"missing {COHORT}")

    df = pd.read_csv(COHORT)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int).to_numpy()

    has_pulm = df[PULMONARY].notna().any(axis=1).to_numpy()
    print(f"children: {len(df):,} | asthma: {y.sum():,} ({y.mean():.1%})")
    print(f"with any spirometry/cotinine: {has_pulm.sum():,} ({has_pulm.mean():.1%})")
    for c in PULMONARY:
        print(f"    {c:16s} present {df[c].notna().mean():5.1%}")

    Xs, Xp = encode(df, SYMPTOM), encode(df, PULMONARY)
    Xall = encode(df, SYMPTOM + PULMONARY)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in ("symptom", "pulmonary", "fusion", "flat")}

    for tr, te in cv.split(Xs, y):
        # --- individual agents, calibrated so their outputs are comparable
        a_s = CalibratedClassifierCV(base_model(), method="sigmoid", cv=3).fit(Xs[tr], y[tr])
        a_p = CalibratedClassifierCV(base_model(), method="sigmoid", cv=3).fit(Xp[tr], y[tr])
        ps_tr, ps_te = a_s.predict_proba(Xs[tr])[:, 1], a_s.predict_proba(Xs[te])[:, 1]
        pp_tr, pp_te = a_p.predict_proba(Xp[tr])[:, 1], a_p.predict_proba(Xp[te])[:, 1]
        oof["symptom"][te], oof["pulmonary"][te] = ps_te, pp_te

        # --- decision agent: learns how much to trust each agent, and is told
        #     explicitly whether the pulmonary evidence exists at all
        Z_tr = np.column_stack([ps_tr, pp_tr, has_pulm[tr].astype(float)])
        Z_te = np.column_stack([ps_te, pp_te, has_pulm[te].astype(float)])
        meta = LogisticRegression(max_iter=5000, class_weight="balanced")
        meta.fit(Z_tr, y[tr])
        oof["fusion"][te] = meta.predict_proba(Z_te)[:, 1]

        # --- flat baseline: no agent structure, one pooled feature table
        flat = CalibratedClassifierCV(base_model(), method="sigmoid", cv=3).fit(Xall[tr], y[tr])
        oof["flat"][te] = flat.predict_proba(Xall[te])[:, 1]

    print(f"\n{'model':22s} {'ROC-AUC':>9s} {'PR-AUC':>8s} {'Brier':>8s}")
    summary = {}
    for name in ("symptom", "pulmonary", "fusion", "flat"):
        s = scores(y, oof[name])
        summary[name] = s
        print(f"{name:22s} {s['roc_auc']:9.3f} {s['pr_auc']:8.3f} {s['brier']:8.3f}")

    lift = summary["fusion"]["roc_auc"] - summary["symptom"]["roc_auc"]
    print(f"\nfusion vs symptom-only : {lift:+.3f} ROC-AUC")
    print(f"fusion vs flat baseline: {summary['fusion']['roc_auc'] - summary['flat']['roc_auc']:+.3f}")

    # --- RQ5: graceful degradation when spirometry is missing
    print("\nmodality availability (fusion model):")
    for label, mask in (("spirometry present", has_pulm), ("spirometry absent", ~has_pulm)):
        if mask.sum() < 30 or len(set(y[mask])) < 2:
            print(f"  {label:20s} n={mask.sum():,} - too few to score")
            continue
        s = scores(y[mask], oof["fusion"][mask])
        ss = scores(y[mask], oof["symptom"][mask])
        print(f"  {label:20s} n={mask.sum():5,} fusion ROC-AUC={s['roc_auc']:.3f} "
              f"(symptom-only {ss['roc_auc']:.3f})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({**{f"p_{k}": v for k, v in oof.items()},
                  "y": y, "has_pulmonary": has_pulm}).to_csv(
        RESULTS / "fusion_oof_predictions.csv", index=False)
    (RESULTS / "fusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote results to {RESULTS}")


if __name__ == "__main__":
    main()
