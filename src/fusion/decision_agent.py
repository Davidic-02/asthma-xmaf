"""Decision Agent: evidence fusion, abstention, and verified explanation.

What this adds beyond the meta-learner in fuse_clinical_modalities.py:

  1. Evidence statements  - each agent reports a calibrated probability, the
     features driving it, and whether its modality was observed at all.
  2. Conflict detection   - when agents disagree strongly the case is flagged
     rather than silently averaged away.
  3. Abstention           - cases in the uncertain band are referred rather
     than predicted, and we measure what accuracy that buys on the rest.
  4. Explanation fusion   - one clinician-readable rationale assembled from
     per-agent attributions.
  5. Faithfulness         - measured, not asserted (see below).

On measuring faithfulness honestly
----------------------------------
The narrative is *generated from* the attributions, so "does the narrative
cite features the model used?" is true by construction and worth nothing.
The real question is whether the cited features actually drive the
prediction. We therefore use deletion-based tests:

  comprehensiveness = p(full) - p(cited features removed)
      high is good: removing what we cited should change the decision.
  sufficiency       = p(full) - p(only cited features kept)
      low (near zero) is good: the cited features alone should nearly
      reproduce the decision.

Both are compared against removing the same number of *random* features. An
explanation that scores no better than random is not faithful, whatever it
says.

SHAP for the linear agents is computed in closed form: for a standardised
linear model the exact Shapley value of feature i is coef_i * (x_i - mean_i).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "clinical_agent_cohort.csv"
RESULTS = ROOT / "results"
SEED = 42
N_SPLITS = 5
TOP_K = 3                 # features cited in the narrative
ABSTAIN_FRACTION = 0.15   # share of cases referred rather than predicted
CONFLICT_GAP = 0.35       # agent disagreement that triggers a flag

SYMPTOM = [
    "wheeze_12mo", "wheeze_attacks", "exercise_wheeze", "nocturnal_dry_cough",
    "family_hx_asthma", "household_smoker", "n_household_smokers",
    "age", "sex", "income_poverty_ratio", "household_size", "bmi", "height",
]
PULMONARY = ["fev1", "fvc", "pef", "fev1_fvc_ratio", "fev1_per_ht",
             "pef_per_ht", "log_cotinine"]
TARGET = "current_asthma"

# Clinician-facing wording for each feature.
LABEL = {
    "wheeze_12mo": "wheezing in the past 12 months",
    "wheeze_attacks": "frequency of wheezing attacks",
    "exercise_wheeze": "exercise-induced wheezing",
    "nocturnal_dry_cough": "nocturnal dry cough",
    "family_hx_asthma": "family history of asthma",
    "household_smoker": "household smoke exposure",
    "n_household_smokers": "number of household smokers",
    "age": "age", "sex": "sex", "bmi": "body mass index",
    "height": "height", "household_size": "household size",
    "income_poverty_ratio": "income-to-poverty ratio",
    "fev1": "FEV1", "fvc": "FVC", "pef": "peak expiratory flow",
    "fev1_fvc_ratio": "FEV1/FVC ratio", "fev1_per_ht": "height-adjusted FEV1",
    "pef_per_ht": "height-adjusted peak flow",
    "log_cotinine": "serum cotinine (smoke exposure)",
}


def prep(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return X.to_numpy(float)


def fit_agent(Xtr, ytr):
    """Impute + scale + calibrated LR, returning the pieces SHAP needs."""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr = sc.transform(imp.transform(Xtr))
    raw = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced").fit(Ztr, ytr)
    cal = CalibratedClassifierCV(
        LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"),
        method="sigmoid", cv=3).fit(Ztr, ytr)
    return {"imp": imp, "sc": sc, "raw": raw, "cal": cal}


def agent_transform(a, X):
    return a["sc"].transform(a["imp"].transform(X))


def agent_prob(a, X):
    return a["cal"].predict_proba(agent_transform(a, X))[:, 1]


def agent_shap(a, X):
    """Exact Shapley values for a standardised linear model."""
    Z = agent_transform(a, X)
    coef = a["raw"].coef_.ravel()
    return (Z - Z.mean(axis=0, keepdims=True)) * coef


def masked_prob(a, X, cols_idx, mode):
    """Probability with a subset of features removed or isolated.

    Removal means replacing with the training median, i.e. the model's
    'no information' value for that feature after imputation.
    """
    Z = agent_transform(a, X).copy()
    Zbase = np.zeros_like(Z)  # standardised median == 0
    if mode == "remove":
        for r, idx in enumerate(cols_idx):
            Z[r, idx] = Zbase[r, idx]
    elif mode == "only":
        keep = Z.copy()
        Z = Zbase.copy()
        for r, idx in enumerate(cols_idx):
            Z[r, idx] = keep[r, idx]
    return a["cal"].predict_proba(Z)[:, 1]


def narrate(row, feat_names, shap_row, p_fused, decision, has_pulm, conflict):
    """Assemble the unified explanation from the fused evidence."""
    order = np.argsort(-np.abs(shap_row))[:TOP_K]
    cited = [feat_names[i] for i in order]
    parts = []
    for i in order:
        name = LABEL.get(feat_names[i], feat_names[i])
        direction = "increasing" if shap_row[i] > 0 else "decreasing"
        parts.append(f"{name} ({direction} risk)")

    text = (f"Asthma likelihood {p_fused:.0%} - {decision}. "
            f"Principal contributing factors: {'; '.join(parts)}.")
    if not has_pulm:
        text += (" Spirometry was not available for this child; the assessment "
                 "rests on reported clinical history alone.")
    if conflict:
        text += (" Note: the clinical and pulmonary agents disagree materially; "
                 "clinical review is recommended.")
    return text, cited


def main() -> None:
    if not COHORT.exists():
        sys.exit(f"missing {COHORT}")

    df = pd.read_csv(COHORT).dropna(subset=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int).to_numpy()
    Xs, Xp = prep(df, SYMPTOM), prep(df, PULMONARY)
    has_pulm = df[PULMONARY].notna().any(axis=1).to_numpy()

    n = len(df)
    p_sym = np.zeros(n); p_pul = np.zeros(n); p_fus = np.zeros(n)
    shap_all = np.zeros((n, len(SYMPTOM)))
    comp, suff, comp_rnd, suff_rnd = [], [], [], []

    rng = np.random.default_rng(SEED)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for tr, te in cv.split(Xs, y):
        a_s = fit_agent(Xs[tr], y[tr])
        a_p = fit_agent(Xp[tr], y[tr])
        ps_tr, ps_te = agent_prob(a_s, Xs[tr]), agent_prob(a_s, Xs[te])
        pp_tr, pp_te = agent_prob(a_p, Xp[tr]), agent_prob(a_p, Xp[te])
        p_sym[te], p_pul[te] = ps_te, pp_te

        meta = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
            np.column_stack([ps_tr, pp_tr, has_pulm[tr].astype(float)]), y[tr])
        p_fus[te] = meta.predict_proba(
            np.column_stack([ps_te, pp_te, has_pulm[te].astype(float)]))[:, 1]

        sv = agent_shap(a_s, Xs[te])
        shap_all[te] = sv

        # --- faithfulness of the cited features, against a random control
        top_idx = np.argsort(-np.abs(sv), axis=1)[:, :TOP_K]
        rnd_idx = np.array([rng.choice(len(SYMPTOM), TOP_K, replace=False)
                            for _ in range(len(te))])
        full = agent_prob(a_s, Xs[te])
        comp.append(np.abs(full - masked_prob(a_s, Xs[te], top_idx, "remove")))
        suff.append(np.abs(full - masked_prob(a_s, Xs[te], top_idx, "only")))
        comp_rnd.append(np.abs(full - masked_prob(a_s, Xs[te], rnd_idx, "remove")))
        suff_rnd.append(np.abs(full - masked_prob(a_s, Xs[te], rnd_idx, "only")))

    comp = np.concatenate(comp); suff = np.concatenate(suff)
    comp_rnd = np.concatenate(comp_rnd); suff_rnd = np.concatenate(suff_rnd)

    # --- decisions with abstention
    # The uncertain band is the middle of the predicted-risk distribution,
    # not a fixed window around 0.5, since prevalence is only 12.7%.
    band_lo = np.quantile(p_fus, 0.5 - ABSTAIN_FRACTION / 2)
    band_hi = np.quantile(p_fus, 0.5 + ABSTAIN_FRACTION / 2)
    abstain = (p_fus > band_lo) & (p_fus < band_hi)
    conflict = (np.abs(p_sym - p_pul) > CONFLICT_GAP) & has_pulm

    print(f"cohort {n:,} | asthma {y.sum():,} ({y.mean():.1%})")
    print(f"\nfused ROC-AUC (all cases)      {roc_auc_score(y, p_fus):.3f}")
    if (~abstain).sum() and len(set(y[~abstain])) > 1:
        print(f"fused ROC-AUC (non-abstained)  {roc_auc_score(y[~abstain], p_fus[~abstain]):.3f}"
              f"   coverage {(~abstain).mean():.0%}")
    print(f"abstained {abstain.sum():,} ({abstain.mean():.1%})")
    print(f"conflict flagged {conflict.sum():,} ({conflict.mean():.1%})"
          f"  | asthma rate in flagged {y[conflict].mean():.1%}" if conflict.sum() else "")

    print("\n--- explanation faithfulness (symptom agent, top-3 cited)")
    print(f"  comprehensiveness  cited {comp.mean():.4f}  vs random {comp_rnd.mean():.4f}"
          f"   ratio {comp.mean() / max(comp_rnd.mean(), 1e-9):.1f}x")
    print(f"  sufficiency (lower better)  cited {suff.mean():.4f}  vs random {suff_rnd.mean():.4f}")

    # --- worked examples
    print("\n--- example explanations")
    order = np.argsort(-p_fus)
    picks = [order[0], order[n // 2], order[-1]]
    if conflict.sum():
        picks.append(int(np.flatnonzero(conflict)[0]))
    for i in picks:
        decision = ("refer for review" if abstain[i]
                    else ("asthma likely" if p_fus[i] >= 0.5 else "asthma unlikely"))
        text, cited = narrate(df.iloc[i], SYMPTOM, shap_all[i], p_fus[i],
                              decision, has_pulm[i], conflict[i])
        print(f"\n  [child {i}] true label = {'asthma' if y[i] else 'no asthma'}")
        print(f"  {text}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "p_symptom": p_sym, "p_pulmonary": p_pul, "p_fused": p_fus,
        "has_pulmonary": has_pulm, "abstain": abstain, "conflict": conflict,
        "y": y,
    })
    out.to_csv(RESULTS / "decision_agent_predictions.csv", index=False)
    (RESULTS / "decision_agent_summary.json").write_text(json.dumps({
        "roc_auc_all": float(roc_auc_score(y, p_fus)),
        "roc_auc_non_abstained": float(roc_auc_score(y[~abstain], p_fus[~abstain])),
        "coverage": float((~abstain).mean()),
        "n_conflict": int(conflict.sum()),
        "comprehensiveness_cited": float(comp.mean()),
        "comprehensiveness_random": float(comp_rnd.mean()),
        "sufficiency_cited": float(suff.mean()),
        "sufficiency_random": float(suff_rnd.mean()),
    }, indent=2))
    print(f"\nwrote results to {RESULTS}")


if __name__ == "__main__":
    main()
