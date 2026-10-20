"""Does a stronger classifier improve the Symptom Agent?

The reported agent is L2-regularised logistic regression. A reviewer will ask
why, so this measures the alternatives instead of asserting an answer. Every
candidate sees identical folds, identical imputation and scaling, and the same
sigmoid calibration wrapper; only the classifier differs. Per-fold scores are
retained so the comparison carries dispersion rather than a single number.

Usage:  python src/experiments/model_comparison.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "fusion"))
from fuse_clinical_modalities import (  # noqa: E402
    PULMONARY, SYMPTOM, TARGET, base_model, encode,
)

RESULTS = ROOT / "results"
SEED = 42
N_SPLITS = 5


def rf():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
    ])


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


def mlp():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                              max_iter=800, early_stopping=True,
                              random_state=SEED)),
    ])


CANDIDATES = {
    "Logistic regression (reported)": base_model,
    "Random forest": rf,
    "Gradient boosting": hgb,
    "Neural network (64-32)": mlp,
}


def main() -> None:
    warnings.filterwarnings("ignore")
    df = (pd.read_csv(ROOT / "data" / "processed" / "clinical_agent_cohort.csv")
          .dropna(subset=[TARGET]).reset_index(drop=True))
    y = df[TARGET].astype(int).to_numpy()
    Xs, Xp = encode(df, SYMPTOM), encode(df, PULMONARY)
    has_pulm = df[PULMONARY].notna().any(axis=1).to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(Xs, y))

    rows, summary = [], {}
    for name, make in CANDIDATES.items():
        per_fold, oof = [], np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            m = CalibratedClassifierCV(make(), method="sigmoid", cv=3).fit(Xs[tr], y[tr])
            p = m.predict_proba(Xs[te])[:, 1]
            oof[te] = p
            per_fold.append((roc_auc_score(y[te], p),
                             average_precision_score(y[te], p),
                             brier_score_loss(y[te], p)))
            rows.append({"model": name, "fold": k, "roc_auc": per_fold[-1][0],
                         "pr_auc": per_fold[-1][1], "brier": per_fold[-1][2]})
        a = np.array(per_fold)
        summary[name] = {
            "roc_auc_mean": a[:, 0].mean(), "roc_auc_std": a[:, 0].std(),
            "pr_auc_mean": a[:, 1].mean(), "pr_auc_std": a[:, 1].std(),
            "brier_mean": a[:, 2].mean(),
            "roc_auc_oof": roc_auc_score(y, oof),
        }
        print(f"{name:32s} ROC {a[:,0].mean():.3f}±{a[:,0].std():.3f}  "
              f"PR {a[:,1].mean():.3f}±{a[:,1].std():.3f}  Brier {a[:,2].mean():.3f}")

    # Does the best alternative change the fused result?
    best = max((k for k in summary if "reported" not in k),
               key=lambda k: summary[k]["roc_auc_mean"])
    fused = {}
    for label, make in (("logistic", base_model), (best, CANDIDATES[best])):
        oof = np.zeros(len(y))
        for tr, te in folds:
            a_s = CalibratedClassifierCV(make(), method="sigmoid", cv=3).fit(Xs[tr], y[tr])
            a_p = CalibratedClassifierCV(base_model(), method="sigmoid", cv=3).fit(Xp[tr], y[tr])
            ps_tr, ps_te = a_s.predict_proba(Xs[tr])[:, 1], a_s.predict_proba(Xs[te])[:, 1]
            pp_tr, pp_te = a_p.predict_proba(Xp[tr])[:, 1], a_p.predict_proba(Xp[te])[:, 1]
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(max_iter=5000).fit(
                np.column_stack([ps_tr, pp_tr, has_pulm[tr].astype(float)]), y[tr])
            oof[te] = meta.predict_proba(
                np.column_stack([ps_te, pp_te, has_pulm[te].astype(float)]))[:, 1]
        fused[label] = roc_auc_score(y, oof)
        print(f"fusion with {label:22s} ROC-AUC {fused[label]:.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "model_comparison_folds.csv", index=False)
    (RESULTS / "model_comparison.json").write_text(json.dumps(
        {"symptom_agent": summary, "fusion": fused, "best_alternative": best}, indent=2))
    print("\nwrote results/model_comparison.json")


if __name__ == "__main__":
    main()
