"""
Clinical Agent -- tabular pediatric asthma classifier with SHAP explanations.

Mirrors the MATRIACARE stacking design (RF + XGB + SVM + LightGBM with a
meta-learner) so methodology carries across both papers, but adds what this
task requires:

  * class imbalance (12.7% positive) is handled explicitly and PR-AUC is
    reported alongside ROC-AUC, since ROC-AUC flatters imbalanced problems;
  * probability calibration is measured (Brier score), because the Decision
    Agent fuses confidences -- an uncalibrated agent poisons the fusion;
  * no post-diagnosis variables are used as features (see build_nhanes.py).

Outputs per-patient probability + SHAP attribution, which is the evidence
payload consumed downstream by the Decision Agent.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix,
                             classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "clinical_agent_cohort.csv"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

SEED = 42
CAT = ["ethnicity"]
TARGET = "current_asthma"


def load():
    df = pd.read_csv(DATA)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, "SEQN", "cycle"])
    return X, y


def preprocessor(X):
    num = [c for c in X.columns if c not in CAT]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), CAT),
    ])


def models(spw):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000,
                                                  class_weight="balanced",
                                                  random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=8,
                                                min_samples_leaf=5,
                                                class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=400, max_depth=5,
                                   learning_rate=0.05, num_leaves=15,
                                   class_weight="balanced", verbose=-1,
                                   random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True,
                         class_weight="balanced", random_state=SEED),
    }


def evaluate(name, pipe, Xtr, ytr, Xte, yte, cv):
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro",
                                                  zero_division=0)
    pp, pr, pf1, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                     zero_division=0)
    auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "precision_asthma": pp, "recall_asthma": pr, "f1_asthma": pf1,
        "roc_auc": roc_auc_score(yte, proba),
        "pr_auc": average_precision_score(yte, proba),
        "brier": brier_score_loss(yte, proba),
        "cv_roc_auc_mean": auc.mean(), "cv_roc_auc_std": auc.std(),
    }, pipe, proba


def main():
    X, y = load()
    print(f"cohort {X.shape[0]:,} x {X.shape[1]} features | "
          f"positives {y.sum():,} ({y.mean():.1%})\n")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=SEED)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, probas = [], {}, {}
    for name, clf in models(spw).items():
        pipe = Pipeline([("prep", preprocessor(X)), ("clf", clf)])
        row, pipe, proba = evaluate(name, pipe, Xtr, ytr, Xte, yte, cv)
        rows.append(row); fitted[name] = pipe; probas[name] = proba
        print(f"{name:22s} AUC {row['roc_auc']:.4f}  PR-AUC {row['pr_auc']:.4f}"
              f"  recall(asthma) {row['recall_asthma']:.3f}")

    base = [(n.split()[0].lower(),
             Pipeline([("prep", preprocessor(X)), ("clf", c)]))
            for n, c in models(spw).items() if n != "Logistic Regression"]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced",
                                           random_state=SEED),
        cv=cv, n_jobs=-1, passthrough=False)
    row, stack, proba = evaluate("Stacking Ensemble", stack,
                                 Xtr, ytr, Xte, yte, cv)
    rows.append(row); fitted["Stacking Ensemble"] = stack
    probas["Stacking Ensemble"] = proba
    print(f"{'Stacking Ensemble':22s} AUC {row['roc_auc']:.4f}  "
          f"PR-AUC {row['pr_auc']:.4f}  recall(asthma) {row['recall_asthma']:.3f}")

    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    res.to_csv(RESULTS / "clinical_agent_results.csv", index=False)

    print("\n" + "=" * 78)
    print(res[["model", "accuracy", "f1_macro", "roc_auc", "pr_auc",
               "recall_asthma", "brier"]].round(4).to_string(index=False))

    best = res.iloc[0]["model"]
    print(f"\nbest by PR-AUC: {best}")
    print("\nclassification report @0.50 -- " + best)
    print(classification_report(yte, (probas[best] >= .5).astype(int),
                                target_names=["no asthma", "asthma"],
                                zero_division=0))
    print("confusion matrix:\n",
          confusion_matrix(yte, (probas[best] >= .5).astype(int)))

    import joblib
    joblib.dump(fitted[best], MODELS / "clinical_agent.joblib")
    json.dump({"best_model": best, "n": int(len(X)),
               "prevalence": float(y.mean()),
               "features": list(X.columns)},
              open(RESULTS / "clinical_agent_meta.json", "w"), indent=2)
    print(f"\nsaved -> models/clinical_agent.joblib")


if __name__ == "__main__":
    main()
