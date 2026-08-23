"""Train and evaluate the Audio Agent (asthma vs non-asthma from lung sound).

Evaluation protocol
-------------------
Every recording from a given child shares that child's diagnosis, and each
child contributes several recordings. Splitting at the recording level would
put the same patient in train and test and inflate every metric - the most
common flaw in respiratory-audio papers. We therefore use StratifiedGroupKFold
grouped on patient_id, and report mean +/- SD across folds rather than a
single hold-out, because with ~71 asthma patients one split is too noisy to
report.

Two decision levels are scored:
  * recording level - one prediction per audio clip
  * patient level   - clip probabilities averaged per child, which is the
                      unit a clinician actually cares about and the unit the
                      fusion layer consumes.

The agent emits calibrated probabilities so the Decision Agent can compare
its confidence against the Clinical Agent on a common scale.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "audio_features.csv"
RESULTS = ROOT / "results"
N_SPLITS = 5
SEED = 42

# Columns that are metadata, not acoustic features.
META = {
    "record_stem", "patient_id", "age_years", "sex", "position",
    "wav_path", "json_path", "record_annotation", "has_wheeze_event",
    "disease", "disease_raw", "is_asthma",
}


def build_models():
    """Two models: an interpretable linear baseline and a nonlinear one.

    Both are wrapped in probability calibration because the fusion layer
    compares agent confidences directly - uncalibrated scores would make one
    agent systematically louder than the other.
    """
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")),
    ])
    gbm = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
        )),
    ])
    return {"logistic_regression": lr, "gradient_boosting": gbm}


def patient_level(df, y_true_col, prob_col):
    """Aggregate clip probabilities to one score per child."""
    g = df.groupby("patient_id").agg(
        y=(y_true_col, "max"), p=(prob_col, "mean")
    )
    return g["y"].to_numpy(), g["p"].to_numpy()


def evaluate(name, model, X, y, groups, feat_names, df_meta):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rec_rows, pat_rows = [], []
    oof = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        # Calibrate on the training folds only - never on test data.
        clf = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p

        rec_rows.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y[te], p),
            "pr_auc": average_precision_score(y[te], p),
            "recall@0.5": recall_score(y[te], (p >= 0.5).astype(int), zero_division=0),
            "f1@0.5": f1_score(y[te], (p >= 0.5).astype(int), zero_division=0),
            "brier": brier_score_loss(y[te], p),
            "n_test": len(te),
            "n_pos": int(y[te].sum()),
        })

        sub = df_meta.iloc[te].copy()
        sub["_y"], sub["_p"] = y[te], p
        py, pp = patient_level(sub, "_y", "_p")
        pat_rows.append({
            "fold": fold,
            "roc_auc": roc_auc_score(py, pp) if len(set(py)) > 1 else np.nan,
            "pr_auc": average_precision_score(py, pp) if len(set(py)) > 1 else np.nan,
            "recall@0.5": recall_score(py, (pp >= 0.5).astype(int), zero_division=0),
            "n_patients": len(py),
            "n_pos": int(py.sum()),
        })

    rec = pd.DataFrame(rec_rows)
    pat = pd.DataFrame(pat_rows)

    print(f"\n--- {name}")
    for level, tbl in (("recording", rec), ("patient", pat)):
        cols = [c for c in ("roc_auc", "pr_auc", "recall@0.5") if c in tbl]
        desc = "  ".join(
            f"{c}={tbl[c].mean():.3f}+/-{tbl[c].std():.3f}" for c in cols
        )
        print(f"  {level:9s} {desc}")

    return {"recording": rec, "patient": pat, "oof": oof}


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"missing {FEATURES} - run extract_audio_features.py first")

    df = pd.read_csv(FEATURES)
    df = df.dropna(subset=["is_asthma"])
    df["is_asthma"] = df["is_asthma"].astype(int)

    feat_names = [c for c in df.columns if c not in META]
    # Add age and sex: both are clinically relevant and available from the
    # filename, and the fusion layer needs them to align patients anyway.
    df["sex_male"] = (df["sex"] == "male").astype(int)
    feat_names += ["age_years", "sex_male"]

    X = df[feat_names].to_numpy(dtype=float)
    y = df["is_asthma"].to_numpy()
    groups = df["patient_id"].astype(str).to_numpy()

    n_pat = df["patient_id"].nunique()
    n_pos_pat = df[df.is_asthma == 1]["patient_id"].nunique()
    print(f"recordings: {len(df):,} | patients: {n_pat:,}")
    print(f"asthma recordings: {y.sum():,} ({y.mean():.1%}) | asthma patients: {n_pos_pat}")
    print(f"features: {len(feat_names)}")

    if n_pos_pat < N_SPLITS:
        sys.exit(f"only {n_pos_pat} asthma patients - too few for {N_SPLITS}-fold CV")

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, model in build_models().items():
        res = evaluate(name, model, X, y, groups, feat_names, df)
        res["recording"].to_csv(RESULTS / f"audio_{name}_recording_folds.csv", index=False)
        res["patient"].to_csv(RESULTS / f"audio_{name}_patient_folds.csv", index=False)
        summary[name] = {
            "recording_roc_auc_mean": float(res["recording"]["roc_auc"].mean()),
            "recording_roc_auc_std": float(res["recording"]["roc_auc"].std()),
            "recording_pr_auc_mean": float(res["recording"]["pr_auc"].mean()),
            "patient_roc_auc_mean": float(res["patient"]["roc_auc"].mean()),
            "patient_roc_auc_std": float(res["patient"]["roc_auc"].std()),
            "patient_pr_auc_mean": float(res["patient"]["pr_auc"].mean()),
        }
        df[f"oof_{name}"] = res["oof"]

    # Persist out-of-fold probabilities: these are the Audio Agent's evidence
    # output, consumed later by the Decision Agent.
    keep = ["record_stem", "patient_id", "age_years", "sex", "position",
            "disease", "is_asthma", "record_annotation", "has_wheeze_event"]
    keep += [c for c in df.columns if c.startswith("oof_")]
    df[keep].to_csv(RESULTS / "audio_agent_oof_predictions.csv", index=False)

    (RESULTS / "audio_agent_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote results to {RESULTS}")

    # Prevalence baseline, so the numbers above can be read in context.
    print(f"\nreference: predicting the majority class gives PR-AUC ~= {y.mean():.3f}")


if __name__ == "__main__":
    main()
