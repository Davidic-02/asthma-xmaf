"""Two-stage Audio Agent: acoustic event detection -> patient-level asthma.

Rationale
---------
Predicting an asthma diagnosis straight from a 15 s clip asks a model to make
a leap clinicians do not make. A clinician first identifies *sounds* (wheeze,
crackles), then reasons from the pattern of those sounds across auscultation
sites to a diagnosis. This agent does the same:

  Stage 1  acoustic event detectors, trained on the ~2.5k recordings carrying
           record-level annotations (CAS = continuous adventitious sounds, i.e.
           wheeze/rhonchi/stridor; DAS = discontinuous, i.e. crackles). This is
           the well-powered part: ~966 abnormal recordings.

  Stage 2  aggregate Stage 1 probabilities over the child's auscultation sites
           into a small, named feature vector (how strong, how widespread) and
           predict asthma from that plus age and sex. Only this last step
           depends on the scarce 66 asthma diagnoses.

Leakage control
---------------
Stage 1 must never see a patient who appears in the Stage 2 test fold, or the
event probabilities fed to Stage 2 would be optimistically biased. We use an
outer patient-grouped CV; inside each outer fold Stage 1 is refit on the
training patients only, then applied to both sides. Recordings marked
"Poor Quality" are excluded from Stage 1 training but still scored, since a
clinician cannot discard them at inference time either.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "audio_features.csv"
RESULTS = ROOT / "results"
N_SPLITS = 5
SEED = 42

META = {
    "record_stem", "patient_id", "age_years", "sex", "position", "wav_path",
    "json_path", "record_annotation", "has_wheeze_event", "disease",
    "disease_raw", "is_asthma", "sex_male",
}


def acoustic_columns(df):
    return [c for c in df.columns if c not in META]


def make_detector():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")),
    ])


def make_classifier():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")),
    ])


def fit_stage1(df_train, feat_cols):
    """Fit CAS and DAS detectors on annotated, non-poor-quality recordings."""
    ann = df_train[
        df_train["record_annotation"].notna()
        & (df_train["record_annotation"] != "Poor Quality")
    ]
    detectors = {}
    for name, positives in (
        ("cas", {"CAS", "CAS & DAS"}),
        ("das", {"DAS", "CAS & DAS"}),
    ):
        y = ann["record_annotation"].isin(positives).astype(int).to_numpy()
        if y.sum() < 20 or len(set(y)) < 2:
            continue
        X = ann[feat_cols].to_numpy(float)
        det = CalibratedClassifierCV(make_detector(), method="sigmoid", cv=3)
        det.fit(X, y)
        detectors[name] = det
    return detectors


def apply_stage1(detectors, df, feat_cols):
    X = df[feat_cols].to_numpy(float)
    out = pd.DataFrame(index=df.index)
    for name, det in detectors.items():
        out[f"p_{name}"] = det.predict_proba(X)[:, 1]
    return out


def aggregate_patient(df, probs):
    """Summarise a child's recordings into one row of named evidence.

    Averaging alone would dilute a real finding: a child wheezing at one site
    is still wheezing. We therefore keep max and spread as well as mean, and
    count how many sites exceed a moderate-confidence threshold.
    """
    d = df[["patient_id", "age_years", "sex"]].copy()
    d = pd.concat([d, probs], axis=1)
    rows = []
    for pid, g in d.groupby("patient_id"):
        row = {
            "patient_id": pid,
            "age_years": g["age_years"].iloc[0],
            "sex_male": int(g["sex"].iloc[0] == "male"),
            "n_recordings": len(g),
        }
        for col in probs.columns:
            v = g[col].to_numpy()
            row[f"{col}_mean"] = float(np.mean(v))
            row[f"{col}_max"] = float(np.max(v))
            row[f"{col}_std"] = float(np.std(v)) if len(v) > 1 else 0.0
            row[f"{col}_n_sites_gt50"] = int((v >= 0.5).sum())
            row[f"{col}_frac_sites_gt50"] = float((v >= 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"missing {FEATURES} - run extract_audio_features.py first")

    df = pd.read_csv(FEATURES).dropna(subset=["disease"]).reset_index(drop=True)
    df["is_asthma"] = df["is_asthma"].astype(int)
    feat_cols = acoustic_columns(df)

    pat_lbl = df.groupby("patient_id")["is_asthma"].max()
    print(f"recordings {len(df):,} | patients {len(pat_lbl):,} | asthma patients {int(pat_lbl.sum())}")
    ann_n = int((df["record_annotation"].notna() & (df["record_annotation"] != "Poor Quality")).sum())
    print(f"stage-1 training pool: {ann_n:,} annotated recordings")

    # Outer CV over patients, stratified on the patient-level asthma label.
    rec_index = df.index.to_numpy()
    y_rec = df["is_asthma"].to_numpy()
    groups = df["patient_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_rows, oof = [], []
    stage1_auc = {"cas": [], "das": []}

    for fold, (tr, te) in enumerate(cv.split(rec_index, y_rec, groups), start=1):
        tr_df, te_df = df.iloc[tr], df.iloc[te]

        detectors = fit_stage1(tr_df, feat_cols)
        if not detectors:
            sys.exit("stage 1 could not be fit - too few annotated positives")

        # Stage-1 quality on held-out annotated recordings, for reporting.
        te_ann = te_df[
            te_df["record_annotation"].notna()
            & (te_df["record_annotation"] != "Poor Quality")
        ]
        if len(te_ann):
            p_ann = apply_stage1(detectors, te_ann, feat_cols)
            for name, pos in (("cas", {"CAS", "CAS & DAS"}), ("das", {"DAS", "CAS & DAS"})):
                if f"p_{name}" not in p_ann:
                    continue
                yy = te_ann["record_annotation"].isin(pos).astype(int)
                if yy.nunique() > 1:
                    stage1_auc[name].append(roc_auc_score(yy, p_ann[f"p_{name}"]))

        p_tr = apply_stage1(detectors, tr_df, feat_cols).reset_index(drop=True)
        p_te = apply_stage1(detectors, te_df, feat_cols).reset_index(drop=True)

        A_tr = aggregate_patient(tr_df.reset_index(drop=True), p_tr)
        A_te = aggregate_patient(te_df.reset_index(drop=True), p_te)

        lbl = df.groupby("patient_id")["is_asthma"].max()
        A_tr["y"] = A_tr["patient_id"].map(lbl).astype(int)
        A_te["y"] = A_te["patient_id"].map(lbl).astype(int)

        stage2_cols = [c for c in A_tr.columns if c not in {"patient_id", "y"}]
        clf = CalibratedClassifierCV(make_classifier(), method="sigmoid", cv=3)
        clf.fit(A_tr[stage2_cols].to_numpy(float), A_tr["y"].to_numpy())
        p = clf.predict_proba(A_te[stage2_cols].to_numpy(float))[:, 1]

        fold_rows.append({
            "fold": fold,
            "roc_auc": roc_auc_score(A_te["y"], p) if A_te["y"].nunique() > 1 else np.nan,
            "pr_auc": average_precision_score(A_te["y"], p) if A_te["y"].nunique() > 1 else np.nan,
            "n_patients": len(A_te),
            "n_asthma": int(A_te["y"].sum()),
        })
        out = A_te[["patient_id", "y"]].copy()
        out["p_asthma"] = p
        out["fold"] = fold
        oof.append(out)

    res = pd.DataFrame(fold_rows)
    oof = pd.concat(oof, ignore_index=True)

    print("\n--- stage 1 (acoustic event detection, held-out recordings)")
    for name in ("cas", "das"):
        if stage1_auc[name]:
            a = np.array(stage1_auc[name])
            label = "CAS (wheeze/rhonchi)" if name == "cas" else "DAS (crackles)"
            print(f"  {label:24s} AUC={a.mean():.3f}+/-{a.std():.3f}")

    print("\n--- stage 2 (patient-level asthma)")
    print(f"  ROC-AUC {res['roc_auc'].mean():.3f}+/-{res['roc_auc'].std():.3f}")
    print(f"  PR-AUC  {res['pr_auc'].mean():.3f}+/-{res['pr_auc'].std():.3f}")
    print(f"  pooled OOF ROC-AUC {roc_auc_score(oof['y'], oof['p_asthma']):.3f}")

    prev = oof["y"].mean()
    print(f"\n  reference: prevalence (PR-AUC floor) = {prev:.3f}")
    print("  single-stage baseline was ROC-AUC 0.705 +/- 0.104")

    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "audio_2stage_folds.csv", index=False)
    oof.to_csv(RESULTS / "audio_2stage_oof_patient.csv", index=False)
    (RESULTS / "audio_2stage_summary.json").write_text(json.dumps({
        "stage1_cas_auc_mean": float(np.mean(stage1_auc["cas"])) if stage1_auc["cas"] else None,
        "stage1_das_auc_mean": float(np.mean(stage1_auc["das"])) if stage1_auc["das"] else None,
        "stage2_roc_auc_mean": float(res["roc_auc"].mean()),
        "stage2_roc_auc_std": float(res["roc_auc"].std()),
        "stage2_pr_auc_mean": float(res["pr_auc"].mean()),
        "pooled_oof_roc_auc": float(roc_auc_score(oof["y"], oof["p_asthma"])),
    }, indent=2))
    print(f"\nwrote results to {RESULTS}")


if __name__ == "__main__":
    main()
