"""Audio Agent on AST embeddings, compared against handcrafted features.

Reports four feature sets under one protocol (patient-grouped CV, patient-level
aggregation, calibrated probabilities) so the comparison is like-for-like:

  handcrafted   58 named acoustic descriptors  (current agent, 0.717)
  embeddings    768-d frozen AST representation
  combined      both
  embeddings+   embeddings reduced by PCA, which usually helps when the
                representation is far wider than the number of patients

With 66 asthma patients and 768 dimensions, the embedding model is in a
severely underdetermined regime; strong regularisation and PCA are therefore
not optional tuning but a requirement for the comparison to mean anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "audio_features.csv"
EMBEDDINGS = ROOT / "data" / "processed" / "audio_embeddings.npz"
RESULTS = ROOT / "results"
SEED = 42
N_SPLITS = 5

META = {
    "record_stem", "patient_id", "age_years", "sex", "position", "wav_path",
    "json_path", "record_annotation", "has_wheeze_event", "disease",
    "disease_raw", "is_asthma", "sex_male",
}


def model(pca_components=None):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
    steps.append(("clf", LogisticRegression(
        max_iter=5000, C=0.05, class_weight="balanced")))
    return Pipeline(steps)


def patient_aggregate(X, pids):
    """Mean and max per patient - a diagnosis is about the child, not a clip."""
    df = pd.DataFrame(X)
    df["pid"] = pids
    g = df.groupby("pid")
    out = g.mean().join(g.max(), lsuffix="_mean", rsuffix="_max")
    return out.to_numpy(float), out.index.to_numpy()


def evaluate(name, X, y_rec, pids, pat_label, pca=None):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for tr, te in cv.split(X, y_rec, pids):
        clf = CalibratedClassifierCV(model(pca), method="sigmoid", cv=3)
        clf.fit(X[tr], y_rec[tr])
        p = clf.predict_proba(X[te])[:, 1]

        agg = pd.DataFrame({"pid": pids[te], "p": p}).groupby("pid")["p"].mean()
        yy = agg.index.map(pat_label).astype(int)
        if len(set(yy)) < 2:
            continue
        rows.append({
            "roc_auc": roc_auc_score(yy, agg.to_numpy()),
            "pr_auc": average_precision_score(yy, agg.to_numpy()),
        })
    r = pd.DataFrame(rows)
    print(f"  {name:22s} ROC-AUC {r['roc_auc'].mean():.3f}+/-{r['roc_auc'].std():.3f}   "
          f"PR-AUC {r['pr_auc'].mean():.3f}")
    return {"roc_auc_mean": float(r["roc_auc"].mean()),
            "roc_auc_std": float(r["roc_auc"].std()),
            "pr_auc_mean": float(r["pr_auc"].mean())}


def main() -> None:
    if not EMBEDDINGS.exists():
        sys.exit(f"missing {EMBEDDINGS} - embedding run has not finished")

    df = pd.read_csv(FEATURES).dropna(subset=["disease"]).reset_index(drop=True)
    df["is_asthma"] = df["is_asthma"].astype(int)

    z = np.load(EMBEDDINGS, allow_pickle=True)
    emb = pd.DataFrame(z["embeddings"])
    emb["record_stem"] = z["record_stem"]

    merged = df.merge(emb, on="record_stem", how="inner")
    print(f"recordings with both feature sets: {len(merged):,}")

    hand_cols = [c for c in df.columns if c not in META]
    emb_cols = [c for c in merged.columns if isinstance(c, int)]

    Xh = merged[hand_cols].to_numpy(float)
    Xe = merged[emb_cols].to_numpy(float)
    Xc = np.hstack([Xh, Xe])
    y = merged["is_asthma"].to_numpy()
    pids = merged["patient_id"].astype(str).to_numpy()
    pat_label = merged.groupby(merged["patient_id"].astype(str))["is_asthma"].max()

    print(f"patients {len(pat_label):,} | asthma patients {int(pat_label.sum())}")
    print(f"handcrafted dims {Xh.shape[1]} | embedding dims {Xe.shape[1]}\n")

    summary = {}
    summary["handcrafted"] = evaluate("handcrafted", Xh, y, pids, pat_label)
    summary["embeddings"] = evaluate("AST embeddings", Xe, y, pids, pat_label)
    summary["embeddings_pca32"] = evaluate("AST + PCA32", Xe, y, pids, pat_label, pca=32)
    summary["combined"] = evaluate("combined", Xc, y, pids, pat_label)
    summary["combined_pca64"] = evaluate("combined + PCA64", Xc, y, pids, pat_label, pca=64)

    best = max(summary, key=lambda k: summary[k]["roc_auc_mean"])
    print(f"\nbest: {best} ({summary[best]['roc_auc_mean']:.3f})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "audio_embedding_comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS / 'audio_embedding_comparison.json'}")


if __name__ == "__main__":
    main()
