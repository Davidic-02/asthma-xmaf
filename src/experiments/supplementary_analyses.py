"""Reviewer-requested supplementary analyses.

The manuscript build is intentionally kept outside this repository.  Export a
single CSV containing the out-of-fold predictions, label, 14 symptom columns,
the spirometry-availability mask, and (when available) the NHANES survey
columns, then run::

    python src/experiments/supplementary_analyses.py --input exported.csv

The functions are also importable by the manuscript build.  All model fits use
the primary-analysis seed and five-fold stratified cross-validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
ABSTAIN_FRACTION = 0.15
SYMPTOM_COLUMNS = [
    "sex", "age", "ethnicity", "income_poverty_ratio", "household_size",
    "family_hx_asthma", "wheeze_12mo", "wheeze_attacks", "exercise_wheeze",
    "nocturnal_dry_cough", "household_smoker", "n_household_smokers",
    "bmi", "height",
]


def _oof_auc(X: np.ndarray, y: np.ndarray) -> float:
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    p = np.zeros(len(y), dtype=float)
    for train, test in cv.split(X, y):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        )
        model.fit(X[train], y[train])
        p[test] = model.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, p))


def item_removal_sensitivity(X, y, groups):
    """Return OOF AUC and loss after removing each requested item group.

    ``groups`` may be a mapping of group names to column names/indices or an
    iterable of ``(name, columns)`` pairs.  Columns are removed together, so
    the requested three-wheeze-plus-cough sensitivity is represented exactly.
    """
    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    baseline = _oof_auc(X, y)
    if isinstance(groups, dict):
        groups = groups.items()
    rows = [{"group": "none", "removed": [], "auc": baseline, "delta_auc": 0.0}]
    for name, group_columns in groups:
        requested = list(group_columns)
        if feature_names is not None and requested and isinstance(requested[0], str):
            idx = np.asarray([feature_names.index(c) for c in requested], dtype=int)
        else:
            idx = np.asarray(requested, dtype=int)
        keep = np.ones(X.shape[1], dtype=bool)
        keep[idx] = False
        auc = _oof_auc(X[:, keep], y)
        rows.append({
            "group": str(name),
            "removed": [int(i) if isinstance(i, (int, np.integer)) else str(i)
                        for i in idx],
            "auc": auc,
            "delta_auc": auc - baseline,
        })
    return pd.DataFrame(rows)


def _referral_mask(p_fused, thr):
    margin = np.abs(p_fused - thr)
    return margin <= np.quantile(margin, ABSTAIN_FRACTION)


def referral_table(y, p_fused, thr):
    """Pool exact counts and diagnostic rates after the stored referral rule."""
    y, p_fused = np.asarray(y, int), np.asarray(p_fused, float)
    thr = np.broadcast_to(np.asarray(thr, float), y.shape)
    referred = _referral_mask(p_fused, thr)
    retained = ~referred
    predicted = p_fused >= thr
    tn = int(((y == 0) & (predicted == 0) & retained).sum())
    fp = int(((y == 0) & (predicted == 1) & retained).sum())
    fn = int(((y == 1) & (predicted == 0) & retained).sum())
    tp = int(((y == 1) & (predicted == 1) & retained).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "n": int(len(y)),
        "n_retained": int(retained.sum()),
        "n_referred": int(referred.sum()),
        "asthma_retained": int(((y == 1) & retained).sum()),
        "asthma_referred": int(((y == 1) & referred).sum()),
        "specificity_retained": div(tn, tn + fp),
        "ppv_retained": div(tp, tp + fp),
        "sensitivity_retained": div(tp, tp + fn),
        "tn_retained": tn,
        "fp_referred": fp,
        "fn_retained": fn,
        "tp_referred": tp,
    }


def missed_share_ci(y, p_fused, thr, n_boot=10000, seed=SEED):
    """Bootstrap interval for errors absorbed by referral."""
    y, p_fused = np.asarray(y, int), np.asarray(p_fused, float)
    thr = np.broadcast_to(np.asarray(thr, float), y.shape)
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        referred = _referral_mask(p_fused[i], thr[i])
        errors = (p_fused[i] >= thr[i]) != (y[i] == 1)
        shares.append((errors & referred).sum() / max(errors.sum(), 1))
    return [float(v) for v in np.percentile(shares, [2.5, 97.5])]


def simple_average_baseline(p_symptom, p_pulmonary):
    """Plain probability averaging, preserving missing pulmonary predictions."""
    p_symptom, p_pulmonary = np.asarray(p_symptom, float), np.asarray(p_pulmonary, float)
    present = np.isfinite(p_pulmonary)
    out = p_symptom.copy()
    out[present] = (p_symptom[present] + p_pulmonary[present]) / 2
    return out


def missing_modality_comparison(y, p_fused, p_pooled_imputed, p_simple_average):
    """Compare fused, median-imputed pooled, and flat-average predictions."""
    y = np.asarray(y, int)
    out = {}
    for name, p in {
        "fusion": p_fused,
        "pooled_median_imputed": p_pooled_imputed,
        "simple_average": p_simple_average,
    }.items():
        out[name] = {"roc_auc": float(roc_auc_score(y, p))}
    out["fusion_minus_simple_average"] = (
        out["fusion"]["roc_auc"] - out["simple_average"]["roc_auc"]
    )
    return out


def weighted_auc(y, p, weight):
    """Weighted ROC-AUC using the pairwise probability definition."""
    y, p, weight = map(lambda x: np.asarray(x, float), (y, p, weight))
    pos, neg = y == 1, y == 0
    pair_weight = weight[pos, None] * weight[None, neg]
    return float((pair_weight * (p[pos, None] > p[None, neg])).sum()
                 + 0.5 * (pair_weight * (p[pos, None] == p[None, neg])).sum()) / pair_weight.sum()


def design_bootstrap_auc(y, p, weight, strata, psu, n_boot=2000, seed=SEED):
    """PSU-within-stratum bootstrap interval for weighted ROC-AUC."""
    y, p, weight = map(lambda x: np.asarray(x), (y, p, weight))
    strata, psu = np.asarray(strata), np.asarray(psu)
    rng = np.random.default_rng(seed)
    clusters = {(s, u): np.flatnonzero((strata == s) & (psu == u))
                for s, u in zip(strata, psu)}
    by_stratum = {}
    for (s, _), indices in clusters.items():
        by_stratum.setdefault(s, []).append(indices)
    estimates = []
    for _ in range(n_boot):
        sampled = []
        for units in by_stratum.values():
            sampled.extend(units[rng.integers(0, len(units))] for _ in units)
        idx = np.concatenate(sampled)
        if len(np.unique(y[idx])) > 1:
            estimates.append(weighted_auc(y[idx], p[idx], weight[idx]))
    return {
        "auc": weighted_auc(y, p, weight),
        "ci": [float(v) for v in np.percentile(estimates, [2.5, 97.5])],
        "n_boot": len(estimates),
    }


def _read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with labels, predictions, features and survey columns")
    parser.add_argument("--output", type=Path, default=Path("results/supplementary_analyses.json"))
    return parser.parse_args()


def main():
    args = _read_args()
    d = pd.read_csv(args.input)
    required = {"y", "p_symptom", "p_pulmonary", "p_fused", "p_pooled"}
    missing = required - set(d)
    if missing:
        raise SystemExit(f"input is missing required columns: {sorted(missing)}")
    y = d.y.to_numpy(int)
    result = {
        "n": int(len(d)),
        "item_removal": item_removal_sensitivity(
            d[[c if c in d else f"symptom_{c}" for c in SYMPTOM_COLUMNS]].rename(
                columns={f"symptom_{c}": c for c in SYMPTOM_COLUMNS}
            ),
            y,
            {
                "wheeze_items_plus_nocturnal_cough": [
                    "wheeze_12mo", "wheeze_attacks", "exercise_wheeze",
                    "nocturnal_dry_cough",
                ],
                "nocturnal_cough_only": ["nocturnal_dry_cough"],
            },
        ).to_dict(orient="records"),
        "missing_modality": missing_modality_comparison(
            y, d.p_fused, d.p_pooled, simple_average_baseline(d.p_symptom, d.p_pulmonary)
        ),
    }
    if {"threshold", "SDMVSTRA", "SDMVPSU", "WTMEC2YR"} <= set(d):
        result["referral"] = referral_table(y, d.p_fused, d.threshold)
        result["missed_share_ci"] = missed_share_ci(y, d.p_fused, d.threshold)
        result["survey"] = design_bootstrap_auc(
            y, d.p_fused, d.WTMEC2YR / 3, d.SDMVSTRA, d.SDMVPSU
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
