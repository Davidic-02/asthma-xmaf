"""What do abstention and conflict detection actually buy?

Both reviewers object that an AUC at 85% coverage and a high prevalence in the
conflict subgroup do not by themselves show clinical benefit, and Reviewer A
adds that overall performance cannot show the Decision Agent mechanisms are
necessary. This script therefore measures, from the stored out-of-fold
predictions:

  Abstention
    * who is referred: size, asthma count, prevalence against the retained set
    * what the retained set looks like: sensitivity, specificity, PPV, NPV
    * how much of the error the referral band absorbs, against the 15% a
      random referral of the same size would absorb
    * whether the same rule applied to the flat pooled model does as well

  Conflict
    * how many asthma cases the flagged subgroup contains, and what share of
      all asthma that is
    * whether the flag carries information BEYOND the flat model's score
      (logistic regression of the outcome on the flat score plus the flag)
    * where a flat model fails inside the flagged subgroup

Every quantity gets a patient-level bootstrap interval. The per-fold thresholds
are rebuilt from the stored predictions exactly as decision_agent.py builds
them, and the script refuses to continue unless that reproduces the stored
abstention flags.

Usage:  python src/experiments/decision_mechanisms.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SEED, N_SPLITS, N_BOOT = 42, 5, 4000
ABSTAIN_FRACTION, TARGET_SENS = 0.15, 0.90


def fold_thresholds(y, p):
    """Per-fold 90%-sensitivity threshold, as in decision_agent.py."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    thr = np.zeros(len(y))
    for tr, te in cv.split(np.zeros(len(y)), y):
        _, tpr, cuts = roc_curve(y[tr], p[tr])
        thr[te] = cuts[int(np.argmin(np.abs(tpr - TARGET_SENS)))]
    return thr


def referral(p, thr):
    margin = np.abs(p - thr)
    return margin <= np.quantile(margin, ABSTAIN_FRACTION)


def rates(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else float("nan")
    return {"sens": div(tp, tp + fn), "spec": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def analyse(y, p, thr, ref):
    """Everything reported about one (score, referral) pair."""
    pred = (p >= thr).astype(int)
    err = pred != y
    keep = ~ref
    r_all, r_keep = rates(y, pred), rates(y[keep], pred[keep])
    return {
        "n_referred": int(ref.sum()),
        "asthma_in_referred": int(y[ref].sum()),
        "share_of_asthma_referred": float(y[ref].sum() / y.sum()),
        "prevalence_referred": float(y[ref].mean()),
        "prevalence_retained": float(y[keep].mean()),
        "errors_total": int(err.sum()),
        "share_of_errors_in_referred": float(err[ref].sum() / err.sum()),
        "auc_all": float(roc_auc_score(y, p)),
        "auc_retained": float(roc_auc_score(y[keep], p[keep])),
        "all": r_all, "retained": r_keep,
    }


def main():
    d = pd.read_csv(RES / "decision_agent_predictions.csv")
    f = pd.read_csv(RES / "fusion_oof_predictions.csv")
    assert len(d) == len(f) and (d.y.values == f.y.values).all(), "row order differs"
    y = d.y.to_numpy()
    pf, pflat = d.p_fused.to_numpy(), f.p_flat.to_numpy()
    psym, ppul = d.p_symptom.to_numpy(), d.p_pulmonary.to_numpy()
    conflict = d.conflict.to_numpy().astype(bool)

    thr = fold_thresholds(y, pf)
    ref = referral(pf, thr)
    if not (ref == d.abstain.to_numpy().astype(bool)).all():
        raise SystemExit("rebuilt abstention flags do not match the stored ones")
    print("rebuilt thresholds reproduce the stored abstention flags exactly\n")

    out = {"n": int(len(y)), "positives": int(y.sum())}

    # ---- abstention ------------------------------------------------------
    fus = analyse(y, pf, thr, ref)
    thr_flat = fold_thresholds(y, pflat)
    flat = analyse(y, pflat, thr_flat, referral(pflat, thr_flat))
    rng = np.random.default_rng(SEED)
    rand_share = []
    for _ in range(2000):
        r = np.zeros(len(y), bool)
        r[rng.choice(len(y), ref.sum(), replace=False)] = True
        rand_share.append(((pf >= thr).astype(int) != y)[r].sum() / ((pf >= thr).astype(int) != y).sum())
    fus["share_of_errors_random_referral"] = float(np.mean(rand_share))
    out["abstention_fusion"], out["abstention_flat"] = fus, flat

    boot = {k: [] for k in ("share_asthma", "share_err", "auc_gain", "npv_keep", "sens_keep")}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        yi, pi, ti, ri = y[i], pf[i], thr[i], ref[i]
        pred = (pi >= ti).astype(int)
        err = pred != yi
        boot["share_asthma"].append(yi[ri].sum() / yi.sum())
        boot["share_err"].append(err[ri].sum() / max(err.sum(), 1))
        boot["auc_gain"].append(roc_auc_score(yi[~ri], pi[~ri]) - roc_auc_score(yi, pi))
        r = rates(yi[~ri], pred[~ri])
        boot["npv_keep"].append(r["npv"]); boot["sens_keep"].append(r["sens"])
    out["abstention_ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                            for k, v in boot.items()}

    # ---- conflict --------------------------------------------------------
    has = d.has_pulmonary.to_numpy().astype(bool)
    c = {"n_flagged": int(conflict.sum()), "share_of_cohort": float(conflict.mean()),
         "asthma_in_flagged": int(y[conflict].sum()),
         "share_of_all_asthma": float(y[conflict].sum() / y.sum()),
         "prevalence_flagged": float(y[conflict].mean()),
         "prevalence_unflagged": float(y[~conflict].mean())}
    sym_hi = conflict & (psym > ppul)
    pul_hi = conflict & (ppul > psym)
    c["symptom_higher"] = {"n": int(sym_hi.sum()), "prevalence": float(y[sym_hi].mean())}
    c["pulmonary_higher"] = {"n": int(pul_hi.sum()), "prevalence": float(y[pul_hi].mean())}

    pred_flat = (pflat >= thr_flat).astype(int)
    c["flat_miss_rate_in_flagged"] = float(((pred_flat == 0) & (y == 1))[conflict].sum() / y[conflict].sum())
    c["flat_miss_rate_elsewhere"] = float(((pred_flat == 0) & (y == 1))[~conflict].sum() / y[~conflict].sum())

    # Does the flag add anything once the flat model's score is known?
    lp = np.log(np.clip(pflat, 1e-6, 1 - 1e-6) / (1 - np.clip(pflat, 1e-6, 1 - 1e-6)))
    X0 = lp.reshape(-1, 1)
    X1 = np.column_stack([lp, conflict.astype(float)])

    def fit(X, yy):
        return LogisticRegression(C=np.inf, max_iter=5000).fit(X, yy)

    m0, m1 = fit(X0, y), fit(X1, y)
    ll0 = -log_loss(y, m0.predict_proba(X0)[:, 1], normalize=False)
    ll1 = -log_loss(y, m1.predict_proba(X1)[:, 1], normalize=False)
    lr = 2 * (ll1 - ll0)
    ors = []
    for _ in range(N_BOOT // 2):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2 or conflict[i].sum() < 5:
            continue
        ors.append(float(np.exp(fit(X1[i], y[i]).coef_[0][1])))
    c["incremental"] = {
        "odds_ratio": float(np.exp(m1.coef_[0][1])),
        "or_ci": [float(np.percentile(ors, 2.5)), float(np.percentile(ors, 97.5))],
        "lr_chi2": float(lr), "p": float(chi2.sf(lr, 1)),
        "auc_flat_only": float(roc_auc_score(y, m0.predict_proba(X0)[:, 1])),
        "auc_flat_plus_flag": float(roc_auc_score(y, m1.predict_proba(X1)[:, 1])),
    }
    out["conflict"] = c

    # ---- print -----------------------------------------------------------
    A, ci = fus, out["abstention_ci"]
    print("ABSTENTION (fusion, 15% referred)")
    print(f"  referred {A['n_referred']:,}; asthma in referred {A['asthma_in_referred']} "
          f"= {A['share_of_asthma_referred']:.1%} of all asthma "
          f"[{ci['share_asthma'][0]:.1%}-{ci['share_asthma'][1]:.1%}]")
    print(f"  prevalence referred {A['prevalence_referred']:.1%} vs retained {A['prevalence_retained']:.1%}")
    print(f"  referred band absorbs {A['share_of_errors_in_referred']:.1%} of all errors "
          f"[{ci['share_err'][0]:.1%}-{ci['share_err'][1]:.1%}] vs {A['share_of_errors_random_referral']:.1%} "
          "for random referral")
    print(f"  retained: sens {A['retained']['sens']:.3f} spec {A['retained']['spec']:.3f} "
          f"PPV {A['retained']['ppv']:.3f} NPV {A['retained']['npv']:.3f}  "
          f"(all: sens {A['all']['sens']:.3f} spec {A['all']['spec']:.3f} NPV {A['all']['npv']:.3f})")
    print(f"  AUC retained {A['auc_retained']:.3f} vs all {A['auc_all']:.3f}; gain "
          f"{A['auc_retained']-A['auc_all']:+.3f} [{ci['auc_gain'][0]:+.3f}, {ci['auc_gain'][1]:+.3f}]")
    B = flat
    print("ABSTENTION (same rule on the flat pooled model)")
    print(f"  asthma referred {B['share_of_asthma_referred']:.1%}; errors absorbed "
          f"{B['share_of_errors_in_referred']:.1%}; AUC gain {B['auc_retained']-B['auc_all']:+.3f}")
    print("CONFLICT")
    print(f"  flagged {c['n_flagged']} ({c['share_of_cohort']:.1%}); asthma {c['asthma_in_flagged']} "
          f"= {c['share_of_all_asthma']:.1%} of all asthma; prevalence {c['prevalence_flagged']:.1%} "
          f"vs {c['prevalence_unflagged']:.1%} unflagged")
    print(f"  symptom>pulmonary n={c['symptom_higher']['n']} prev {c['symptom_higher']['prevalence']:.1%}; "
          f"pulmonary>symptom n={c['pulmonary_higher']['n']} prev {c['pulmonary_higher']['prevalence']:.1%}")
    print(f"  flat model misses {c['flat_miss_rate_in_flagged']:.1%} of asthma inside the flagged group "
          f"vs {c['flat_miss_rate_elsewhere']:.1%} elsewhere")
    I = c["incremental"]
    print(f"  flag beyond flat score: OR {I['odds_ratio']:.2f} ({I['or_ci'][0]:.2f}-{I['or_ci'][1]:.2f}), "
          f"LR chi2 {I['lr_chi2']:.1f}, p = {I['p']:.3g}; AUC {I['auc_flat_only']:.3f} -> {I['auc_flat_plus_flag']:.3f}")

    (RES / "decision_mechanisms.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/decision_mechanisms.json")


if __name__ == "__main__":
    main()
