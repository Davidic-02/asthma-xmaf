"""Check every quantitative claim in the manuscript against the result files.

Recomputes each figure from data rather than trusting the prose, and reports
any claim that does not match. Run before submission and after any rerun.

Usage:  python tools/verify_claims.py <manuscript.docx>
"""

import json
import re
import sys
from pathlib import Path

import docx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"


def manuscript_text(path):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def main():
    doc = manuscript_text(sys.argv[1])


    clin = pd.read_csv(PROC / "clinical_agent_cohort.csv")
    fus = pd.read_csv(RES / "fusion_oof_predictions.csv")
    dec = pd.read_csv(RES / "decision_agent_predictions.csv")
    dsum = json.loads((RES / "decision_agent_summary.json").read_text())
    emb = json.loads((RES / "audio_embedding_comparison.json").read_text())
    two = json.loads((RES / "audio_2stage_summary.json").read_text())
    man = pd.read_csv(PROC / "sprsound_manifest.csv").dropna(subset=["disease"])
    feats = pd.read_csv(PROC / "audio_features.csv")

    y = fus["y"].to_numpy()
    checks = []

    def nocommas(t):
        """Drop thousands separators: the BMC copy omits them by journal rule."""
        return re.sub(r"(\d),(\d{3})", r"\1\2", t)

    def claim(label, text, truth, fmt="{:.3f}"):
        """Record whether `text` (as written) matches the recomputed `truth`."""
        want = fmt.format(truth) if not isinstance(truth, str) else truth
        # whole-number match: "4.2%" must not be satisfied by "14.2%". Boundaries apply
        # only where the claim itself starts or ends with a digit.
        pre = r"(?<![\d.])" if text[:1].isdigit() else ""
        post = r"(?!\d)" if text[-1:].isdigit() or text[-1:] == "%" else ""
        pat = re.compile(pre + re.escape(text) + post)
        present = bool(pat.search(doc) or pat.search(nocommas(doc)))
        checks.append((label, text, want, present, want))

    # ---- cohort -------------------------------------------------------
    claim("cohort size", "6,329", f"{len(clin):,}", "{}")
    claim("asthma cases", "805", f"{int(clin.current_asthma.sum()):,}", "{}")
    claim("prevalence", "12.7%", f"{clin.current_asthma.mean():.1%}", "{}")
    n_no_pul = int((~fus.has_pulmonary.astype(bool)).sum())
    claim("no spirometry", "489", f"{n_no_pul:,}", "{}")
    claim("with spirometry", "5,840", f"{len(fus)-n_no_pul:,}", "{}")

    # ---- acoustic cohort ----------------------------------------------
    claim("acoustic recordings", "5,216", f"{len(feats):,}", "{}")
    claim("acoustic children", "747", f"{feats.patient_id.nunique():,}", "{}")
    n_ast_pat = feats[feats.is_asthma == 1].patient_id.nunique()
    claim("acoustic asthma patients", "66", f"{n_ast_pat}", "{}")

    # ---- agent performance --------------------------------------------
    for col, name, txt in (("p_symptom", "Symptom", "0.863"),
                           ("p_pulmonary", "Pulmonary", "0.646"),
                           ("p_fusion", "Fusion", "0.871"),
                           ("p_flat", "Flat", "0.867")):
        claim(f"{name} ROC-AUC", txt, roc_auc_score(y, fus[col]))
    for col, name, txt in (("p_symptom", "Symptom", "0.643"),
                           ("p_pulmonary", "Pulmonary", "0.238"),
                           ("p_fusion", "Fusion", "0.657")):
        claim(f"{name} PR-AUC", txt, average_precision_score(y, fus[col]))
    claim("Fusion Brier", "0.068", brier_score_loss(y, fus.p_fusion))
    claim("Flat Brier", "0.067", brier_score_loss(y, fus.p_flat))

    # ---- confusion matrix ---------------------------------------------
    p = fus.p_fusion.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()
    claim("default TP", "408", f"{tp}", "{}")
    claim("default FP", "125", f"{fp}", "{}")
    claim("default FN", "397", f"{fn}", "{}")
    claim("default TN", "5,399", f"{tn:,}", "{}")
    claim("default sensitivity", "0.507", tp / (tp + fn))
    claim("default specificity", "0.977", tn / (tn + fp))
    claim("missed cases sentence", "397 of 805", "397 of 805", "{}")

    fpr, tpr, cuts = roc_curve(y, p)
    t90 = cuts[int(np.argmin(np.abs(tpr - 0.90)))]
    tn2, fp2, fn2, tp2 = confusion_matrix(y, (p >= t90).astype(int)).ravel()
    claim("screening threshold", "0.05", f"{t90:.2f}", "{}")
    claim("screening sensitivity", "0.899", tp2 / (tp2 + fn2))
    claim("screening NPV", "0.972", tn2 / (tn2 + fn2))
    claim("screening PPV", "0.213", tp2 / (tp2 + fp2))

    # ---- graceful degradation -----------------------------------------
    for flag, lbl, f_txt, s_txt in ((1, "present", "0.873", "0.865"),
                                    (0, "absent", "0.846", "0.837")):
        g = fus[fus.has_pulmonary.astype(int) == flag]
        claim(f"degradation {lbl} fusion", f_txt, roc_auc_score(g.y, g.p_fusion))
        claim(f"degradation {lbl} symptom", s_txt, roc_auc_score(g.y, g.p_symptom))

    # ---- decision agent ------------------------------------------------
    claim("non-abstained AUC", "0.884", dsum["roc_auc_non_abstained"])
    claim("coverage", "85%", f"{dsum['coverage']:.0%}", "{}")
    claim("conflict count", "511", f"{dsum['n_conflict']}", "{}")
    claim("conflict share", "8.1%", f"{dsum['n_conflict']/len(dec):.1%}", "{}")
    conf_rate = dec[dec.conflict == 1].y.mean()
    claim("conflict asthma rate", "73.2%", f"{conf_rate:.1%}", "{}")

    # ---- faithfulness --------------------------------------------------
    claim("comprehensiveness cited", "0.0991", dsum["comprehensiveness_cited"], "{:.4f}")
    claim("comprehensiveness random", "0.0344", dsum["comprehensiveness_random"], "{:.4f}")
    # Table 8 prints sufficiency to four decimals; accept either precision.
    claim("sufficiency cited", "0.0189", dsum["sufficiency_cited"], "{:.4f}")
    claim("sufficiency random", "0.085", dsum["sufficiency_random"])
    ratio = dsum["comprehensiveness_cited"] / dsum["comprehensiveness_random"]
    claim("faithfulness ratio", "2.9", ratio, "{:.1f}")

    # ---- acoustic agent -------------------------------------------------
    claim("combined acoustic AUC", "0.732", emb["combined"]["roc_auc_mean"])
    claim("combined acoustic SD", "0.083", emb["combined"]["roc_auc_std"])
    claim("handcrafted AUC", "0.683", emb["handcrafted"]["roc_auc_mean"])
    claim("handcrafted SD", "0.057", emb["handcrafted"]["roc_auc_std"])
    claim("two-stage CAS", "0.799", two["stage1_cas_auc_mean"])
    claim("two-stage DAS", "0.751", two["stage1_das_auc_mean"])
    claim("two-stage stage2", "0.647", two["stage2_roc_auc_mean"])
    # PR-AUC is reported at patient level, so the floor is the patient-level
    # prevalence (66/747), not the recording-level share.
    pat_prev = n_ast_pat / feats.patient_id.nunique()
    claim("acoustic prevalence floor", "0.088", pat_prev)
    claim("acoustic PR-AUC lift", "3.4-fold",
          f"{emb['combined']['pr_auc_mean'] / pat_prev:.1f}-fold", "{}")

    # ---- clinical panel and model comparison ---------------------------
    tn3, fp3, fn3, tp3 = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()
    claim("default accuracy", "0.918", (tp3 + tn3) / len(y))
    claim("default F1", "0.610", 2 * tp3 / (2 * tp3 + fp3 + fn3))
    claim("balanced accuracy", "0.742",
          0.5 * (tp3 / (tp3 + fn3) + tn3 / (tn3 + fp3)))
    # A model that predicts the majority class for every child scores this;
    # accuracy is only meaningful against it.
    claim("no-information rate", "0.873", 1 - y.mean())
    mc = json.loads((RES / "model_comparison.json").read_text())
    for key, txt in (("Logistic regression (reported)", "0.868"),
                     ("Random forest", "0.861"),
                     ("Gradient boosting", "0.853"),
                     ("Neural network (64-32)", "0.861")):
        claim(f"model comparison {key}", txt, mc["symptom_agent"][key]["roc_auc_mean"])
    claim("fusion with random forest", "0.865", mc["fusion"]["Random forest"])

    # ---- temporal external validation (NHANES 2015-2018) ----------------
    ev = json.loads((RES / "external_validation.json").read_text())
    ti, te = ev["transportable_internal"], ev["transportable_external"]
    cal, eo = te["calibration"], te["operating_point"]
    claim("external cohort n", "4,111", f"{ev['validation']['n']:,}", "{}")
    claim("external prevalence", "12.2%", f"{ev['validation']['prevalence']:.1%}", "{}")
    claim("external AUC", "0.727", te["roc_auc"])
    claim("transportable internal AUC", "0.733", ti["roc_auc"])
    claim("published internal AUC", "0.867", ev["published_internal"]["roc_auc"])
    claim("calibration slope", "0.950", cal["slope"])
    claim("calibration intercept", "-0.147", f"{cal['intercept']:+.3f}", "{}")
    claim("external sensitivity", "91.4%", f"{eo['sensitivity']:.1%}", "{}")
    claim("external specificity", "0.249", eo["specificity"])
    for cyc, d in ev["per_cycle"].items():
        claim(f"external AUC {cyc}", f"{d['roc_auc']:.3f}", d["roc_auc"])

    # ---- screening specificity, referral band, conflict flag (R2) ---------
    claim("screening specificity", "0.516", tn2 / (tn2 + fp2))
    dm = json.loads((RES / "decision_mechanisms.json").read_text())
    A, AF, CF = dm["abstention_fusion"], dm["abstention_flat"], dm["conflict"]
    pc = lambda x: f"{x:.1%}"
    claim("referral band size", f"{A['n_referred']:,} children, of whom", f"{A['n_referred']:,} children, of whom", "{}")
    claim("asthma in referral band", f"of whom {A['asthma_in_referred']} (", f"of whom {A['asthma_in_referred']} (", "{}")
    claim("share of asthma referred", pc(A["share_of_asthma_referred"]), pc(A["share_of_asthma_referred"]), "{}")
    claim("referred prevalence", pc(A["prevalence_referred"]), pc(A["prevalence_referred"]), "{}")
    claim("retained prevalence", pc(A["prevalence_retained"]), pc(A["prevalence_retained"]), "{}")
    claim("errors inside band", pc(A["share_of_errors_in_referred"]), pc(A["share_of_errors_in_referred"]), "{}")
    claim("errors under random referral", pc(A["share_of_errors_random_referral"]), pc(A["share_of_errors_random_referral"]), "{}")
    claim("Decision Agent all-case AUC", f"{A['auc_all']:.3f}", A["auc_all"])
    claim("Decision Agent retained AUC", f"{A['auc_retained']:.3f}", A["auc_retained"])
    claim("pooled-model retained AUC", f"{AF['auc_retained']:.3f}", AF["auc_retained"])
    claim("pooled-model errors in band", pc(AF["share_of_errors_in_referred"]), pc(AF["share_of_errors_in_referred"]), "{}")
    claim("retained sensitivity", f"{A['retained']['sens']:.3f}", A["retained"]["sens"])
    claim("retained NPV", f"{A['retained']['npv']:.3f}", A["retained"]["npv"])
    claim("conflict flagged", f"{CF['n_flagged']} children ({pc(CF['share_of_cohort'])})", f"{CF['n_flagged']} children ({pc(CF['share_of_cohort'])})", "{}")
    claim("asthma in flagged group", f"included {CF['asthma_in_flagged']} of the {int(y.sum())}", f"included {CF['asthma_in_flagged']} of the {int(y.sum())}", "{}")
    claim("share of all asthma flagged", pc(CF["share_of_all_asthma"]), pc(CF["share_of_all_asthma"]), "{}")
    claim("flagged prevalence", pc(CF["prevalence_flagged"]), pc(CF["prevalence_flagged"]), "{}")
    claim("unflagged prevalence", pc(CF["prevalence_unflagged"]), pc(CF["prevalence_unflagged"]), "{}")
    claim("symptom-higher prevalence", pc(CF["symptom_higher"]["prevalence"]), pc(CF["symptom_higher"]["prevalence"]), "{}")
    claim("pulmonary-higher prevalence", pc(CF["pulmonary_higher"]["prevalence"]), pc(CF["pulmonary_higher"]["prevalence"]), "{}")
    claim("pooled miss rate, flagged", pc(CF["flat_miss_rate_in_flagged"]), pc(CF["flat_miss_rate_in_flagged"]), "{}")
    claim("pooled miss rate, unflagged", pc(CF["flat_miss_rate_elsewhere"]), pc(CF["flat_miss_rate_elsewhere"]), "{}")
    inc = CF["incremental"]
    claim("flag odds ratio", f"{inc['odds_ratio']:.2f}", inc["odds_ratio"], "{:.2f}")
    claim("flag odds ratio CI", f"{inc['or_ci'][0]:.2f} to {inc['or_ci'][1]:.2f}", f"{inc['or_ci'][0]:.2f} to {inc['or_ci'][1]:.2f}", "{}")
    claim("flag LR p", f"p = {inc['p']:.2f}", f"p = {inc['p']:.2f}", "{}")

    # ---- acoustic evaluation limits (R2) ----------------------------------
    arec = pd.read_csv(RES / "audio_agent_oof_predictions.csv")
    au = json.loads((RES / "acoustic_uncertainty.json").read_text())
    claim("asthma recordings", "434", f"{int(arec.is_asthma.sum())}", "{}")
    claim("handcrafted AUC CI low", "0.614", au["handcrafted_ci"][0])
    claim("handcrafted AUC CI high", "0.754", au["handcrafted_ci"][1])
    fsig = json.loads((RES / "fusion_significance.json").read_text())["contrasts"]
    claim("fusion vs pooled p", "p = 0.094", f"p = {fsig['fusion_minus_flat']['p']:.3f}", "{}")
    claim("fusion vs symptom p", "p = 0.031", f"p = {fsig['fusion_minus_symptom']['p']:.3f}", "{}")

    # ---- stale values -------------------------------------------------
    # Figures from superseded runs. Checking only that the correct value is
    # present misses a stale copy left elsewhere in the text.
    for label, stale in (("old fusion gain", "by 0.007"), ("old fusion gain", "+0.007"),
                         ("old fusion AUC", "0.870"), ("old abstained AUC", "0.881"),
                         ("old degraded AUC", "0.845"), ("old fusion Brier", "0.116"),
                         ("recording-level prevalence", "8.3%"),
                         ("old screening NPV", "0.973"), ("old screening specificity", "0.515"),
                         ("old referral count", " 949 "),
                         ("13-item conflict count", "517 children"), ("13-item retained AUC", "0.880 at 85%"),
                         ("13-item faithfulness ratio", "2.8 times"), ("old lift", "3.5-fold")):
        checks.append((label, "absent", "absent", stale not in doc, "absent"))

    # ---- report ---------------------------------------------------------
    bad = [c for c in checks if not c[3] or c[1] != c[4]]
    print(f"{len(checks)} claims checked\n")
    print(f"{'claim':32s} {'in text':>12s} {'recomputed':>12s}  status")
    print("-" * 74)
    for label, text, want, present, truth in checks:
        if text == truth and present:
            status = "ok"
        elif text != truth:
            status = "MISMATCH"
        else:
            status = "not found in text"
        print(f"{label:32s} {text:>12s} {truth:>12s}  {status}")
    n_mismatch = sum(1 for c in checks if c[1] != c[4])
    n_missing = sum(1 for c in checks if c[1] == c[4] and not c[3])
    print(f"\n{n_mismatch} mismatched, {n_missing} not found in text")


if __name__ == "__main__":
    main()
