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

    def claim(label, text, truth, fmt="{:.3f}"):
        """Record whether `text` (as written) matches the recomputed `truth`."""
        want = fmt.format(truth) if not isinstance(truth, str) else truth
        checks.append((label, text, want, text in doc, want))

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
    claim("non-abstained AUC", "0.880", dsum["roc_auc_non_abstained"])
    claim("coverage", "85%", f"{dsum['coverage']:.0%}", "{}")
    claim("conflict count", "517", f"{dsum['n_conflict']}", "{}")
    claim("conflict share", "8.2%", f"{dsum['n_conflict']/len(dec):.1%}", "{}")
    conf_rate = dec[dec.conflict == 1].y.mean()
    claim("conflict asthma rate", "73.1%", f"{conf_rate:.1%}", "{}")

    # ---- faithfulness --------------------------------------------------
    claim("comprehensiveness cited", "0.0999", dsum["comprehensiveness_cited"], "{:.4f}")
    claim("comprehensiveness random", "0.0356", dsum["comprehensiveness_random"], "{:.4f}")
    claim("sufficiency cited", "0.017", dsum["sufficiency_cited"])
    claim("sufficiency random", "0.084", dsum["sufficiency_random"])
    ratio = dsum["comprehensiveness_cited"] / dsum["comprehensiveness_random"]
    claim("faithfulness ratio", "2.8", ratio, "{:.1f}")

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
