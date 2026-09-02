"""Update the manuscript's reported numbers after the calibration and
abstention fixes, editing the Word file in place so the author's formatting
and equations survive.

Two substantive changes drive all of this:

  * the fusion meta-learner no longer uses class_weight="balanced", which was
    fitting it to a 50/50 prior and inflating every fused probability
    (Brier 0.118 -> 0.069);
  * abstention now refers cases nearest the operating threshold rather than
    cases near the median prediction, which at 12.7% prevalence had been
    referring the model's most confident negatives.

Usage:  python tools/update_results_text.py <in.docx> <out.docx>
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# (label, old, new) - each `old` must appear exactly once.
EDITS = [
    ("abstract: fusion metrics",
     "reached ROC-AUC 0.870 (PR-AUC 0.656), a gain of +0.007 over symptoms alone and +0.003 over a flat pooled-feature model",
     "reached ROC-AUC 0.871 (PR-AUC 0.657), a gain of +0.008 over symptoms alone and +0.004 over a flat pooled-feature model"),

    ("abstract: degradation",
     "fused performance fell only to 0.845 (n=489)",
     "fused performance fell only to 0.846 (n=489)"),

    ("abstract: abstention",
     "raised discrimination on the remainder to 0.881",
     "raised discrimination on the remainder to 0.880"),

    ("methods: abstention rule",
     " Cases falling in the central 15% of the predicted-risk distribution are referred for clinical review rather than classified. The band is defined on distribution quantiles rather than as a fixed window around 0.5, because prevalence is 12.7% and a symmetric window around 0.5 would abstain on almost nothing.",
     " The 15% of cases lying closest to the operating threshold are referred for clinical review rather than classified. Uncertainty is measured as distance from that threshold rather than from the median prediction: at 12.7% prevalence the median predicted probability is approximately 0.05, so a band defined on quantiles of the predicted distribution would refer the model's most confident negatives instead of its genuinely borderline cases. The threshold is set for 90% sensitivity, the operating point a case-finding tool requires, and is fitted within training folds only."),

    ("results: fusion gains",
     "Fusion improved on the symptom agent by +0.007 ROC-AUC and on the flat baseline by +0.003.",
     "Fusion improved on the symptom agent by +0.008 ROC-AUC and on the flat baseline by +0.004."),

    ("results: calibration",
     "The fusion model is less well calibrated (Brier 0.116) than the flat model (0.067); calibration of the meta-learner is an outstanding item, and is visible as the downward deviation of the fusion curve in Figure 3(c).",
     "The fused model is well calibrated (Brier 0.068), matching the flat baseline (0.067), and Figure 3(c) shows both tracking the diagonal closely. An earlier configuration of the meta-learner used class-weighting to balance the outcome classes, which fitted it to a 50/50 prior and inflated every fused probability (Brier 0.118); because its inputs are already calibrated probabilities, removing the class weighting restored calibration at no cost to discrimination."),

    ("results: threshold discussion",
     "At the default 0.50 cut-off the model misses 284 of 805 children with asthma. For a case-finding application in a setting where asthma is already under-diagnosed, that false-negative rate is the binding failure: the children missed are precisely those the system exists to identify. Lowering the threshold to 0.25 recovers 90% of cases and raises negative predictive value to 0.972, so a negative result becomes genuinely reassuring, at the cost of referring roughly half the non-asthmatic children for review.",
     "At the default 0.50 cut-off the model misses 397 of 805 children with asthma. Because the fused probabilities are calibrated to the observed 12.7% prevalence rather than to a balanced prior, comparatively few cases exceed 0.50 at all, and sensitivity is correspondingly low at 0.507. For a case-finding application in a setting where asthma is already under-diagnosed, that false-negative rate is the binding failure: the children missed are precisely those the system exists to identify. This is a property of the threshold, not of the model, and it illustrates why 0.50 is the wrong operating point for a calibrated screening tool. Lowering the threshold to 0.05 recovers 90% of cases and raises negative predictive value to 0.972, so a negative result becomes genuinely reassuring, at the cost of referring roughly half the non-asthmatic children for review."),

    ("figure 7 caption",
     "reaching 0.881 at 85% coverage",
     "reaching 0.880 at 85% coverage"),

    ("discussion: honest metrics",
     "We regard ROC-AUC 0.870 without such variables",
     "We regard ROC-AUC 0.871 without such variables"),

    ("discussion: architecture",
     "Given a fusion gain of +0.007",
     "Given a fusion gain of +0.008"),

    ("conclusions",
     "The framework achieved ROC-AUC 0.870 with fusion, 0.881 at 85% coverage under calibrated abstention, retained 0.845 when spirometry was unavailable",
     "The framework achieved ROC-AUC 0.871 with fusion, 0.880 at 85% coverage under calibrated abstention, retained 0.846 when spirometry was unavailable"),
]

# Table cells: (label, regex over the cell run text, replacement)
CELL_EDITS = [
    ("Table 3 fusion row: ROC",   r"(<w:t[^>]*>)0\.870(</w:t>)", r"\g<1>0.871\g<2>"),
    ("Table 3 fusion row: PR",    r"(<w:t[^>]*>)0\.656(</w:t>)", r"\g<1>0.657\g<2>"),
    ("Table 3 fusion row: Brier", r"(<w:t[^>]*>)0\.116(</w:t>)", r"\g<1>0.068\g<2>"),
    ("Table 4 default: TP",       r"(<w:t[^>]*>)521(</w:t>)",    r"\g<1>408\g<2>"),
    ("Table 4 default: FP",       r"(<w:t[^>]*>)355(</w:t>)",    r"\g<1>125\g<2>"),
    ("Table 4 default: FN",       r"(<w:t[^>]*>)284(</w:t>)",    r"\g<1>397\g<2>"),
    ("Table 4 default: TN",       r"(<w:t[^>]*>)5,169(</w:t>)",  r"\g<1>5,399\g<2>"),
    ("Table 4 default: sens",     r"(<w:t[^>]*>)0\.647(</w:t>)", r"\g<1>0.507\g<2>"),
    ("Table 4 default: spec",     r"(<w:t[^>]*>)0\.936(</w:t>)", r"\g<1>0.977\g<2>"),
    ("Table 4 default: PPV",      r"(<w:t[^>]*>)0\.595(</w:t>)", r"\g<1>0.765\g<2>"),
    ("Table 4 default: NPV",      r"(<w:t[^>]*>)0\.948(</w:t>)", r"\g<1>0.932\g<2>"),
    ("Table 4 screening label",   r"(<w:t[^>]*>)Screening \(0\.25\)(</w:t>)", r"\g<1>Screening (0.05)\g<2>"),
    ("Table 7 present",           r"(<w:t[^>]*>)0\.872(</w:t>)", r"\g<1>0.873\g<2>"),
    ("Table 7 absent",            r"(<w:t[^>]*>)0\.845(</w:t>)", r"\g<1>0.846\g<2>"),
    ("Table 8 non-abstained",     r"(<w:t[^>]*>)0\.881(</w:t>)", r"\g<1>0.880\g<2>"),
]


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)

    doc = tmp / "word" / "document.xml"
    xml = doc.read_text(encoding="utf8")

    ok, warn = 0, []
    for label, old, new in EDITS:
        n = xml.count(old)
        if n == 1:
            xml = xml.replace(old, new)
            ok += 1
        else:
            warn.append(f"{label}: found {n} occurrences, skipped")

    for label, pat, rep in CELL_EDITS:
        xml, n = re.subn(pat, rep, xml, count=1)
        if n == 1:
            ok += 1
        else:
            warn.append(f"{label}: no match, skipped")

    doc.write_text(xml, encoding="utf8")
    if dst.exists():
        dst.unlink()
    subprocess.run(["zip", "-Xrq", str(dst.resolve()), "."], cwd=tmp, check=True)
    shutil.rmtree(tmp)

    print(f"applied {ok}/{len(EDITS) + len(CELL_EDITS)} edits")
    for w in warn:
        print("  ! " + w)


if __name__ == "__main__":
    main()
