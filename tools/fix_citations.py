"""Repair citation hygiene in the manuscript.

Every reference was verified against Crossref, arXiv or the ACL Anthology, and
all 17 entries are accurate. The problems are not fabricated sources but
usage: four entries appear in the list without ever being cited in the text,
and two of those describe methods the study actually uses (SHAP and the
ERASER faithfulness metrics). Using a method without citing its source is a
substantive omission, not a formatting nicety.

This script:
  * cites [16] where Shapley values are introduced;
  * cites [17] where comprehensiveness and sufficiency are defined;
  * cites [13] where NHANES is first named in the Methods;
  * cites [15] alongside the other clinical ML studies in Related Works;
  * moves editorial commentary out of the reference entries, where it does not
    belong, keeping the substantive caveat about ERASER in the Methods text;
  * completes two entries with the issue numbers confirmed by Crossref.

Usage:  python tools/fix_citations.py <in.docx> <out.docx>
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EDITS = [
    # --- inline citations for methods the study relies on -----------------
    ("cite SHAP [16] at the Shapley definition",
     "agents the exact Shapley value of feature ",
     "agents the exact Shapley value [16] of feature "),

    ("cite ERASER [17] at the faithfulness definitions",
     "We instead apply deletion-based tests. ",
     "We instead apply the deletion-based tests introduced by the ERASER benchmark [17]. "),

    ("cite NHANES [13] at first use in Methods",
     "Data were drawn from NHANES cycles 2007–2008, 2009–2010 and 2011–2012",
     "Data were drawn from the National Health and Nutrition Examination Survey (NHANES) [13], cycles 2007–2008, 2009–2010 and 2011–2012"),

    # --- tidy the reference entries themselves ----------------------------
    ("remove editorial note from [16]",
     ". 2017. — Primary source for SHAP.",
     ". 2017:4765-4774. arXiv:1705.07874"),

    ("remove editorial note from [17]",
     "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)</w:t>",
     "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics</w:t>"),

    ("trim editorial note from [17] tail",
     ". 2020:4443-4458. arXiv:1911.03429. \u2014 Origin of the comprehensiveness and sufficiency metrics used in Section 3.4. Note that subsequent work criticises these metrics for lacking random baselines; our use of a random-feature control (Section 4.7) directly addresses that critique.",
     ". 2020:4443-4458. doi:10.18653/v1/2020.acl-main.408"),

    ("cite [15] among the clinical ML studies",
     "with SHAP interpretation [1, 2, 9].",
     "with SHAP interpretation [1, 2, 9, 15]."),

    # --- complete two entries with details confirmed by Crossref ----------
    # Journal titles sit in their own italic runs, so the volume text is
    # matched separately from the name.
    ("correct journal name for [11]",
     "<w:t>Biomedica</w:t>",
     "<w:t>Biom\u00e9dica</w:t>"),

    ("add issue number to [11]",
     ". 2025;45:277-285.",
     ". 2025;45(2):277-285."),

    ("add issue number to [12]",
     ". 2026;12:01246-2025.",
     ". 2026;12(3):01246-2025."),
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
            warn.append(f"{label}: {n} occurrences, skipped")

    doc.write_text(xml, encoding="utf8")
    if dst.exists():
        dst.unlink()
    subprocess.run(["zip", "-Xrq", str(dst.resolve()), "."], cwd=tmp, check=True)
    shutil.rmtree(tmp)

    print(f"applied {ok}/{len(EDITS)} edits")
    for w in warn:
        print("  ! " + w)


if __name__ == "__main__":
    main()
