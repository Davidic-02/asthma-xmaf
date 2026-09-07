"""Insert the manuscript's display equations into the Word file as OMML.

Each equation states what the code in src/ actually computes; the comment
above each one names the file it corresponds to, so the two can be checked
against each other.

Usage:  python tools/add_equations.py <in.docx> <out.docx>
"""

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omml import (  # noqa: E402
    delim, equation_paragraph, frac, func, nary, op, r, sub, subsup, sup,
)

# --------------------------------------------------------------- equations
# (1) standardisation - src/fusion/decision_agent.py, fit_agent()
EQ1 = (
    sub(r("z"), op("ij")) + op(" = ") +
    frac(sub(r("x"), op("ij")) + op(" − ") + sub(r("μ"), op("j")),
         sub(r("σ"), op("j")))
)

# (2) agent linear score - LogisticRegression on standardised features
EQ2 = (
    sub(r("η"), op("a")) + delim(r("x")) + op(" = ") +
    subsup(r("β"), op("a"), op("0")) + op(" + ") +
    nary("∑", r("j") + op(" = ") + op("1"), op("p"),
         sub(r("β"), op("aj")) + sub(r("z"), op("ij")))
)

# (3) sigmoid (Platt) calibration - CalibratedClassifierCV(method="sigmoid")
EQ3 = (
    sub(r("p"), op("a")) + delim(r("x")) + op(" = ") +
    frac(op("1"),
         op("1 + ") + func("exp") +
         delim(sub(r("A"), op("a")) + sub(r("η"), op("a")) + delim(r("x")) +
               op(" + ") + sub(r("B"), op("a"))))
)

# (4) exact Shapley values for a standardised linear model - agent_shap()
EQ4 = (
    sub(r("φ"), op("aj")) + delim(r("x")) + op(" = ") +
    sub(r("β"), op("aj")) + delim(sub(r("z"), op("ij")) + op(" − ") +
                                  sub(op("z̄"), op("j")))
)

# (5) decision-agent fusion over agent evidence - meta learner
EQ5 = (
    sub(r("p"), op("fused")) + op(" = ") + r("σ") +
    delim(sub(r("γ"), op("0")) + op(" + ") +
          sub(r("γ"), op("1")) + sub(r("p"), op("sym")) + op(" + ") +
          sub(r("γ"), op("2")) + sub(r("p"), op("pul")) + op(" + ") +
          sub(r("γ"), op("3")) + r("m"))
)

# (6) abstention - refer cases nearest the operating threshold t
EQ6 = (
    func("refer") + delim(r("x")) + op(" = ") + op("1") + op("  ⟺  ") +
    delim(sub(r("p"), op("fused")) + op(" − ") + r("t"), "|", "|") +
    op(" ≤ ") + sub(r("Q"), r("α")) +
    delim(delim(sub(r("p"), op("fused")) + op(" − ") + r("t"), "|", "|"))
)

# (7) inter-agent conflict flag - CONFLICT_GAP = 0.35
EQ7 = (
    func("conflict") + delim(r("x")) + op(" = ") + op("1") + op("  ⟺  ") +
    delim(sub(r("p"), op("sym")) + op(" − ") + sub(r("p"), op("pul")),
          "|", "|") + op(" > ") + r("τ") + op("   and   ") + r("m") +
    op(" = 1")
)

# (8) comprehensiveness - deletion of the cited feature set R
EQ8 = (
    func("comp") + op(" = ") +
    frac(op("1"), r("N")) +
    nary("∑", r("i") + op(" = ") + op("1"), r("N"),
         delim(r("p") + delim(sub(r("x"), op("i"))) + op(" − ") +
               r("p") + delim(sub(r("x"), op("i")) + op(" ∖ ") +
                              sub(r("R"), op("i")))))
)

# (9) sufficiency - retaining only the cited feature set
EQ9 = (
    func("suff") + op(" = ") +
    frac(op("1"), r("N")) +
    nary("∑", r("i") + op(" = ") + op("1"), r("N"),
         delim(r("p") + delim(sub(r("x"), op("i"))) + op(" − ") +
               r("p") + delim(subsup(r("x"), op("i"), sub(r("R"), op("i"))))))
)

# (10) band energy ratio - src/agents/extract_audio_features.py
EQ10 = (
    sub(r("E"), r("b")) + op(" = ") +
    frac(nary("∑", r("t"), op(""),
              nary("∑", r("f") + op(" ∈ ") + r("b"), op(""),
                   sup(delim(r("S") + delim(r("t") + op(", ") + r("f"))), op("2")))),
         nary("∑", r("t"), op(""),
              nary("∑", r("f"), op(""),
                   sup(delim(r("S") + delim(r("t") + op(", ") + r("f"))), op("2")))))
)

# (11) wheeze tonality - peak-to-mean spectral ratio in 100-1600 Hz
EQ11 = (
    r("T") + delim(r("t")) + op(" = ") +
    frac(func("max") + op(" ") +
         r("S") + delim(r("t") + op(", ") + r("f")),
         frac(op("1"), delim(r("F"), "|", "|")) +
         nary("∑", r("f") + op(" ∈ ") + r("F"), op(""),
              r("S") + delim(r("t") + op(", ") + r("f"))))
)

# (12) patient-level aggregation of clip probabilities
EQ12 = (
    sub(r("p"), op("patient")) + delim(r("i")) + op(" = ") +
    frac(op("1"), delim(sub(r("R"), op("i")), "|", "|")) +
    nary("∑", r("r") + op(" ∈ ") + sub(r("R"), op("i")), op(""),
         r("p") + delim(sub(r("x"), r("r"))))
)

EQUATIONS = [EQ1, EQ2, EQ3, EQ4, EQ5, EQ6, EQ7, EQ8, EQ9, EQ10, EQ11, EQ12]

# Anchor text -> equations to insert after that paragraph. Display numbers are
# assigned from where each anchor actually falls in the document, so the
# equations read (1), (2), (3)... in order regardless of how this list is
# arranged.
PLACEMENT = [
    ("Band-energy ratios over six physiologically motivated bands", [EQ10, EQ11]),
    ("All agents use L2-regularised logistic regression", [EQ1, EQ2, EQ3]),
    ("Clip probabilities are averaged per child", [EQ12]),
    ("A logistic meta-learner over the agents", [EQ5]),
    ("The 15% of cases lying closest to the operating threshold", [EQ6, EQ7]),
    ("For the standardised linear agents the exact Shaple", [EQ4]),
    ("equal number of randomly chosen features is manipulated", [EQ8, EQ9]),
]


def paragraph_end_after(xml: str, needle: str) -> int:
    """Index just past the </w:p> of the paragraph containing `needle`."""
    i = xml.find(needle)
    if i < 0:
        return -1
    j = xml.find("</w:p>", i)
    return j + len("</w:p>") if j > 0 else -1


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)

    doc = tmp / "word" / "document.xml"
    xml = doc.read_text(encoding="utf8")

    # Word needs the math namespace declared on the document root.
    if 'xmlns:m=' not in xml.split(">", 2)[1]:
        xml = re.sub(
            r"(<w:document\b)",
            r'\1 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"',
            xml, count=1)

    # Order the anchors by where they actually occur, then number the
    # equations sequentially in that reading order.
    located, missing = [], []
    for needle, eqs in PLACEMENT:
        pos = paragraph_end_after(xml, needle)
        if pos < 0:
            missing.append(needle)
        else:
            located.append((pos, needle, eqs))
    located.sort()

    numbered, n = [], 0
    for pos, needle, eqs in located:
        block = []
        for eq in eqs:
            n += 1
            block.append(equation_paragraph(eq, n))
        numbered.append((pos, "".join(block)))

    inserted = n
    # Insert from the end backwards so earlier offsets stay valid.
    for pos, block in reversed(numbered):
        xml = xml[:pos] + block + xml[pos:]

    doc.write_text(xml, encoding="utf8")

    if dst.exists():
        dst.unlink()
    subprocess.run(["zip", "-Xrq", str(dst.resolve()), "."], cwd=tmp, check=True)
    shutil.rmtree(tmp)

    print(f"inserted {inserted} equations")
    for m in missing:
        print(f"  ! anchor not found: {m[:60]!r}")


if __name__ == "__main__":
    main()
