"""Structural and consistency checks for a manuscript .docx. Exit 1 on any failure.

Catches the defects that have reached reviewers before: tables cited only by
their own caption, section references to renumbered sections, citations out of
first-appearance order, uncited references, stale or over-strong phrases, and
a package that does not open.

Usage:  python tools/check_manuscript.py FILE.docx [--highlighted]
"""
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docxblocks as D  # noqa: E402

BANNED = ["cannot abstain", "cannot report", "only expressible", "cannot be evaluated for graceful",
          "approximately six times", "six-fold", "cross-modal", "multi-agent multimodal",
          "Research questions", "RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "A reviewer may", "reviewer",
          "Reviewer", "0.973", "0.515", " 949 ", "Section 3.12", "Section 3.10", "TODO", "K:"]


def main(path, highlighted):
    fails = []
    z = zipfile.ZipFile(path)
    if z.testzip() is not None:
        fails.append("zip is corrupt")
    from docx import Document
    Document(path)
    xml = z.read("word/document.xml").decode()
    _, blocks, _ = D.split(xml)
    texts = [D.text(b) for b in blocks]
    i_ref = texts.index("REFERENCES")
    i_app = next(i for i, x in enumerate(texts) if x.startswith("APPENDIX A"))
    body_t = texts[:i_ref]
    body = "\n".join(body_t)

    heads = {m.group(1) for x in texts if len(x) < 110
             for m in [re.match(r'^(\d+(?:\.\d+)?)\.? [A-Z]', x)] if m}
    for m in re.finditer(r'Sections? (\d+(?:\.\d+)?)', body):
        if m.group(1) not in heads:
            fails.append(f"unresolved section reference: {m.group(0)}")

    caps = {}
    for x in body_t:
        m = re.match(r'^(Table|Figure) (\d+)\.', x)
        if m:
            caps.setdefault(m.group(1), []).append(int(m.group(2)))
    nocap = "\n".join(x for x in body_t if not re.match(r'^(Table|Figure) \d+\.', x))
    for kind, nums in caps.items():
        if nums != sorted(nums) or nums != list(range(1, len(nums) + 1)):
            fails.append(f"{kind} numbers out of order or gapped: {nums}")
        for k in nums:
            if not re.search(rf'\b{kind}s? (?:\d+(?:, | and | to |-))*{k}\b|{kind} {k}\(', nocap):
                fails.append(f"{kind} {k} is never cited in the running text")

    order = []
    for m in re.finditer(r'\[(\d+(?:, \d+)*)\]', body):
        for k in map(int, m.group(1).split(', ')):
            if k not in order:
                order.append(k)
    if order != sorted(order):
        fails.append("citations are not in order of first appearance")
    refs = [int(m.group(1)) for x in texts[i_ref + 1:i_app] for m in [re.match(r'\[(\d+)\]', x)] if m]
    if refs != list(range(1, len(refs) + 1)):
        fails.append("reference list is not numbered 1..N in order")
    if set(order) != set(refs):
        fails.append(f"cited vs listed mismatch: {set(order) ^ set(refs)}")

    full = "\n".join(texts)
    for ph in BANNED:
        if ph in full:
            at = full.index(ph)
            fails.append(f"banned/stale phrase {ph!r}: ...{full[max(0, at - 40):at + 60]!r}")

    n_hl = xml.count('<w:highlight w:val="yellow"/>')
    if highlighted and n_hl == 0:
        fails.append("highlighted copy contains no highlighting")
    if not highlighted and n_hl:
        fails.append(f"clean copy contains {n_hl} highlight marks")
    # schema order: highlight must follow szCs inside a run property block
    if re.search(r'<w:highlight[^>]*/>(?:<w:szCs)', xml):
        fails.append("highlight precedes szCs (schema order)")

    words = len(full.split())
    print(f"{Path(path).name}: {len(blocks)} blocks, {len(refs)} references, "
          f"tables {caps.get('Table', [])[-1:]}, figures {caps.get('Figure', [])[-1:]}, "
          f"{words:,} words, {n_hl} highlight marks")
    for f in fails:
        print("  FAIL:", f)
    print("  ALL CHECKS PASSED" if not fails else f"  {len(fails)} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--highlighted" in sys.argv))
