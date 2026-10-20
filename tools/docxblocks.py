"""Top-level block parsing for word/document.xml.

A body is a flat run of <w:p> and <w:tbl> children followed by one <w:sectPr>.
Table cells contain their own paragraphs, so a non-greedy regex over
<w:p>...</w:p> splits tables apart; this scanner walks tag depth instead and
returns each table whole. split() and join() round-trip byte for byte, which
callers should assert before editing.
"""
import re

TXT = re.compile(r'<w:t(?: [^>]*)?>(.*?)</w:t>', re.S)
_OPEN = re.compile(r'<(/?)w:(p|tbl)(?=[\s>/])')


def split(xml):
    """Return (head, blocks, tail): xml before the body content, the top-level
    blocks, and xml after them (the final sectPr and closing tags)."""
    b0 = xml.index('<w:body>') + len('<w:body>')
    final = xml.rindex('<w:sectPr')
    head, body, tail = xml[:b0], xml[b0:final], xml[final:]
    blocks, i = [], 0
    while i < len(body):
        m = re.compile(r'<w:(p|tbl)(?=[\s>/])').match(body, i)
        if not m:
            raise ValueError(f"unexpected content at {i}: {body[i:i+60]!r}")
        tag, depth, j = m.group(1), 0, i
        for t in _OPEN.finditer(body, i):
            if t.group(2) != tag:
                continue
            depth += -1 if t.group(1) else 1
            if depth == 0:
                j = body.index('>', t.end()) + 1
                break
        blocks.append(body[i:j])
        i = j
    assert ''.join(blocks) == body, "block split does not round-trip"
    return head, blocks, tail


def join(head, blocks, tail):
    return head + ''.join(blocks) + tail


def text(block):
    import html
    return html.unescape(''.join(TXT.findall(block))).strip()


def kind(block):
    if block.startswith('<w:tbl'):
        return 'TBL'
    if '<w:drawing' in block or '<w:pict' in block:
        return 'IMG'
    if '<m:oMath' in block:
        return 'EQ'
    if '<w:sectPr' in block:
        return 'SECT'
    return 'P'
