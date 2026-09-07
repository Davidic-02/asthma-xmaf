"""Build Word OMML (Office Math Markup) equations in Cambria Math.

Word stores equations as OMML, not as images, so equations produced here stay
editable in Word's equation editor and print as real maths rather than
rasterised pictures. Every mathematical run must carry the Cambria Math font,
which is what makes Word treat it as an equation object.

Only the constructs this manuscript needs are implemented: runs, subscripts,
superscripts, fractions, n-ary sums, delimiters and function names.
"""

M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
CAMBRIA = ('<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
           'w:cs="Cambria Math"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>')


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def r(text: str, *, italic: bool = True) -> str:
    """A mathematical run. Variables are italic; operators and digits are not."""
    style = "" if italic else '<m:sty m:val="p"/>'
    props = f"<m:rPr>{style}</m:rPr>" if style else ""
    return f"<m:r>{props}{CAMBRIA}<m:t>{esc(text)}</m:t></m:r>"


def op(text: str) -> str:
    """An upright operator, digit or punctuation run."""
    return r(text, italic=False)


def func(name: str) -> str:
    """An upright function name such as exp, max, min."""
    return op(name)


def sub(base: str, s: str) -> str:
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(base: str, s: str) -> str:
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>"


def subsup(base: str, sb: str, sp: str) -> str:
    return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sb}</m:sub>"
            f"<m:sup>{sp}</m:sup></m:sSubSup>")


def frac(num: str, den: str) -> str:
    return f"<m:f><m:fPr><m:type m:val=\"bar\"/></m:fPr><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def delim(inner: str, left: str = "(", right: str = ")") -> str:
    return (f'<m:d><m:dPr><m:begChr m:val="{left}"/><m:endChr m:val="{right}"/>'
            f"</m:dPr><m:e>{inner}</m:e></m:d>")


def nary(chr_: str, lo: str, hi: str, body: str) -> str:
    """An n-ary operator (sum, product) with optional limits.

    An empty limit must be *hidden*, not supplied as an empty element: Word
    renders a vacant math slot as a dotted placeholder box, which is what an
    empty <m:sup/> would produce on screen and in print.
    """
    sub_hide = "1" if not lo.strip() else "0"
    sup_hide = "1" if not hi.strip() else "0"
    return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>'
            f'<m:subHide m:val="{sub_hide}"/><m:supHide m:val="{sup_hide}"/>'
            f"</m:naryPr>"
            f"<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{body}</m:e></m:nary>")


def equation_paragraph(body: str, number: int, *, col_twips: int = 4320) -> str:
    """A centred display equation with a right-aligned number.

    Tab stops rather than spaces position the number, so it stays at the
    column edge when the text reflows.
    """
    centre = col_twips // 2
    return (
        "<w:p>"
        "<w:pPr>"
        "<w:tabs>"
        f'<w:tab w:val="center" w:pos="{centre}"/>'
        f'<w:tab w:val="right" w:pos="{col_twips}"/>'
        "</w:tabs>"
        '<w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/>'
        "</w:pPr>"
        f"<w:r>{CAMBRIA}<w:tab/></w:r>"
        f"<m:oMath>{body}</m:oMath>"
        f"<w:r>{CAMBRIA}<w:tab/></w:r>"
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
        '<w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
        f"<w:t>({number})</w:t></w:r>"
        "</w:p>"
    )
