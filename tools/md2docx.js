/**
 * Convert the manuscript Markdown into a two-column journal-style Word file.
 *
 * Layout follows the template used for the author's brain-tumour manuscript:
 * a single-column front matter (title, authors, boxed abstract, keywords),
 * then a two-column body. Tables and figures break out to full width via
 * continuous section breaks, which is how journals typeset wide floats.
 *
 * Only the Markdown constructs this manuscript uses are handled: ATX headings,
 * paragraphs with inline bold/italic/code, pipe tables, images with captions,
 * blockquote notes, bullet and numbered lists, horizontal rules.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, SectionType, LevelFormat, convertInchesToTwip,
} = require("./node_modules/docx");

const ROOT = path.resolve(__dirname, "..");
const SRC = path.join(ROOT, "paper", "manuscript.md");
const OUT = path.join(ROOT, "paper", "Asthma_Manuscript.docx");

const CONTENT_DXA = 9360;                  // Letter 12240 - 2 x 1440 margins
const COL_DXA = Math.floor((CONTENT_DXA - 400) / 2);

const SERIF = "Times New Roman";
const MONO = "Consolas";

const NAVY = "1F3864";                     // headings, table header fill
const BLUE = "1F4E79";                     // subheadings, accent bar
const BANNER = "DCE6F1";                   // section heading background
const BOXBG = "F7FAFC";                    // abstract box fill
const INK = "1A1A1A";
const MUTED = "5A5A5A";
const RULE = "AFC1D6";

const BODY_PT = 19;                        // half-points -> 9.5pt (two-column)

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

/** Split inline markdown into styled runs. */
function runs(text, o = {}) {
  const base = {
    font: o.font || SERIF,
    size: o.size || BODY_PT,
    color: o.color || INK,
    bold: o.bold || false,
    italics: o.italics || false,
  };
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ ...base, text: tok.slice(2, -2), bold: true }));
    else if (tok.startsWith("`")) out.push(new TextRun({ ...base, text: tok.slice(1, -1), font: MONO, size: base.size - 2 }));
    else out.push(new TextRun({ ...base, text: tok.slice(1, -1), italics: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out.length ? out : [new TextRun({ ...base, text: "" })];
}

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const HAIR = { style: BorderStyle.SINGLE, size: 3, color: "B7C4D4" };

function bodyPara(text, o = {}) {
  return new Paragraph({
    children: runs(text, o),
    alignment: o.alignment || AlignmentType.JUSTIFIED,
    spacing: { line: o.line || 230, after: o.after == null ? 110 : o.after, before: o.before || 0 },
  });
}

/** Section heading: navy text on a pale banner with an accent bar. */
function sectionHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 130 },
    shading: { type: ShadingType.CLEAR, fill: BANNER, color: "auto" },
    border: {
      left: { style: BorderStyle.SINGLE, size: 18, color: BLUE, space: 6 },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 2 },
    },
    indent: { left: 90 },
    children: [new TextRun({
      text: text.toUpperCase(), bold: true, font: SERIF, size: 21, color: NAVY,
    })],
  });
}

function subHeading(text, level) {
  return new Paragraph({
    heading: level === 3 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: 190, after: 90 },
    children: [new TextRun({
      text, bold: true, font: SERIF, size: level === 3 ? 20 : 19, color: BLUE,
    })],
  });
}

/** Full-width table with a navy header row. */
function buildTable(rows) {
  const nCols = Math.max(...rows.map((r) => r.length));
  const first = nCols >= 5 ? Math.round(CONTENT_DXA * 0.15) : Math.round(CONTENT_DXA / nCols);
  const rest = Math.floor((CONTENT_DXA - first) / (nCols - 1 || 1));
  const widths = [first, ...Array(nCols - 1).fill(rest)];
  widths[widths.length - 1] += CONTENT_DXA - widths.reduce((a, b) => a + b, 0);

  const size = nCols >= 6 ? 13 : nCols >= 4 ? 16 : 17;

  const trs = rows.map((cells, ri) => new TableRow({
    tableHeader: ri === 0,
    children: Array.from({ length: nCols }, (_, ci) => {
      const raw = (cells[ci] || "").trim();
      const head = ri === 0;
      return new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: head
          ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" }
          : { type: ShadingType.CLEAR, fill: ri % 2 ? "FFFFFF" : "F4F7FA", color: "auto" },
        margins: { top: 55, bottom: 55, left: 85, right: 85 },
        borders: { top: HAIR, bottom: HAIR, left: HAIR, right: HAIR },
        children: [new Paragraph({
          children: runs(raw, {
            size, color: head ? "FFFFFF" : INK, bold: head,
          }),
          spacing: { line: 210, after: 0 },
          alignment: AlignmentType.LEFT,
        })],
      });
    }),
  }));

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    rows: trs,
  });
}

function figureParagraph(rel) {
  const file = path.join(ROOT, "paper", rel);
  if (!fs.existsSync(file)) return bodyPara(`[missing figure: ${rel}]`, { color: "AA0000" });
  const { width, height } = pngSize(file);
  const maxW = 630;                                    // ~6.5in at 96dpi
  const w = Math.min(maxW, width);
  const h = Math.round((height / width) * w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 60 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(file),
      transformation: { width: w, height: h },
    })],
  });
}

function caption(text, opts = {}) {
  return new Paragraph({
    children: runs(text, { size: 17, color: BLUE, bold: true }),
    alignment: opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
    spacing: { line: 210, before: opts.left ? 140 : 40, after: opts.left ? 60 : 200 },
  });
}

function rule() {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE } },
    children: [new TextRun("")],
  });
}

// ------------------------------------------------------------------ parse
const lines = fs.readFileSync(SRC, "utf8").split("\n");
const blocks = [];
let i = 0;

const isTableLine = (s) => /^\s*\|.*\|\s*$/.test(s);
const isSep = (s) => /^\s*\|[\s:|-]+\|\s*$/.test(s);

while (i < lines.length) {
  const raw = lines[i];
  const t = raw.trim();
  if (!t) { i++; continue; }

  if (t === "---") { i++; continue; }                  // rules handled by layout

  const img = t.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
  if (img) { blocks.push({ kind: "figure", src: img[1] }); i++; continue; }

  const h = t.match(/^(#{1,4})\s+(.*)$/);
  if (h) {
    blocks.push({ kind: "heading", level: h[1].length, text: h[2].replace(/\*\*/g, "") });
    i++; continue;
  }

  if (isTableLine(raw)) {
    const rows = [];
    while (i < lines.length && isTableLine(lines[i])) {
      if (!isSep(lines[i])) rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|"));
      i++;
    }
    if (rows.length) blocks.push({ kind: "table", rows });
    continue;
  }

  if (t.startsWith(">")) {
    const buf = [];
    while (i < lines.length && lines[i].trim().startsWith(">")) {
      buf.push(lines[i].trim().replace(/^>\s?/, "")); i++;
    }
    const text = buf.join(" ").replace(/\s+/g, " ").trim();
    if (text) blocks.push({ kind: "note", text });
    continue;
  }

  const bullet = t.match(/^[-*]\s+(.*)$/);
  if (bullet) { blocks.push({ kind: "bullet", text: bullet[1] }); i++; continue; }

  const num = t.match(/^(\d+)\.\s+(.*)$/);
  if (num) {
    let text = num[2];
    while (i + 1 < lines.length && /^\s{2,}\S/.test(lines[i + 1]) && !isTableLine(lines[i + 1])) {
      text += " " + lines[i + 1].trim(); i++;
    }
    blocks.push({ kind: "numbered", n: num[1], text });
    i++; continue;
  }

  const buf = [t]; i++;
  while (i < lines.length) {
    const n = lines[i].trim();
    if (!n || n === "---" || /^#{1,4}\s/.test(n) || isTableLine(lines[i]) ||
        n.startsWith(">") || /^!\[/.test(n) || /^[-*]\s/.test(n) || /^\d+\.\s/.test(n)) break;
    buf.push(n); i++;
  }
  blocks.push({ kind: "para", text: buf.join(" ") });
}

// --------------------------------------------------- front matter / body
const titleBlock = blocks.find((b) => b.kind === "heading" && b.level === 1);
const startIdx = blocks.findIndex((b) => b.kind === "heading" && /INTRODUCTION/i.test(b.text));

const front = [];
front.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 160 },
  children: [new TextRun({
    text: (titleBlock ? titleBlock.text : "Manuscript").toUpperCase(),
    bold: true, font: SERIF, size: 30, color: NAVY,
  })],
}));
front.push(rule());
front.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Adekoya David Olusegun, R. S. Akinbo", font: SERIF, size: 21, color: INK })],
}));
front.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({
    text: "Department of Information Technology",
    italics: true, font: SERIF, size: 18, color: MUTED,
  })],
}));
front.push(rule());

// Draft note, abstract and keywords, all still single column.
for (const b of blocks.slice(0, startIdx)) {
  if (b.kind === "note") {
    front.push(new Paragraph({
      children: runs(b.text, { size: 16, color: MUTED }),
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: 200, before: 60, after: 140 },
      indent: { left: 200, right: 200 },
      shading: { type: ShadingType.CLEAR, fill: "FFF7E6", color: "auto" },
      border: { left: { style: BorderStyle.SINGLE, size: 14, color: "D9A441", space: 6 } },
    }));
  } else if (b.kind === "heading" && /ABSTRACT/i.test(b.text)) {
    front.push(sectionHeading("Abstract"));
  } else if (b.kind === "para") {
    const kw = b.text.match(/^\*\*Keywords:\*\*\s*(.*)$/);
    if (kw) {
      front.push(new Paragraph({
        spacing: { before: 120, after: 60 },
        children: [
          new TextRun({ text: "Keywords: ", bold: true, font: SERIF, size: 18, color: BLUE }),
          new TextRun({ text: kw[1], font: SERIF, size: 18, color: INK }),
        ],
      }));
    } else {
      front.push(new Paragraph({
        children: runs(b.text, { size: 18 }),
        alignment: AlignmentType.JUSTIFIED,
        spacing: { line: 220, after: 100 },
        indent: { left: 160, right: 160 },
        shading: { type: ShadingType.CLEAR, fill: BOXBG, color: "auto" },
        border: {
          top: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 },
          left: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 },
          right: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 },
        },
      }));
    }
  }
}

/** Render one body block into the flowing two-column stream. */
function renderFlow(b) {
  switch (b.kind) {
    case "heading":
      return b.level === 2 ? sectionHeading(b.text) : subHeading(b.text, b.level);
    case "para": {
      if (/^\*\*Table\s/.test(b.text)) {
        const p = caption(b.text, { left: true });
        p.__isTableCaption = true;
        return p;
      }
      if (/^\*\*Figure\s/.test(b.text)) return caption(b.text);
      return bodyPara(b.text);
    }
    case "note":
      return new Paragraph({
        children: runs(b.text, { size: 16, color: MUTED, italics: true }),
        spacing: { line: 200, after: 120 },
      });
    case "bullet":
      return new Paragraph({
        children: runs(b.text), bullet: { level: 0 },
        alignment: AlignmentType.JUSTIFIED,
        spacing: { line: 220, after: 60 },
      });
    case "numbered":
      return new Paragraph({
        children: [
          new TextRun({ text: `${b.n}.  `, font: SERIF, size: BODY_PT, color: INK }),
          ...runs(b.text, { size: BODY_PT }),
        ],
        alignment: AlignmentType.LEFT,
        spacing: { line: 215, after: 70 },
        indent: { left: 300, hanging: 300 },
      });
    default:
      return null;
  }
}

// Tables and figures break out of the columns; text flows in two columns.
const sections = [{
  properties: {
    page: {
      size: { width: 12240, height: 15840 },
      margin: {
        top: convertInchesToTwip(0.9), bottom: convertInchesToTwip(0.9),
        left: convertInchesToTwip(0.85), right: convertInchesToTwip(0.85),
      },
    },
  },
  children: front,
}];

const twoCol = {
  type: SectionType.CONTINUOUS,
  column: { count: 2, space: 380, equalWidth: true },
};
const oneCol = { type: SectionType.CONTINUOUS, column: { count: 1 } };

let buf = [];
const flushTwoCol = () => {
  if (buf.length) { sections.push({ properties: twoCol, children: buf }); buf = []; }
};

for (let k = startIdx; k < blocks.length; k++) {
  const b = blocks[k];
  if (b.kind === "table" || b.kind === "figure") {
    flushTwoCol();
    const kids = [];
    if (b.kind === "table") {
      // A caption paragraph immediately before the table belongs above it.
      const prev = blocks[k - 1];
      if (prev && prev.kind === "para" && /^\*\*Table\s/.test(prev.text)) {
        if (buf.length && buf[buf.length - 1].__isTableCaption) buf.pop();
        kids.push(caption(prev.text, { left: true }));
      }
      kids.push(buildTable(b.rows));
      kids.push(new Paragraph({ spacing: { after: 140 }, children: [new TextRun("")] }));
    } else {
      kids.push(figureParagraph(b.src));
      // Pull the caption paragraph up so it stays with its figure.
      const nxt = blocks[k + 1];
      if (nxt && nxt.kind === "para" && /^\*\*Figure\s/.test(nxt.text)) {
        kids.push(caption(nxt.text)); k++;
      }
    }
    sections.push({ properties: oneCol, children: kids });
    continue;
  }
  const p = renderFlow(b);
  if (p) buf.push(p);
}
flushTwoCol();

const doc = new Document({
  creator: "Adekoya David Olusegun",
  title: titleBlock ? titleBlock.text : "Manuscript",
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 200 } } },
      }],
    }],
  },
  sections,
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(OUT, b);
  console.log(`wrote ${OUT} (${(b.length / 1024).toFixed(0)} KB, ${sections.length} sections)`);
});
