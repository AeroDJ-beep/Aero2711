const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, LevelFormat,
  TableOfContents, PageBreak, Footer, PageNumber, TabStopType,
} = require('docx');

const FIG = path.join(__dirname, 'figs');
const OUT = process.argv[2] || path.join(__dirname, 'AERO2711_Progress_Report.docx');

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold**, *italic*, ^sup^ , ~sub~
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('*')) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else if (t.startsWith('^')) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: 120 }, ...(o.p || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: /^(Abstract|Table of Contents|References)/.test(t) });
const H1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: 'bul', level: lvl }, children: runs(t), alignment: AlignmentType.LEFT, spacing: { after: 60 } });
const N = (t, ref = 'num') => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), alignment: AlignmentType.LEFT, spacing: { after: 60 } });
let eqN = 0;
const EQ = (t, _latex, num = ++eqN) => new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { before: 60, after: 120 },
  tabStops: [{ type: TabStopType.CENTER, position: 4200 }, { type: TabStopType.RIGHT, position: 8900 }],
  children: [new TextRun({ text: '\t' }), ...runs(t, { font: 'Cambria Math' }), new TextRun({ text: num ? `\t(${num})` : '' })],
});
let figN = 0;
function FIGURE(file, widthCm, caption) {
  caption = `Figure ${++figN}. ` + caption;
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthCm / 2.54 * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: buf, transformation: { width: wpx, height: wpx * h / w } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200, line: 276 }, children: runs(caption, { size: 20 }) }),
  ];
}
const TW = 9072; // text width DXA: A4 11906 - 1701 - 1134 ≈ 9071
const border = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const borders = { top: border, bottom: border, left: border, right: border };
let tabN = 0;
function TABLE(widths, rows, caption) {
  caption = `Table ${++tabN}. ` + caption;
  const tot = widths.reduce((a, b) => a + b, 0);
  const W = widths.map((w) => Math.round(w / tot * TW));
  W[W.length - 1] += TW - W.reduce((a, b) => a + b, 0);
  const trs = rows.map((r, i) => new TableRow({ cantSplit: true,
    tableHeader: i === 0,
    children: r.map((c, j) => new TableCell({
      borders, width: { size: W[j], type: WidthType.DXA },
      shading: i === 0 ? { fill: 'DCE6F1', type: ShadingType.CLEAR } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ spacing: { line: 240, after: 0 }, alignment: AlignmentType.LEFT,
        children: runs(String(c), { size: 19, bold: i === 0 }) })],
    })),
  }));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80, line: 276 }, keepNext: true, children: runs(caption, { size: 20 }) }),
    new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: W, rows: trs }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const PH = (t) => new TextRun({ text: t, highlight: 'yellow' });


const C = [];
// Title page
C.push(
  new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'AERO2711 Research Project — Progress Report', size: 26, color: '555555' })] }),
  new Paragraph({ spacing: { before: 400, after: 200, line: 300 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Visual Active Tracking of Manoeuvring Targets under Corner Occlusion', bold: true, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 }, children: [new TextRun({ text: 'Map-constrained prediction and decoupled blind-phase control for a drone-mounted camera', italics: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Student: ', bold: true }), PH('[Full name], SID [xxxxxxxxx]')] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Supervisor: ', bold: true }), PH('[Title] Zi [Surname]')] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Co-supervisor: ', bold: true }), PH('[name, or delete this line]')] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [PH('[School / Faculty], [University]')] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun('September 2026')] }),
);


const build = require('./content.js');
const TOC = () => [H1('Table of Contents'), new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-2' })];
const res = build({ P, B, N, H1, H2, EQ, FIGURE, TABLE, TOC });
for (const b of res.C) C.push(...(Array.isArray(b) ? b : [b]));
C.push(H1('References'));
res.refs.forEach((r, i) => C.push(new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { after: 80, line: 276 },
  indent: { left: 567, hanging: 567 },
  children: [new TextRun({ text: `[${i + 1}]	`, size: 22 }), ...runs(r, { size: 22 })],
  tabStops: [{ type: TabStopType.LEFT, position: 567 }],
})));

// ---------- document ----------
const doc = new Document({
  creator: 'AERO2711 student',
  title: 'AERO2711 Progress Report',
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 24 }, paragraph: { spacing: { line: 360 } } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } },
                                   { level: 1, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1134, hanging: 283 } } } }] },
      { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 397 } } } }] },
      { reference: 'contrib', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: 'C%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 680, hanging: 510 } } } }] },
      { reference: 'num2', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 397 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1701, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
