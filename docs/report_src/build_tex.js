// Generate a LaTeX version of the progress report from the same content
// used by build_report.js (content is shared, only the helpers differ).
const fs = require('fs');
const path = require('path');
const OUTDIR = process.argv[2] || path.join(__dirname, 'latex');

// ---------- text conversion ----------
const UNI = [
  ['ρ̂', '$\\hat{\\rho}$'], ['θ', '$\\theta$'], ['ω', '$\\omega$'], ['ψ', '$\\psi$'], ['ρ', '$\\rho$'],
  ['σ', '$\\sigma$'], ['χ', '$\\chi$'], ['β', '$\\beta$'], ['Δ', '$\\Delta$'], ['≤', '$\\le$'], ['≥', '$\\ge$'],
  ['≈', '$\\approx$'], ['±', '$\\pm$'], ['×', '$\\times$'], ['−', '$-$'], ['→', '$\\rightarrow$'], ['∈', '$\\in$'],
  ['°', '\\textdegree{}'], ['²', '\\textsuperscript{2}'], ['½', '$\\frac{1}{2}$'], ['√', '$\\surd$'], ['·', '$\\cdot$'],
  ['✔', '$\\checkmark$'], ['○', '$\\square$'], ['◐', '$\\boxminus$'],
];
function esc(s) {
  s = s.replace(/\\/g, '\\textbackslash{}')
    .replace(/([&%$#_{}])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}').replace(/\^/g, '\\textasciicircum{}')
    .replace(/--/g, '-{}-')
    .replace(/"([^"]*)"/g, '``$1\'\'');
  for (const [u, l] of UNI) s = s.split(u).join(l);
  // citations: [1], [2]–[5], [19], [20]
  s = s.replace(/\[\d+\](?:(?:–|, )\[\d+\])*/g, (m) => {
    const parts = m.split(/(–|, )/); const ids = []; let prev = null, range = false;
    for (const p of parts) {
      if (p === '–') { range = true; continue; } if (p === ', ') continue;
      const n = +p.slice(1, -1);
      if (range && prev !== null) { for (let k = prev + 1; k <= n; k++) ids.push(k); range = false; } else ids.push(n);
      prev = n;
    }
    return '\\cite{' + ids.map((k) => 'r' + k).join(',') + '}';
  });
  return s;
}
function tex(text) {
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(text)) !== null) {
    out += esc(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**')) out += '\\textbf{' + esc(t.slice(2, -2)) + '}';
    else if (t.startsWith('*')) out += '\\emph{' + esc(t.slice(1, -1)) + '}';
    else if (t.startsWith('^')) out += '\\textsuperscript{' + esc(t.slice(1, -1)) + '}';
    else out += '\\textsubscript{' + esc(t.slice(1, -1)) + '}';
    last = m.index + t.length;
  }
  return out + esc(text.slice(last));
}
const stripNum = (t) => t.replace(/^(Appendix\s+A\s+|\d+(\.\d+)?\s+)/, '').trim();
const stripCap = (c) => c.replace(/^(Figure|Table)\s+[A-Z]?\d+\.\s*/, '');

// ---------- shims (same names as build_report.js helpers) ----------
const C = [];
const AlignmentType = { LEFT: 'l', CENTER: 'c', JUSTIFIED: 'j' };
const P = (t) => ({ t: 'p', s: tex(t) });
const H1 = (t) => (t === 'Table of Contents' ? { t: 'skip' } : { t: 'h1', s: t });
const H1n = H1;
const H2 = (t) => ({ t: 'h2', s: t });
const H3 = (t) => ({ t: 'h3', s: t });
const B = (t) => ({ t: 'item', k: 'bul', s: tex(t) });
const N = (t, ref = 'num') => ({ t: 'item', k: ref, s: tex(t) });
const EQ = (t, latex) => ({ t: 'eq', s: latex });
const FIGURE = (file, w, cap) => [{ t: 'fig', f: file, w: Math.min(w, 15.5), cap: tex(cap) }];
const TABLE = (widths, rows, cap) => [{ t: 'table', widths, rows, cap: tex(cap) }];
const TOC = () => ({ t: 'toc' });
const res = require('./content.js')({ P, B, N, H1, H2, EQ, FIGURE, TABLE, TOC });
for (const b of res.C) C.push(...(Array.isArray(b) ? b : [b]));
C.push({ t: 'refs' });
const REFS = res.refs;

// ---------- emit ----------
let L = [];
let open = null; const used = {};
const closeList = () => { if (open) { L.push(open === 'bul' ? '\\end{itemize}' : '\\end{enumerate}'); open = null; } };
let inAppendix = false; let pendingToc = false;
for (const b of C.flat()) {
  if (b.t !== 'item') closeList();
  switch (b.t) {
    case 'skip': break;
    case 'toc': L.push('\\clearpage', '\\tableofcontents', '\\clearpage'); break;
    case 'h1':
      if (b.s === 'Abstract') { L.push('\\clearpage', '\\section*{Abstract}', '\\addcontentsline{toc}{section}{Abstract}'); break; }
      if (b.s === 'References') break;
      if (b.s.startsWith('Appendix')) { inAppendix = true; L.push('\\clearpage', '\\appendix', '\\renewcommand{\\thetable}{A\\arabic{table}}', '\\setcounter{table}{0}'); }
      L.push(`\\section{${esc(stripNum(b.s))}}`); break;
    case 'h2': L.push(`\\subsection{${esc(stripNum(b.s))}}`); break;
    case 'h3': L.push(`\\subsubsection{${esc(b.s)}}`); break;
    case 'p': L.push(b.s, ''); break;
    case 'item':
      if (open !== b.k) {
        closeList();
        if (b.k === 'bul') L.push('\\begin{itemize}');
        else if (b.k === 'contrib') L.push('\\begin{enumerate}[label=\\textbf{C\\arabic*.}]');
        else L.push(used[b.k] ? '\\begin{enumerate}[resume]' : '\\begin{enumerate}');
        used[b.k] = true; open = b.k;
      }
      L.push('  \\item ' + b.s); break;
    case 'eq':
      L.push('\\begin{equation}', '  ' + b.s, '\\end{equation}'); break;
    case 'fig':
      L.push('\\begin{figure}[htbp]', '  \\centering', `  \\includegraphics[width=${b.w}cm]{figs/${b.f}}`, `  \\caption{${b.cap}}`, '\\end{figure}', ''); break;
    case 'table': {
      const tot = b.widths.reduce((a, c) => a + c, 0);
      const spec = b.widths.map((w) => `>{\\raggedright\\arraybackslash}p{${(w / tot * 0.97 - 0.02).toFixed(3)}\\linewidth}`).join('');
      L.push('\\begin{table}[htbp]', '  \\centering', '  \\small', `  \\caption{${b.cap}}`, '  \\renewcommand{\\arraystretch}{1.15}', `  \\begin{tabular}{${spec}}`, '  \\toprule');
      b.rows.forEach((r, i) => {
        L.push('  ' + r.map((c) => (i === 0 ? '\\textbf{' + tex(String(c)) + '}' : tex(String(c)))).join(' & ') + ' \\\\');
        if (i === 0) L.push('  \\midrule');
      });
      L.push('  \\bottomrule', '  \\end{tabular}', '\\end{table}', ''); break;
    }
    case 'refs':
      L.push('\\clearpage', '\\begin{thebibliography}{99}', '\\addcontentsline{toc}{section}{References}');
      REFS.forEach((r, i) => L.push(`\\bibitem{r${i + 1}} ${tex(r)}`));
      L.push('\\end{thebibliography}'); break;
    case 'code':
      L.push('\\begin{lstlisting}', b.s, '\\end{lstlisting}'); break;
    default: throw new Error('unknown block ' + JSON.stringify(b).slice(0, 80));
  }
}
closeList();

const PRE = String.raw`% AERO2711 progress report -- LaTeX version.
% Compile with pdfLaTeX (Overleaf default) or tectonic. Figures are in figs/.
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{newtxtext,newtxmath}
\usepackage[a4paper,left=3cm,right=2cm,top=2.54cm,bottom=2.54cm]{geometry}
\usepackage{setspace}
\onehalfspacing
\usepackage{graphicx}
\usepackage{array,booktabs}
\usepackage{enumitem}
\usepackage[table]{xcolor}
\usepackage{textcomp}
\usepackage{listings}
\usepackage{cite}
\usepackage[skip=6pt]{caption}
\usepackage[hidelinks]{hyperref}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{4pt}
\newcommand{\ph}[1]{\colorbox{yellow}{#1}}

\begin{document}

\begin{titlepage}
  \centering
  \vspace*{3cm}
  {\large\color{gray} AERO2711 Research Project --- Progress Report\par}
  \vspace{1cm}
  {\LARGE\bfseries Visual Active Tracking of Manoeuvring Targets\\ under Corner Occlusion\par}
  \vspace{0.6cm}
  {\large\itshape Map-constrained prediction and decoupled blind-phase control\\ for a drone-mounted camera\par}
  \vspace{3cm}
  \textbf{Student:} \ph{[Full name], SID [xxxxxxxxx]}\par
  \textbf{Supervisor:} \ph{[Title] Zi [Surname]}\par
  \textbf{Co-supervisor:} \ph{[name, or delete this line]}\par
  \vspace{1.5cm}
  \ph{[School / Faculty], [University]}\par
  September 2026\par
\end{titlepage}
`;
fs.mkdirSync(path.join(OUTDIR, 'figs'), { recursive: true });
for (const f of fs.readdirSync(path.join(__dirname, 'figs'))) fs.copyFileSync(path.join(__dirname, 'figs', f), path.join(OUTDIR, 'figs', f));
fs.writeFileSync(path.join(OUTDIR, 'AERO2711_Progress_Report.tex'), PRE + '\n' + L.join('\n') + '\n\n\\end{document}\n');
console.log('wrote', OUTDIR, L.length, 'lines');
