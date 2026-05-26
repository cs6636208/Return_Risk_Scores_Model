const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const ROOT = path.resolve(__dirname, '..');
const INPUT_MD = path.join(ROOT, 'docs', 'analysis', 'all_feature_model_versions_summary.md');
const OUTPUT_HTML = path.join(ROOT, 'docs', 'analysis', 'all_feature_model_versions_summary.html');
const OUTPUT_PDF = path.join(ROOT, 'docs', 'analysis', 'all_feature_model_versions_summary.pdf');

const browserCandidates = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
];

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function renderTable(lines, startIndex) {
  const header = splitTableRow(lines[startIndex]);
  const rows = [];
  let i = startIndex + 2;
  while (i < lines.length && lines[i].trim().startsWith('|')) {
    rows.push(splitTableRow(lines[i]));
    i += 1;
  }
  const html = [
    '<div class="table-wrap"><table>',
    '<thead><tr>',
    ...header.map((cell) => `<th>${inlineMarkdown(cell)}</th>`),
    '</tr></thead><tbody>',
    ...rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join('')}</tr>`),
    '</tbody></table></div>',
  ].join('');
  return { html, nextIndex: i };
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  const parts = [];
  let paragraph = [];
  let list = [];

  function flushParagraph() {
    if (paragraph.length) {
      parts.push(`<p>${inlineMarkdown(paragraph.join(' '))}</p>`);
      paragraph = [];
    }
  }

  function flushList() {
    if (list.length) {
      parts.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join('')}</ul>`);
      list = [];
    }
  }

  for (let i = 0; i < lines.length;) {
    const raw = lines[i];
    const line = raw.trim();

    if (!line) {
      flushParagraph();
      flushList();
      i += 1;
      continue;
    }

    if (line.startsWith('|') && i + 1 < lines.length && /^\|\s*:?-{3,}/.test(lines[i + 1].trim())) {
      flushParagraph();
      flushList();
      const table = renderTable(lines, i);
      parts.push(table.html);
      i = table.nextIndex;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      parts.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (line.startsWith('- ')) {
      flushParagraph();
      list.push(line.slice(2));
      i += 1;
      continue;
    }

    paragraph.push(line);
    i += 1;
  }

  flushParagraph();
  flushList();
  return parts.join('\n');
}

function buildHtml(markdown) {
  return `<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <title>All Feature And Model Versions Summary</title>
  <style>
    @page { size: A4 landscape; margin: 12mm; }
    body {
      font-family: "Noto Sans Thai", "Tahoma", "Segoe UI", Arial, sans-serif;
      color: #1f2933;
      line-height: 1.55;
      font-size: 12px;
      margin: 0;
    }
    h1, h2, h3 {
      color: #102a43;
      break-after: avoid;
      margin: 18px 0 8px;
    }
    h1 {
      font-size: 24px;
      border-bottom: 2px solid #486581;
      padding-bottom: 8px;
      margin-top: 0;
    }
    h2 { font-size: 18px; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; }
    h3 { font-size: 14px; }
    p { margin: 7px 0; }
    code {
      font-family: Consolas, "Courier New", monospace;
      background: #f0f4f8;
      border: 1px solid #d9e2ec;
      border-radius: 3px;
      padding: 1px 4px;
      font-size: 0.92em;
    }
    .table-wrap {
      width: 100%;
      overflow: visible;
      break-inside: avoid;
      margin: 8px 0 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 9.2px;
    }
    th, td {
      border: 1px solid #bcccdc;
      padding: 5px 6px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      background: #d9e2ec;
      color: #102a43;
      font-weight: 700;
    }
    tr:nth-child(even) td { background: #f8fafc; }
    ul { margin: 6px 0 10px 18px; padding: 0; }
    strong { color: #102a43; }
  </style>
</head>
<body>
${markdownToHtml(markdown)}
</body>
</html>`;
}

async function main() {
  if (!fs.existsSync(INPUT_MD)) {
    throw new Error(`Missing input markdown: ${INPUT_MD}`);
  }
  const browserPath = browserCandidates.find((candidate) => fs.existsSync(candidate));
  if (!browserPath) {
    throw new Error('Could not find Chrome or Edge executable.');
  }

  const markdown = fs.readFileSync(INPUT_MD, 'utf8');
  fs.writeFileSync(OUTPUT_HTML, buildHtml(markdown), 'utf8');

  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: 'new',
    protocolTimeout: 180000,
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);
  await page.goto(`file:///${OUTPUT_HTML.replace(/\\/g, '/')}`, {
    waitUntil: 'domcontentloaded',
    timeout: 120000,
  });
  await page.emulateMediaType('screen');
  await page.pdf({
    path: OUTPUT_PDF,
    format: 'A4',
    landscape: true,
    printBackground: true,
    margin: { top: '12mm', right: '10mm', bottom: '12mm', left: '10mm' },
    timeout: 180000,
  });
  await browser.close();

  console.log(`HTML written: ${OUTPUT_HTML}`);
  console.log(`PDF written: ${OUTPUT_PDF}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
