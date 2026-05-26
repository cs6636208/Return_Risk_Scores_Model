from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = ROOT / "docs" / "analysis" / "all_feature_model_versions_summary.md"
OUTPUT_PDF = ROOT / "docs" / "analysis" / "all_feature_model_versions_summary.pdf"


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/tahoma.ttf")
    bold = Path("C:/Windows/Fonts/tahomabd.ttf")
    if regular.exists():
        pdfmetrics.registerFont(TTFont("TahomaThai", str(regular)))
        font_name = "TahomaThai"
    else:
        font_name = "Helvetica"
    if bold.exists():
        pdfmetrics.registerFont(TTFont("TahomaThai-Bold", str(bold)))
        bold_name = "TahomaThai-Bold"
    else:
        bold_name = font_name
    return font_name, bold_name


def inline_markup(text: str) -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # Keep simple inline code readable without relying on monospace Thai fallback.
    parts = safe.split("`")
    for i in range(1, len(parts), 2):
        parts[i] = f'<font color="#334e68">{parts[i]}</font>'
    safe = "".join(parts)
    while "**" in safe:
        safe = safe.replace("**", "<b>", 1)
        if "**" in safe:
            safe = safe.replace("**", "</b>", 1)
        else:
            break
    return safe


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} and "-" in cell for cell in cells)


def build_styles(font_name: str, bold_name: str):
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "ThaiBase",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    h1 = ParagraphStyle(
        "ThaiH1",
        parent=base,
        fontName=bold_name,
        fontSize=17,
        leading=22,
        textColor=colors.HexColor("#102a43"),
        spaceBefore=0,
        spaceAfter=9,
    )
    h2 = ParagraphStyle(
        "ThaiH2",
        parent=base,
        fontName=bold_name,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#102a43"),
        spaceBefore=8,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "ThaiH3",
        parent=base,
        fontName=bold_name,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#243b53"),
        spaceBefore=6,
        spaceAfter=4,
    )
    table_cell = ParagraphStyle(
        "ThaiTableCell",
        parent=base,
        fontSize=6.8,
        leading=8.5,
        spaceAfter=0,
        wordWrap="CJK",
    )
    table_head = ParagraphStyle(
        "ThaiTableHead",
        parent=table_cell,
        fontName=bold_name,
        textColor=colors.HexColor("#102a43"),
    )
    bullet = ParagraphStyle(
        "ThaiBullet",
        parent=base,
        leftIndent=4,
        firstLineIndent=0,
    )
    return {
        "base": base,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "table_cell": table_cell,
        "table_head": table_head,
        "bullet": bullet,
    }


def table_widths(num_cols: int, page_width: float) -> list[float]:
    if num_cols <= 3:
        return [page_width / num_cols] * num_cols
    if num_cols <= 6:
        return [page_width * 0.18] * (num_cols - 1) + [page_width * (1 - 0.18 * (num_cols - 1))]
    return [page_width / num_cols] * num_cols


def render_table(lines: list[str], index: int, styles: dict, page_width: float):
    header = split_table_row(lines[index])
    rows = []
    i = index + 2
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append(split_table_row(lines[i]))
        i += 1
    max_cols = max([len(header), *[len(row) for row in rows]])
    header += [""] * (max_cols - len(header))
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

    data = [
        [Paragraph(inline_markup(cell), styles["table_head"]) for cell in header],
        *[
            [Paragraph(inline_markup(cell), styles["table_cell"]) for cell in row]
            for row in normalized_rows
        ],
    ]
    table = Table(data, colWidths=table_widths(max_cols, page_width), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ec")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9fb3c8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table, i


def markdown_to_flowables(markdown: str, styles: dict, page_width: float):
    flow = []
    lines = markdown.splitlines()
    paragraph: list[str] = []
    bullets: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            flow.append(Paragraph(inline_markup(" ".join(paragraph)), styles["base"]))
            paragraph = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [
                ListItem(Paragraph(inline_markup(item), styles["bullet"]), bulletColor=colors.HexColor("#486581"))
                for item in bullets
            ]
            flow.append(ListFlowable(items, bulletType="bullet", leftIndent=12, spaceAfter=5))
            bullets = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush_paragraph()
            flush_bullets()
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            flush_paragraph()
            flush_bullets()
            table, next_i = render_table(lines, i, styles, page_width)
            flow.append(table)
            flow.append(Spacer(1, 5))
            i = next_i
            continue

        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            flow.append(Paragraph(inline_markup(line[2:]), styles["h1"]))
            flow.append(Spacer(1, 3))
        elif line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            if flow:
                flow.append(Spacer(1, 4))
            flow.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            flush_paragraph()
            flush_bullets()
            flow.append(Paragraph(inline_markup(line[4:]), styles["h3"]))
        elif line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:])
        else:
            paragraph.append(line)
        i += 1

    flush_paragraph()
    flush_bullets()
    return flow


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#627d98"))
    canvas.drawRightString(doc.pagesize[0] - 10 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    if not INPUT_MD.exists():
        raise FileNotFoundError(INPUT_MD)

    font_name, bold_name = register_fonts()
    styles = build_styles(font_name, bold_name)
    page_size = landscape(A4)
    page_width = page_size[0] - 20 * mm
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=page_size,
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title="All Feature And Model Versions Summary",
        author="Return Risk Prediction Project",
    )
    flowables = markdown_to_flowables(INPUT_MD.read_text(encoding="utf-8"), styles, page_width)
    doc.build(flowables, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF written: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
