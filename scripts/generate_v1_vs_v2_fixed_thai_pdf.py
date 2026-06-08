from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PIL import JpegImagePlugin  # noqa: F401 - registers JPEG writer used by Pillow PDF export.
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "Version Reports"
OUT_PDF = OUT_DIR / "V1_vs_V2_Comparison_Report_FIXED_THAI.pdf"
OUT_PREVIEW = OUT_DIR / "V1_vs_V2_Comparison_Report_FIXED_THAI_page1_preview.png"

FONT_REGULAR = Path("C:/Windows/Fonts/tahoma.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/tahomabd.ttf")

# A4 portrait at 200 DPI. Image-based PDF avoids Thai text encoding issues in PDF viewers.
PAGE_W = 1654
PAGE_H = 2339
MARGIN_X = 120
MARGIN_TOP = 105
LINE = 48

BLUE = "#1F4E79"
GREEN = "#2E7D32"
GRAY = "#F2F5F7"
DARK = "#222222"
MUTED = "#5A6670"
ORANGE = "#F5A623"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


F_TITLE = font(52, True)
F_SUBTITLE = font(34, True)
F_H1 = font(36, True)
F_H2 = font(29, True)
F_BODY = font(27)
F_BODY_BOLD = font(27, True)
F_SMALL = font(22)
F_TABLE = font(22)
F_TABLE_BOLD = font(22, True)


def pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def num(value: object) -> str:
    return f"{float(value):,.0f}"


def load_metrics() -> tuple[dict[str, object], dict[str, object]]:
    comp = pd.read_csv(ROOT / "docs" / "Comparison Version" / "version_1_to_4_selected_model_comparison.csv")
    v1 = comp[comp["display_version"].eq("V1")].iloc[0].to_dict()
    v2_path = (
        ROOT
        / "docs"
        / "version 2"
        / "v2_xgboost_safe_plus_rolling"
        / "reports"
        / "v2_xgboost_safe_plus_rolling_metrics.csv"
    )
    v2 = pd.read_csv(v2_path).iloc[0].to_dict()
    return v1, v2


def text_bbox(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_center(draw: ImageDraw.ImageDraw, y: int, text: str, fnt: ImageFont.ImageFont, fill: str = DARK) -> int:
    width, height = text_bbox(draw, text, fnt)
    draw.text(((PAGE_W - width) // 2, y), text, font=fnt, fill=fill)
    return y + height + 18


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fnt: ImageFont.ImageFont = F_BODY,
    fill: str = DARK,
    max_chars: int = 82,
    line_gap: int = 10,
) -> int:
    # Thai has no spaces between every word; this project text is mixed Thai/English, so char wrapping is safer.
    lines: list[str] = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(wrap(para, width=max_chars, break_long_words=True, replace_whitespace=False))
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def h1(draw: ImageDraw.ImageDraw, y: int, text: str) -> int:
    draw.text((MARGIN_X, y), text, font=F_H1, fill=BLUE)
    return y + 58


def h2(draw: ImageDraw.ImageDraw, y: int, text: str) -> int:
    draw.text((MARGIN_X, y), text, font=F_H2, fill=GREEN)
    return y + 48


def bullet(draw: ImageDraw.ImageDraw, y: int, text: str, color: str = DARK) -> int:
    draw.ellipse((MARGIN_X + 8, y + 12, MARGIN_X + 21, y + 25), fill=GREEN)
    return draw_wrapped(draw, MARGIN_X + 42, y, text, F_BODY, color, max_chars=76, line_gap=9) + 6


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, PAGE_W, 32), fill=BLUE)
    draw.rectangle((0, PAGE_H - 28, PAGE_W, PAGE_H), fill=GREEN)
    return img, draw


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    widths: list[int],
    rows: list[list[str]],
    header_fill: str = BLUE,
    row_h: int = 96,
) -> int:
    cur_y = y
    for r, row in enumerate(rows):
        cur_x = x
        fill = header_fill if r == 0 else ("#FFFFFF" if r % 2 else GRAY)
        for c, cell in enumerate(row):
            draw.rectangle((cur_x, cur_y, cur_x + widths[c], cur_y + row_h), fill=fill, outline="#9AA5AD", width=2)
            fnt = F_TABLE_BOLD if r == 0 else F_TABLE
            color = "white" if r == 0 else DARK
            max_chars = max(8, widths[c] // 14)
            cell_y = cur_y + 14
            for line in wrap(str(cell), width=max_chars, break_long_words=True, replace_whitespace=False)[:3]:
                draw.text((cur_x + 12, cell_y), line, font=fnt, fill=color)
                cell_y += fnt.size + 6
            cur_x += widths[c]
        cur_y += row_h
    return cur_y + 22


def page_1(v1: dict[str, object], v2: dict[str, object]) -> Image.Image:
    img, draw = new_page()
    y = MARGIN_TOP
    y = draw_center(draw, y, "Return Risk Prediction", F_TITLE, BLUE)
    y = draw_center(draw, y, "รายงานแก้ภาษาไทย: เปรียบเทียบ Version 1 กับ Version 2", F_SUBTITLE, GREEN)
    y = draw_center(draw, y, "Version 2 ปัจจุบันมีโมเดลเดียว: v2_xgboost_safe_plus_rolling", F_SMALL, MUTED)
    y += 42

    y = h1(draw, y, "1. สรุปภาพรวม")
    summary = (
        "Version 1 คือ baseline model ที่ใช้ feature พื้นฐานจาก clean_dataset.csv เพื่อวัด performance ตั้งต้น "
        "ส่วน Version 2 คือการต่อยอดโดยเพิ่ม customer history และ rolling history แบบ point-in-time "
        "ทำให้โมเดลดูพฤติกรรมลูกค้าย้อนหลังก่อน order ปัจจุบันได้"
    )
    y = draw_wrapped(draw, MARGIN_X, y, summary, F_BODY, max_chars=78)
    y += 22

    draw.rounded_rectangle((MARGIN_X, y, PAGE_W - MARGIN_X, y + 340), radius=22, fill="#EEF7F0", outline="#B7D9BD", width=3)
    y_box = y + 32
    y_box = draw_wrapped(
        draw,
        MARGIN_X + 34,
        y_box,
        f"Accuracy: V1 {pct(v1['accuracy'])} → V2 {pct(v2['accuracy'])}",
        F_BODY_BOLD,
        BLUE,
        max_chars=72,
    )
    y_box = draw_wrapped(
        draw,
        MARGIN_X + 34,
        y_box + 8,
        f"Recall: V1 {pct(v1['recall'])} → V2 {pct(v2['recall'])}",
        F_BODY_BOLD,
        GREEN,
        max_chars=72,
    )
    y_box = draw_wrapped(
        draw,
        MARGIN_X + 34,
        y_box + 8,
        f"F1-score: V1 {pct(v1['f1'])} → V2 {pct(v2['f1'])}",
        F_BODY_BOLD,
        GREEN,
        max_chars=72,
    )
    draw_wrapped(
        draw,
        MARGIN_X + 34,
        y_box + 18,
        "แปลว่า V2 ไม่ได้เด่นแค่ Accuracy แต่เด่นตรงจับเคสคืนสินค้าได้มากขึ้นอย่างชัดเจน",
        F_BODY,
        DARK,
        max_chars=72,
    )
    y += 390

    y = h1(draw, y, "2. ทำไม Accuracy เพิ่มไม่มาก แต่ V2 ดีกว่า")
    y = bullet(draw, y, "V1 กับ V2 ใช้ฐานข้อมูล clean_dataset.csv เดิม จึงไม่ได้มี data signal ใหม่ที่ทำให้ Accuracy กระโดดแรง")
    y = bullet(draw, y, "V2 เพิ่ม feature ที่ตรงโจทย์ return-risk เช่น hist_return_rate และ rolling history หลายช่วงเวลา")
    y = bullet(draw, y, "V2 ลด feature จาก 136 เหลือ 60 แต่เลือก feature ที่อธิบายพฤติกรรมคืนสินค้าได้ดีกว่า")
    y = bullet(draw, y, "Recall เพิ่มมาก แปลว่าโมเดลจับลูกค้าที่มีโอกาสคืนสินค้าได้มากกว่า V1")

    return img


def page_2(v1: dict[str, object], v2: dict[str, object]) -> Image.Image:
    img, draw = new_page()
    y = MARGIN_TOP
    y = h1(draw, y, "3. ตาราง Metric เปรียบเทียบ")
    rows = [
        ["Metric", "Version 1", "Version 2", "ความหมาย"],
        ["Model", "XGBoost baseline", "XGBoost safe rolling", "ใช้ model ตระกูลเดียวกัน แต่ V2 มี feature ดีกว่า"],
        ["Accuracy", pct(v1["accuracy"]), pct(v2["accuracy"]), "ทายถูกโดยรวมเพิ่มเล็กน้อย"],
        ["Recall", pct(v1["recall"]), pct(v2["recall"]), "V2 จับเคส return ได้มากขึ้น"],
        ["Precision", pct(v1["precision"]), pct(v2["precision"]), "ความแม่นเมื่อทายว่า return ใกล้เคียงกัน"],
        ["F1-score", pct(v1["f1"]), pct(v2["f1"]), "V2 balance precision/recall ดีกว่า"],
        ["AUC", pct(v1["auc"]), pct(v2["auc"]), "V2 แยก class return/not return ได้ดีกว่า"],
        ["Cost", num(v1["cost"]), num(v2["cost"]), "V2 ลด expected cost ลง"],
        ["Feature count", num(v1["feature_count"]), num(v2["raw_feature_count"]), "V2 ใช้ feature น้อยกว่า"],
        ["Threshold", str(v1["threshold"]), str(v2["selected_threshold"]), "V2 tune threshold เพื่อ balance recall/cost"],
    ]
    y = draw_table(draw, MARGIN_X, y, [250, 310, 330, 520], rows, header_fill=BLUE, row_h=110)
    y += 18
    y = draw_wrapped(
        draw,
        MARGIN_X,
        y,
        "ข้อสรุป: ถ้าดู Accuracy อย่างเดียว V1 กับ V2 ใกล้กัน แต่ถ้าดู Recall/F1/Cost สำหรับโจทย์ return-risk แล้ว V2 เหมาะกว่าชัดเจน",
        F_BODY_BOLD,
        GREEN,
        max_chars=82,
    )
    return img


def page_3(v1: dict[str, object], v2: dict[str, object]) -> Image.Image:
    img, draw = new_page()
    y = MARGIN_TOP
    y = h1(draw, y, "4. Feature Engineering ต่างกันอย่างไร")
    rows = [
        ["ประเด็น", "Version 1", "Version 2"],
        ["บทบาท", "Baseline จาก feature พื้นฐาน", "Return-risk feature ที่เน้นประวัติลูกค้า"],
        ["ข้อมูลตั้งต้น", "clean_dataset.csv 5,000 rows", "clean_dataset.csv 5,000 rows"],
        ["แนวคิด", "ดูข้อมูล order/customer/product ทั่วไป", "ดูประวัติย้อนหลังของลูกค้าก่อน order ปัจจุบัน"],
        ["History", "ยังไม่เป็นแกนหลัก", "hist_order_count, hist_return_rate"],
        ["Rolling", "มีน้อย/ยังไม่ชัด", "30/60/90/180/365 วัน"],
        ["Interaction", "feature พื้นฐานและ encoding", "category_payment, category_channel, province_payment"],
        ["Feature count", "136 features", "60 features"],
    ]
    y = draw_table(draw, MARGIN_X, y, [300, 520, 590], rows, header_fill=GREEN, row_h=118)
    y += 12

    y = h2(draw, y, "ตัวอย่าง logic ของ V2")
    y = draw_wrapped(
        draw,
        MARGIN_X,
        y,
        "ถ้าลูกค้ากำลังจะสั่ง order ที่ 3 และมีประวัติก่อนหน้า 2 order โดย order แรกไม่คืนสินค้า และ order ที่สองคืนสินค้า",
        F_BODY,
        max_chars=82,
    )
    y += 10
    draw.rounded_rectangle((MARGIN_X, y, PAGE_W - MARGIN_X, y + 145), radius=18, fill="#FFF7E6", outline="#F0C36A", width=3)
    draw_wrapped(draw, MARGIN_X + 34, y + 34, "hist_return_rate = 1 / 2 = 0.5 = 50%", F_BODY_BOLD, ORANGE, max_chars=74)
    y += 185
    y = draw_wrapped(
        draw,
        MARGIN_X,
        y,
        "จุดนี้ทำให้ V2 เหมาะกับข้อมูล order ใหม่มากกว่า เพราะสามารถคำนวณจากประวัติของลูกค้าคนนั้นเท่านั้น ไม่จำเป็นต้องคำนวณทุก record ในระบบ",
        F_BODY,
        max_chars=82,
    )
    return img


def page_4(v1: dict[str, object], v2: dict[str, object]) -> Image.Image:
    img, draw = new_page()
    y = MARGIN_TOP
    y = h1(draw, y, "5. สรุปสำหรับอธิบายอาจารย์")
    final_text = (
        "Version 1 เป็น baseline ที่ใช้ feature พื้นฐานจาก clean_dataset.csv เพื่อวัด performance ตั้งต้น "
        "ส่วน Version 2 พัฒนาต่อโดยเพิ่ม customer history และ rolling history แบบ point-in-time "
        "ทำให้โมเดลดูพฤติกรรมลูกค้าย้อนหลังก่อน order ปัจจุบันได้ เช่น จำนวน order ก่อนหน้า จำนวนครั้งที่คืน "
        "และ return rate ในช่วง 30/60/90/180/365 วัน"
    )
    y = draw_wrapped(draw, MARGIN_X, y, final_text, F_BODY, max_chars=78)
    y += 24
    y = draw_wrapped(
        draw,
        MARGIN_X,
        y,
        f"ผลคือ Accuracy เพิ่มจาก {pct(v1['accuracy'])} เป็น {pct(v2['accuracy'])} แต่จุดสำคัญคือ Recall เพิ่มจาก {pct(v1['recall'])} เป็น {pct(v2['recall'])} และ F1 เพิ่มจาก {pct(v1['f1'])} เป็น {pct(v2['f1'])}",
        F_BODY_BOLD,
        GREEN,
        max_chars=78,
    )
    y += 18
    y = draw_wrapped(
        draw,
        MARGIN_X,
        y,
        "ดังนั้น V2 เหมาะกับโจทย์ return-risk มากกว่า เพราะจับเคสคืนสินค้าได้ดีขึ้น ใช้ feature น้อยลง และลด cost ได้ดีกว่า V1",
        F_BODY_BOLD,
        BLUE,
        max_chars=78,
    )
    y += 55

    y = h1(draw, y, "6. ข้อควรจำ")
    y = bullet(draw, y, "รายงานนี้เปรียบเทียบ V1 กับ V2 ตัวจริงในโฟลเดอร์ version 2 เท่านั้น")
    y = bullet(draw, y, "Version 2 ตอนนี้มีโมเดลเดียวคือ v2_xgboost_safe_plus_rolling")
    y = bullet(draw, y, "ไม่ได้ใช้ V2 HIGH_ACCURACY ในรายงานนี้ เพราะถูกแยกออกไปเป็น version อื่นแล้ว")
    y = bullet(draw, y, "ไฟล์นี้เป็น image-based PDF เพื่อแก้ปัญหาภาษาไทยเพี้ยน จึงอ่านได้ชัวร์แต่ copy text จาก PDF ไม่ได้")

    y += 40
    y = h2(draw, y, "Artifact อ้างอิง")
    refs = [
        "V1 metrics: docs/Comparison Version/version_1_to_4_selected_model_comparison.csv",
        "V2 metrics: docs/version 2/v2_xgboost_safe_plus_rolling/reports/v2_xgboost_safe_plus_rolling_metrics.csv",
        "V2 features: docs/version 2/v2_xgboost_safe_plus_rolling/data/v2_xgboost_safe_plus_rolling_used_features.csv",
    ]
    for ref in refs:
        y = bullet(draw, y, ref, MUTED)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1, v2 = load_metrics()
    pages = [page_1(v1, v2), page_2(v1, v2), page_3(v1, v2), page_4(v1, v2)]
    pages[0].save(OUT_PREVIEW)
    pages[0].save(OUT_PDF, save_all=True, append_images=pages[1:], resolution=200.0)

    reader = PdfReader(str(OUT_PDF))
    print(OUT_PDF)
    print(OUT_PREVIEW)
    print(f"pages={len(reader.pages)}")
    print(f"size={OUT_PDF.stat().st_size}")


if __name__ == "__main__":
    main()
