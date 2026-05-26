from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "production"
MD_PATH = OUT_DIR / "production_readiness_plan.md"
PDF_PATH = OUT_DIR / "production_readiness_plan.pdf"


MODEL_ROWS = [
    ["V1", "70.80%", "26.80%", "34.82%", "68.82%", "35,900", "ข้อมูลจริง", "ไม่แนะนำเป็นตัวหลัก เพราะ Recall ต่ำและ Cost สูง"],
    ["V2", "67.87%", "65.60%", "54.27%", "73.66%", "19,550", "ข้อมูลจริง", "แนะนำเป็น baseline production candidate"],
    ["V3", "66.67%", "63.76%", "52.65%", "71.90%", "20,400", "ข้อมูลจริง", "ใกล้ V2 แต่ Cost สูงกว่าเล็กน้อย"],
    ["V4", "83.45%", "46.39%", "45.69%", "85.38%", "31,650", "generated/synthetic", "ใช้เป็น experiment/stress test ต้อง validate กับ real data ก่อน deploy"],
]


def register_font() -> str:
    for font_path in [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/THSarabunNew.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("ThaiFont", str(font_path)))
            return "ThaiFont"
    return "Helvetica"


def md_table(rows: list[list[str]], headers: list[str]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_md() -> None:
    content = f"""# Production Readiness Plan - Return Risk Prediction

## ข้อสรุปสำหรับใช้งานจริง

ถ้าจะใช้ระดับ production จริงหลังจบโปรเจ็กต์ ไม่ควรเลือกจาก Accuracy อย่างเดียว เพราะโจทย์ return-risk ต้องลดเคสคืนสินค้าที่โมเดลพลาดและลด Cost Matrix ด้วย

ข้อเสนอปัจจุบัน:

- ใช้ **V2** เป็น baseline production candidate เพราะมาจากข้อมูลจริงและมี Cost ต่ำสุดในกลุ่ม V1-V3
- ใช้ **V4 Generated** เป็น experiment/stress test เท่านั้น จนกว่าจะ retrain และ validate กับข้อมูลจริง
- ก่อน deploy จริง ต้องทำ temporal split, leakage check, threshold tuning และ monitoring plan

## เปรียบเทียบ Model Versions

{md_table(MODEL_ROWS, ["Version", "Accuracy", "Recall", "F1", "AUC", "Cost", "Data Type", "Production Note"])}

## คำแนะนำ Model สำหรับ Production

### 1. Production Baseline

ใช้ `V2` เป็น baseline เพราะแม้ Accuracy ไม่สูงสุด แต่ Recall/F1/Cost ดีกว่าสำหรับข้อมูลจริง:

- Accuracy: 67.87%
- Recall: 65.60%
- F1: 54.27%
- AUC: 73.66%
- Cost: 19,550

### 2. Research/Experiment Candidate

ใช้ `V4 Generated` เป็น candidate เพื่อทดลองแนวทางใหม่:

- XGBoost + SMOTE + Optuna
- Accuracy: 83.45%
- AUC: 85.38%
- แต่ข้อมูลเป็น generated/synthetic imbalance dataset
- ต้องทดสอบกับ real holdout data ก่อน deploy จริง

## Feature Policy สำหรับ Production

### Feature ที่ใช้ได้ตอน Order เข้าระบบ

- customer history ที่คำนวณแบบ as-of order date เช่น `hist_order_count`, `hist_return_rate`, `days_since_last_order`
- product/category/brand feature
- price, discount, promotion
- payment method, channel
- province/location
- expected delivery days
- product rating และ courier historical risk

### Feature ที่ต้องระวัง Leakage

ห้ามใช้ในโมเดลที่ทำนายตอน order เพิ่งเข้าระบบ ถ้าค่านั้นรู้หลังเหตุการณ์:

- `return_id`
- `return_date`
- `refund_amount`
- `return_reason`
- `risk_score`
- `risk_tier`
- `shap_values`
- `delivery_days`
- `delay_days`

หมายเหตุ: `delivery_days` และ `delay_days` ใช้ได้เฉพาะโมเดลหลังจัดส่งแล้ว เช่น post-delivery risk follow-up model

## Production Workflow ที่ควรใช้

1. ดึงข้อมูลจริงจาก Order, Return, Customer, Product, Courier, Promotion
2. ทำ data validation ทุกครั้งก่อนสร้าง feature
3. สร้าง feature แบบ point-in-time เพื่อกัน future leakage
4. แบ่ง train/test แบบเวลา เช่น train เดือนก่อนหน้า test เดือนล่าสุด
5. Train หลาย model เช่น Logistic Regression, Random Forest, XGBoost, LightGBM
6. Tune threshold จาก Cost Matrix ไม่ใช่จาก Accuracy อย่างเดียว
7. เลือก model จาก Recall, F1, AUC และ Cost ร่วมกัน
8. Export model พร้อม feature list และ preprocessing pipeline
9. ทำ batch scoring หรือ API scoring
10. Monitor production drift, recall, cost และ approval outcome

## Go/No-Go Checklist

- [ ] มี real holdout test set ที่ไม่ได้ใช้ train/tune
- [ ] ไม่มี leakage feature เข้า model
- [ ] มี feature list เวอร์ชันเดียวกับ model artifact
- [ ] มี preprocessing pipeline ที่ reproduce ได้
- [ ] มี threshold ที่เลือกตาม business cost
- [ ] มี fallback rule ถ้า feature ขาดหรือ data quality fail
- [ ] มี monitoring report รายสัปดาห์/รายเดือน
- [ ] มี retraining schedule

## ไฟล์ Production Candidate ที่เกี่ยวข้อง

- `docs/version 2/models/best_model_v2.pkl`
- `docs/version 2/data/features/train_test_sets_v2.pkl`
- `docs/version 2/reports/thresholds/v2_threshold_accuracy_tradeoff.csv`
- `docs/version 4/models/best_model_v4_generated.pkl`
- `docs/version 4/reports/model_evaluation/v4_generated_model_metrics.csv`
- `docs/model comparison/model_versions_v1_to_v4_latest_generated_comparison.csv`

## สรุปสุดท้าย

สำหรับ production จริงตอนนี้ให้ถือว่า **V2 คือ production baseline ที่ปลอดภัยกว่า** เพราะใช้ข้อมูลจริงและลด Cost ได้ดีที่สุดในกลุ่ม real-data versions

ส่วน **V4 คือแนวทางทดลองที่ Accuracy สูงกว่า** แต่ต้อง retrain/validate ด้วยข้อมูลจริงก่อนใช้จริง ไม่ควร deploy จาก synthetic result เพียงอย่างเดียว
"""
    MD_PATH.write_text(content, encoding="utf-8")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def write_pdf() -> None:
    font_name = register_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ThaiTitle", parent=styles["Heading1"], fontName=font_name, fontSize=18, leading=22)
    h2 = ParagraphStyle("ThaiH2", parent=styles["Heading2"], fontName=font_name, fontSize=13, leading=17)
    body = ParagraphStyle("ThaiBody", parent=styles["BodyText"], fontName=font_name, fontSize=9, leading=13)
    header = ParagraphStyle("ThaiHeader", parent=body, textColor=colors.white, fontSize=8, leading=10)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    story = [
        p("Production Readiness Plan - Return Risk Prediction", title),
        p(
            "ถ้าจะใช้ระดับ production จริง ควรใช้ V2 เป็น baseline จากข้อมูลจริง และใช้ V4 Generated เป็น experiment ที่ต้อง validate กับ real holdout data ก่อน deploy",
            body,
        ),
        Spacer(1, 0.3 * cm),
        p("Model Version Summary", h2),
    ]

    rows = [["Version", "Accuracy", "Recall", "F1", "AUC", "Cost", "Production Note"]]
    rows.extend([row[0], row[1], row[2], row[3], row[4], row[5], row[7]] for row in MODEL_ROWS)
    table_data = [[p(str(cell), header if i == 0 else body) for cell in row] for i, row in enumerate(rows)]
    table = Table(table_data, colWidths=[1.4 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 8.2 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d7dee5")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    story.append(table)
    story.extend(
        [
            Spacer(1, 0.35 * cm),
            p("Production Recommendation", h2),
            p(
                "ใช้ V2 เป็น production baseline เพราะมาจากข้อมูลจริงและมี Cost ต่ำสุดในกลุ่ม V1-V3 ส่วน V4 มี Accuracy สูงกว่าแต่เป็น generated/synthetic data จึงต้อง retrain และ validate กับข้อมูลจริงก่อนใช้งานจริง",
                body,
            ),
            Spacer(1, 0.25 * cm),
            p("Feature Policy", h2),
            p(
                "ใช้ feature ที่รู้ได้ตอน order เข้า เช่น customer history แบบ as-of order date, category, price, discount, payment, channel, province และ expected delivery days. หลีกเลี่ยง leakage fields เช่น return_id, return_date, refund_amount, return_reason, risk_score, risk_tier, shap_values, delivery_days และ delay_days สำหรับโมเดลที่ต้องทำนายก่อนส่งสินค้า",
                body,
            ),
            Spacer(1, 0.25 * cm),
            p("Go/No-Go Checklist", h2),
            p(
                "ก่อน deploy ต้องมี real holdout test set, leakage check, reproducible preprocessing pipeline, threshold ที่เลือกจาก Cost Matrix, fallback rule เมื่อข้อมูลขาด, monitoring report และ retraining schedule",
                body,
            ),
        ]
    )
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_md()
    write_pdf()
    print(MD_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
