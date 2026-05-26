from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "version 4" / "docs"
MD_PATH = OUT_DIR / "project_progress_status_1_1_to_1_4.md"
PDF_PATH = OUT_DIR / "project_progress_status_1_1_to_1_4.pdf"


PHASES = [
    {
        "phase": "1.1 Data Collection & Understanding (เก็บข้อมูลและทำความเข้าใจ)",
        "period": "Week 1-2: 27 เม.ย. - 10 พ.ค. 2569",
        "status": "เสร็จแล้ว",
        "done": [
            "เตรียม SQL template สำหรับดึงและ join ข้อมูล Order, Return, Customer, Product, Courier และ Promotion",
            "ศึกษา schema/context จาก ER diagram และเอกสารอ้างอิงของโปรเจ็กต์",
            "สร้าง Data Dictionary สำหรับชุดข้อมูล V4 generated",
            "จัดการ Missing Value, Duplicate และกฎตรวจความสอดคล้องของข้อมูล",
            "สร้าง Clean Dataset พร้อมใช้สำหรับ train model",
        ],
        "outputs": [
            "docs/version 4/docs/v4_generated_data_collection_sql.sql",
            "docs/version 4/docs/data_dictionary_v4_generated.csv",
            "docs/version 4/data/generated/v4_synthetic_orders_returns.csv",
            "docs/version 4/data/processed/clean_dataset_v4_generated.csv",
        ],
    },
    {
        "phase": "1.2 Exploratory Data Analysis (EDA) - วิเคราะห์เชิงสำรวจ",
        "period": "Week 3: 11-17 พ.ค. 2569",
        "status": "เสร็จแล้ว",
        "done": [
            "วิเคราะห์สัดส่วนการคืนสินค้าหลังสร้าง imbalance dataset",
            "วิเคราะห์ pattern การคืนสินค้าแยกตาม Category, Channel และ Payment",
            "สร้าง Correlation Heatmap",
            "สรุป Business/EDA Insight ที่ผูกกับ feature ได้",
        ],
        "outputs": [
            "docs/version 4/reports/eda/01_target_distribution.png",
            "docs/version 4/reports/eda/02_return_rate_by_category.png",
            "docs/version 4/reports/eda/03_return_rate_by_channel.png",
            "docs/version 4/reports/eda/04_return_rate_by_payment.png",
            "docs/version 4/reports/eda/05_correlation_heatmap.png",
            "docs/version 4/docs/eda_insight_v4_generated.md",
        ],
    },
    {
        "phase": "1.3 Feature Engineering & Preprocessing (สร้าง Feature และเตรียมข้อมูล)",
        "period": "Week 4: 18-24 พ.ค. 2569",
        "status": "เสร็จแล้ว",
        "done": [
            "สร้าง engineered features มากกว่า 30 ตัว เช่น customer/category/brand/province/payment/channel/courier history",
            "สร้าง interaction features เช่น category_payment, category_channel และ province_payment",
            "Encode categorical variables ด้วย one-hot encoding",
            "สร้าง train/test split",
            "ใช้ SMOTE เฉพาะ training split เพื่อจัดการ imbalanced data",
        ],
        "outputs": [
            "docs/version 4/data/features/df_engineered_v4_generated.csv",
            "docs/version 4/data/features/train_test_sets_v4_generated.pkl",
        ],
    },
    {
        "phase": "1.4 Model Training & Evaluation (สร้าง Train Model และประเมิน Model)",
        "period": "Week 5-6: 25 พ.ค. - 7 มิ.ย. 2569",
        "status": "เสร็จแล้ว",
        "done": [
            "Train Logistic Regression, Random Forest, XGBoost และ LightGBM",
            "Tune XGBoost และ LightGBM ด้วย Optuna",
            "ประเมินผลด้วย Accuracy, Recall, F1, AUC-ROC, Average Precision และ Cost Matrix",
            "สร้าง SHAP Explainability สำหรับอธิบาย feature importance",
            "บันทึก best model artifact และ final report",
        ],
        "outputs": [
            "docs/version 4/models/best_model_v4_generated.pkl",
            "docs/version 4/models/best_model_v4_generated_metadata.json",
            "docs/version 4/reports/model_evaluation/v4_generated_model_metrics.csv",
            "docs/version 4/reports/model_evaluation/v4_generated_model_metrics.png",
            "docs/version 4/reports/model_evaluation/v4_generated_shap_summary.png",
            "docs/version 4/reports/model_evaluation/v4_generated_shap_feature_importance.csv",
            "docs/version 4/docs/v4_generated_end_to_end_report.pdf",
        ],
    },
]


def md_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def write_md() -> None:
    content = """# สถานะโปรเจ็กต์ 1.1-1.4

## สรุปสถานะปัจจุบัน

ตอนนี้ทำครบแล้วตั้งแต่ **1.1 Data Collection & Understanding** ถึง **1.4 Model Training & Evaluation** สำหรับชุดทดลองล่าสุด **V4 Generated Imbalanced Data Experiment**

## Best Current V4 Generated Model

| Model | Accuracy | Recall | F1 | AUC | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost + SMOTE + Optuna | 83.45% | 46.39% | 45.69% | 85.38% | 31,650 |

## หมายเหตุเรื่องข้อมูล

V4 ใช้ข้อมูล generated/synthetic imbalance dataset:

- แถวข้อมูลตั้งต้น: 5,000
- แถว non-return ที่ generate เพิ่ม: 4,700
- จำนวนแถวสุดท้าย: 9,700
- Return rate: 15%
- ใช้ SMOTE เฉพาะ training split

"""
    for phase in PHASES:
        content += f"""## {phase["phase"]}

**ช่วงเวลา:** {phase["period"]}  
**สถานะ:** {phase["status"]}

### งานที่ทำแล้ว

{md_list(phase["done"])}

### ไฟล์ผลลัพธ์

{md_list(phase["outputs"])}

"""
    content += """## งานเสริมที่ยังทำเพิ่มได้

- ต่อ real database จริง ถ้ามีสิทธิ์เข้าถึง production DB
- เพิ่ม EDA Notebook ถ้าต้องส่งเป็นไฟล์ notebook แยก
- ตรวจสมมติฐานของ generated data กับ order จริงในอนาคต
- ถ้าจะใช้ production ควรเทียบกับ V2 real-data model อีกครั้งก่อน deploy
"""
    MD_PATH.write_text(content, encoding="utf-8")


def register_font() -> str:
    font_candidates = [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/THSarabunNew.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("ThaiFont", str(font_path)))
            return "ThaiFont"
    return "Helvetica"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def write_pdf() -> None:
    font_name = register_font()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=0.9 * cm,
        bottomMargin=0.9 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ThaiTitle",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=18,
        leading=22,
    )
    body_style = ParagraphStyle(
        "ThaiBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=13,
    )
    header_style = ParagraphStyle(
        "ThaiHeader",
        parent=body_style,
        textColor=colors.white,
        fontSize=8,
        leading=11,
    )
    story: list = []
    story.append(Paragraph("สถานะโปรเจ็กต์ 1.1-1.4", title_style))
    story.append(
        Paragraph(
            "ทำครบแล้วตั้งแต่ Data Collection & Understanding ถึง Model Training & Evaluation สำหรับ V4 Generated Imbalanced Data Experiment",
            body_style,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    summary = [
        ["Best model", "XGBoost + SMOTE + Optuna"],
        ["Accuracy", "83.45%"],
        ["Recall", "46.39%"],
        ["F1", "45.69%"],
        ["AUC", "85.38%"],
        ["Cost", "31,650"],
        ["Dataset", "Generated/synthetic imbalance dataset"],
        ["Rows", "9,700 final rows; 15% return rate"],
    ]
    summary = [[para(str(cell), body_style) for cell in row] for row in summary]
    table = Table(summary, colWidths=[5 * cm, 13 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dee5")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))

    phase_rows = [[
        para("Phase", header_style),
        para("ช่วงเวลา", header_style),
        para("สถานะ", header_style),
        para("Main Outputs", header_style),
    ]]
    for phase in PHASES:
        phase_rows.append(
            [
                para(phase["phase"], body_style),
                para(phase["period"], body_style),
                para(phase["status"], body_style),
                para("\n".join(phase["outputs"][:4]), body_style),
            ]
        )
    table = Table(phase_rows, colWidths=[7.2 * cm, 5.2 * cm, 3.0 * cm, 12.0 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d7dee5")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "งานเสริม: ต่อ production DB จริง, เพิ่ม EDA notebook ถ้าต้องส่งแยก, validate generated-data assumptions กับ future real orders และเทียบ deployment choice กับ V2 real-data model ก่อนใช้งานจริง",
            body_style,
        )
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
