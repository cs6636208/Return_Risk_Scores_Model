from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
VERSION_DIR = ROOT / "docs" / "version 2"
FEATURE_DOC_DIR = VERSION_DIR / "feature_documentation"
SELECTED_PACKAGE_DOC_DIR = VERSION_DIR / "v2_xgboost_safe_plus_rolling" / "docs"

CLEAN_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
DF_FEATURED_PATH = VERSION_DIR / "data" / "features" / "df_featured.csv"
DF_ENGINEERED_ALIAS_PATH = VERSION_DIR / "data" / "features" / "df_engineered.csv"
V2_PREVIEW_PATH = VERSION_DIR / "data" / "features" / "df_engineered_v2_preview.csv"
USED_FEATURES_PATH = VERSION_DIR / "v2_xgboost_safe_plus_rolling" / "data" / "v2_xgboost_safe_plus_rolling_used_features.csv"

OUT_PDF = FEATURE_DOC_DIR / "v2_xgboost_safe_plus_rolling_clean_vs_df_engineered_feature_comparison.pdf"
OUT_CSV = FEATURE_DOC_DIR / "v2_xgboost_safe_plus_rolling_clean_vs_df_engineered_feature_comparison.csv"
OUT_USED_CSV = FEATURE_DOC_DIR / "v2_xgboost_safe_plus_rolling_used_features_detailed.csv"
OUT_DROPPED_CSV = FEATURE_DOC_DIR / "v2_xgboost_safe_plus_rolling_dropped_features_detailed.csv"


LEAKAGE_OR_POST_EVENT = {
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
    "score_id",
    "risk_score",
    "risk_tier",
    "scored_at",
    "shap_values",
    "delivery_date",
    "delivery_days",
    "delay_days",
}

IDENTIFIER_FIELDS = {
    "order_id",
    "customer_id",
    "customer_name",
    "customer_phone",
    "product_id",
    "supplier_id",
    "courier_id",
    "promo_id",
}


def register_font() -> str:
    for candidate in [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/THSarabunNew.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("V2FeatureFont", str(candidate)))
            return "V2FeatureFont"
    return "Helvetica"


FONT = register_font()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName=FONT, fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=FONT, fontSize=14, leading=18, spaceBefore=10, spaceAfter=7),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=FONT, fontSize=11.5, leading=15, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=13.5, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName=FONT, fontSize=7.4, leading=10.2, alignment=TA_LEFT),
    }


S = build_styles()


def p(text: Any, style: str = "body") -> Paragraph:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe.replace("\n", "<br/>"), S[style])


def make_table(rows: list[list[Any]], widths: list[float] | None = None, repeat_rows: int = 1) -> Table:
    table = Table([[p(cell, "small") for cell in row] for row in rows], colWidths=widths, repeatRows=repeat_rows)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def feature_group(feature: str) -> str:
    low = feature.lower()
    groups = [
        ("customer_history", ["hist_", "cust_", "orders_before", "returns_before", "return_ratio", "days_since"]),
        ("product_category", ["category", "brand", "product", "rating", "fragile"]),
        ("payment_channel", ["payment", "channel", "cod"]),
        ("promotion_discount", ["promo", "discount", "campaign"]),
        ("logistics", ["courier", "delivery", "delay", "damage", "logistics"]),
        ("customer_profile", ["gender", "age", "membership", "province", "tenure"]),
        ("time", ["hour", "month", "dayofweek", "weekend", "date"]),
        ("price_amount", ["price", "amount", "quantity", "spend"]),
    ]
    for group, tokens in groups:
        if any(token in low for token in tokens):
            return group
    return "other"


def clean_status(feature: str, used_features: set[str]) -> tuple[str, str]:
    if feature == "is_returned":
        return "target", "เป็น target label ไม่ใช่ input feature"
    if feature in used_features:
        return "used_exact", "ใช้เข้า model โดยตรง"
    if any(used.startswith(f"{feature}_") for used in used_features):
        return "used_after_encoding_or_derivation", "ใช้เป็น source แล้วถูกแตก/derive เป็น feature ใหม่"
    if feature in LEAKAGE_OR_POST_EVENT:
        return "dropped", "เป็น post-event/leakage field จึงตัดออกเพื่อกันข้อมูลอนาคต"
    if feature in IDENTIFIER_FIELDS:
        return "query_or_audit_only", "ใช้ join/query/audit ได้ แต่ไม่ส่งให้ model จำ identity โดยตรง"
    return "not_used_or_replaced", "ไม่ได้ใช้ตรง หรือถูกแทนด้วย engineered feature อื่น"


def engineered_status(feature: str, used_features: set[str]) -> tuple[str, str]:
    if feature == "is_returned":
        return "target", "target label"
    if feature == "dataset_split":
        return "metadata", "ใช้บอก split/full frame ไม่ใช่ input model"
    if feature in used_features:
        return "used", "เป็น input feature ของ v2_xgboost_safe_plus_rolling"
    if feature in LEAKAGE_OR_POST_EVENT:
        return "dropped", "เป็นข้อมูลหลังเหตุการณ์หรือมี leakage risk"
    return "not_used", "ไม่อยู่ใน selected feature set"


def ensure_df_engineered_alias() -> pd.DataFrame:
    df_featured = pd.read_csv(DF_FEATURED_PATH, low_memory=False)
    df_engineered = df_featured.drop(columns=["dataset_split"], errors="ignore")
    df_engineered.to_csv(DF_ENGINEERED_ALIAS_PATH, index=False, encoding="utf-8-sig")
    return df_engineered


def build_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    FEATURE_DOC_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_PACKAGE_DOC_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(CLEAN_PATH, nrows=5, low_memory=False)
    clean_cols = list(clean.columns)
    df_engineered = ensure_df_engineered_alias()
    engineered_cols = list(df_engineered.columns)
    preview_cols = list(pd.read_csv(V2_PREVIEW_PATH, nrows=5, low_memory=False).columns) if V2_PREVIEW_PATH.exists() else []
    used_features = set(pd.read_csv(USED_FEATURES_PATH)["feature"].astype(str).tolist())

    comparison_rows = []
    used_rows = []
    dropped_rows = []

    for feature in clean_cols:
        status, reason = clean_status(feature, used_features)
        row = {
            "source": "clean_dataset.csv",
            "feature": feature,
            "feature_group": feature_group(feature),
            "status": status,
            "reason": reason,
            "exists_in_df_engineered": feature in engineered_cols,
            "exists_in_v2_preview": feature in preview_cols,
        }
        comparison_rows.append(row)
        if status.startswith("used"):
            used_rows.append(row)
        elif status != "target":
            dropped_rows.append(row)

    for feature in engineered_cols:
        status, reason = engineered_status(feature, used_features)
        row = {
            "source": "df_engineered.csv",
            "feature": feature,
            "feature_group": feature_group(feature),
            "status": status,
            "reason": reason,
            "exists_in_clean_dataset": feature in clean_cols,
            "exists_in_v2_preview": feature in preview_cols,
        }
        comparison_rows.append(row)
        if status == "used":
            used_rows.append(row)
        elif status not in {"target", "metadata"}:
            dropped_rows.append(row)

    for feature in preview_cols:
        if feature not in engineered_cols and feature != "is_returned":
            status = "dropped_from_selected_v2"
            reason = "มีใน V2 preview/intermediate แต่ไม่อยู่ใน selected v2_xgboost_safe_plus_rolling feature set"
            if feature in LEAKAGE_OR_POST_EVENT:
                reason = "มีใน V2 preview แต่เป็น post-event/leakage field จึงตัดออกจาก selected model"
            row = {
                "source": "df_engineered_v2_preview.csv",
                "feature": feature,
                "feature_group": feature_group(feature),
                "status": status,
                "reason": reason,
                "exists_in_clean_dataset": feature in clean_cols,
                "exists_in_df_engineered": feature in engineered_cols,
            }
            comparison_rows.append(row)
            dropped_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    used = pd.DataFrame(used_rows).drop_duplicates(subset=["source", "feature"])
    dropped = pd.DataFrame(dropped_rows).drop_duplicates(subset=["source", "feature"])

    summary = {
        "clean_feature_count": len(clean_cols),
        "df_engineered_columns": len(engineered_cols),
        "model_used_feature_count": len(used_features),
        "dropped_or_not_used_count": len(dropped),
        "post_event_dropped": ", ".join(sorted((LEAKAGE_OR_POST_EVENT & set(clean_cols)) | (LEAKAGE_OR_POST_EVENT & set(preview_cols)))),
        "df_engineered_path": str(DF_ENGINEERED_ALIAS_PATH.relative_to(ROOT)),
        "clean_path": str(CLEAN_PATH.relative_to(ROOT)),
        "model_name": "v2_xgboost_safe_plus_rolling / XGBoost",
    }
    return comparison, used, dropped, summary


def write_outputs() -> None:
    comparison, used, dropped, summary = build_comparison_tables()
    comparison.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    used.to_csv(OUT_USED_CSV, index=False, encoding="utf-8-sig")
    dropped.to_csv(OUT_DROPPED_CSV, index=False, encoding="utf-8-sig")

    story: list[Any] = [p("V2 Feature Comparison: clean_dataset.csv vs df_engineered.csv", "title")]
    story.append(
        p(
            "เอกสารนี้สรุป feature ของ Version 2 สำหรับ model v2_xgboost_safe_plus_rolling ว่าเริ่มจาก clean_dataset.csv แล้วแปลงเป็น df_engineered.csv อย่างไร feature ไหนใช้เข้า model และ feature ไหนตัดทิ้งหรือไม่ใช้",
            "body",
        )
    )

    story.append(p("1. Summary", "h1"))
    story.append(
        make_table(
            [
                ["Item", "Value"],
                ["Model", summary["model_name"]],
                ["Source clean data", summary["clean_path"]],
                ["V2 df_engineered output", summary["df_engineered_path"]],
                ["Clean dataset columns", summary["clean_feature_count"]],
                ["df_engineered columns", summary["df_engineered_columns"]],
                ["Model input features", summary["model_used_feature_count"]],
                ["Dropped / not used rows in audit", summary["dropped_or_not_used_count"]],
                ["Post-event/leakage fields dropped", summary["post_event_dropped"]],
            ],
            widths=[6.5 * cm, 16.5 * cm],
        )
    )

    story.append(p("2. Feature Engineering Concept", "h1"))
    for text in [
        "V2 selected ใช้แนวคิด order-time safe: ใช้เฉพาะข้อมูลที่รู้ได้ตอน order เข้ามา และไม่ใช้ข้อมูลอนาคตของ order นั้น",
        "ตัด delivery_days และ delay_days เพราะเป็นข้อมูลหลังส่งสินค้า",
        "เพิ่ม rolling customer history 30/60/90/180/365 วัน เช่น hist_return_rate_90d = return_count_90d / order_count_90d",
        "เพิ่ม interaction features เช่น category_payment, category_channel, province_payment เพื่อให้ model เห็น pattern แบบผสม",
        "ใช้ customer/product/payment/channel/logistics/promotion/order-time features รวม 60 ตัวเข้า XGBoost",
    ]:
        story.append(p(f"- {text}", "body"))

    story.append(p("3. Used Features", "h1"))
    used_features = used[used["source"].eq("df_engineered.csv") & used["status"].eq("used")].copy()
    used_rows = [["Feature", "Group", "Reason"]]
    for _, row in used_features.iterrows():
        used_rows.append([row["feature"], row["feature_group"], row["reason"]])
    story.append(make_table(used_rows, widths=[6.0 * cm, 4.0 * cm, 12.5 * cm]))

    story.append(PageBreak())
    story.append(p("4. Dropped / Not Used Features", "h1"))
    story.append(p("ตารางนี้แสดง field จาก clean_dataset.csv และ intermediate V2 preview ที่ไม่ได้ใช้เข้า selected model พร้อมเหตุผล เช่น leakage, query-only หรือถูกแทนด้วย engineered feature", "body"))
    dropped_rows = [["Source", "Feature", "Status", "Reason", "Group"]]
    for _, row in dropped.iterrows():
        dropped_rows.append([row["source"], row["feature"], row["status"], row["reason"], row["feature_group"]])
    story.append(make_table(dropped_rows[:120], widths=[5.0 * cm, 4.0 * cm, 4.0 * cm, 6.5 * cm, 3.2 * cm]))
    if len(dropped_rows) > 120:
        story.append(p(f"หมายเหตุ: PDF แสดง 119 rows แรก รายละเอียดเต็มอยู่ใน CSV: {OUT_DROPPED_CSV.relative_to(ROOT)}", "body"))

    story.append(PageBreak())
    story.append(p("5. Clean Dataset Field Status", "h1"))
    clean_rows = [["Clean Feature", "Status", "Reason"]]
    clean_part = comparison[comparison["source"].eq("clean_dataset.csv")]
    for _, row in clean_part.iterrows():
        clean_rows.append([row["feature"], row["status"], row["reason"]])
    story.append(make_table(clean_rows, widths=[5.0 * cm, 4.5 * cm, 12.0 * cm]))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="V2 Feature Comparison",
    )
    doc.build(story)

    package_pdf = SELECTED_PACKAGE_DOC_DIR / OUT_PDF.name
    package_csv = SELECTED_PACKAGE_DOC_DIR / OUT_CSV.name
    package_used = SELECTED_PACKAGE_DOC_DIR / OUT_USED_CSV.name
    package_dropped = SELECTED_PACKAGE_DOC_DIR / OUT_DROPPED_CSV.name
    package_pdf.write_bytes(OUT_PDF.read_bytes())
    package_csv.write_bytes(OUT_CSV.read_bytes())
    package_used.write_bytes(OUT_USED_CSV.read_bytes())
    package_dropped.write_bytes(OUT_DROPPED_CSV.read_bytes())

    print(OUT_PDF)
    print(OUT_CSV)
    print(OUT_USED_CSV)
    print(OUT_DROPPED_CSV)
    print(DF_ENGINEERED_ALIAS_PATH)


if __name__ == "__main__":
    write_outputs()
