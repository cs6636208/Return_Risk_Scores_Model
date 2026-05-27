from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = ROOT / "docs" / "Comparison Version"
COMPARISON_IMG_DIR = COMPARISON_DIR / "images"
AUDIT_DIR = ROOT / "reports" / "feature_audit"
CLEAN_DATA = ROOT / "data" / "processed" / "clean_dataset.csv"


def ensure_dirs() -> None:
    COMPARISON_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for version in ["version 1", "version 2", "version 3", "version 4"]:
        (ROOT / "docs" / version / "feature_documentation").mkdir(parents=True, exist_ok=True)


def register_font() -> str:
    for candidate in [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/THSarabunNew.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("DocFont", str(candidate)))
            return "DocFont"
    return "Helvetica"


FONT = register_font()


def style_sheet() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleThai", parent=base["Title"], fontName=FONT, fontSize=19, leading=25, alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1Thai", parent=base["Heading1"], fontName=FONT, fontSize=14, leading=19, spaceBefore=10, spaceAfter=7),
        "h2": ParagraphStyle("H2Thai", parent=base["Heading2"], fontName=FONT, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("BodyThai", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=13.5, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("SmallThai", parent=base["BodyText"], fontName=FONT, fontSize=7.5, leading=10.5, alignment=TA_LEFT),
    }


S = style_sheet()


def para(text: Any, style: str = "body") -> Paragraph:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe.replace("\n", "<br/>"), S[style])


def table(rows: list[list[Any]], widths: list[float] | None = None, repeat_rows: int = 1) -> Table:
    t = Table([[para(cell, "small") for cell in row] for row in rows], colWidths=widths, repeatRows=repeat_rows)
    t.setStyle(
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
    return t


def build_pdf(output: Path, title: str, story: list[Any], landscape_mode: bool = False) -> None:
    doc = SimpleDocTemplate(
        str(output),
        pagesize=landscape(A4) if landscape_mode else A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        title=title,
    )
    doc.build(story)


def read_csv_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=5, low_memory=False).columns)


def feature_names_from_pkl(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = joblib.load(path)
    return list(data.get("feature_names", []))


def read_feature_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df["feature"].astype(str).tolist()
    return list(df.columns)


def model_summary() -> pd.DataFrame:
    return pd.read_csv(AUDIT_DIR / "model_version_completion_summary.csv")


def representative_models() -> pd.DataFrame:
    df = model_summary().copy()
    selected_versions = ["V1", "V2 XGBoost safe rolling", "V3 stacking", "V4 generated"]
    out = df[df["version"].isin(selected_versions)].copy()
    out["display_version"] = out["version"].replace({"V2 XGBoost safe rolling": "V2 selected"})
    return out


def feature_sources() -> dict[str, dict[str, Any]]:
    return {
        "V1": {
            "folder": ROOT / "docs" / "version 1",
            "engineered_path": ROOT / "docs" / "version 1" / "data" / "features" / "df_engineered.csv",
            "engineered_features": read_csv_columns(ROOT / "docs" / "version 1" / "data" / "features" / "df_engineered.csv"),
            "used_features": feature_names_from_pkl(ROOT / "docs" / "version 1" / "data" / "features" / "train_test_sets.pkl"),
            "code_files": [
                ROOT / "docs" / "version 1" / "feature_engineering.py",
                ROOT / "docs" / "version 1" / "model_training.py",
                ROOT / "docs" / "version 1" / "model_evaluation.py",
            ],
            "model_note": "Baseline feature engineering: สร้าง feature จำนวนมากและ encode จนได้ feature model 136 ตัว",
            "process": [
                "Load clean_dataset.csv และเลือก field ที่เกี่ยวกับ order, customer, product, promotion, channel, payment และ history",
                "สร้าง numeric/log features เช่น log_unit_price และ log_total_amount เพื่อลด skew ของราคาและยอดรวม",
                "สร้าง binary flags เช่น is_peak_hour, is_cod, is_high_discount, is_first_order, low_rating_alert",
                "สร้าง interaction features เช่น category_payment, category_channel, province_payment",
                "สร้าง customer history features เช่น total_orders_before, total_returns_before, customer_return_ratio และ rolling history 30/60/180 วัน",
                "Encode categorical variables และสร้าง train_test_sets.pkl",
                "Train หลาย model แล้ว save best_model.pkl",
            ],
            "reasoning": "V1 ใช้เป็น baseline กว้าง ๆ เพื่อดูว่าการสร้าง feature จำนวนมากช่วย model ได้แค่ไหน แต่ recall ต่ำและ cost สูง จึงยังไม่ใช่ candidate หลัก",
        },
        "V2": {
            "folder": ROOT / "docs" / "version 2",
            "engineered_path": ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv",
            "engineered_features": read_csv_columns(ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv"),
            "used_features": read_feature_csv(ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "data" / "v2_xgboost_safe_plus_rolling_used_features.csv"),
            "code_files": [
                ROOT / "docs" / "version 2" / "feature_engineering_v2.py",
                ROOT / "docs" / "version 2" / "train_v2_optimized_model.py",
                ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "scripts" / "train_v2_optimized_model.py",
            ],
            "model_note": "Selected candidate: v2_xgboost_safe_plus_rolling ใช้ XGBoost และ feature 60 ตัวแบบ order-time safe",
            "process": [
                "เริ่มจาก clean_dataset.csv แล้วลด feature ให้ตีความง่ายกว่า V1",
                "สร้าง customer_tenure_months, order_month, order_dayofweek, is_weekend, age_group และ logistics_risk",
                "ตัด post-delivery fields สำหรับ selected candidate เช่น delivery_days และ delay_days",
                "เพิ่ม rolling customer history 30/60/90/180/365 วัน เช่น hist_return_rate_30d และ hist_spend_sum_365d",
                "เพิ่ม interaction features เช่น discount_amount_ratio, category_payment, category_channel, province_payment",
                "ใช้ target encoding/smoothing สำหรับ categorical feature",
                "Train XGBoost และเลือก threshold 0.49 จาก balanced metric",
            ],
            "reasoning": "V2 selected ถูกเลือกเพราะใช้ข้อมูลจริง, order-time safe, accuracy สูงสุดในกลุ่มข้อมูลจริง, precision ดีขึ้น, feature สอดคล้องกับ Feature Store และไม่มี delivery_days/delay_days leakage",
        },
        "V3": {
            "folder": ROOT / "docs" / "version 3",
            "engineered_path": ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv",
            "engineered_features": read_csv_columns(ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv"),
            "used_features": read_feature_csv(ROOT / "docs" / "version 3" / "stacking_model_v3" / "data" / "v3_used_features.csv"),
            "code_files": [
                ROOT / "docs" / "version 3" / "stacking_model_v3" / "scripts" / "model_training_v3_stacking.py",
                ROOT / "docs" / "version 3" / "stacking_model_v3" / "scripts" / "model_evaluation_v3.py",
            ],
            "model_note": "V3 เป็น model architecture experiment: Stacking XGBoost + LightGBM + CatBoost โดย reuse feature จาก V2",
            "process": [
                "Load train_test_sets_v2.pkl จาก V2",
                "ไม่ได้สร้าง feature engineering ใหม่เอง แต่ใช้ V2 feature 38 ตัว",
                "Train base models: XGBoost, LightGBM และ CatBoost",
                "ใช้ LogisticRegression เป็น meta learner ใน StackingClassifier",
                "Evaluate threshold scenarios, recall และ cost matrix",
            ],
            "reasoning": "V3 พิสูจน์ว่าการเปลี่ยน model architecture เป็น ensemble ช่วย recall ได้ แต่ยัง reuse V2 feature ที่มี delivery_days/delay_days และไม่ได้แก้ order-time safety",
        },
        "V4": {
            "folder": ROOT / "docs" / "version 4",
            "engineered_path": ROOT / "docs" / "version 4" / "data" / "features" / "df_engineered_v4_generated.csv",
            "engineered_features": read_csv_columns(ROOT / "docs" / "version 4" / "data" / "features" / "df_engineered_v4_generated.csv"),
            "used_features": read_feature_csv(ROOT / "docs" / "version 4" / "reports" / "model_evaluation" / "v4_used_features.csv"),
            "code_files": [
                ROOT / "docs" / "version 4" / "scripts" / "clean_dataset_v4.py",
                ROOT / "docs" / "version 4" / "scripts" / "run_v4_generated_end_to_end_pipeline.py",
            ],
            "model_note": "V4 เป็น generated/synthetic end-to-end pipeline ใช้ XGBoost + SMOTE + Optuna",
            "process": [
                "Generate synthetic order/return data เพื่อจำลอง imbalance data",
                "Clean data และสร้าง clean_dataset_v4_generated.csv",
                "ทำ EDA: target distribution, category, channel, payment และ correlation heatmap",
                "สร้าง point-in-time aggregate features เช่น customer/category/brand/province/payment/channel/courier return rates",
                "สร้าง one-hot encoded features และ feature interactions",
                "ใช้ SMOTE สำหรับ imbalanced data",
                "Train LogisticRegression, RandomForest, XGBoost, LightGBM และ tune ด้วย Optuna",
                "Evaluate ด้วย Accuracy, Recall, F1, AUC, Cost Matrix และ SHAP",
            ],
            "reasoning": "V4 เหมาะเป็น showcase end-to-end pipeline เพราะ Accuracy สูง แต่เป็น generated data และ cost สูงกว่า V2 จึงไม่ใช่ production winner บนข้อมูลจริง",
        },
    }


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
IDENTIFIER_FIELDS = {"order_id", "customer_id", "customer_name", "customer_phone", "product_id", "supplier_id", "courier_id", "promo_id"}


def infer_source_used(clean_cols: list[str], used_features: list[str]) -> set[str]:
    used = set(used_features)
    source = set()
    for col in clean_cols:
        if col in used:
            source.add(col)
            continue
        prefix = f"{col}_"
        if any(feature.startswith(prefix) for feature in used):
            source.add(col)
    return source


def feature_group(feature: str) -> str:
    low = feature.lower()
    groups = [
        ("customer_history", ["hist_", "cust_", "orders_before", "returns_before", "return_ratio", "days_since"]),
        ("product_category", ["category", "brand", "product", "rating", "fragile"]),
        ("payment_channel", ["payment", "channel", "cod"]),
        ("promotion_discount", ["promo", "discount", "campaign"]),
        ("logistics", ["courier", "delivery", "delay", "damage", "logistics"]),
        ("customer_profile", ["gender", "age", "membership", "province", "tenure"]),
        ("time", ["hour", "month", "dayofweek", "weekend"]),
        ("price_amount", ["price", "amount", "quantity", "spend"]),
    ]
    for group, keys in groups:
        if any(key in low for key in keys):
            return group
    return "other"


def summarize_feature_groups(features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        rows.append({"feature_group": feature_group(feature)})
    if not rows:
        return pd.DataFrame(columns=["feature_group", "count"])
    return pd.DataFrame(rows).value_counts("feature_group").reset_index(name="count")


def comma_list(values: list[str], max_items: int = 30) -> str:
    values = [str(v) for v in values]
    if not values:
        return "-"
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f", ... (+{len(values) - max_items} more)"


def save_comparison_charts(models: pd.DataFrame) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    versions = models["display_version"].tolist()
    metric_cols = ["accuracy", "recall", "precision", "f1", "auc"]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
    colors = ["#2f6fbb", "#d65f5f", "#4f9d69", "#8064a2", "#d49a3a"]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(models))
    width = 0.15
    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - 2) * width, models[col].astype(float), width, label=metric_labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(versions, rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison by Version")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    charts["performance"] = COMPARISON_IMG_DIR / "version_1_to_4_performance_metrics.png"
    fig.savefig(charts["performance"], dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(versions, models["cost"].astype(float), color=["#8d99ae", "#2a9d8f", "#457b9d", "#e76f51"])
    ax.set_ylabel("Cost")
    ax.set_title("Cost Matrix Comparison")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, models["cost"].astype(float)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 500, f"{int(value):,}", ha="center", fontsize=9)
    fig.tight_layout()
    charts["cost"] = COMPARISON_IMG_DIR / "version_1_to_4_cost_comparison.png"
    fig.savefig(charts["cost"], dpi=170)
    plt.close(fig)

    fs = pd.read_csv(AUDIT_DIR / "feature_audit_summary_by_version.csv")
    feature_rows = [
        ("V1", "v1_model_used"),
        ("V2 selected", "v2_xgboost_safe_plus_rolling"),
        ("V3", "v3_stacking"),
        ("V4", "v4_generated_model_used"),
    ]
    labels = []
    counts = []
    for label, key in feature_rows:
        row = fs[fs["version"].eq(key)]
        if not row.empty:
            labels.append(label)
            counts.append(int(float(row.iloc[0]["used_feature_count"])))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color=["#6c757d", "#2a9d8f", "#457b9d", "#e9c46a"])
    ax.set_ylabel("Used feature count")
    ax.set_title("Feature Count Used by Representative Version")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    charts["features"] = COMPARISON_IMG_DIR / "version_1_to_4_feature_count.png"
    fig.savefig(charts["features"], dpi=170)
    plt.close(fig)

    return charts


def write_version_comparison_pdf() -> None:
    ensure_dirs()
    models = representative_models()
    charts = save_comparison_charts(models)
    models.to_csv(COMPARISON_DIR / "version_1_to_4_selected_model_comparison.csv", index=False, encoding="utf-8-sig")

    rows = [["Version", "Model", "Order-time safety", "Accuracy", "Recall", "Precision", "F1", "AUC", "Cost", "Rating", "Interpretation"]]
    safety = {
        "V1": "No clear order-time guarantee",
        "V2 selected": "Yes",
        "V3 stacking": "No, reuses V2 baseline features",
        "V4 generated": "Synthetic pipeline, not production comparable",
    }
    for _, row in models.iterrows():
        label = row["display_version"]
        rows.append(
            [
                label,
                row["model"],
                safety.get(label, ""),
                f"{float(row['accuracy']):.2%}",
                f"{float(row['recall']):.2%}",
                f"{float(row['precision']):.2%}",
                f"{float(row['f1']):.2%}",
                f"{float(row['auc']):.2%}",
                f"{int(float(row['cost'])):,}",
                row["rating"],
                row["note"],
            ]
        )

    story: list[Any] = [para("Detailed Version Comparison: V1 - V4", "title")]
    story.append(para("เอกสารนี้เปรียบเทียบ version 1 ถึง version 4 โดยใช้ metric ล่าสุดจาก project artifacts และอธิบายเหตุผลว่าทำไม candidate ที่เหมาะกับการต่อยอดที่สุดคือ V2 v2_xgboost_safe_plus_rolling", "body"))
    story.append(para("Executive Summary", "h1"))
    for text in [
        "V1 เป็น baseline ที่ช่วยให้เห็นภาพ feature engineering กว้าง ๆ แต่ recall ต่ำและ cost สูง จึงไม่เหมาะเป็น candidate หลัก",
        "V2 baseline มี recall และ cost ดีมากบนข้อมูลจริง แต่ยังมี delivery_days/delay_days จึงไม่ order-time safe",
        "V2 v2_xgboost_safe_plus_rolling ใช้ XGBoost, feature 60 ตัว, ตัดข้อมูลหลังเหตุการณ์ และเพิ่ม rolling history จึงเหมาะกับ Feature Store และ real-time direction มากที่สุด",
        "V3 ใช้ stacking ensemble และ recall ดี แต่ไม่ได้สร้าง feature ใหม่เองและ reuse V2 feature ที่ยังมี post-delivery fields",
        "V4 ได้ Accuracy สูงที่สุด แต่เป็น generated/synthetic data และ cost สูง จึงใช้เป็น pipeline showcase มากกว่า production winner",
    ]:
        story.append(para(f"- {text}", "body"))

    story.append(para("Performance Table", "h1"))
    story.append(table(rows, widths=[2.0 * cm, 3.2 * cm, 3.1 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.2 * cm, 4.5 * cm]))
    story.append(PageBreak())

    story.append(para("Performance Graphs", "h1"))
    for key in ["performance", "cost", "features"]:
        story.append(Image(str(charts[key]), width=24.5 * cm, height=10.5 * cm, kind="proportional"))
        story.append(Spacer(1, 0.35 * cm))

    story.append(PageBreak())
    story.append(para("Why Choose V2 v2_xgboost_safe_plus_rolling", "h1"))
    reasons = [
        ("Uses real project data", "V4 มี Accuracy สูงกว่า แต่เป็น synthetic/generated data จึงไม่ควรเอามาตัดสิน production บนข้อมูลจริงโดยตรง"),
        ("Order-time safe", "ตัด delivery_days และ delay_days ออก ทำให้ใช้กับ order ที่เพิ่งเข้ามาได้ ไม่แอบใช้ข้อมูลอนาคต"),
        ("Feature Store ready", "rolling history 30/60/90/180/365 วันสามารถคำนวณจาก SQL DB หรือ customer_feature_snapshot ได้"),
        ("Best real-data accuracy among practical candidates", "Accuracy 71.07% สูงกว่า V1 และ V3 และยังรักษา F1/Precision ได้ดีกว่า baseline หลายตัว"),
        ("Business explainability", "feature กลุ่ม customer history, category/payment/channel, discount และ province อธิบายได้ด้วย Business Insight ที่มีรูปประกอบ"),
        ("Resource control", "feature 60 ตัวมากกว่า V2 baseline 38 ตัว แต่ยังน้อยกว่า V1 encoded 136 และ V4 180 ทำให้เหมาะกับการต่อยอดมากกว่า full-heavy feature set"),
    ]
    story.append(table([["Reason", "Explanation"], *reasons], widths=[5.5 * cm, 18.5 * cm]))

    story.append(para("Decision", "h1"))
    story.append(para("เลือก V2 v2_xgboost_safe_plus_rolling เป็น candidate หลักสำหรับการอธิบาย production direction เพราะให้สมดุลที่ดีที่สุดระหว่างข้อมูลจริง, order-time safety, feature engineering ที่มีเหตุผลทางธุรกิจ, performance ที่แข็งแรง และการต่อยอดเป็น Feature Store", "body"))

    build_pdf(COMPARISON_DIR / "version_1_to_4_detailed_comparison.pdf", "Version 1 to 4 Detailed Comparison", story, landscape_mode=True)


def feature_audit(version_label: str, info: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_cols = read_csv_columns(CLEAN_DATA)
    engineered = list(info["engineered_features"])
    used = list(info["used_features"])
    clean_set = set(clean_cols)
    engineered_set = set(engineered)
    used_set = set(used)
    source_used = infer_source_used(clean_cols, used)
    clean_not_used = sorted((clean_set - source_used) - {"is_returned"})
    risky_not_used = sorted(set(clean_not_used) & (LEAKAGE_OR_POST_EVENT | IDENTIFIER_FIELDS))
    engineered_used_exact = sorted(engineered_set & used_set)
    engineered_not_used_exact = sorted((engineered_set - used_set) - {"is_returned"})
    new_used = sorted([f for f in used if f not in clean_set])

    rows = []
    for feature in clean_cols:
        if feature == "is_returned":
            status = "target"
            reason = "target label ไม่ใช่ input feature"
        elif feature in source_used:
            status = "used_source_or_encoded"
            reason = "ใช้ตรงหรือถูก encode/derive เป็น model feature"
        elif feature in LEAKAGE_OR_POST_EVENT:
            status = "dropped"
            reason = "post-event/leakage field ไม่ควรใช้กับ training หรือ order-time scoring"
        elif feature in IDENTIFIER_FIELDS:
            status = "query_or_audit_only"
            reason = "ใช้ join/query/audit ได้ แต่ไม่ควรให้ model จำ identity โดยตรง"
        else:
            status = "not_used_or_replaced"
            reason = "ไม่ได้ใช้ตรง หรือถูกแทนด้วย engineered/encoded feature"
        rows.append({"source": "clean_dataset.csv", "feature": feature, "status": status, "reason": reason, "group": feature_group(feature)})

    for feature in engineered:
        if feature == "is_returned":
            status = "target"
            reason = "target label"
        elif feature in used_set:
            status = "used_exact"
            reason = "อยู่ใน model feature list โดยตรง"
        elif any(u.startswith(f"{feature}_") for u in used):
            status = "used_after_encoding"
            reason = "ถูก one-hot/encoding แตกเป็น feature ย่อย"
        else:
            status = "not_used"
            reason = "ไม่อยู่ใน selected model feature set หรือถูกตัดเพื่อลด feature/resource"
        rows.append({"source": str(info["engineered_path"].relative_to(ROOT)), "feature": feature, "status": status, "reason": reason, "group": feature_group(feature)})

    for feature in used:
        if feature not in clean_set and feature not in engineered_set:
            rows.append({"source": "model_feature_set", "feature": feature, "status": "used_encoded_or_generated", "reason": "feature ที่เกิดหลัง encoding, rolling aggregation, one-hot หรือ generated pipeline", "group": feature_group(feature)})

    summary = {
        "clean_count": len(clean_cols),
        "engineered_count": len(engineered),
        "used_count": len(used),
        "source_clean_used_count": len(source_used),
        "clean_not_used_count": len(clean_not_used),
        "risky_not_used_count": len(risky_not_used),
        "engineered_used_exact_count": len(engineered_used_exact),
        "engineered_not_used_exact_count": len(engineered_not_used_exact),
        "new_used_count": len(new_used),
        "source_used": sorted(source_used),
        "clean_not_used": clean_not_used,
        "risky_not_used": risky_not_used,
        "engineered_used_exact": engineered_used_exact,
        "engineered_not_used_exact": engineered_not_used_exact,
        "new_used": new_used,
    }
    return pd.DataFrame(rows), summary


def write_feature_audit_pdf(version_label: str, info: dict[str, Any]) -> None:
    df, summary = feature_audit(version_label, info)
    out_dir = info["folder"] / "feature_documentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{version_label.lower()}_feature_used_unused_audit.csv", index=False, encoding="utf-8-sig")

    story: list[Any] = [para(f"{version_label} Feature Used / Not Used Audit", "title")]
    story.append(para(f"เอกสารนี้เทียบ feature ของ {version_label} กับ clean_dataset.csv และ engineered dataset ของ version นี้ เพื่ออธิบายว่า feature ไหนถูกใช้ ตัดทิ้ง หรือใช้เฉพาะ query/audit", "body"))
    story.append(para("Summary", "h1"))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Clean dataset columns", summary["clean_count"]],
                ["Engineered columns", summary["engineered_count"]],
                ["Model used features", summary["used_count"]],
                ["Clean source fields used or encoded", summary["source_clean_used_count"]],
                ["Clean source fields not used", summary["clean_not_used_count"]],
                ["Risky/leakage fields excluded", summary["risky_not_used_count"]],
                ["New/encoded/engineered used features", summary["new_used_count"]],
            ],
            widths=[8.5 * cm, 7.5 * cm],
        )
    )
    story.append(para("Important Used Feature Groups", "h1"))
    group_df = summarize_feature_groups(info["used_features"])
    story.append(table([["Feature group", "Count"], *group_df.values.tolist()], widths=[8 * cm, 3 * cm]))
    story.append(para("Used Features", "h1"))
    story.append(para(comma_list(info["used_features"], max_items=120), "small"))
    story.append(para("Clean Dataset Fields Not Used / Dropped", "h1"))
    story.append(para(comma_list(summary["clean_not_used"], max_items=120), "small"))
    story.append(para("Risky / Leakage / Identifier Fields Excluded or Query-only", "h1"))
    story.append(para(comma_list(summary["risky_not_used"], max_items=120), "small"))
    story.append(para("Engineered Features Not Used Exactly", "h1"))
    story.append(para("หมายเหตุ: ถ้า feature ถูก one-hot/encoding แล้วชื่อจะเปลี่ยน จึงอาจไม่ match แบบ exact แต่ยังถือว่า source feature ถูกใช้ผ่าน encoded feature ได้", "body"))
    story.append(para(comma_list(summary["engineered_not_used_exact"], max_items=120), "small"))

    story.append(PageBreak())
    story.append(para("Detailed Audit Table", "h1"))
    detail_rows = [["Source", "Feature", "Status", "Reason", "Group"]]
    for _, row in df.head(220).iterrows():
        detail_rows.append([row["source"], row["feature"], row["status"], row["reason"], row["group"]])
    story.append(table(detail_rows, widths=[5.0 * cm, 4.2 * cm, 3.0 * cm, 5.5 * cm, 2.8 * cm]))
    if len(df) > 220:
        story.append(para(f"ตารางใน PDF แสดง 220 rows แรก รายละเอียดเต็มอยู่ใน CSV: {out_dir.relative_to(ROOT) / (version_label.lower() + '_feature_used_unused_audit.csv')}", "body"))

    build_pdf(out_dir / f"{version_label.lower()}_feature_used_unused_audit.pdf", f"{version_label} Feature Audit", story)


def write_feature_engineering_process_pdf(version_label: str, info: dict[str, Any], metrics: pd.Series | None) -> None:
    out_dir = info["folder"] / "feature_documentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_refs = [str(path.relative_to(ROOT)) for path in info["code_files"] if path.exists()]

    story: list[Any] = [para(f"{version_label} Feature Engineering Process", "title")]
    story.append(para(info["model_note"], "body"))
    story.append(para("Input / Output", "h1"))
    io_rows = [
        ["Input clean data", str(CLEAN_DATA.relative_to(ROOT))],
        ["Engineered dataset", str(info["engineered_path"].relative_to(ROOT))],
        ["Used feature count", len(info["used_features"])],
        ["Code files", "\n".join(code_refs)],
    ]
    story.append(table([["Item", "Path / Value"], *io_rows], widths=[5 * cm, 11.5 * cm]))

    story.append(para("Feature Engineering Steps", "h1"))
    for idx, step in enumerate(info["process"], start=1):
        story.append(para(f"{idx}. {step}", "body"))

    story.append(para("Feature Groups Created / Used", "h1"))
    group_df = summarize_feature_groups(info["used_features"])
    story.append(table([["Feature group", "Count"], *group_df.values.tolist()], widths=[8 * cm, 3 * cm]))

    story.append(para("Reasoning", "h1"))
    story.append(para(info["reasoning"], "body"))

    if metrics is not None:
        story.append(para("Model Result Snapshot", "h1"))
        metric_rows = [
            ["Metric", "Value"],
            ["Model", metrics["model"]],
            ["Accuracy", f"{float(metrics['accuracy']):.2%}"],
            ["Recall", f"{float(metrics['recall']):.2%}"],
            ["Precision", f"{float(metrics['precision']):.2%}"],
            ["F1", f"{float(metrics['f1']):.2%}"],
            ["AUC", f"{float(metrics['auc']):.2%}"],
            ["Cost", f"{int(float(metrics['cost'])):,}"],
            ["Rating", metrics["rating"]],
        ]
        story.append(table(metric_rows, widths=[5 * cm, 5 * cm]))

    story.append(para("Pseudo-code / Code Logic", "h1"))
    pseudo = pseudo_code_for(version_label)
    story.append(para(pseudo, "small"))

    md = [
        f"# {version_label} Feature Engineering Process",
        "",
        info["model_note"],
        "",
        "## Code Files",
        *[f"- `{ref}`" for ref in code_refs],
        "",
        "## Steps",
        *[f"{i}. {step}" for i, step in enumerate(info["process"], start=1)],
        "",
        "## Reasoning",
        info["reasoning"],
        "",
        "## Pseudo-code",
        "```python",
        pseudo,
        "```",
    ]
    (out_dir / f"{version_label.lower()}_feature_engineering_process.md").write_text("\n".join(md), encoding="utf-8")
    build_pdf(out_dir / f"{version_label.lower()}_feature_engineering_process.pdf", f"{version_label} Feature Engineering Process", story)


def pseudo_code_for(version_label: str) -> str:
    snippets = {
        "V1": """
df = read_csv("clean_dataset.csv")
df["log_unit_price"] = log1p(unit_price)
df["log_total_amount"] = log1p(total_amount)
df["is_cod"] = payment_method == "COD"
df["is_high_discount"] = total_discount_pct > 0.20
df["category_payment"] = category + "_" + payment_method
df["category_channel"] = category + "_" + channel_type
df = add_customer_history(df)
X = encode_and_scale(df.drop("is_returned"))
y = df["is_returned"]
train_test_sets = train_test_split(X, y, stratify=y, random_state=42)
""",
        "V2": """
df = read_csv("clean_dataset.csv")
df = add_v2_basic_features(df)
df = add_customer_tenure_order_time_age_group(df)
df = add_logistics_risk(df)
df = drop_post_event_for_safe_candidate(["delivery_days", "delay_days"])
df = add_rolling_history_windows([30, 60, 90, 180, 365])
df = add_interactions(["category_payment", "category_channel", "province_payment"])
X = target_encode_with_smoothing(df[selected_features])
model = XGBClassifier(...)
model.fit(X_train, y_train)
""",
        "V3": """
data = joblib.load("train_test_sets_v2.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
base_models = [XGBClassifier(), LGBMClassifier(), CatBoostClassifier()]
model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
model.fit(X_train, y_train)
evaluate_thresholds(model.predict_proba(X_test))
""",
        "V4": """
raw = generate_synthetic_orders_returns()
clean = clean_dataset_v4(raw)
features = add_point_in_time_aggregates(clean)
features = add_one_hot_interactions(features)
X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target)
X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)
model = tune_xgboost_with_optuna(X_train_smote, y_train_smote)
evaluate_cost_auc_shap(model, X_test, y_test)
""",
    }
    return textwrap.dedent(snippets[version_label]).strip()


def generate_per_version_docs() -> None:
    models = model_summary()
    version_to_metric = {
        "V1": "V1",
        "V2": "V2 XGBoost safe rolling",
        "V3": "V3 stacking",
        "V4": "V4 generated",
    }
    for version_label, info in feature_sources().items():
        write_feature_audit_pdf(version_label, info)
        metric_row = models[models["version"].eq(version_to_metric[version_label])]
        metric = metric_row.iloc[0] if not metric_row.empty else None
        write_feature_engineering_process_pdf(version_label, info, metric)


def main() -> None:
    ensure_dirs()
    write_version_comparison_pdf()
    generate_per_version_docs()
    print(COMPARISON_DIR.relative_to(ROOT))
    print((COMPARISON_DIR / "version_1_to_4_detailed_comparison.pdf").relative_to(ROOT))
    for version in ["version 1", "version 2", "version 3", "version 4"]:
        out_dir = ROOT / "docs" / version / "feature_documentation"
        for path in sorted(out_dir.glob("*.pdf")):
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
