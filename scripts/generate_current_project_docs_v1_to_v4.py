from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import shorten

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
from reportlab.platypus import (
    Image,
    KeepTogether,
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
DOCS = ROOT / "docs"
COMPARISON_DIR = DOCS / "Comparison Version"
COMPARISON_IMG_DIR = COMPARISON_DIR / "images"
DATA_PROCESSED = ROOT / "data" / "processed"


def register_fonts() -> tuple[str, str]:
    """Register a Thai-capable font when available."""
    regular_candidates = [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/tahomabd.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]

    regular = next((p for p in regular_candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), regular)
    if regular:
        pdfmetrics.registerFont(TTFont("DocFont", str(regular)))
        pdfmetrics.registerFont(TTFont("DocFont-Bold", str(bold)))
        return "DocFont", "DocFont-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=18,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.5,
            leading=11,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7,
            leading=9,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), S[style])


def bullet(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if abs(float(value) - int(float(value))) < 1e-9:
        return f"{int(float(value)):,}"
    return f"{float(value):,.2f}"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def read_feature_list(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    df = pd.read_csv(path)
    for col in ["feature", "feature_name", "features", "column"]:
        if col in df.columns:
            return [str(v) for v in df[col].dropna().tolist()]
    if len(df.columns) == 1:
        return [str(v) for v in df.iloc[:, 0].dropna().tolist()]
    return [str(v) for v in df.iloc[:, 0].dropna().tolist()]


def read_columns(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def clean_metric_frame() -> pd.DataFrame:
    old_cmp = read_csv_if_exists(COMPARISON_DIR / "version_1_to_4_selected_model_comparison.csv")
    v2_meta = read_json_if_exists(
        DOCS
        / "version 2"
        / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY"
        / "models"
        / "best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY_metadata.json"
    )
    v3_csv = read_csv_if_exists(DOCS / "version 3" / "stacking_model_v3" / "reports" / "metrics_summary_v3.csv")
    v4_csv = read_csv_if_exists(DOCS / "version 4" / "reports" / "model_evaluation" / "v4_generated_model_metrics.csv")

    def old_row(version: str) -> dict:
        if old_cmp is None or "version" not in old_cmp.columns:
            return {}
        rows = old_cmp[old_cmp["version"].astype(str).eq(version)]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    def metric_values(row: dict) -> dict:
        keep = [
            "accuracy",
            "recall",
            "precision",
            "f1",
            "auc",
            "avg_precision",
            "cost",
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        return {key: row[key] for key in keep if key in row}

    v1 = metric_values(old_row("V1")) or {
        "accuracy": 0.708,
        "recall": 0.268041237113402,
        "precision": 0.4968152866242038,
        "f1": 0.3482142857142857,
        "auc": 0.6882352085847644,
        "avg_precision": 0.4463599411314076,
        "cost": 35900,
        "threshold": 0.5,
        "tn": 630,
        "fp": 79,
        "fn": 213,
        "tp": 78,
    }

    if v3_csv is not None and not v3_csv.empty:
        r3 = v3_csv.iloc[0].to_dict()
        v3 = {
            "accuracy": r3.get("optimal_cost_accuracy", 0.6666666667),
            "recall": r3.get("recall", 0.6376146789),
            "precision": r3.get("precision", 0.4483870968),
            "f1": r3.get("f1_score", 0.5265151515),
            "auc": r3.get("auc_roc", 0.7189849624),
            "avg_precision": r3.get("avg_precision", 0.4878256192),
            "cost": r3.get("optimal_cost", 19500),
            "threshold": r3.get("optimal_cost_threshold", r3.get("threshold", 0.45)),
            "tn": r3.get("tn"),
            "fp": r3.get("fp"),
            "fn": r3.get("fn"),
            "tp": r3.get("tp"),
        }
    else:
        v3 = old_row("V3 stacking")

    if v4_csv is not None and not v4_csv.empty:
        rv4 = v4_csv[v4_csv["model"].astype(str).str.contains("XGBoost", case=False, na=False)]
        r4 = (rv4.iloc[0] if not rv4.empty else v4_csv.iloc[0]).to_dict()
        v4 = {
            "accuracy": r4.get("accuracy", 0.8345360825),
            "recall": r4.get("recall", 0.4639175258),
            "precision": r4.get("precision", 0.45),
            "f1": r4.get("f1", 0.4568527919),
            "auc": r4.get("auc", 0.8538112237),
            "avg_precision": r4.get("avg_precision", 0.4446689081),
            "cost": r4.get("cost_thb", r4.get("cost", 31650)),
            "threshold": r4.get("threshold", 0.34),
            "tn": r4.get("tn"),
            "fp": r4.get("fp"),
            "fn": r4.get("fn"),
            "tp": r4.get("tp"),
        }
    else:
        v4 = old_row("V4 generated")

    rows = [
        {
            "display_version": "V1",
            "version": "v1_baseline_xgboost",
            "model": "XGBoost baseline",
            "dataset": "clean_dataset.csv",
            "rows": 5000,
            "test_rows": 1000,
            "feature_count": len(read_feature_list(DOCS / "version 1" / "feature_documentation" / "used_features.csv")) or 136,
            "order_time_safe": "Partial",
            "rating": "D",
            "note": "baseline feature engineering; recall ต่ำและ cost สูง จึงใช้เป็นจุดเทียบ",
            **v1,
        },
        {
            "display_version": "V2",
            "version": "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY",
            "model": "XGBoost safe rolling",
            "dataset": "clean_dataset_v2_high_signal.csv",
            "rows": v2_meta.get("rows", 50000),
            "test_rows": v2_meta.get("test_rows", 10000),
            "feature_count": v2_meta.get("feature_count", 71),
            "order_time_safe": "Yes",
            "rating": "A",
            "note": "current selected model; rolling history + insight features + leakage exclusion",
            "accuracy": v2_meta.get("accuracy", 0.8888),
            "recall": v2_meta.get("recall", 0.7602599179206566),
            "precision": v2_meta.get("precision", 0.8439635535307517),
            "f1": v2_meta.get("f1", 0.799928031666067),
            "auc": v2_meta.get("auc", 0.9465833236024899),
            "avg_precision": v2_meta.get("avg_precision", 0.9002822163562983),
            "cost": v2_meta.get("cost", 371050),
            "threshold": v2_meta.get("threshold", 0.71),
            "tn": v2_meta.get("tn", 6665),
            "fp": v2_meta.get("fp", 411),
            "fn": v2_meta.get("fn", 701),
            "tp": v2_meta.get("tp", 2223),
        },
        {
            "display_version": "V3",
            "version": "v3_stacking_from_v2",
            "model": "Stacking XGB+LGBM+CatBoost",
            "dataset": "V2 engineered feature set",
            "rows": 3750,
            "test_rows": 750,
            "feature_count": len(read_feature_list(DOCS / "version 3" / "stacking_model_v3" / "data" / "v3_used_features.csv")) or 38,
            "order_time_safe": "Yes",
            "rating": "B",
            "note": "tests model architecture; feature base มาจาก V2 จึงซับซ้อนขึ้นแต่ไม่ชนะ V2",
            **v3,
        },
        {
            "display_version": "V4",
            "version": "v4_generated_xgboost_smote_optuna",
            "model": "XGBoost + SMOTE + Optuna",
            "dataset": "clean_dataset_v4_generated.csv",
            "rows": 9700,
            "test_rows": 1940,
            "feature_count": len(read_feature_list(DOCS / "version 4" / "reports" / "model_evaluation" / "v4_used_features.csv")) or 180,
            "order_time_safe": "No/Benchmark",
            "rating": "C",
            "note": "generated-data benchmark; accuracy สูงแต่ recall และ feature volume ยังไม่เหมาะกว่า V2",
            **v4,
        },
    ]
    df = pd.DataFrame(rows)
    for col in ["accuracy", "recall", "precision", "f1", "auc", "avg_precision", "cost", "test_rows"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cost_per_1k_test_orders"] = df["cost"] / df["test_rows"].replace(0, pd.NA) * 1000
    return df


def table_from_df(df: pd.DataFrame, col_widths: list[float] | None = None, header_color="#1f4e79") -> Table:
    data = [[p(col, "table_header") for col in df.columns]]
    for _, row in df.iterrows():
        data.append([p(row[col], "table_cell") for col in df.columns])
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def save_comparison_outputs(metrics: pd.DataFrame) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_IMG_DIR.mkdir(parents=True, exist_ok=True)

    export = metrics.copy()
    export.to_csv(COMPARISON_DIR / "version_1_to_4_selected_model_comparison.csv", index=False, encoding="utf-8-sig")

    labels = metrics["display_version"].tolist()
    colors_map = ["#65737e", "#2e7d32", "#f9a825", "#1565c0"]

    perf_metrics = [
        ("accuracy", "Accuracy"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1-score"),
        ("auc", "AUC"),
    ]
    metric_names = [name for _, name in perf_metrics]
    values = np.array([metrics[col].astype(float).to_numpy() * 100 for col, _ in perf_metrics])
    y = np.arange(len(perf_metrics))
    bar_height = 0.16
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * bar_height

    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    for version_idx, label in enumerate(labels):
        edge_color = "#111111" if label == "V2" else "white"
        line_width = 1.8 if label == "V2" else 0.6
        alpha = 1.0 if label == "V2" else 0.88
        bars = ax.barh(
            y + offsets[version_idx],
            values[:, version_idx],
            height=bar_height * 0.88,
            color=colors_map[version_idx],
            edgecolor=edge_color,
            linewidth=line_width,
            alpha=alpha,
            label=label,
        )
        for bar, value in zip(bars, values[:, version_idx]):
            label_x = min(value + 1.0, 98.0)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold" if label == "V2" else "normal",
                color="#111111",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        fontsize=13,
        title="Model Version",
        title_fontsize=12,
    )
    ax.set_title("Version 1-4 Performance Comparison (Selected: V2)", fontsize=22, fontweight="bold", pad=42)
    fig.savefig(COMPARISON_IMG_DIR / "version_1_to_4_performance_metrics.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    raw_cost = metrics["cost"].astype(float).to_numpy()
    cost_1k = metrics["cost_per_1k_test_orders"].astype(float).to_numpy()
    for ax, values, title, label_fmt in [
        (axes[0], raw_cost, "Raw Cost", "{:,.0f}"),
        (axes[1], cost_1k, "Cost per 1,000 Test Orders", "{:,.0f}"),
    ]:
        bars = ax.bar(labels, values, color=colors_map, width=0.58)
        ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelsize=13, rotation=0)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.25)
        y_pad = max(values) * 0.03 if max(values) else 1
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                label_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    fig.suptitle("Cost Matrix Comparison", fontsize=21, fontweight="bold")
    fig.savefig(COMPARISON_IMG_DIR / "version_1_to_4_cost_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    features = metrics["feature_count"].astype(float).to_numpy()
    bars = ax.bar(labels, features, color=colors_map, width=0.56)
    ax.set_title("Feature Count by Version", fontsize=20, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, features):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(features) * 0.02,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.savefig(COMPARISON_IMG_DIR / "version_1_to_4_feature_count.png", dpi=180)
    plt.close(fig)


def build_comparison_pdf(metrics: pd.DataFrame) -> None:
    pdf_path = COMPARISON_DIR / "version_1_to_4_detailed_comparison.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    story = [
        p("Version 1-4 Model & Feature Comparison", "title"),
        p(f"อัปเดตล่าสุด: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Project: Return / Refund Risk Prediction", "subtitle"),
        p(
            "เอกสารนี้เปรียบเทียบแนวทางของ Version 1-4 โดยเน้น 3 แกนหลัก: Feature Engineering, Model Performance, "
            "และความเหมาะสมต่อการนำไปต่อยอด production/feature store ในอนาคต.",
        ),
        p("Executive Summary", "h1"),
        bullet(
            [
                "Version 2: v2_xgboost_safe_plus_rolling_HIGH_ACCURACY เป็น candidate ล่าสุด เพราะ Accuracy, Recall, F1 และ AUC สูงสุดในชุดเปรียบเทียบ และยังคุม leakage/post-event fields ออกจาก training.",
                "V1 เป็น baseline สำหรับพิสูจน์ว่าการทำ feature engineering ช่วยได้ แต่ recall ต่ำมาก จึงยังไม่พอสำหรับ return-risk.",
                "V3 ทดลองเพิ่มความซับซ้อนของ model ด้วย stacking แต่ไม่ได้สร้าง signal ใหม่พอที่จะชนะ V2.",
                "V4 ใช้ generated/synthetic + SMOTE + Optuna ทำให้ accuracy สูง แต่ recall และ feature volume ยังไม่สมดุลเท่า V2 ในฐานะ final candidate.",
            ]
        ),
        p("Selected Model Metrics", "h1"),
    ]

    table_df = metrics[
        [
            "display_version",
            "model",
            "dataset",
            "feature_count",
            "order_time_safe",
            "accuracy",
            "recall",
            "precision",
            "f1",
            "auc",
            "cost",
            "cost_per_1k_test_orders",
            "rating",
        ]
    ].copy()
    for col in ["accuracy", "recall", "precision", "f1", "auc"]:
        table_df[col] = table_df[col].map(fmt_pct)
    table_df["cost"] = table_df["cost"].map(fmt_num)
    table_df["cost_per_1k_test_orders"] = table_df["cost_per_1k_test_orders"].map(fmt_num)
    table_df.columns = [
        "Version",
        "Model",
        "Dataset",
        "Features",
        "Order-time Safe",
        "Accuracy",
        "Recall",
        "Precision",
        "F1",
        "AUC",
        "Cost",
        "Cost / 1k",
        "Rating",
    ]
    story.append(
        table_from_df(
            table_df,
            col_widths=[1.2 * cm, 3.2 * cm, 3.8 * cm, 1.4 * cm, 2.1 * cm, 1.8 * cm, 1.7 * cm, 1.7 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.2 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "หมายเหตุ: Cost raw เปรียบเทียบข้าม version ต้องดูร่วมกับ test size เพราะ V2 HIGH_ACCURACY ใช้ test set 10,000 rows; "
            "จึงใส่ Cost / 1k เพื่อให้เห็น scale ที่เทียบกันง่ายขึ้น.",
            "small",
        )
    )
    story.append(PageBreak())

    story += [
        p("Performance Graphs", "h1"),
        Image(str(COMPARISON_IMG_DIR / "version_1_to_4_performance_metrics.png"), width=25 * cm, height=15 * cm),
        Spacer(1, 0.2 * cm),
        Image(str(COMPARISON_IMG_DIR / "version_1_to_4_cost_comparison.png"), width=25 * cm, height=9.2 * cm),
        PageBreak(),
        p("Feature Count & Version Logic", "h1"),
        Image(str(COMPARISON_IMG_DIR / "version_1_to_4_feature_count.png"), width=18 * cm, height=9 * cm),
        p("Why Each Version Is Different", "h1"),
    ]

    diff_rows = pd.DataFrame(
        [
            {
                "Version": "V1",
                "Different Point": "baseline feature set จาก clean_dataset.csv",
                "Reason": "ใช้เป็นจุดเริ่มต้น วัดผลจาก feature พื้นฐานและ encoded columns",
                "Limitation": "recall ต่ำ แปลว่าพลาด return cases เยอะ",
            },
            {
                "Version": "V2",
                "Different Point": "safe rolling history + high-signal insight features",
                "Reason": "ใช้ข้อมูลที่รู้ก่อน/ขณะ order เข้า เช่น customer history, category, payment, discount, rating, province",
                "Limitation": "ชุด high-signal ให้ accuracy สูง ควร validate กับข้อมูลจริงก่อน production เต็มรูปแบบ",
            },
            {
                "Version": "V3",
                "Different Point": "stacking model บน feature base ใกล้ V2",
                "Reason": "ทดสอบว่าเพิ่ม architecture complexity แล้วดีขึ้นหรือไม่",
                "Limitation": "complexity สูงขึ้น แต่ performance ไม่ชนะ V2",
            },
            {
                "Version": "V4",
                "Different Point": "generated dataset + SMOTE + Optuna",
                "Reason": "ทดสอบการเพิ่มข้อมูลและ imbalance handling",
                "Limitation": "feature เยอะและ synthetic distribution อาจไม่ตรง production จริงเท่า V2",
            },
        ]
    )
    story.append(table_from_df(diff_rows, col_widths=[1.4 * cm, 6.5 * cm, 8.8 * cm, 8.8 * cm]))

    story.append(p("Why Select Version 2", "h1"))
    story.append(
        bullet(
            [
                "V2 ชนะด้าน Accuracy 88.88%, F1 79.99%, AUC 94.66% และ Recall 76.03% ใน comparison ล่าสุด.",
                "feature หลักสอดคล้องกับ business insight: customer return history, rolling windows, category/channel/payment/discount/rating/province.",
                "ตัด leakage fields เช่น return_date, refund_amount, return_reason, delivery_days, delay_days, risk_score, shap_values ออกจาก input model.",
                "แนวคิดพร้อมต่อยอด real-time inference: order ใหม่คำนวณเฉพาะประวัติ customer_id คนนั้นแบบ point-in-time ไม่ต้อง scan ทุก record.",
                "feature count 71 ตัวไม่ใหญ่เกินไปเมื่อเทียบกับ V4 ที่มากกว่าและมี synthetic dependency สูงกว่า.",
            ]
        )
    )
    story.append(p("Recommended Next Validation Before Real Production", "h1"))
    story.append(
        bullet(
            [
                "นำข้อมูลจริงใหม่เข้ามา backtest แบบ time-based split เพื่อดูว่า 88.88% ยังอยู่ในช่วงที่รับได้หรือไม่.",
                "ติดตาม drift ของ return rate ราย category/payment/province ทุกเดือน.",
                "เตรียม feature store สำหรับ rolling history เช่น 30/60/90/180/365 days เพื่อให้ online inference เร็ว.",
                "ถ้าจะใช้ production จริง ให้เก็บ V2 realistic baseline ไว้เทียบกับ V2 HIGH_ACCURACY เพื่อแยกผลจาก synthetic/high-signal signal.",
            ]
        )
    )
    doc.build(story)


def feature_config() -> dict[str, dict]:
    return {
        "1": {
            "title": "Version 1 - Baseline XGBoost",
            "version_label": "V1",
            "version_dir": DOCS / "version 1",
            "source_data": DATA_PROCESSED / "clean_dataset.csv",
            "engineered_data": DOCS / "version 1" / "data" / "features" / "df_engineered.csv",
            "used_features": DOCS / "version 1" / "feature_documentation" / "used_features.csv",
            "model": "XGBoost baseline",
            "code_paths": [
                DOCS / "version 1" / "feature_engineering.py",
                DOCS / "version 1" / "model_training.py",
                DOCS / "version 1" / "model_evaluation.py",
            ],
            "process": [
                "อ่าน clean_dataset.csv เป็น source หลัก",
                "clean missing/outlier/duplicate จาก dataset ที่เตรียมไว้",
                "สร้าง feature พื้นฐาน เช่น order_hour, price/log price, customer/product/category fields และ encoded categorical columns",
                "drop target/post-event/identifier columns ก่อน train",
                "train XGBoost เป็น baseline เพื่อวัดว่าฟีเจอร์ชุดแรกให้ performance เท่าไร",
            ],
            "decision": "ใช้เป็น baseline เท่านั้น เพราะ recall ต่ำและพลาด return cases มาก",
        },
        "2": {
            "title": "Version 2 - XGBoost Safe Plus Rolling HIGH_ACCURACY",
            "version_label": "V2",
            "version_dir": DOCS / "version 2",
            "source_data": DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "data" / "clean_dataset_v2_high_signal.csv",
            "engineered_data": DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "data" / "df_engineered_v2_HIGH_ACCURACY.csv",
            "used_features": DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "data" / "used_features_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
            "dropped_features": DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "reports" / "dropped_or_not_used_features_v2_HIGH_ACCURACY.csv",
            "model": "XGBoost safe rolling",
            "code_paths": [
                DOCS
                / "version 2"
                / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY"
                / "scripts"
                / "feature_engineered_v2_HIGH_ACCURACY.py",
                DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "data" / "train_test_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl",
                DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "models" / "best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl",
            ],
            "process": [
                "ทำ EDA จาก clean_dataset_v2/high-signal เพื่อดู pattern: history, category, channel, payment, discount, rating, province",
                "สร้าง customer historical features แบบ point-in-time โดยไม่ใช้ order ปัจจุบันเป็นประวัติของตัวเอง",
                "เพิ่ม rolling windows 30/60/90/180/365 days เช่น cust_return_rate_30d, cust_order_count_90d, cust_spend_sum_180d",
                "เพิ่ม business interaction features เช่น category x payment, channel x category, discount band, rating band, province risk",
                "ตัด leakage/post-event fields และ identifier fields ก่อน train",
                "train XGBoost พร้อม threshold tuning เพื่อ balance Accuracy, Recall, F1, AUC และ Cost Matrix",
            ],
            "decision": "เลือกเป็น candidate หลัก เพราะ performance สูงสุดและ feature logic ตรงกับโจทย์ order-time prediction",
        },
        "3": {
            "title": "Version 3 - Stacking Model from V2 Features",
            "version_label": "V3",
            "version_dir": DOCS / "version 3",
            "source_data": DOCS / "version 2" / "data" / "features" / "df_engineered.csv",
            "engineered_data": DOCS / "version 3" / "stacking_model_v3" / "data" / "df_featured.csv",
            "used_features": DOCS / "version 3" / "stacking_model_v3" / "data" / "v3_used_features.csv",
            "model": "Stacking XGB + LGBM + CatBoost",
            "code_paths": [
                DOCS / "version 3" / "stacking_model_v3" / "scripts" / "model_training_v3_stacking.py",
                DOCS / "version 3" / "stacking_model_v3" / "scripts" / "model_evaluation_v3.py",
            ],
            "process": [
                "นำ feature base จาก V2 มาใช้ต่อเพื่อทดสอบผลของ model architecture",
                "สร้าง stacking ensemble จาก XGBoost, LightGBM, CatBoost และ meta learner",
                "ใช้ train/test split จาก V2 เพื่อให้เทียบผลจาก model ได้ยุติธรรม",
                "ประเมิน threshold/cost/recall trade-off",
            ],
            "decision": "ไม่เลือกเป็น final เพราะซับซ้อนกว่า แต่ performance ไม่ชนะ V2",
        },
        "4": {
            "title": "Version 4 - Generated Data + SMOTE + Optuna",
            "version_label": "V4",
            "version_dir": DOCS / "version 4",
            "source_data": DOCS / "version 4" / "data" / "processed" / "clean_dataset_v4_generated.csv",
            "engineered_data": DOCS / "version 4" / "data" / "features" / "df_engineered_v4_generated.csv",
            "used_features": DOCS / "version 4" / "reports" / "model_evaluation" / "v4_used_features.csv",
            "model": "XGBoost + SMOTE + Optuna",
            "code_paths": [
                DOCS / "version 4" / "scripts" / "clean_dataset_v4.py",
                DOCS / "version 4" / "scripts" / "run_v4_generated_end_to_end_pipeline.py",
            ],
            "process": [
                "generate synthetic order/return data เพื่อทดลองเพิ่มปริมาณข้อมูล",
                "clean missing/outlier/duplicate และสร้าง clean_dataset_v4_generated.csv",
                "ทำ EDA category/channel/payment/correlation",
                "สร้าง feature จำนวนมาก รวม interaction/encoded features",
                "ใช้ SMOTE สำหรับ imbalance และ Optuna สำหรับ tuning",
                "ประเมิน XGBoost/LightGBM/RandomForest/Logistic และเลือก XGBoost_SMOTE_Optuna",
            ],
            "decision": "ใช้เป็น benchmark ของ generated data แต่ยังไม่เลือก final เพราะ V2 ให้ balance ดีกว่า",
        },
    }


def classify_feature(feature: str, used: set[str], clean_cols: set[str], engineered_cols: set[str]) -> tuple[str, str]:
    f = feature.lower()
    if feature in used:
        return "USED", "model input feature"
    leakage_tokens = [
        "return_date",
        "return_reason",
        "return_status",
        "return_id",
        "refund",
        "risk_score",
        "risk_tier",
        "shap",
        "score_id",
        "scored_at",
        "delivery_date",
        "delivery_days",
        "delay_days",
        "item_condition",
        "is_returned",
    ]
    identity_tokens = ["_id", "customer_name", "customer_phone", "product_name", "supplier", "courier_name", "promo_name", "order_id"]
    if any(t in f for t in leakage_tokens):
        return "DROPPED", "post-event/target/leakage field"
    if any(t in f for t in identity_tokens):
        return "DROPPED", "identifier or high-cardinality identity field"
    if feature in clean_cols and feature not in engineered_cols:
        return "DROPPED", "raw field replaced by engineered/encoded feature or not required"
    if feature in engineered_cols and feature not in clean_cols:
        return "NOT USED", "engineered candidate not selected for this model"
    return "NOT USED", "not selected in final feature set"


def make_feature_audit(version: str, cfg: dict) -> pd.DataFrame:
    clean_cols = set(read_columns(cfg["source_data"]))
    engineered_cols = set(read_columns(cfg["engineered_data"]))
    used = set(read_feature_list(cfg.get("used_features")))

    if "dropped_features" in cfg and Path(cfg["dropped_features"]).exists():
        drop_df = pd.read_csv(cfg["dropped_features"])
        if "feature" in drop_df.columns:
            base_features = set(drop_df["feature"].dropna().astype(str).tolist()) | clean_cols | engineered_cols | used
        else:
            base_features = clean_cols | engineered_cols | used
    else:
        base_features = clean_cols | engineered_cols | used

    rows = []
    for feature in sorted(base_features):
        status, reason = classify_feature(feature, used, clean_cols, engineered_cols)
        rows.append(
            {
                "version": f"V{version}",
                "feature": feature,
                "in_clean_source": feature in clean_cols,
                "in_df_engineered": feature in engineered_cols,
                "used_in_model": feature in used,
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def source_summary(cfg: dict) -> pd.DataFrame:
    rows = []
    for label, path in [
        ("source_data", cfg["source_data"]),
        ("df_engineered", cfg["engineered_data"]),
        ("used_feature_list", cfg.get("used_features")),
    ]:
        if path and Path(path).exists():
            try:
                df_head = pd.read_csv(path, nrows=5)
                rows.append(
                    {
                        "artifact": label,
                        "path": str(Path(path).relative_to(ROOT)),
                        "columns": len(df_head.columns),
                        "sample_rows_read": len(df_head),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "artifact": label,
                        "path": str(Path(path).relative_to(ROOT)),
                        "columns": "-",
                        "sample_rows_read": "-",
                    }
                )
        else:
            rows.append({"artifact": label, "path": "missing", "columns": "-", "sample_rows_read": "-"})
    return pd.DataFrame(rows)


def build_feature_docs(metrics: pd.DataFrame) -> None:
    configs = feature_config()
    for version, cfg in configs.items():
        out_dir = cfg["version_dir"] / "feature_documentation"
        out_dir.mkdir(parents=True, exist_ok=True)

        audit = make_feature_audit(version, cfg)
        audit_csv = out_dir / f"v{version}_feature_used_unused_audit.csv"
        audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")

        used_df = audit[audit["used_in_model"]].copy()
        dropped_df = audit[~audit["used_in_model"]].copy()
        used_df.to_csv(out_dir / "used_features_current.csv", index=False, encoding="utf-8-sig")
        dropped_df.to_csv(out_dir / "dropped_or_not_used_features_current.csv", index=False, encoding="utf-8-sig")
        source_summary(cfg).to_csv(out_dir / "source_data_summary_current.csv", index=False, encoding="utf-8-sig")

        build_feature_audit_pdf(version, cfg, audit, metrics, out_dir)
        build_feature_process_pdf(version, cfg, audit, metrics, out_dir)


def feature_group_summary(used_features: list[str]) -> pd.DataFrame:
    def group_name(feature: str) -> str:
        f = feature.lower()
        if "return_rate" in f or "return_count" in f or "order_count" in f or "hist_" in f or "cust_" in f:
            return "customer_history"
        if "price" in f or "amount" in f or "discount" in f or "spend" in f:
            return "price_discount"
        if "category" in f or "brand" in f or "rating" in f or "fragile" in f:
            return "product"
        if "payment" in f or "channel" in f:
            return "order_channel_payment"
        if "province" in f or "region" in f:
            return "geography"
        if "day" in f or "month" in f or "hour" in f or "week" in f:
            return "time"
        if "tier" in f or "age" in f or "gender" in f:
            return "customer_profile"
        return "other"

    rows = []
    for feature in used_features:
        rows.append({"group": group_name(feature), "feature": feature})
    if not rows:
        return pd.DataFrame(columns=["group", "feature_count", "example_features"])
    df = pd.DataFrame(rows)
    return (
        df.groupby("group")["feature"]
        .agg(feature_count="count", example_features=lambda s: ", ".join(list(s.head(5))))
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )


def build_feature_audit_pdf(version: str, cfg: dict, audit: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    pdf_path = out_dir / f"v{version}_feature_used_unused_audit.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    metric = metrics[metrics["display_version"].eq(f"V{version}")].iloc[0].to_dict()
    used_features = audit[audit["used_in_model"]]["feature"].tolist()
    dropped = audit[~audit["used_in_model"]]
    leakage = dropped[dropped["reason"].str.contains("leakage|post-event|target", case=False, na=False)]
    identity = dropped[dropped["reason"].str.contains("identifier", case=False, na=False)]

    story = [
        p(f"{cfg['title']} - Feature Used / Dropped Audit", "title"),
        p("เทียบ source clean dataset กับ df_engineered และระบุว่า feature ใดใช้ train model / feature ใดตัดทิ้ง", "subtitle"),
        p("Model Summary", "h1"),
        table_from_df(
            pd.DataFrame(
                [
                    {
                        "Model": cfg["model"],
                        "Dataset": Path(cfg["source_data"]).name,
                        "df_engineered": Path(cfg["engineered_data"]).name,
                        "Used Features": len(used_features),
                        "Accuracy": fmt_pct(metric["accuracy"]),
                        "Recall": fmt_pct(metric["recall"]),
                        "F1": fmt_pct(metric["f1"]),
                        "AUC": fmt_pct(metric["auc"]),
                        "Rating": metric["rating"],
                    }
                ]
            ),
            col_widths=[3.2 * cm, 3.0 * cm, 3.0 * cm, 1.8 * cm, 1.8 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.3 * cm],
        ),
        p("Source Artifacts", "h1"),
        table_from_df(source_summary(cfg), col_widths=[3.2 * cm, 10.0 * cm, 2.0 * cm, 2.5 * cm]),
        p("Used Feature Groups", "h1"),
    ]
    group_df = feature_group_summary(used_features)
    if not group_df.empty:
        story.append(table_from_df(group_df, col_widths=[4.0 * cm, 2.2 * cm, 11.0 * cm]))
    else:
        story.append(p("ไม่พบ used feature list ใน artifact", "body"))

    story += [
        p("Used Features (sample)", "h1"),
        p(", ".join(used_features[:60]) + (" ..." if len(used_features) > 60 else ""), "small"),
        p("Dropped / Not Used Feature Logic", "h1"),
        bullet(
            [
                f"ตัด leakage/post-event/target fields จำนวนประมาณ {len(leakage)} feature เช่น {', '.join(leakage['feature'].head(10).tolist())}",
                f"ตัด identifier/high-cardinality fields จำนวนประมาณ {len(identity)} feature เช่น {', '.join(identity['feature'].head(10).tolist())}",
                "feature ที่เป็น raw date หรือ raw text จะถูกแปลงเป็น numeric/encoded/rolling feature ก่อน หรือไม่ใช้ถ้าไม่ช่วย performance",
                "feature ที่เพิ่มแล้ว performance ไม่ดีขึ้นหรือซ้ำซ้อนกับ feature สำคัญ จะถูกจัดเป็น NOT USED เพื่อลด resource/query cost",
            ]
        ),
        p("Feature Audit Detail (sample rows)", "h1"),
    ]
    sample = pd.concat([audit[audit["used_in_model"]].head(12), dropped.head(18)], ignore_index=True)
    sample_df = sample[["feature", "in_clean_source", "in_df_engineered", "used_in_model", "status", "reason"]].copy()
    sample_df.columns = ["Feature", "In Clean", "In Engineered", "Used", "Status", "Reason"]
    story.append(table_from_df(sample_df, col_widths=[4.0 * cm, 1.7 * cm, 2.0 * cm, 1.4 * cm, 1.8 * cm, 6.0 * cm]))
    story.append(p(f"Full CSV: {audit_csv_relative(out_dir, version)}", "small"))
    doc.build(story)


def audit_csv_relative(out_dir: Path, version: str) -> str:
    return str((out_dir / f"v{version}_feature_used_unused_audit.csv").relative_to(ROOT))


def build_feature_process_pdf(version: str, cfg: dict, audit: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    pdf_path = out_dir / f"v{version}_feature_engineering_process.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    metric = metrics[metrics["display_version"].eq(f"V{version}")].iloc[0].to_dict()
    used_features = audit[audit["used_in_model"]]["feature"].tolist()
    code_rows = []
    for path in cfg["code_paths"]:
        code_rows.append(
            {
                "Code / Artifact": path.name,
                "Path": str(path.relative_to(ROOT)) if path.exists() else str(path.relative_to(ROOT)) + " (missing/check)",
            }
        )

    story = [
        p(f"{cfg['title']} - Feature Engineering Process", "title"),
        p("เอกสารอธิบายกระบวนการสร้าง feature แบบอ่านเข้าใจง่าย พร้อม code/artifact ที่เกี่ยวข้อง", "subtitle"),
        p("Process Overview", "h1"),
        bullet(cfg["process"]),
        p("Code / Artifact Reference", "h1"),
        table_from_df(pd.DataFrame(code_rows), col_widths=[5.0 * cm, 12.0 * cm]),
        p("Input / Output", "h1"),
        table_from_df(
            pd.DataFrame(
                [
                    {"Stage": "Input clean data", "Artifact": str(Path(cfg["source_data"]).relative_to(ROOT))},
                    {"Stage": "Engineered dataset", "Artifact": str(Path(cfg["engineered_data"]).relative_to(ROOT))},
                    {"Stage": "Used feature list", "Artifact": str(Path(cfg["used_features"]).relative_to(ROOT)) if cfg.get("used_features") else "-"},
                ]
            ),
            col_widths=[4.0 * cm, 13.0 * cm],
        ),
        p("Feature Engineering Logic", "h1"),
    ]
    if version == "2":
        story.append(
            bullet(
                [
                    "Rolling history คำนวณย้อนหลังแบบ point-in-time: ใช้เฉพาะ order_date ก่อน order ปัจจุบันเท่านั้น",
                    "ตัวอย่าง: ลูกค้ามี 2 order ก่อนหน้า, คืน 1 order, ไม่คืน 1 order => return_rate = 1 / 2 = 0.5 = 50%",
                    "lookback windows: 30, 60, 90, 180, 365 days เพื่อให้ model เรียนรู้ว่าควรดูประวัติสั้น/กลาง/ยาวแค่ไหน",
                    "ถ้าลูกค้าใหม่ไม่มีประวัติ จะ fill history count/rate/spend เป็น 0 เพื่อให้ predict ได้",
                    "ตัด delivery_days และ delay_days เพราะเป็นข้อมูลหลังส่งสินค้า ไม่รู้ตอน order เพิ่งเข้า",
                ]
            )
        )
    else:
        story.append(
            bullet(
                [
                    "เริ่มจาก clean dataset ของ version นั้น แล้วสร้าง numeric/encoded/interaction features",
                    "ใช้ train/test split เดียวใน version เพื่อเปรียบเทียบ model อย่างยุติธรรม",
                    "drop leakage/identifier fields ก่อน train",
                    "ประเมินด้วย Accuracy, Recall, Precision, F1, AUC และ Cost Matrix",
                ]
            )
        )

    story += [
        p("Performance Result", "h1"),
        table_from_df(
            pd.DataFrame(
                [
                    {
                        "Accuracy": fmt_pct(metric["accuracy"]),
                        "Recall": fmt_pct(metric["recall"]),
                        "Precision": fmt_pct(metric["precision"]),
                        "F1": fmt_pct(metric["f1"]),
                        "AUC": fmt_pct(metric["auc"]),
                        "Cost": fmt_num(metric["cost"]),
                        "Rating": metric["rating"],
                        "Used Features": len(used_features),
                    }
                ]
            ),
            col_widths=[2.1 * cm, 2.0 * cm, 2.0 * cm, 1.7 * cm, 1.7 * cm, 2.1 * cm, 1.4 * cm, 2.0 * cm],
        ),
        p("Decision", "h1"),
        p(cfg["decision"]),
        p("Important Notes", "h1"),
        bullet(
            [
                "เอกสารนี้ไม่ทับ model artifact เดิม แต่รีเจน documentation ให้ตรงกับสถานะล่าสุดของโปรเจ็กต์",
                "ถ้า metrics/model เปลี่ยน ให้รัน scripts/generate_current_project_docs_v1_to_v4.py ใหม่เพื่อ sync เอกสาร",
                "สำหรับ production จริงควร validate ด้วยข้อมูลจริงแบบ time-based holdout ก่อน deploy",
            ]
        ),
    ]

    process_md = out_dir / f"v{version}_feature_engineering_process.md"
    process_md.write_text(
        "\n".join(
            [
                f"# {cfg['title']} - Feature Engineering Process",
                "",
                "## Process",
                *[f"- {item}" for item in cfg["process"]],
                "",
                "## Code / Artifact Reference",
                *[f"- `{row['Path']}`" for row in code_rows],
                "",
                "## Decision",
                cfg["decision"],
                "",
                f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            ]
        ),
        encoding="utf-8",
    )
    doc.build(story)


def write_readme(metrics: pd.DataFrame) -> None:
    lines = [
        "# Version 1-4 Comparison",
        "",
        "โฟลเดอร์นี้รวมเอกสารเปรียบเทียบ model/feature engineering ของ Version 1-4 จากสถานะล่าสุดของโปรเจ็กต์",
        "",
        "## Main Files",
        "- `version_1_to_4_detailed_comparison.pdf` - เอกสาร PDF เปรียบเทียบละเอียด",
        "- `version_1_to_4_selected_model_comparison.csv` - metric table สำหรับ V1-V4",
        "- `images/version_1_to_4_performance_metrics.png` - กราฟ Accuracy / Recall / F1 / AUC",
        "- `images/version_1_to_4_cost_comparison.png` - กราฟ Cost Matrix",
        "- `images/version_1_to_4_feature_count.png` - กราฟจำนวน feature",
        "",
        "## Current Decision",
        "เลือก `V2 - v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` เป็น candidate หลัก เพราะให้ performance สูงสุดและ feature logic ตรงกับ return-risk prediction แบบ order-time safe มากที่สุด",
        "",
        "## Metrics Snapshot",
    ]
    table = metrics[
        ["display_version", "model", "accuracy", "recall", "precision", "f1", "auc", "cost", "feature_count", "rating"]
    ].copy()
    for col in ["accuracy", "recall", "precision", "f1", "auc"]:
        table[col] = table[col].map(lambda x: f"{x*100:.2f}%")
    headers = list(table.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    (COMPARISON_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics = clean_metric_frame()
    save_comparison_outputs(metrics)
    build_comparison_pdf(metrics)
    build_feature_docs(metrics)
    write_readme(metrics)
    print("Generated current V1-V4 comparison docs")
    print(COMPARISON_DIR / "version_1_to_4_detailed_comparison.pdf")
    for version in ["1", "2", "3", "4"]:
        print(DOCS / f"version {version}" / "feature_documentation" / f"v{version}_feature_used_unused_audit.pdf")
        print(DOCS / f"version {version}" / "feature_documentation" / f"v{version}_feature_engineering_process.pdf")


if __name__ == "__main__":
    main()
