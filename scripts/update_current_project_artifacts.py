from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import math
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
COMPARISON = DOCS / "Comparison Version"
COMPARISON_IMAGES = COMPARISON / "images"
BUSINESS_DIR = ROOT / "reports" / "business_insights"
BUSINESS_ASSOC = ROOT / "reports" / "business_insights_association"
TEST_DIR = DOCS / "test"
SQL_DIR = DOCS / "sql"

PYTHON_BENCHMARK_SCRIPT = ROOT / "scripts" / "run_same_xgboost_feature_version_benchmark.py"


@dataclass
class VersionMetric:
    display_version: str
    version: str
    model: str
    dataset: str
    rows: int | None
    train_rows: int | None
    test_rows: int | None
    feature_count: int | None
    order_time_safe: str
    accuracy: float
    recall: float
    precision: float
    f1: float
    auc: float
    avg_precision: float | None
    cost: float
    threshold: float | None
    tn: int | None
    fp: int | None
    fn: int | None
    tp: int | None
    performance_rating: str
    note: str


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    fnt: ImageFont.ImageFont,
    fill: str = "#111111",
    max_chars: int = 95,
    line_spacing: int = 8,
) -> int:
    x, y = xy
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y += getattr(fnt, "size", 20) + line_spacing
            continue
        for line in wrap(paragraph, max_chars):
            draw.text((x, y), line, font=fnt, fill=fill)
            y += getattr(fnt, "size", 20) + line_spacing
    return y


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def rating_from_metrics(accuracy: float, recall: float, f1: float, cost: float | None = None) -> str:
    if accuracy >= 0.85 and recall >= 0.70 and f1 >= 0.70:
        return "A"
    if accuracy >= 0.80 and auc_like(f1, recall) >= 0.45:
        return "B+"
    if accuracy >= 0.70 and recall >= 0.50 and f1 >= 0.50:
        return "B"
    if accuracy >= 0.65:
        return "C"
    return "D"


def auc_like(*values: float) -> float:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return sum(vals) / len(vals) if vals else 0.0


def read_csv_row(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if len(df) else {}


def df_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
    except OSError:
        return None


def collect_version_metrics() -> list[VersionMetric]:
    v1 = read_csv_row(COMPARISON / "01_Version_1_Baseline_XGBoost" / "v1_metrics_summary.csv")
    v2 = read_csv_row(DOCS / "version 2" / "v2_xgboost_safe_plus_rolling" / "reports" / "v2_xgboost_safe_plus_rolling_metrics.csv")
    v3 = read_csv_row(DOCS / "version 3" / "model_evaluation_v3" / "metrics_summary_v3.csv")
    v4_all = pd.read_csv(DOCS / "version 4" / "reports" / "model_evaluation" / "v4_generated_model_metrics.csv")
    v4 = v4_all.sort_values(["performance_score", "accuracy"], ascending=False).iloc[0].to_dict()
    v3_tn, v3_fp, v3_fn, v3_tp = int(v3["tn"]), int(v3["fp"]), int(v3["fn"]), int(v3["tp"])
    v3_accuracy = (v3_tn + v3_tp) / max(1, v3_tn + v3_fp + v3_fn + v3_tp)
    v3_cost = v3_fp * 50 + v3_fn * 200

    out = [
        VersionMetric(
            display_version="V1",
            version="v1_baseline_xgboost",
            model="XGBoost baseline",
            dataset="clean_dataset.csv",
            rows=int(v1.get("rows", df_rows(ROOT / "data" / "processed" / "clean_dataset.csv") or 0)),
            train_rows=None,
            test_rows=int(v1.get("test_rows", 0)),
            feature_count=int(v1.get("feature_count", 0)),
            order_time_safe="Partial",
            accuracy=float(v1["accuracy"]),
            recall=float(v1["recall"]),
            precision=float(v1["precision"]),
            f1=float(v1["f1"]),
            auc=float(v1["auc"]),
            avg_precision=float(v1.get("avg_precision", np.nan)),
            cost=float(v1["cost"]),
            threshold=float(v1["threshold"]),
            tn=int(v1["tn"]),
            fp=int(v1["fp"]),
            fn=int(v1["fn"]),
            tp=int(v1["tp"]),
            performance_rating=str(v1.get("rating", "D")),
            note="Baseline feature set from clean_dataset.csv; includes customer history but less complete rolling windows.",
        ),
        VersionMetric(
            display_version="V2",
            version="v2_xgboost_safe_plus_rolling",
            model="XGBoost safe plus rolling",
            dataset="clean_dataset.csv",
            rows=df_rows(DOCS / "version 2" / "v2_xgboost_safe_plus_rolling" / "data" / "df_featured.csv"),
            train_rows=None,
            test_rows=(int(v2["tn"]) + int(v2["fp"]) + int(v2["fn"]) + int(v2["tp"])) if v2 else None,
            feature_count=int(v2.get("raw_feature_count", v2.get("encoded_feature_count", 0))),
            order_time_safe="Yes",
            accuracy=float(v2["accuracy"]),
            recall=float(v2["recall"]),
            precision=float(v2["precision"]),
            f1=float(v2["f1"]),
            auc=float(v2["auc"]),
            avg_precision=float(v2.get("avg_precision", np.nan)),
            cost=float(v2["cost"]),
            threshold=float(v2["selected_threshold"]),
            tn=int(v2["tn"]),
            fp=int(v2["fp"]),
            fn=int(v2["fn"]),
            tp=int(v2["tp"]),
            performance_rating=str(v2.get("performance_rating", "B")),
            note="Current V2. Uses order-time-safe features and rolling history 30/60/90/180/365d; does not use HIGH_ACCURACY synthetic signal.",
        ),
        VersionMetric(
            display_version="V3",
            version="v3_stacking_from_v2",
            model="Stacking XGBoost + LightGBM + CatBoost",
            dataset="V2 engineered feature set",
            rows=df_rows(DOCS / "version 3" / "data" / "features" / "df_featured.csv"),
            train_rows=None,
            test_rows=(v3_tn + v3_fp + v3_fn + v3_tp) if v3 else None,
            feature_count=None,
            order_time_safe="Depends on V2 feature input",
            accuracy=v3_accuracy,
            recall=float(v3["recall"]),
            precision=float(v3["precision"]),
            f1=float(v3["f1_score"]),
            auc=float(v3["auc_roc"]),
            avg_precision=float(v3.get("avg_precision", np.nan)),
            cost=float(v3_cost),
            threshold=float(v3["threshold"]),
            tn=v3_tn,
            fp=v3_fp,
            fn=v3_fn,
            tp=v3_tp,
            performance_rating=rating_from_metrics(v3_accuracy, float(v3["recall"]), float(v3["f1_score"])),
            note="Stacking model at threshold 0.5. It improves recall over V2 but adds complexity and lowers precision.",
        ),
        VersionMetric(
            display_version="V4",
            version="v4_generated_xgboost_smote_optuna",
            model=str(v4["model"]),
            dataset="clean_dataset_v4_generated.csv",
            rows=None,
            train_rows=None,
            test_rows=(int(v4["tn"]) + int(v4["fp"]) + int(v4["fn"]) + int(v4["tp"])),
            feature_count=None,
            order_time_safe="Synthetic/generated-data experiment",
            accuracy=float(v4["accuracy"]),
            recall=float(v4["recall"]),
            precision=float(v4["precision"]),
            f1=float(v4["f1"]),
            auc=float(v4["auc"]),
            avg_precision=float(v4.get("avg_precision", np.nan)),
            cost=float(v4["cost_thb"]),
            threshold=float(v4["threshold"]),
            tn=int(v4["tn"]),
            fp=int(v4["fp"]),
            fn=int(v4["fn"]),
            tp=int(v4["tp"]),
            performance_rating=rating_from_metrics(float(v4["accuracy"]), float(v4["recall"]), float(v4["f1"])),
            note="Generated-data experiment with SMOTE/Optuna. Good accuracy/AUC, but lower recall than current V2.",
        ),
    ]
    return out


def write_current_comparison(metrics: list[VersionMetric]) -> pd.DataFrame:
    COMPARISON.mkdir(parents=True, exist_ok=True)
    COMPARISON_IMAGES.mkdir(parents=True, exist_ok=True)
    rows = [m.__dict__ for m in metrics]
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON / "version_1_to_4_selected_model_comparison.csv", index=False, encoding="utf-8-sig")
    (COMPARISON / "00_Overall_Comparison").mkdir(exist_ok=True)
    df.to_csv(COMPARISON / "00_Overall_Comparison" / "version_1_to_4_selected_model_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def draw_metric_chart(df: pd.DataFrame) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("auc", "AUC"),
    ]
    colors = {"V1": "#6D7C85", "V2": "#2E7D32", "V3": "#F9A825", "V4": "#1565C0"}
    width, height = 1900, 1250
    img = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(img)
    title_f = font(50)
    h_f = font(32)
    label_f = font(25)
    small_f = font(21)
    d.text((width // 2, 45), "Version 1-4 Current Performance Metrics", font=title_f, fill="#111111", anchor="ma")
    d.text((width // 2, 105), "V2 is current v2_xgboost_safe_plus_rolling; HIGH_ACCURACY is separated as V5 archive.", font=small_f, fill="#455A64", anchor="ma")

    pad_x, pad_y = 80, 155
    panel_w = (width - pad_x * 3) // 2
    panel_h = (height - pad_y - 80 - 35) // 2
    for idx, (col, label) in enumerate(metrics):
        row, col_idx = divmod(idx, 2)
        x0 = pad_x + col_idx * (panel_w + pad_x)
        y0 = pad_y + row * (panel_h + 35)
        x1, y1 = x0 + panel_w, y0 + panel_h
        d.rectangle((x0, y0, x1, y1), outline="#CFD8DC", width=2)
        d.text(((x0 + x1) // 2, y0 + 18), label, font=h_f, fill="#111111", anchor="ma")
        axis_y = y1 - 55
        axis_x = x0 + 95
        d.line((axis_x, y0 + 75, axis_x, axis_y), fill="#263238", width=2)
        d.line((axis_x, axis_y, x1 - 30, axis_y), fill="#263238", width=2)
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = int(axis_y - tick * (panel_h - 135))
            d.line((axis_x - 6, yy, x1 - 30, yy), fill="#ECEFF1", width=1)
            d.text((axis_x - 12, yy), f"{int(tick*100)}", font=small_f, fill="#546E7A", anchor="rm")
        vals = df[col].astype(float).tolist()
        bar_space = (panel_w - 160) // 4
        for i, (_, r) in enumerate(df.iterrows()):
            v = float(r[col])
            version = r["display_version"]
            bw = min(90, bar_space - 35)
            bx = axis_x + 35 + i * bar_space
            by = int(axis_y - v * (panel_h - 135))
            d.rounded_rectangle((bx, by, bx + bw, axis_y), radius=6, fill=colors.get(version, "#607D8B"))
            d.text((bx + bw // 2, by - 8), pct(v), font=small_f, fill="#111111", anchor="ms")
            d.text((bx + bw // 2, axis_y + 14), version, font=label_f, fill="#111111", anchor="ma")
    out = COMPARISON_IMAGES / "version_1_to_4_performance_metrics.png"
    img.save(out)
    img.save(COMPARISON_IMAGES / "version_1_to_4_current_performance_metrics.png")


def draw_feature_count_chart(df: pd.DataFrame) -> None:
    width, height = 1500, 820
    img = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.text((width // 2, 45), "Feature Count / Feature Set Size by Version", font=font(42), fill="#111111", anchor="ma")
    d.text((width // 2, 95), "Feature count is recorded where available from current artifacts.", font=font(22), fill="#455A64", anchor="ma")
    x0, y0, x1, y1 = 150, 160, 1420, 700
    d.line((x0, y1, x1, y1), fill="#263238", width=2)
    d.line((x0, y0, x0, y1), fill="#263238", width=2)
    counts = [0 if pd.isna(v) else int(v) for v in df["feature_count"]]
    max_count = max(counts + [1])
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0"]
    bar_space = (x1 - x0 - 80) // len(counts)
    for i, (count, version) in enumerate(zip(counts, df["display_version"])):
        bx = x0 + 55 + i * bar_space
        bw = 120
        by = int(y1 - (count / max_count) * (y1 - y0 - 30)) if count else y1 - 8
        d.rounded_rectangle((bx, by, bx + bw, y1), radius=6, fill=colors[i])
        label = str(count) if count else "N/A"
        d.text((bx + bw // 2, by - 10), label, font=font(26), fill="#111111", anchor="ms")
        d.text((bx + bw // 2, y1 + 20), version, font=font(27), fill="#111111", anchor="ma")
    img.save(COMPARISON_IMAGES / "version_1_to_4_feature_count.png")


def make_comparison_pdf(df: pd.DataFrame) -> None:
    pages: list[Image.Image] = []
    page_w, page_h = 1800, 1300
    title_f, h_f, body_f, small_f = font(42), font(30), font(24), font(20)

    page = Image.new("RGB", (page_w, page_h), "#FFFFFF")
    d = ImageDraw.Draw(page)
    d.text((80, 55), "Version 1-4 Comparison: Current Project State", font=title_f, fill="#111111")
    y = draw_wrapped(
        d,
        "เอกสารนี้แก้สถานะล่าสุดให้ Version 2 เป็น v2_xgboost_safe_plus_rolling เท่านั้น ส่วน HIGH_ACCURACY ถูกแยกเป็น Version 5 archive/experiment แล้ว. "
        "ดังนั้นการเปรียบเทียบ V1-V4 ด้านล่างใช้ metric จาก artifact ปัจจุบันที่มีอยู่จริงในโปรเจ็กต์.",
        (80, 125),
        body_f,
        fill="#263238",
        max_chars=105,
    )
    d.text((80, y + 35), "Metric Summary", font=h_f, fill="#0D47A1")
    y += 90
    headers = ["Version", "Model", "Accuracy", "Recall", "F1", "AUC", "Cost", "Rating"]
    widths = [110, 470, 130, 130, 130, 130, 130, 115]
    x = 80
    for h, w in zip(headers, widths):
        d.rectangle((x, y, x + w, y + 44), fill="#ECEFF1", outline="#B0BEC5")
        d.text((x + 8, y + 10), h, font=small_f, fill="#111111")
        x += w
    y += 44
    for _, r in df.iterrows():
        x = 80
        row_vals = [
            r["display_version"],
            r["model"],
            pct(float(r["accuracy"])),
            pct(float(r["recall"])),
            pct(float(r["f1"])),
            pct(float(r["auc"])),
            f"{float(r['cost']):,.0f}",
            r["performance_rating"],
        ]
        for val, w in zip(row_vals, widths):
            d.rectangle((x, y, x + w, y + 58), outline="#CFD8DC")
            d.text((x + 8, y + 14), str(val)[:42], font=small_f, fill="#111111")
            x += w
        y += 58
    y += 35
    draw_wrapped(
        d,
        "เหตุผลที่เลือก V2 ในฐานะ production candidate: V2 ใช้ feature ที่รู้ได้ตอน order เข้า, มี rolling history หลายช่วงเวลา, recall/F1 ดีขึ้นจาก V1 มาก และไม่พึ่ง high-signal synthetic artifact. "
        "V4 มี accuracy/AUC สูงกว่าในบางมุม แต่เป็น generated-data experiment และ recall ต่ำกว่า V2.",
        (80, y),
        body_f,
        fill="#263238",
        max_chars=105,
    )
    pages.append(page)

    for chart in [
        COMPARISON_IMAGES / "version_1_to_4_performance_metrics.png",
        COMPARISON_IMAGES / "version_1_to_4_feature_count.png",
    ]:
        p = Image.new("RGB", (page_w, page_h), "#FFFFFF")
        d = ImageDraw.Draw(p)
        d.text((80, 45), chart.stem.replace("_", " ").title(), font=title_f, fill="#111111")
        if chart.exists():
            im = Image.open(chart).convert("RGB")
            im.thumbnail((1640, 1120), Image.Resampling.LANCZOS)
            p.paste(im, ((page_w - im.width) // 2, 140))
        pages.append(p)

    out = COMPARISON / "version_1_to_4_detailed_comparison.pdf"
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)


def write_comparison_readme(df: pd.DataFrame) -> None:
    text = """# Comparison Version - Current State

โฟลเดอร์นี้เป็นเอกสารเปรียบเทียบ Version 1-4 ตามสถานะล่าสุดของโปรเจ็กต์

## Current Mapping

- V1 = `v1_baseline_xgboost`
- V2 = `v2_xgboost_safe_plus_rolling`
- V3 = `v3_stacking_from_v2`
- V4 = `v4_generated_xgboost_smote_optuna`
- V5 = `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` ถูกแยกออกจาก V2 แล้ว เพราะเป็น high-signal/generated-data experiment

## Important Note

ไฟล์ comparison รุ่นเก่าบางตัวเคยอ้าง V2 เป็น HIGH_ACCURACY ตอนนี้แก้ master CSV/PDF/graph ล่าสุดแล้ว ให้ใช้ไฟล์ root-level ต่อไปนี้เป็นแหล่งอ้างอิงหลัก:

- `version_1_to_4_selected_model_comparison.csv`
- `version_1_to_4_detailed_comparison.pdf`
- `images/version_1_to_4_performance_metrics.png`

## Current Metric Snapshot

"""
    for _, r in df.iterrows():
        text += (
            f"- {r['display_version']} `{r['version']}`: model={r['model']}, "
            f"Accuracy={pct(float(r['accuracy']))}, Recall={pct(float(r['recall']))}, "
            f"F1={pct(float(r['f1']))}, AUC={pct(float(r['auc']))}, Cost={float(r['cost']):,.0f}\n"
        )
    (COMPARISON / "README.md").write_text(text, encoding="utf-8")


def safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sync_comparison_version_folders() -> None:
    current_v2 = COMPARISON / "02_Version_2_XGBoost_Safe_Plus_Rolling_CURRENT"
    current_v2.mkdir(parents=True, exist_ok=True)
    (current_v2 / "README.md").write_text(
        """# Version 2 - Current Production Candidate

Current V2 คือ `v2_xgboost_safe_plus_rolling`

- Dataset: `clean_dataset.csv`
- Model: XGBoost safe plus rolling
- Policy: order-time safe, ตัด `delivery_days` และ `delay_days`
- Rolling history: 30/60/90/180/365 วัน
- ใช้ไฟล์ metric จาก `docs/version 2/v2_xgboost_safe_plus_rolling/reports`

หมายเหตุ: `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` ไม่ใช่ V2 current แล้ว ถูกแยกไปเป็น Version 5 archive/experiment.
""",
        encoding="utf-8",
    )
    v2_source = DOCS / "version 2" / "v2_xgboost_safe_plus_rolling"
    for src, dst in [
        (v2_source / "reports" / "v2_xgboost_safe_plus_rolling_metrics.csv", current_v2 / "csv" / "v2_xgboost_safe_plus_rolling_metrics.csv"),
        (v2_source / "reports" / "v2_xgboost_safe_plus_rolling_confusion_matrix.csv", current_v2 / "csv" / "v2_xgboost_safe_plus_rolling_confusion_matrix.csv"),
        (v2_source / "reports" / "v2_xgboost_safe_plus_rolling_test_predictions.csv", current_v2 / "csv" / "v2_xgboost_safe_plus_rolling_test_predictions.csv"),
        (v2_source / "data" / "v2_xgboost_safe_plus_rolling_used_features.csv", current_v2 / "csv" / "v2_xgboost_safe_plus_rolling_used_features.csv"),
        (v2_source / "docs" / "v2_xgboost_safe_plus_rolling_clean_vs_df_engineered_feature_comparison.csv", current_v2 / "csv" / "v2_clean_vs_engineered_feature_comparison.csv"),
        (v2_source / "docs" / "v2_xgboost_safe_plus_rolling_clean_vs_df_engineered_feature_comparison.pdf", current_v2 / "docs" / "v2_clean_vs_engineered_feature_comparison.pdf"),
        (v2_source / "scripts" / "feature_engineered_v2.py", current_v2 / "code" / "feature_engineered_v2.py"),
    ]:
        safe_copy(src, dst)

    old_v2_high = COMPARISON / "02_Version_2_XGBoost_Safe_Rolling_HIGH_ACCURACY"
    if old_v2_high.exists():
        (old_v2_high / "ARCHIVED_AS_VERSION_5.md").write_text(
            """# Archived Folder

โฟลเดอร์นี้เป็น artifact เก่าที่เคยถูกวางไว้ใต้ Version 2 แต่ตามสถานะล่าสุดไม่ใช่ V2 current แล้ว

- Current V2 = `02_Version_2_XGBoost_Safe_Plus_Rolling_CURRENT`
- HIGH_ACCURACY = `05_Version_5_HIGH_ACCURACY_ARCHIVE`

ให้ใช้ root comparison CSV/PDF ล่าสุดแทน:

- `docs/Comparison Version/version_1_to_4_selected_model_comparison.csv`
- `docs/Comparison Version/version_1_to_4_detailed_comparison.pdf`
""",
            encoding="utf-8",
        )

    v5_archive = COMPARISON / "05_Version_5_HIGH_ACCURACY_ARCHIVE"
    v5_archive.mkdir(parents=True, exist_ok=True)
    (v5_archive / "README.md").write_text(
        """# Version 5 - HIGH_ACCURACY Archive / Experiment

โฟลเดอร์นี้เก็บ `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY`

ใช้สำหรับอธิบาย experiment ที่ได้ Accuracy สูงจาก high-signal/generated dataset ไม่ใช่ V2 current production candidate.
""",
        encoding="utf-8",
    )
    v5_source = DOCS / "version 5" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY"
    for src, dst in [
        (v5_source / "reports" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv", v5_archive / "csv" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv"),
        (v5_source / "reports" / "confusion_matrix_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv", v5_archive / "csv" / "confusion_matrix_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv"),
        (v5_source / "reports" / "feature_importance_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv", v5_archive / "csv" / "feature_importance_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv"),
        (v5_source / "docs" / "model_report_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.md", v5_archive / "docs" / "model_report_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.md"),
        (v5_source / "images" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.png", v5_archive / "images" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.png"),
        (v5_source / "sql" / "query_dataset_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.sql", v5_archive / "sql" / "query_dataset_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.sql"),
    ]:
        safe_copy(src, dst)


def load_business_dataset() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "clean_dataset_v2.csv"
    if not path.exists():
        path = ROOT / "data" / "processed" / "clean_dataset.csv"
    df = pd.read_csv(path, low_memory=False)
    df["is_returned"] = pd.to_numeric(df["is_returned"], errors="coerce").fillna(0).astype(int)
    return df


def cramer_v(df: pd.DataFrame, column: str, target: str = "is_returned") -> float:
    table = pd.crosstab(df[column].fillna("Unknown").astype(str), df[target])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    expected[expected == 0] = np.nan
    chi2 = np.nansum((observed - expected) ** 2 / expected)
    denom = n * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(math.sqrt(chi2 / denom)) if denom else 0.0


def summarize_group(
    df: pd.DataFrame,
    graph_id: str,
    graph_name: str,
    fields: list[str],
    baseline_rate: float,
    min_count: int = 30,
    filter_expr: tuple[str, str] | None = None,
) -> list[dict]:
    work = df.copy()
    if filter_expr:
        col, value = filter_expr
        work = work[work[col].astype(str).eq(value)]
    available = [f for f in fields if f in work.columns]
    if not available or work.empty:
        return []
    grouped = (
        work.groupby(available, dropna=False)["is_returned"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "order_count", "sum": "return_count", "mean": "return_rate"})
    )
    grouped = grouped[grouped["order_count"] >= min_count].copy()
    if grouped.empty:
        return []
    grouped["baseline_return_rate"] = baseline_rate
    grouped["lift_vs_baseline"] = grouped["return_rate"] / baseline_rate if baseline_rate else np.nan
    grouped["direction"] = np.where(
        grouped["lift_vs_baseline"] >= 1.05,
        "positive",
        np.where(grouped["lift_vs_baseline"] <= 0.95, "negative", "neutral"),
    )
    grouped["abs_lift_gap"] = (grouped["lift_vs_baseline"] - 1).abs()
    selected = pd.concat(
        [
            grouped.sort_values("lift_vs_baseline", ascending=False).head(4),
            grouped.sort_values("lift_vs_baseline", ascending=True).head(4),
        ],
        ignore_index=True,
    ).drop_duplicates()
    rows = []
    for _, r in selected.sort_values("abs_lift_gap", ascending=False).iterrows():
        label = " | ".join(f"{f}={r[f]}" for f in available)
        rows.append(
            {
                "graph_id": graph_id,
                "graph_name": graph_name,
                "method": "return_rate_lift",
                "fields": "+".join(available),
                "segment": label,
                "order_count": int(r["order_count"]),
                "return_count": int(r["return_count"]),
                "return_rate": float(r["return_rate"]),
                "baseline_return_rate": float(baseline_rate),
                "lift_vs_baseline": float(r["lift_vs_baseline"]),
                "direction": str(r["direction"]),
                "strength": "high" if abs(float(r["lift_vs_baseline"]) - 1) >= 0.25 else "medium" if abs(float(r["lift_vs_baseline"]) - 1) >= 0.10 else "low",
                "interpretation": f"{label} มี return rate {r['return_rate']:.2%} เทียบ baseline {baseline_rate:.2%} (lift {r['lift_vs_baseline']:.2f}x)",
            }
        )
    return rows


def build_business_association() -> pd.DataFrame:
    BUSINESS_ASSOC.mkdir(parents=True, exist_ok=True)
    df = load_business_dataset()
    baseline = float(df["is_returned"].mean())
    rows: list[dict] = []

    # Graph 01: categorical impact uses Cramer's V and top lift segments.
    for col in ["category", "channel_type", "payment_method", "province", "membership_tier", "promo_type", "courier_type"]:
        if col in df.columns:
            v = cramer_v(df, col)
            rows.append(
                {
                    "graph_id": "01",
                    "graph_name": "Categorical Impact",
                    "method": "cramers_v",
                    "fields": col,
                    "segment": col,
                    "order_count": int(len(df)),
                    "return_count": int(df["is_returned"].sum()),
                    "return_rate": baseline,
                    "baseline_return_rate": baseline,
                    "lift_vs_baseline": 1.0,
                    "direction": "association",
                    "strength": "high" if v >= 0.25 else "medium" if v >= 0.10 else "low",
                    "interpretation": f"{col} มี Cramer's V = {v:.3f} กับ is_returned",
                    "association_value": v,
                }
            )
            rows.extend(summarize_group(df, "01", "Categorical Impact", [col], baseline))

    rows.extend(summarize_group(df, "02", "Category x Payment Return Rate", ["category", "payment_method"], baseline))
    rows.extend(summarize_group(df, "03", "Province x Gender Return Rate", ["province", "gender"], baseline))
    rows.extend(summarize_group(df, "04", "Province x Payment Return Rate", ["province", "payment_method"], baseline))

    corr_path = BUSINESS_DIR / "05_feature_importance_correlation_direction.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        for _, r in corr_df.sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False).head(12).iterrows():
            rows.append(
                {
                    "graph_id": "05",
                    "graph_name": "Feature Importance Correlation Direction",
                    "method": "pearson_correlation",
                    "fields": r["feature"],
                    "segment": r["feature"],
                    "order_count": int(r.get("non_null_rows", len(df))),
                    "return_count": int(df["is_returned"].sum()),
                    "return_rate": baseline,
                    "baseline_return_rate": baseline,
                    "lift_vs_baseline": np.nan,
                    "direction": r["direction"],
                    "strength": "high" if abs(float(r["pearson_corr"])) >= 0.20 else "medium" if abs(float(r["pearson_corr"])) >= 0.08 else "low",
                    "interpretation": r["interpretation"],
                    "association_value": float(r["pearson_corr"]),
                }
            )

    for graph_id, category in [
        ("06", "Cosmetics"),
        ("07", "Fashion"),
        ("08", "Electronics"),
        ("09", "Home_Appliance"),
        ("10", "Supplement"),
    ]:
        category_baseline = baseline
        if "category" in df.columns and not df[df["category"].astype(str).eq(category)].empty:
            category_baseline = float(df[df["category"].astype(str).eq(category)]["is_returned"].mean())
        rows.extend(
            summarize_group(
                df,
                graph_id,
                f"{category} Return Rate by Province",
                ["province"],
                category_baseline,
                min_count=20,
                filter_expr=("category", category),
            )
        )

    assoc = pd.DataFrame(rows)
    if "association_value" not in assoc.columns:
        assoc["association_value"] = np.nan
    assoc.to_csv(BUSINESS_ASSOC / "business_insight_01_to_10_association_summary.csv", index=False, encoding="utf-8-sig")
    return assoc


def draw_association_graphs(assoc: pd.DataFrame) -> None:
    for graph_id, group in assoc.groupby("graph_id"):
        graph_name = str(group["graph_name"].iloc[0])
        plot = group[group["method"].eq("return_rate_lift")].copy()
        if plot.empty:
            plot = group.copy()
            value_col = "association_value"
            axis_label = "Association / correlation value"
        else:
            plot = plot.sort_values("lift_vs_baseline", ascending=True).tail(10)
            value_col = "lift_vs_baseline"
            axis_label = "Lift vs baseline return rate"
        plot = plot.dropna(subset=[value_col]).tail(10)
        width, height = 1900, 950
        img = Image.new("RGB", (width, height), "#FFFFFF")
        d = ImageDraw.Draw(img)
        d.text((width // 2, 38), f"Graph {graph_id}: {graph_name}", font=font(40), fill="#111111", anchor="ma")
        d.text((width // 2, 88), axis_label, font=font(22), fill="#455A64", anchor="ma")
        if plot.empty:
            d.text((80, 200), "No association rows available for this graph.", font=font(28), fill="#111111")
            img.save(BUSINESS_ASSOC / f"{graph_id}_association_direction.png")
            continue
        x0, y0, x1, y1 = 780, 155, 1800, 850
        zero_or_one = 1.0 if value_col == "lift_vs_baseline" else 0.0
        vals = plot[value_col].astype(float).tolist()
        vmin = min(vals + [zero_or_one])
        vmax = max(vals + [zero_or_one])
        if math.isclose(vmin, vmax):
            vmin -= 0.1
            vmax += 0.1
        pad = (vmax - vmin) * 0.12
        vmin -= pad
        vmax += pad

        def xpx(v: float) -> int:
            return int(x0 + (v - vmin) / (vmax - vmin) * (x1 - x0))

        ref_x = xpx(zero_or_one)
        d.line((ref_x, y0, ref_x, y1), fill="#263238", width=3)
        n = len(plot)
        row_h = (y1 - y0) / max(n, 1)
        for i, (_, r) in enumerate(plot.iterrows()):
            y = int(y0 + i * row_h + row_h / 2)
            value = float(r[value_col])
            direction = str(r.get("direction", "neutral"))
            color = "#2E7D32" if direction == "positive" else "#C62828" if direction == "negative" else "#78909C"
            label = str(r["segment"])
            if len(label) > 70:
                label = label[:67] + "..."
            d.text((x0 - 16, y), label, font=font(20), fill="#111111", anchor="rm")
            bx0, bx1 = sorted((ref_x, xpx(value)))
            d.rounded_rectangle((bx0, y - 17, bx1, y + 17), radius=5, fill=color)
            value_label = f"{value:.2f}x" if value_col == "lift_vs_baseline" else f"{value:+.3f}"
            d.text((xpx(value) + (10 if value >= zero_or_one else -10), y), value_label, font=font(20), fill="#111111", anchor="lm" if value >= zero_or_one else "rm")
        img.save(BUSINESS_ASSOC / f"{graph_id}_association_direction.png")


def make_business_association_pdf(assoc: pd.DataFrame) -> None:
    pages: list[Image.Image] = []
    page_w, page_h = 1800, 1300
    cover = Image.new("RGB", (page_w, page_h), "#FFFFFF")
    d = ImageDraw.Draw(cover)
    d.text((80, 55), "Business Insight 1-10: Association / Correlation Supplement", font=font(42), fill="#111111")
    draw_wrapped(
        d,
        "เอกสารเสริมนี้ตอบคำถามว่าแต่ละกราฟสัมพันธ์กับการคืนสินค้าอย่างไร ไม่ใช่ดูเพียงว่า return rate สูงหรือต่ำ. "
        "กราฟ categorical ใช้ return-rate lift และ Cramer's V ส่วนกราฟ feature importance ใช้ Pearson correlation direction.",
        (80, 130),
        font(25),
        fill="#263238",
        max_chars=105,
    )
    y = 270
    for graph_id in sorted(assoc["graph_id"].unique()):
        top = assoc[assoc["graph_id"].eq(graph_id)].copy()
        name = str(top["graph_name"].iloc[0])
        d.text((95, y), f"Graph {graph_id}: {name}", font=font(24), fill="#0D47A1")
        y += 38
        if y > 1180:
            pages.append(cover)
            cover = Image.new("RGB", (page_w, page_h), "#FFFFFF")
            d = ImageDraw.Draw(cover)
            y = 70
    pages.append(cover)

    for graph_id in sorted(assoc["graph_id"].unique()):
        p = Image.new("RGB", (page_w, page_h), "#FFFFFF")
        d = ImageDraw.Draw(p)
        graph = BUSINESS_ASSOC / f"{graph_id}_association_direction.png"
        d.text((80, 45), f"Graph {graph_id} Association Direction", font=font(38), fill="#111111")
        if graph.exists():
            im = Image.open(graph).convert("RGB")
            im.thumbnail((1660, 1040), Image.Resampling.LANCZOS)
            p.paste(im, ((page_w - im.width) // 2, 130))
        detail = assoc[assoc["graph_id"].eq(graph_id)].head(3)
        y = 1160
        for _, r in detail.iterrows():
            y = draw_wrapped(d, f"- {r['interpretation']}", (80, y), font(18), fill="#263238", max_chars=130, line_spacing=4)
        pages.append(p)

    out = BUSINESS_ASSOC / "business_insights_01_to_10_association_supplement.pdf"
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)

    source_pdf = ROOT / "notebooks" / "eda" / "business_insights_report.pdf"
    merged_pdf = ROOT / "notebooks" / "eda" / "business_insights_report_with_all_association.pdf"
    if source_pdf.exists():
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        for pdf in [source_pdf, out]:
            reader = PdfReader(str(pdf))
            for page in reader.pages:
                writer.add_page(page)
        with merged_pdf.open("wb") as f:
            writer.write(f)


def write_sql_files() -> None:
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    sql = """-- Customer sample + Return / Not Returned split for model/EDA validation
-- Change the FROM public."order_history_complete_v2_NEW" line if the production table name changes.
-- Uses only one query and a deterministic MOD bucket so the sample is repeatable.

WITH params AS (
    SELECT
        50::int AS return_customer_quota,
        50::int AS not_return_customer_quota
),
base AS (
    SELECT
        order_id,
        customer_id,
        order_date,
        category,
        province,
        payment_method,
        total_amount,
        is_returned,
        ABS(MOD(HASHTEXT(customer_id), 100)) AS customer_mod_100
    FROM public."order_history_complete_v2_NEW"
),
customer_summary AS (
    SELECT
        customer_id,
        COUNT(*) AS order_count,
        SUM(is_returned::int) AS return_count,
        AVG(is_returned::int)::numeric(10,4) AS customer_return_rate,
        MIN(customer_mod_100) AS customer_mod_100
    FROM base
    GROUP BY customer_id
),
ranked_customers AS (
    SELECT
        *,
        CASE WHEN return_count > 0 THEN 'Return' ELSE 'Not Returned' END AS customer_group,
        ROW_NUMBER() OVER (
            PARTITION BY CASE WHEN return_count > 0 THEN 'Return' ELSE 'Not Returned' END
            ORDER BY customer_mod_100, customer_id
        ) AS group_rank
    FROM customer_summary
    WHERE customer_mod_100 BETWEEN 0 AND 99
),
selected_customers AS (
    SELECT *
    FROM ranked_customers, params
    WHERE
        (customer_group = 'Return' AND group_rank <= return_customer_quota)
        OR
        (customer_group = 'Not Returned' AND group_rank <= not_return_customer_quota)
)
SELECT
    b.*,
    s.customer_group,
    s.order_count AS customer_order_count,
    s.return_count AS customer_return_count,
    s.customer_return_rate
FROM base b
JOIN selected_customers s USING (customer_id)
ORDER BY s.customer_group, s.customer_id, b.order_date;
"""
    (SQL_DIR / "customer_sample_100_return_not_returned_split.sql").write_text(sql, encoding="utf-8")

    gen_sql = """-- Synthetic-style sampling helper using MOD % 100 buckets.
-- Change selected_bucket_start/end to control how much data enters an experiment.

WITH sampled_orders AS (
    SELECT
        *,
        ABS(MOD(HASHTEXT(order_id), 100)) AS order_mod_100,
        ABS(MOD(HASHTEXT(customer_id), 100)) AS customer_mod_100
    FROM public."order_history_complete_v2_NEW"
),
bucketed AS (
    SELECT
        *,
        CASE
            WHEN order_mod_100 < 70 THEN 'train'
            WHEN order_mod_100 < 85 THEN 'validation'
            ELSE 'test'
        END AS dataset_split
    FROM sampled_orders
)
SELECT *
FROM bucketed
WHERE customer_mod_100 BETWEEN 0 AND 99
ORDER BY dataset_split, customer_id, order_date;
"""
    (SQL_DIR / "mod_100_train_validation_test_split.sql").write_text(gen_sql, encoding="utf-8")


def write_test_readiness_report(df: pd.DataFrame) -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset_size_plan": "current_artifact",
                "version": r["display_version"],
                "model": r["model"],
                "dataset": r["dataset"],
                "accuracy": r["accuracy"],
                "recall": r["recall"],
                "f1": r["f1"],
                "auc": r["auc"],
                "cost": r["cost"],
                "status": "measured_from_existing_artifact",
            }
        )
    for size in [5000, 50000]:
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset_size_plan": f"same_xgboost_{size}_rows",
                    "version": r["display_version"],
                    "model": "XGBoost same-model benchmark",
                    "dataset": r["dataset"],
                    "accuracy": "",
                    "recall": "",
                    "f1": "",
                    "auc": "",
                    "cost": "",
                    "status": "pending_run_requires_sklearn_xgboost",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TEST_DIR / "version_1_to_4_test_readiness_5k_50k.csv", index=False, encoding="utf-8-sig")
    md = """# Test Folder - Version 1-4 Readiness

ไฟล์นี้จัดสถานะการทดสอบตาม requirement ล่าสุด:

- `current_artifact` = metric ที่มีอยู่จริงในโปรเจ็กต์แล้ว
- `same_xgboost_5000_rows` และ `same_xgboost_50000_rows` = ช่องสำหรับ benchmark แบบใช้ XGBoost รุ่นเดียวกันทุก version

ตอนนี้ runtime ที่ใช้ใน Codex ไม่มี `sklearn` และ `xgboost` จึงยังไม่สามารถ train benchmark ใหม่ได้จาก environment นี้โดยตรง แต่เพิ่มสคริปต์ `scripts/run_same_xgboost_feature_version_benchmark.py` ให้พร้อมรันแล้ว เมื่อ environment ติดตั้ง dependency ตาม `requirements.txt`.
"""
    (TEST_DIR / "README.md").write_text(md, encoding="utf-8")


def main() -> None:
    metrics = collect_version_metrics()
    df = write_current_comparison(metrics)
    draw_metric_chart(df)
    draw_feature_count_chart(df)
    make_comparison_pdf(df)
    write_comparison_readme(df)
    sync_comparison_version_folders()

    assoc = build_business_association()
    draw_association_graphs(assoc)
    make_business_association_pdf(assoc)

    write_sql_files()
    write_test_readiness_report(df)
    print("Updated current comparison, business associations, SQL, and test readiness outputs.")


if __name__ == "__main__":
    main()
