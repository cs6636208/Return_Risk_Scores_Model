from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".ml_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TEST_DATA = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "real_dataset_s1.csv"
MODEL_ROOT = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1"
OUT_ROOT = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "S3"
TARGET = "is_returned"
MODEL_PREFIX = "lgbm_s1"


def load_seq_module() -> Any:
    module_path = ROOT / "scripts" / "run_dataset_5000_sequential_version_pipeline.py"
    spec = importlib.util.spec_from_file_location("seq_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sequential pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["seq_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def apply_version_feature_engineering(seq: Any, version: int, df: pd.DataFrame) -> pd.DataFrame:
    if version == 1:
        return seq.add_v1_features(df)
    if version == 2:
        return seq.add_customer_history(df)
    if version == 3:
        return seq.add_v3_features(df)
    if version == 4:
        return seq.add_v4_features(df)
    if version == 5:
        return df.copy()
    raise ValueError(version)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    regular = Path("C:/Windows/Fonts/tahoma.ttf")
    bold_path = Path("C:/Windows/Fonts/tahomabd.ttf")
    path = bold_path if bold and bold_path.exists() else regular
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        (OUT_ROOT / f"V{version}" / "reports").mkdir(parents=True, exist_ok=True)


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            (df[col].astype(str).str.strip() == "").sum()
            for col in df.select_dtypes(include=["object", "string"]).columns
        )
    )


def test_data_validation(df: pd.DataFrame) -> dict[str, object]:
    return {
        "file": str(TEST_DATA.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "blank_text_cells": blank_text_count(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()) if "order_id" in df.columns else None,
        "distinct_order_id": int(df["order_id"].nunique()) if "order_id" in df.columns else None,
        "distinct_customer_id": int(df["customer_id"].nunique()) if "customer_id" in df.columns else None,
        "first_order_id": str(df["order_id"].iloc[0]) if "order_id" in df.columns else "",
        "last_order_id": str(df["order_id"].iloc[-1]) if "order_id" in df.columns else "",
        "min_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").min()) if "order_date" in df.columns else "",
        "max_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").max()) if "order_date" in df.columns else "",
        "returned_count": int((df[TARGET] == 1).sum()),
        "not_returned_count": int((df[TARGET] == 0).sum()),
        "return_rate": float(df[TARGET].mean()),
        "note": "External full-dataset test for LightGBM SETC S1 models. Do not split this file again.",
    }


def load_used_features(version: int) -> list[str]:
    path = MODEL_ROOT / f"V{version}" / "features" / f"used_features_{MODEL_PREFIX}_v{version}.csv"
    return pd.read_csv(path)["feature"].astype(str).tolist()


def load_model_metrics(version: int) -> dict[str, float]:
    path = MODEL_ROOT / f"V{version}" / "reports" / f"metrics_{MODEL_PREFIX}_v{version}.csv"
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "threshold": float(row["threshold"]),
        "holdout_accuracy": float(row["accuracy"]),
        "holdout_recall": float(row["recall"]),
        "holdout_precision": float(row["precision"]),
        "holdout_f1": float(row["f1"]),
        "holdout_auc": float(row["auc"]),
        "holdout_cost": float(row["cost"]),
    }


def evaluate_version(version: int, df_featured: pd.DataFrame) -> dict[str, object]:
    report_dir = OUT_ROOT / f"V{version}" / "reports"
    features = load_used_features(version)
    missing_features = [feature for feature in features if feature not in df_featured.columns]
    if missing_features:
        raise KeyError(f"V{version} missing test features: {missing_features}")

    model_path = MODEL_ROOT / f"V{version}" / "models" / f"model_{MODEL_PREFIX}_v{version}_lightgbm.pkl"
    model = joblib.load(model_path)
    x_test = df_featured[features].copy()
    y_true = df_featured[TARGET].astype(int).to_numpy()

    model_metrics = load_model_metrics(version)
    threshold = model_metrics["threshold"]
    proba = model.predict_proba(x_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float(accuracy_score(y_true, y_pred))
    metrics = {
        "version": f"V{version}",
        "train_dataset": "LightGBM_SETC_S1_clean_dataset_s1",
        "test_dataset": "LightGBM_SETD_real_dataset_s1",
        "evaluation_type": "external_full_dataset_accuracy",
        "rows": int(len(df_featured)),
        "feature_count": int(len(features)),
        "threshold": float(threshold),
        **model_metrics,
        "external_accuracy": accuracy,
        "external_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "external_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "external_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "external_auc": float(roc_auc_score(y_true, proba)),
        "external_avg_precision": float(average_precision_score(y_true, proba)),
        "external_cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy_gap_points": float((accuracy - model_metrics["holdout_accuracy"]) * 100),
        "model_path": str(model_path.relative_to(ROOT)),
        "test_data_path": str(TEST_DATA.relative_to(ROOT)),
    }

    predictions = pd.DataFrame(
        {
            "order_id": df_featured["order_id"].astype(str).to_numpy(),
            "customer_id": df_featured["customer_id"].astype(str).to_numpy(),
            "actual_is_returned": y_true,
            "predict_probability_return": proba,
            "predicted_is_returned": y_pred,
            "threshold": threshold,
            "correct_prediction": (y_true == y_pred).astype(int),
        }
    )
    predictions.to_csv(
        report_dir / f"test_predictions_{MODEL_PREFIX}_v{version}_on_real_dataset_s1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metrics]).to_csv(
        report_dir / f"test_metrics_{MODEL_PREFIX}_v{version}_on_real_dataset_s1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def draw_accuracy_chart(summary: pd.DataFrame) -> None:
    width, height = 1650, 900
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    draw.text((width // 2, 48), "LightGBM S1 Models on real_dataset_s1.csv", font=font(38, bold=True), fill="#111111", anchor="ma")
    draw.text((width // 2, 96), "Holdout Accuracy vs External Full-Dataset Accuracy; no 20% split on SETD", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 120, 170, 1530, 720
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        y = int(y1 - (tick / 100) * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 14, y), str(tick), font=font(18), fill="#455A64", anchor="rm")

    group_gap = (x1 - x0 - 140) // len(summary)
    for i, row in summary.reset_index(drop=True).iterrows():
        holdout = float(row["holdout_accuracy"]) * 100
        external = float(row["external_accuracy"]) * 100
        bx = x0 + 70 + i * group_gap
        bw = 55
        holdout_y = int(y1 - (holdout / 100) * (y1 - y0))
        external_y = int(y1 - (external / 100) * (y1 - y0))
        draw.rounded_rectangle((bx, holdout_y, bx + bw, y1), radius=6, fill="#B0BEC5")
        draw.rounded_rectangle((bx + bw + 12, external_y, bx + (bw * 2) + 12, y1), radius=6, fill=colors[i])
        draw.text((bx + bw // 2, holdout_y - 10), f"{holdout:.1f}%", font=font(15), fill="#263238", anchor="mb")
        draw.text((bx + bw + 12 + bw // 2, external_y - 10), f"{external:.1f}%", font=font(15, bold=True), fill="#111111", anchor="mb")
        draw.text((bx + bw + 6, y1 + 34), row["version"], font=font(25), fill="#111111", anchor="ma")
        draw.text((bx + bw + 6, y1 + 65), f"gap {float(row['accuracy_gap_points']):+.1f}pt", font=font(16), fill="#455A64", anchor="ma")

    draw.rounded_rectangle((1125, 120, 1170, 145), radius=4, fill="#B0BEC5")
    draw.text((1180, 132), "S1 holdout", font=font(18), fill="#263238", anchor="lm")
    draw.rounded_rectangle((1325, 120, 1370, 145), radius=4, fill="#2E7D32")
    draw.text((1380, 132), "SETD real_dataset_s1", font=font(18), fill="#263238", anchor="lm")
    image.save(OUT_ROOT / "images" / "lgbm_s1_models_real_dataset_s1_test_accuracy_v1_to_v5.png")


def write_readme(summary: pd.DataFrame, validation: dict[str, object]) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {float(row['threshold']):.2f} | "
            f"{float(row['holdout_accuracy']) * 100:.2f}% | {float(row['external_accuracy']) * 100:.2f}% | "
            f"{float(row['accuracy_gap_points']):+.2f}pt | {float(row['external_recall']) * 100:.2f}% | "
            f"{float(row['external_precision']) * 100:.2f}% | {float(row['external_f1']) * 100:.2f}% | "
            f"{float(row['external_auc']) * 100:.2f}% | {int(row['external_cost']):,} |"
        )

    content = f"""# LightGBM SETC S1 Model Test on SETD real_dataset_s1.csv

Test data: `{TEST_DATA.relative_to(ROOT)}`

Train model source: `{MODEL_ROOT.relative_to(ROOT)}`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_dataset_s1.csv` file through already-trained LightGBM S1 V1-V5 models. It does not split the SETD test data again and does not retrain the models.

## Test Data Validation

- Rows: `{int(validation["rows"]):,}`
- Columns: `{int(validation["columns"])}`
- Missing/null cells: `{int(validation["missing_or_null_cells"])}`
- Duplicate order_id: `{int(validation["duplicate_order_id"])}`
- Returned: `{int(validation["returned_count"]):,}`
- Not Returned: `{int(validation["not_returned_count"]):,}`
- Return rate: `{float(validation["return_rate"]) * 100:.2f}%`

## Results

| Version | Features | Threshold | S1 Holdout Accuracy | real_dataset_s1 Accuracy | Gap | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

`S1 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s1.csv`.

`real_dataset_s1 Accuracy` is the result from sending the full external test file through the same model after applying the matching V1-V5 feature engineering.

If the gap is small, the generated SETD test data is close to the S1 model distribution. If the gap is large, the test data distribution or signal strength is different from the S1 holdout split.
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not TEST_DATA.exists():
        raise FileNotFoundError(TEST_DATA)
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(MODEL_ROOT)

    seq = load_seq_module()
    raw_test = pd.read_csv(TEST_DATA)
    validation = test_data_validation(raw_test)
    pd.DataFrame([validation]).to_csv(OUT_ROOT / "real_dataset_s1_validation_for_lgbm_s1_model_test.csv", index=False, encoding="utf-8-sig")

    current = seq.clean_dataset(raw_test)
    summary_rows: list[dict[str, object]] = []
    for version in range(1, 6):
        current = apply_version_feature_engineering(seq, version, current)
        summary_rows.append(evaluate_version(version, current))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "lgbm_s1_v1_to_v5_real_dataset_s1_test_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "lgbm_s1_v1_to_v5_real_dataset_s1_test_summary.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison = summary[
        [
            "version",
            "holdout_accuracy",
            "external_accuracy",
            "accuracy_gap_points",
            "external_recall",
            "external_precision",
            "external_f1",
            "external_auc",
            "external_cost",
        ]
    ].copy()
    for col in ["holdout_accuracy", "external_accuracy", "external_recall", "external_precision", "external_f1", "external_auc"]:
        comparison[col] = (comparison[col] * 100).round(2)
    comparison["accuracy_gap_points"] = comparison["accuracy_gap_points"].round(2)
    comparison.to_csv(OUT_ROOT / "lgbm_s1_holdout_vs_real_dataset_s1_comparison.csv", index=False, encoding="utf-8-sig")

    draw_accuracy_chart(summary)
    write_readme(summary, validation)
    print(
        summary[
            [
                "version",
                "feature_count",
                "threshold",
                "holdout_accuracy",
                "external_accuracy",
                "accuracy_gap_points",
                "external_recall",
                "external_precision",
                "external_f1",
                "external_auc",
                "external_cost",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
