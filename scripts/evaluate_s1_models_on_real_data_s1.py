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


TEST_DATA = ROOT / "docs" / "XGBoost" / "SETB" / "real_data" / "real_data_s1.csv"
S1_ROOT = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "S1"
OUT_ROOT = ROOT / "docs" / "XGBoost" / "SETB" / "real_data" / "S3"
TARGET = "is_returned"


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


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        (OUT_ROOT / f"V{version}" / "reports").mkdir(parents=True, exist_ok=True)


def test_data_validation(df: pd.DataFrame) -> dict[str, object]:
    blank_text = int(
        sum((df[col].astype(str).str.strip() == "").sum() for col in df.select_dtypes(include=["object", "string"]).columns)
    )
    return {
        "file": str(TEST_DATA.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "blank_text_cells": blank_text,
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
        "note": "S1-aligned benchmark data. Do not use this file for train or hyperparameter tuning.",
    }


def load_used_features(version: int) -> list[str]:
    path = S1_ROOT / f"V{version}" / "features" / f"used_features_s1_v{version}.csv"
    return pd.read_csv(path)["feature"].astype(str).tolist()


def load_threshold(version: int) -> float:
    path = S1_ROOT / f"V{version}" / "reports" / f"metrics_s1_v{version}.csv"
    metrics = pd.read_csv(path)
    return float(metrics.loc[0, "threshold"])


def load_holdout_accuracy(version: int) -> float:
    path = S1_ROOT / f"V{version}" / "reports" / f"metrics_s1_v{version}.csv"
    metrics = pd.read_csv(path)
    return float(metrics.loc[0, "accuracy"])


def evaluate_version(version: int, df_featured: pd.DataFrame) -> dict[str, object]:
    report_dir = OUT_ROOT / f"V{version}" / "reports"
    features = load_used_features(version)
    missing_features = [feature for feature in features if feature not in df_featured.columns]
    if missing_features:
        raise KeyError(f"V{version} missing test features: {missing_features}")

    model_path = S1_ROOT / f"V{version}" / "models" / f"model_s1_v{version}_xgboost.pkl"
    model = joblib.load(model_path)
    x_test = df_featured[features].copy()
    y_true = df_featured[TARGET].astype(int).to_numpy()

    proba = model.predict_proba(x_test)[:, 1]
    threshold = load_threshold(version)
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "version": f"V{version}",
        "train_dataset": "SETA_S1_clean_dataset_s1",
        "test_dataset": "SETB_real_data_s1",
        "evaluation_type": "s1_calibrated_full_dataset_accuracy",
        "rows": int(len(df_featured)),
        "feature_count": int(len(features)),
        "threshold": float(threshold),
        "holdout_accuracy": load_holdout_accuracy(version),
        "external_accuracy": float(accuracy_score(y_true, y_pred)),
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
        report_dir / f"test_predictions_s1_v{version}_on_real_data_s1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metrics]).to_csv(
        report_dir / f"test_metrics_s1_v{version}_on_real_data_s1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def draw_accuracy_chart(summary: pd.DataFrame) -> None:
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    draw.text((width // 2, 50), "S1 Models on real_data_s1.csv", font=font(38), fill="#111111", anchor="ma")
    draw.text((width // 2, 96), "External full-dataset accuracy on 55,000 rows; no 20% split", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 130, 160, 1400, 700
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        y = int(y1 - (tick / 100) * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 14, y), str(tick), font=font(18), fill="#455A64", anchor="rm")

    bar_gap = (x1 - x0 - 160) // len(summary)
    for i, row in summary.reset_index(drop=True).iterrows():
        value = float(row["external_accuracy"]) * 100
        bx = x0 + 80 + i * bar_gap
        bw = 120
        by = int(y1 - (value / 100) * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=8, fill=colors[i])
        draw.text((bx + bw // 2, by - 16), f"{value:.2f}%", font=font(23), fill="#111111", anchor="mb")
        draw.text((bx + bw // 2, y1 + 34), row["version"], font=font(25), fill="#111111", anchor="ma")
        draw.text((bx + bw // 2, y1 + 66), f"Recall {float(row['external_recall']) * 100:.1f}%", font=font(17), fill="#455A64", anchor="ma")

    image.save(OUT_ROOT / "images" / "s1_models_real_data_s1_test_accuracy_v1_to_v5.png")


def write_readme(summary: pd.DataFrame, validation: dict[str, object]) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {float(row['threshold']):.2f} | "
            f"{float(row['holdout_accuracy']) * 100:.2f}% | {float(row['external_accuracy']) * 100:.2f}% | "
            f"{float(row['external_recall']) * 100:.2f}% | {float(row['external_precision']) * 100:.2f}% | "
            f"{float(row['external_f1']) * 100:.2f}% | {float(row['external_auc']) * 100:.2f}% | {int(row['external_cost']):,} |"
        )

    content = f"""# S1 Model Test on real_data_s1.csv

Test data: `{TEST_DATA.relative_to(ROOT)}`

Train model source: `{S1_ROOT.relative_to(ROOT)}`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_data_s1.csv` file through already-trained S1 V1-V5 models. It does not split the test data again and does not retrain the models.

## Test Data Validation

- Rows: `{int(validation["rows"]):,}`
- Columns: `{int(validation["columns"])}`
- Missing/null cells: `{int(validation["missing_or_null_cells"])}`
- Duplicate order_id: `{int(validation["duplicate_order_id"])}`
- Returned: `{int(validation["returned_count"]):,}`
- Not Returned: `{int(validation["not_returned_count"]):,}`
- Return rate: `{float(validation["return_rate"]) * 100:.2f}%`

## Results

| Version | Features | Threshold | S1 Holdout Accuracy | real_data_s1 Accuracy | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

`S1 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s1.csv`.

`real_data_s1 Accuracy` is measured by sending all 55,000 rows through each already-trained S1 model after applying the matching version feature engineering. This external test uses the full file, not a 20% split.
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not TEST_DATA.exists():
        raise FileNotFoundError(TEST_DATA)
    if not S1_ROOT.exists():
        raise FileNotFoundError(S1_ROOT)

    seq = load_seq_module()
    raw_test = pd.read_csv(TEST_DATA)
    validation = test_data_validation(raw_test)
    pd.DataFrame([validation]).to_csv(OUT_ROOT / "real_data_s1_validation_for_s1_model_test.csv", index=False, encoding="utf-8-sig")

    current = seq.clean_dataset(raw_test)
    summary_rows: list[dict[str, object]] = []
    for version in range(1, 6):
        current = apply_version_feature_engineering(seq, version, current)
        metrics = evaluate_version(version, current)
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "s1_v1_to_v5_real_data_s1_test_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "s1_v1_to_v5_real_data_s1_test_summary.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison = summary[
        [
            "version",
            "holdout_accuracy",
            "external_accuracy",
            "external_recall",
            "external_precision",
            "external_f1",
            "external_auc",
            "external_cost",
        ]
    ].copy()
    comparison["accuracy_gap_points"] = (comparison["external_accuracy"] - comparison["holdout_accuracy"]) * 100
    for col in ["holdout_accuracy", "external_accuracy", "external_recall", "external_precision", "external_f1", "external_auc"]:
        comparison[col] = (comparison[col] * 100).round(2)
    comparison["accuracy_gap_points"] = comparison["accuracy_gap_points"].round(2)
    comparison.to_csv(OUT_ROOT / "s1_holdout_vs_real_data_s1_comparison.csv", index=False, encoding="utf-8-sig")
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
