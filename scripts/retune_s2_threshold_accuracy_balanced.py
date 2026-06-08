from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

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


OUT_ROOT = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "S2"
SOURCE = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s2.csv"
TARGET = "is_returned"

THRESHOLD_STRATEGY = "accuracy_balanced_recall_floor_70"
MIN_RECALL = 0.70


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def metric_row(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    fallback: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.25, 0.80, 56):
        row = metric_row(y_true, proba, float(threshold))
        row["score"] = (
            float(row["accuracy"]) * 0.60
            + float(row["f1"]) * 0.25
            + float(row["precision"]) * 0.10
            + float(row["recall"]) * 0.05
            - int(row["cost"]) / 20_000_000
        )
        fallback.append(row)
        if float(row["recall"]) >= MIN_RECALL:
            candidates.append(row)

    pool = candidates if candidates else fallback
    best = max(pool, key=lambda row: (float(row["score"]), float(row["accuracy"]), float(row["f1"])))
    return float(best["threshold"]), best


def update_train_full_pickle(path: Path, threshold: float) -> None:
    if not path.exists():
        return
    payload = joblib.load(path)
    if isinstance(payload, dict):
        payload["threshold"] = threshold
        payload["threshold_strategy"] = THRESHOLD_STRATEGY
        payload["minimum_recall_for_threshold"] = MIN_RECALL
        joblib.dump(payload, path)


def update_json(path: Path, updates: dict[str, object]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_version_readme(version: int, metrics: dict[str, object], feature_count: int) -> None:
    version_dir = OUT_ROOT / f"V{version}"
    content = f"""# S2 V{version} - XGBoost Accuracy-Balanced Threshold

Clean dataset source: `{SOURCE.relative_to(ROOT)}`

Evaluation type: `full_training_in_sample_no_holdout`

Meaning: train and predict on the full 50,000-row dataset. This is **not** holdout test accuracy.

## Current Result

- Model: `XGBoost`
- Feature count: `{feature_count}`
- Threshold strategy: `{THRESHOLD_STRATEGY}`
- Threshold: `{float(metrics["threshold"]):.2f}`
- Accuracy: `{float(metrics["accuracy"]) * 100:.2f}%`
- Recall: `{float(metrics["recall"]) * 100:.2f}%`
- Precision: `{float(metrics["precision"]) * 100:.2f}%`
- F1: `{float(metrics["f1"]) * 100:.2f}%`
- AUC: `{float(metrics["auc"]) * 100:.2f}%`
- Cost: `{int(metrics["cost"]):,}`

## Why This Was Adjusted

The previous threshold was too aggressive for return prediction. It caught many returned orders, but it also produced too many false positives, so Accuracy stayed low.

This version keeps Recall at approximately 70% or higher where possible, then chooses the threshold that gives the best combined Accuracy, F1, Precision, Recall, and Cost trade-off.

## New Data Policy

The model can predict new rows only when the same feature schema is generated. It does not learn automatically, does not discard the old model, and does not jump to a new model by itself. If new production data changes pattern, retrain/tune a new version.
"""
    (version_dir / "README.md").write_text(content, encoding="utf-8")


def draw_accuracy_chart(summary: pd.DataFrame) -> None:
    image_dir = OUT_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    draw.text((width // 2, 48), "S2 Accuracy-Balanced Full-Dataset Prediction", font=font(38), fill="#111111", anchor="ma")
    draw.text((width // 2, 96), "Threshold retuned with Recall floor around 70%", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 130, 160, 1400, 700
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        y = int(y1 - (tick / 100) * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 14, y), str(tick), font=font(18), fill="#455A64", anchor="rm")

    bar_gap = (x1 - x0 - 160) // len(summary)
    for i, row in summary.reset_index(drop=True).iterrows():
        value = float(row["accuracy"]) * 100
        bx = x0 + 80 + i * bar_gap
        bw = 120
        by = int(y1 - (value / 100) * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=8, fill=colors[i])
        draw.text((bx + bw // 2, by - 16), f"{value:.2f}%", font=font(23), fill="#111111", anchor="mb")
        draw.text((bx + bw // 2, y1 + 34), row["version"], font=font(25), fill="#111111", anchor="ma")
        draw.text((bx + bw // 2, y1 + 66), f"Recall {float(row['recall']) * 100:.1f}%", font=font(17), fill="#455A64", anchor="ma")

    image.save(image_dir / "s2_accuracy_balanced_accuracy_v1_to_v5.png")
    image.save(image_dir / "s2_full_training_accuracy_v1_to_v5.png")


def write_summary_readme(summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {float(row['threshold']):.2f} | "
            f"{float(row['accuracy']) * 100:.2f}% | {float(row['recall']) * 100:.2f}% | "
            f"{float(row['precision']) * 100:.2f}% | {float(row['f1']) * 100:.2f}% | "
            f"{float(row['auc']) * 100:.2f}% | {int(row['cost']):,} |"
        )

    content = f"""# S2 Sequential V1-V5 - Accuracy-Balanced Update

Source clean dataset: `{SOURCE.relative_to(ROOT)}`

Flow: `clean_dataset_s2.csv -> V1 -> V2 -> V3 -> V4 -> V5`

Evaluation type: `full_training_in_sample_no_holdout`

Important: this trains and predicts on the full 50,000-row dataset. It is useful for full-dataset prediction accuracy, but it is not unbiased holdout test accuracy.

## Updated Metrics

Threshold strategy: `{THRESHOLD_STRATEGY}`

| Version | Features | Threshold | Accuracy | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## What Changed

The previous S2 models used a low threshold around 0.41-0.42, which made the model predict many orders as Returned. That increased Recall but created many false positives and kept Accuracy low.

This update retunes only the decision threshold. The XGBoost model and feature engineering files remain the same. The new threshold keeps Recall around 70% where possible, while improving Accuracy and Precision.

## New Data Policy

These models support new incoming data only if the same feature schema is built. They do not learn automatically, do not discard old knowledge, and do not jump to a new model by themselves. If new real production data changes the pattern, create a new retrained/tuned version.
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    summary_path = OUT_ROOT / "s2_v1_to_v5_full_train_summary.csv"
    backup_path = OUT_ROOT / "s2_v1_to_v5_full_train_summary_before_accuracy_balanced.csv"
    if summary_path.exists() and not backup_path.exists():
        shutil.copyfile(summary_path, backup_path)

    summary_rows: list[dict[str, object]] = []
    for version in range(1, 6):
        version_dir = OUT_ROOT / f"V{version}"
        predictions_path = version_dir / "reports" / f"full_train_predictions_s2_v{version}.csv"
        metrics_path = version_dir / "reports" / f"metrics_s2_v{version}.csv"
        used_path = version_dir / "features" / f"used_features_s2_v{version}.csv"
        train_full_path = version_dir / "features" / f"train_full_sets_s2_v{version}.pkl"
        metadata_path = version_dir / "models" / f"model_s2_v{version}_metadata.json"
        schema_path = version_dir / "features" / f"feature_schema_s2_v{version}.json"

        df = pd.read_csv(predictions_path)
        y = df["actual_is_returned"].astype(int).to_numpy()
        proba = df["predict_probability_return"].astype(float).to_numpy()
        threshold, metrics = choose_threshold(y, proba)
        pred = (proba >= threshold).astype(int)

        feature_count = len(pd.read_csv(used_path))
        row = {
            "version": f"V{version}",
            "dataset": "S2",
            "model": "XGBoost",
            "evaluation_type": "full_training_in_sample_no_holdout",
            "rows": len(df),
            "train_rows": len(df),
            "test_rows": 0,
            "feature_count": feature_count,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "minimum_recall_for_threshold": MIN_RECALL,
            **{key: value for key, value in metrics.items() if key != "score"},
            "threshold_score": float(metrics["score"]),
            "source_clean_dataset": str(SOURCE.relative_to(ROOT)),
            "df_featured_path": str((version_dir / "features" / f"df_featured_s2_v{version}.csv").relative_to(ROOT)),
            "model_path": str((version_dir / "models" / f"model_s2_v{version}_xgboost.pkl").relative_to(ROOT)),
            "train_full_path": str(train_full_path.relative_to(ROOT)),
        }

        df["predicted_is_returned"] = pred
        df["threshold"] = threshold
        df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        update_train_full_pickle(train_full_path, threshold)
        update_json(metadata_path, row)
        update_json(
            schema_path,
            {
                "threshold": threshold,
                "threshold_strategy": THRESHOLD_STRATEGY,
                "minimum_recall_for_threshold": MIN_RECALL,
            },
        )
        write_version_readme(version, row, feature_count)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (OUT_ROOT / "s2_v1_to_v5_full_train_summary.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_accuracy_chart(summary)
    write_summary_readme(summary)
    print(summary[["version", "feature_count", "threshold", "accuracy", "recall", "precision", "f1", "auc", "cost", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    main()
