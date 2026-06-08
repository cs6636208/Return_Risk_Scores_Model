from __future__ import annotations

import argparse
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


TARGET = "is_returned"
COST_FN = 500
COST_FP = 50
VERSIONS = [1, 2, 3, 4, 5]


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def dataset_base(dataset_size: int) -> Path:
    return ROOT / "docs" / "test" / f"dataset_{dataset_size}" / "sequential_pipeline"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def required_paths(base: Path, version: int) -> dict[str, Path]:
    version_dir = base / f"version_{version}"
    return {
        "model": version_dir / "models" / f"model_version_{version}_xgboost.pkl",
        "featured": version_dir / "data" / f"df_featured_version_{version}.csv",
        "train_test": version_dir / "data" / f"train_test_sets_version_{version}.pkl",
        "reports": version_dir / "reports",
    }


def validate_required(paths: dict[str, Path], dataset_size: int, version: int) -> None:
    missing = [name for name, path in paths.items() if name != "reports" and not path.exists()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Dataset {dataset_size} V{version} missing required artifact(s): {detail}")


def metrics_from_probability(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "cost": int(fn * COST_FN + fp * COST_FP),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_version(base: Path, dataset_size: int, version: int) -> dict[str, object]:
    paths = required_paths(base, version)
    validate_required(paths, dataset_size, version)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    model = joblib.load(paths["model"])
    train_test = joblib.load(paths["train_test"])
    feature_names = list(train_test["feature_names"])
    threshold = float(train_test["threshold"])

    featured = pd.read_csv(paths["featured"], low_memory=False)
    missing_features = [feature for feature in feature_names if feature not in featured.columns]
    if missing_features:
        raise KeyError(f"Dataset {dataset_size} V{version} missing feature columns: {missing_features[:10]}")

    x = featured[feature_names].copy()
    y = pd.to_numeric(featured[TARGET], errors="coerce").fillna(0).astype(int).to_numpy()
    proba = model.predict_proba(x)[:, 1]
    metrics = metrics_from_probability(y, proba, threshold)
    pred = (proba >= threshold).astype(int)

    split = np.array(["unknown"] * len(featured), dtype=object)
    train_index = np.asarray(train_test.get("train_index", []), dtype=int)
    test_index = np.asarray(train_test.get("test_index", []), dtype=int)
    split[train_index] = "train"
    split[test_index] = "holdout_test"

    prediction_cols = {
        "row_index": np.arange(len(featured)),
        "dataset_split": split,
        "actual_is_returned": y,
        "predict_probability_return": proba,
        "predicted_is_returned": pred,
        "threshold": threshold,
    }
    for col in ["order_id", "customer_id", "order_date"]:
        if col in featured.columns:
            prediction_cols[col] = featured[col].to_numpy()

    predictions = pd.DataFrame(prediction_cols)
    ordered_cols = [col for col in ["row_index", "order_id", "customer_id", "order_date", "dataset_split"] if col in predictions.columns]
    ordered_cols += [
        "actual_is_returned",
        "predict_probability_return",
        "predicted_is_returned",
        "threshold",
    ]
    predictions = predictions[ordered_cols]
    predictions_path = paths["reports"] / f"full_dataset_predictions_version_{version}.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    row = {
        "dataset_size": dataset_size,
        "version": f"V{version}",
        "model": "XGBoost",
        "rows_predicted": int(len(featured)),
        "feature_count": int(len(feature_names)),
        "threshold": threshold,
        "full_dataset_prediction_accuracy": metrics["accuracy"],
        "full_dataset_prediction_recall": metrics["recall"],
        "full_dataset_prediction_precision": metrics["precision"],
        "full_dataset_prediction_f1": metrics["f1"],
        "full_dataset_prediction_auc": metrics["auc"],
        "full_dataset_prediction_avg_precision": metrics["avg_precision"],
        "full_dataset_prediction_cost": metrics["cost"],
        "full_dataset_prediction_tn": metrics["tn"],
        "full_dataset_prediction_fp": metrics["fp"],
        "full_dataset_prediction_fn": metrics["fn"],
        "full_dataset_prediction_tp": metrics["tp"],
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "note": "Full-dataset prediction accuracy is in-sample + holdout mixed, not unbiased test accuracy.",
    }
    pd.DataFrame([row]).to_csv(
        paths["reports"] / f"full_dataset_metrics_version_{version}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return row


def draw_full_accuracy_chart(base: Path, dataset_size: int, summary: pd.DataFrame) -> None:
    image_dir = base / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((width // 2, 50), f"Dataset {dataset_size} Full-Dataset Prediction Accuracy", font=font(38), fill="#111111", anchor="ma")
    draw.text((width // 2, 92), "100% rows predicted after training: in-sample + holdout mixed", font=font(22), fill="#B71C1C", anchor="ma")

    x0, y0, x1, y1 = 120, 160, 1500, 720
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        yy = int(y1 - tick / 100 * (y1 - y0))
        draw.line((x0 - 6, yy, x1, yy), fill="#ECEFF1", width=1)
        draw.text((x0 - 12, yy), str(tick), font=font(18), fill="#455A64", anchor="rm")

    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]
    bar_space = (x1 - x0 - 80) // len(summary)
    for i, (_, row) in enumerate(summary.iterrows()):
        value = float(row["full_dataset_prediction_accuracy"]) * 100
        bx = x0 + 55 + i * bar_space
        bw = 120
        by = int(y1 - value / 100 * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=7, fill=colors[i])
        draw.text((bx + bw // 2, by - 10), f"{value:.2f}%", font=font(24), fill="#111111", anchor="ms")
        draw.text((bx + bw // 2, y1 + 20), str(row["version"]), font=font(25), fill="#111111", anchor="ma")
    img.save(image_dir / f"dataset_{dataset_size}_full_dataset_accuracy_v1_to_v5.png")


def draw_test_vs_full_chart(base: Path, dataset_size: int, combined: pd.DataFrame) -> None:
    image_dir = base / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1000
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((width // 2, 50), f"Dataset {dataset_size}: Holdout Test vs Full-Dataset Accuracy", font=font(36), fill="#111111", anchor="ma")
    draw.text((width // 2, 92), "Holdout = 20% unseen rows | Full = 100% rows after training", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 130, 160, 1690, 780
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        yy = int(y1 - tick / 100 * (y1 - y0))
        draw.line((x0 - 6, yy, x1, yy), fill="#ECEFF1", width=1)
        draw.text((x0 - 12, yy), str(tick), font=font(18), fill="#455A64", anchor="rm")

    group_space = (x1 - x0 - 80) // len(combined)
    bw = 68
    for i, (_, row) in enumerate(combined.iterrows()):
        center = x0 + 100 + i * group_space
        for offset, col, color in [
            (-0.55, "test_accuracy", "#607D8B"),
            (0.55, "full_dataset_prediction_accuracy", "#2E7D32"),
        ]:
            value = float(row[col]) * 100
            bx = center + int(offset * (bw + 18))
            by = int(y1 - value / 100 * (y1 - y0))
            draw.rounded_rectangle((bx, by, bx + bw, y1), radius=5, fill=color)
            draw.text((bx + bw // 2, by - 8), f"{value:.1f}%", font=font(17), fill="#111111", anchor="ms")
        draw.text((center, y1 + 20), str(row["version"]), font=font(24), fill="#111111", anchor="ma")

    legend_x = 1210
    draw.rectangle((legend_x, 850, legend_x + 24, 874), fill="#607D8B")
    draw.text((legend_x + 34, 846), "Holdout Test Accuracy", font=font(22), fill="#111111")
    draw.rectangle((legend_x, 890, legend_x + 24, 914), fill="#2E7D32")
    draw.text((legend_x + 34, 886), "Full-Dataset Prediction Accuracy", font=font(22), fill="#111111")
    img.save(image_dir / f"dataset_{dataset_size}_test_vs_full_dataset_accuracy_v1_to_v5.png")


def build_combined_summary(base: Path, dataset_size: int, full_summary: pd.DataFrame) -> pd.DataFrame:
    test_summary_path = base / "sequential_pipeline_summary.csv"
    test_summary = pd.read_csv(test_summary_path)
    holdout_rows: list[int] = []
    for version in VERSIONS:
        metrics_path = base / f"version_{version}" / "reports" / f"metrics_version_{version}.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).iloc[0]
            holdout_rows.append(int(metrics.get("test_rows", 0)))
        else:
            holdout_rows.append(0)
    combined = pd.DataFrame(
        {
            "dataset_size": dataset_size,
            "version": test_summary["version"],
            "feature_count": test_summary["feature_count"].astype(int),
            "holdout_test_rows": holdout_rows,
            "full_dataset_rows": full_summary["rows_predicted"].astype(int),
            "test_accuracy": test_summary["accuracy"].astype(float),
            "full_dataset_prediction_accuracy": full_summary["full_dataset_prediction_accuracy"].astype(float),
            "accuracy_delta_full_minus_test": full_summary["full_dataset_prediction_accuracy"].astype(float) - test_summary["accuracy"].astype(float),
            "test_recall": test_summary["recall"].astype(float),
            "full_dataset_prediction_recall": full_summary["full_dataset_prediction_recall"].astype(float),
            "test_f1": test_summary["f1"].astype(float),
            "full_dataset_prediction_f1": full_summary["full_dataset_prediction_f1"].astype(float),
            "test_auc": test_summary["auc"].astype(float),
            "full_dataset_prediction_auc": full_summary["full_dataset_prediction_auc"].astype(float),
            "test_cost": test_summary["cost"].astype(int),
            "full_dataset_prediction_cost": full_summary["full_dataset_prediction_cost"].astype(int),
        }
    )
    return combined


def table_rows_for_readme(combined: pd.DataFrame) -> str:
    rows = []
    for _, row in combined.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {pct(float(row['test_accuracy']))} | "
            f"{pct(float(row['full_dataset_prediction_accuracy']))} | {pct(float(row['accuracy_delta_full_minus_test']))} | "
            f"{pct(float(row['test_f1']))} | {pct(float(row['full_dataset_prediction_f1']))} | "
            f"{int(row['test_cost']):,} | {int(row['full_dataset_prediction_cost']):,} |"
        )
    return "\n".join(rows)


def append_readme_section(path: Path, dataset_size: int, combined: pd.DataFrame) -> None:
    marker = "## 3. Full-Dataset Prediction Accuracy"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# Dataset {dataset_size}\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    section = f"""

{marker}

ส่วนนี้คือการเอา dataset ทั้งก้อน 100% ไป predict ด้วย model ที่ train ไว้แล้วในแต่ละ version

คำเตือน: ค่านี้เป็น `in-sample + holdout mixed accuracy` เพราะมีข้อมูล train รวมอยู่ด้วย จึงไม่ควรใช้แทน holdout test accuracy สำหรับวัด generalization

| Version | Features | Holdout Test Accuracy | Full-Dataset Prediction Accuracy | Delta | Holdout F1 | Full-Dataset F1 | Holdout Cost | Full-Dataset Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table_rows_for_readme(combined)}

ไฟล์ที่เกี่ยวข้อง:

- `sequential_pipeline/full_dataset_prediction_summary.csv`
- `sequential_pipeline/test_vs_full_dataset_prediction_summary.csv`
- `sequential_pipeline/images/dataset_{dataset_size}_full_dataset_accuracy_v1_to_v5.png`
- `sequential_pipeline/images/dataset_{dataset_size}_test_vs_full_dataset_accuracy_v1_to_v5.png`
"""
    path.write_text(text.rstrip() + section, encoding="utf-8")


def evaluate_dataset(dataset_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = dataset_base(dataset_size)
    if not base.exists():
        raise FileNotFoundError(f"Missing sequential pipeline folder: {base}")

    rows = [evaluate_version(base, dataset_size, version) for version in VERSIONS]
    full_summary = pd.DataFrame(rows)
    full_summary.to_csv(base / "full_dataset_prediction_summary.csv", index=False, encoding="utf-8-sig")

    combined = build_combined_summary(base, dataset_size, full_summary)
    combined.to_csv(base / "test_vs_full_dataset_prediction_summary.csv", index=False, encoding="utf-8-sig")

    draw_full_accuracy_chart(base, dataset_size, full_summary)
    draw_test_vs_full_chart(base, dataset_size, combined)

    append_readme_section(base / "README.md", dataset_size, combined)
    append_readme_section(ROOT / "docs" / "test" / f"dataset_{dataset_size}" / "README.md", dataset_size, combined)
    return full_summary, combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-dataset prediction accuracy for sequential V1-V5 pipelines.")
    parser.add_argument(
        "--dataset-size",
        type=int,
        action="append",
        choices=[5000, 50000],
        help="Dataset size to evaluate. Can be provided more than once. Defaults to both 5000 and 50000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_sizes = args.dataset_size or [5000, 50000]
    for dataset_size in dataset_sizes:
        full_summary, combined = evaluate_dataset(dataset_size)
        print(f"\nDataset {dataset_size} full-dataset prediction")
        print(
            combined[
                [
                    "version",
                    "test_accuracy",
                    "full_dataset_prediction_accuracy",
                    "accuracy_delta_full_minus_test",
                    "test_f1",
                    "full_dataset_prediction_f1",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
