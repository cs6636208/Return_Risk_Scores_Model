from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "v2_random_forest_full_38"
PACKAGE_DIR = ROOT / "docs" / "version 2" / "optimized_model" / CANDIDATE

SOURCE_FILES = {
    "model": ROOT / "models" / "best_model_v2_optimized.pkl",
    "metadata": ROOT / "models" / "best_model_v2_optimized_metadata.json",
    "train_test": ROOT / "data" / "features" / "train_test_sets_v2_optimized.pkl",
    "features": ROOT / "data" / "features" / "v2_optimized_used_features.csv",
    "predictions": ROOT / "reports" / "model_evaluation_v2_optimized" / "v2_optimized_test_predictions.csv",
    "comparison": ROOT / "reports" / "model_evaluation_v2_optimized" / "v2_optimized_model_comparison.csv",
    "script": ROOT / "scripts" / "train_v2_optimized_model.py",
}


def ensure_dirs() -> None:
    for child in ["models", "data", "reports", "scripts", "docs", "images"]:
        (PACKAGE_DIR / child).mkdir(parents=True, exist_ok=True)
    stale_all_candidate_plot = PACKAGE_DIR / "images" / "v2_optimized_model_comparison.png"
    if stale_all_candidate_plot.exists():
        stale_all_candidate_plot.unlink()


def copy_artifacts() -> None:
    shutil.copy2(SOURCE_FILES["model"], PACKAGE_DIR / "models" / "best_model_v2_random_forest_full_38.pkl")
    shutil.copy2(SOURCE_FILES["metadata"], PACKAGE_DIR / "models" / "best_model_v2_random_forest_full_38_metadata.json")
    shutil.copy2(SOURCE_FILES["train_test"], PACKAGE_DIR / "data" / "train_test_sets_v2_random_forest_full_38.pkl")
    shutil.copy2(SOURCE_FILES["features"], PACKAGE_DIR / "data" / "v2_random_forest_full_38_used_features.csv")
    shutil.copy2(SOURCE_FILES["predictions"], PACKAGE_DIR / "reports" / "v2_random_forest_full_38_test_predictions.csv")
    shutil.copy2(SOURCE_FILES["script"], PACKAGE_DIR / "scripts" / "train_v2_optimized_model.py")
    shutil.copy2(Path(__file__), PACKAGE_DIR / "scripts" / "package_v2_random_forest_full_38.py")


def write_selected_metrics() -> dict:
    comparison = pd.read_csv(SOURCE_FILES["comparison"])
    selected = comparison.loc[comparison["candidate"].eq(CANDIDATE)].copy()
    if selected.empty:
        raise ValueError(f"Candidate {CANDIDATE} not found in {SOURCE_FILES['comparison']}")

    selected.to_csv(PACKAGE_DIR / "reports" / "v2_random_forest_full_38_metrics.csv", index=False, encoding="utf-8-sig")
    row = selected.iloc[0].to_dict()
    (PACKAGE_DIR / "reports" / "v2_random_forest_full_38_metrics.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return row


def save_metric_chart(metrics: dict) -> None:
    labels = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
    values = [
        metrics["accuracy"],
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        metrics["auc"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#2f6fbb", "#d65f5f", "#4f9d69", "#8064a2", "#d49a3a"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("V2 RandomForest Full 38 - Selected Metrics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PACKAGE_DIR / "images" / "v2_random_forest_full_38_metrics.png", dpi=160)
    plt.close(fig)

    confusion = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual no_return", "Actual return"],
        columns=["Pred no_return", "Pred return"],
    )
    confusion.to_csv(PACKAGE_DIR / "reports" / "v2_random_forest_full_38_confusion_matrix.csv", encoding="utf-8-sig")


def write_docs(metrics: dict) -> None:
    features = pd.read_csv(SOURCE_FILES["features"])["feature"].tolist()
    feature_lines = "\n".join(f"- `{feature}`" for feature in features)
    content = f"""# V2 RandomForest Full 38

## Summary

โฟลเดอร์นี้เก็บเฉพาะ candidate ที่เลือกจาก V2 Optimized:

- Candidate: `{CANDIDATE}`
- Model: `RandomForestClassifier`
- Feature set: V2 original 38 features
- Threshold: `{metrics["selected_threshold"]:.2f}`
- Performance Rating: `{metrics["performance_rating"]}`

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {metrics["accuracy"]:.2%} |
| Recall | {metrics["recall"]:.2%} |
| Precision | {metrics["precision"]:.2%} |
| F1 | {metrics["f1"]:.2%} |
| AUC | {metrics["auc"]:.2%} |
| Cost Matrix | {int(metrics["cost"]):,} |

## Confusion Matrix

|  | Pred no_return | Pred return |
| --- | ---: | ---: |
| Actual no_return | {int(metrics["tn"])} | {int(metrics["fp"])} |
| Actual return | {int(metrics["fn"])} | {int(metrics["tp"])} |

## Feature List

{feature_lines}

## Process

1. ใช้ `data/processed/clean_dataset.csv` เป็น input หลัก
2. สร้าง feature ตามกรอบ V2 เช่น customer history, discount, delivery context, promotion, product, channel และ customer profile
3. ใช้ target encoding สำหรับ categorical feature
4. แบ่ง train/validation/test ด้วย `random_state=42`
5. Train candidates หลายตัวใน V2 optimized แล้วเลือกตัวนี้เพราะได้ performance รวมดีที่สุด
6. Export model, metadata, prediction test rows, feature list และ metrics ไว้ในโฟลเดอร์นี้

## Production Note

ตัวนี้เป็น V2 full 38 features และมี `delivery_days` / `delay_days` อยู่ใน feature list เหมือน V2 เดิม จึงเหมาะกับ project experiment หรือ post-delivery scoring มากกว่า real-time order-time scoring
"""
    (PACKAGE_DIR / "README.md").write_text(content, encoding="utf-8")
    (PACKAGE_DIR / "docs" / "v2_random_forest_full_38_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_artifacts()
    metrics = write_selected_metrics()
    save_metric_chart(metrics)
    write_docs(metrics)
    print(PACKAGE_DIR)
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
