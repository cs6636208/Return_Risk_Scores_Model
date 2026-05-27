from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_v2_optimized_model import (  # noqa: E402
    RANDOM_STATE,
    build_feature_sets,
    candidate_specs,
    load_v2_frame,
    train_candidate,
)


CANDIDATE = "v2_xgboost_safe_plus_rolling"
PACKAGE_DIR = ROOT / "docs" / "version 2" / "xgboost v2" / CANDIDATE
SOURCE_COMPARISON = ROOT / "reports" / "model_evaluation_v2_optimized" / "v2_optimized_model_comparison.csv"
SOURCE_TRAIN_SCRIPT = ROOT / "scripts" / "train_v2_optimized_model.py"


def ensure_dirs() -> None:
    for child in ["models", "data", "reports", "scripts", "docs", "images"]:
        (PACKAGE_DIR / child).mkdir(parents=True, exist_ok=True)


def selected_spec(specs: list[Any]) -> Any:
    for spec in specs:
        if spec.name == CANDIDATE:
            return spec
    raise ValueError(f"{CANDIDATE} not found in V2 optimized candidate specs")


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def save_model_package(
    pipeline: Any,
    metrics: dict[str, Any],
    feature_columns: list[str],
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    target: pd.Series,
) -> None:
    artifact = {
        "model_pipeline": pipeline,
        "threshold": float(metrics["selected_threshold"]),
        "feature_columns": feature_columns,
        "metadata": metrics,
        "split": {
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        },
    }
    joblib.dump(artifact, PACKAGE_DIR / "models" / "best_model_v2_xgboost_safe_plus_rolling.pkl")
    metadata = {
        "candidate": CANDIDATE,
        "model": "XGBClassifier",
        "order_time_safe": True,
        "feature_columns": feature_columns,
        "metrics": {key: to_jsonable(value) for key, value in metrics.items()},
        "target_distribution": {
            "all_return_rate": float(target.mean()),
            "test_return_rate": float(target.iloc[test_idx].mean()),
        },
        "split": artifact["split"],
    }
    (PACKAGE_DIR / "models" / "best_model_v2_xgboost_safe_plus_rolling_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_tables_and_data(df: pd.DataFrame, metrics: dict[str, Any], predictions: pd.DataFrame, feature_columns: list[str], test_idx: Any, target: pd.Series) -> None:
    pd.DataFrame([metrics]).to_csv(
        PACKAGE_DIR / "reports" / "v2_xgboost_safe_plus_rolling_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (PACKAGE_DIR / "reports" / "v2_xgboost_safe_plus_rolling_metrics.json").write_text(
        json.dumps({key: to_jsonable(value) for key, value in metrics.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(
        PACKAGE_DIR / "reports" / "v2_xgboost_safe_plus_rolling_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"feature": feature_columns}).to_csv(
        PACKAGE_DIR / "data" / "v2_xgboost_safe_plus_rolling_used_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(
        {
            "X_test_raw": df.iloc[test_idx][feature_columns],
            "y_test": target.iloc[test_idx].to_numpy(),
            "feature_names": feature_columns,
            "test_index": test_idx,
        },
        PACKAGE_DIR / "data" / "train_test_sets_v2_xgboost_safe_plus_rolling.pkl",
    )

    confusion = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual no_return", "Actual return"],
        columns=["Pred no_return", "Pred return"],
    )
    confusion.to_csv(
        PACKAGE_DIR / "reports" / "v2_xgboost_safe_plus_rolling_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def save_metric_chart(metrics: dict[str, Any]) -> None:
    labels = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
    values = [metrics["accuracy"], metrics["recall"], metrics["precision"], metrics["f1"], metrics["auc"]]
    colors = ["#2f6fbb", "#d65f5f", "#4f9d69", "#8064a2", "#d49a3a"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("V2 XGBoost Safe Plus Rolling - Selected Metrics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PACKAGE_DIR / "images" / "v2_xgboost_safe_plus_rolling_metrics.png", dpi=160)
    plt.close(fig)


def write_report(metrics: dict[str, Any], feature_columns: list[str]) -> None:
    feature_lines = "\n".join(f"- `{feature}`" for feature in feature_columns)
    content = f"""# V2 XGBoost Safe Plus Rolling

## Summary

โฟลเดอร์นี้เก็บเฉพาะ candidate `v2_xgboost_safe_plus_rolling` จากกระบวนการ V2 Optimized

- Candidate: `{CANDIDATE}`
- Model: `XGBClassifier`
- Order-time Safe: `Yes`
- Feature count: `{len(feature_columns)}`
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
2. สร้าง V2 order-time-safe features โดยตัด `delivery_days` และ `delay_days` ออก
3. เพิ่ม rolling customer history เช่น 30d, 60d, 90d, 180d, 365d
4. เพิ่ม interaction features เช่น discount ratio, category-payment, category-channel และ province-payment
5. ใช้ target encoding สำหรับ categorical features
6. แบ่ง train/validation/test ด้วย `random_state=42`
7. Train เฉพาะ `XGBClassifier` candidate นี้ และ export model, features, metrics, predictions, chart และ report เข้ามาในโฟลเดอร์นี้

## Production Note

ตัวนี้เหมาะกว่า `v2_random_forest_full_38` สำหรับ real-time หรือ order-time scoring เพราะ feature set ไม่ใช้ `delivery_days` และ `delay_days` ซึ่งเป็นข้อมูลหลังส่งสินค้า
"""
    (PACKAGE_DIR / "README.md").write_text(content, encoding="utf-8")
    (PACKAGE_DIR / "docs" / "v2_xgboost_safe_plus_rolling_report.md").write_text(content, encoding="utf-8")


def copy_scripts() -> None:
    shutil.copy2(SOURCE_TRAIN_SCRIPT, PACKAGE_DIR / "scripts" / "train_v2_optimized_model.py")
    shutil.copy2(Path(__file__), PACKAGE_DIR / "scripts" / "package_v2_xgboost_safe_plus_rolling.py")


def write_source_comparison_row() -> None:
    if not SOURCE_COMPARISON.exists():
        return
    comparison = pd.read_csv(SOURCE_COMPARISON)
    selected = comparison.loc[comparison["candidate"].eq(CANDIDATE)].copy()
    if selected.empty:
        return
    selected.to_csv(
        PACKAGE_DIR / "reports" / "v2_xgboost_safe_plus_rolling_source_comparison_row.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    ensure_dirs()
    df = load_v2_frame()
    target = df["is_returned"].astype(int).reset_index(drop=True)
    all_idx = pd.RangeIndex(len(df)).to_numpy()
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.15, random_state=RANDOM_STATE, stratify=target)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=target.iloc[train_val_idx],
    )
    scale_pos_weight = float(target.iloc[train_idx].eq(0).sum() / max(target.iloc[train_idx].eq(1).sum(), 1))
    feature_sets = build_feature_sets(df)
    spec = selected_spec(candidate_specs(feature_sets, scale_pos_weight))
    pipeline, metrics, predictions = train_candidate(df, spec, train_idx, val_idx, test_idx)

    save_model_package(pipeline, metrics, spec.feature_columns, train_idx, val_idx, test_idx, target)
    write_tables_and_data(df, metrics, predictions, spec.feature_columns, test_idx, target)
    save_metric_chart(metrics)
    write_report(metrics, spec.feature_columns)
    copy_scripts()
    write_source_comparison_row()

    print(PACKAGE_DIR)
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
