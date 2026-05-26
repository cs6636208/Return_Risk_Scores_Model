from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "run_high_accuracy_feature_search.py"
DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
OUT_DIR = ROOT / "reports" / "model_experiments"
DOC_PATH = ROOT / "docs" / "analysis" / "accuracy_maximization_report.md"

COST_FN = 150
COST_FP = 50
RANDOM_STATE = 42


def load_helper():
    spec = importlib.util.spec_from_file_location("high_accuracy_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = load_helper()


@dataclass
class ModelSpec:
    name: str
    production_safe: bool
    leakage_level: str
    columns: list[str]
    model_params: dict
    top_k: int | None = None
    note: str = ""


def load_feature_frame() -> pd.DataFrame:
    df = helper.add_point_in_time_features(helper.load_dataset())
    return df


def leakage_columns(df: pd.DataFrame) -> list[str]:
    # Intentionally includes fields unavailable at order time, only to prove the leakage ceiling.
    cols = helper.production_columns(df) + [
        "return_id",
        "return_reason",
        "return_scenario",
        "item_condition",
        "return_status",
        "refund_amount",
        "risk_score",
        "risk_tier",
        "delivery_days",
        "delay_days",
    ]
    return [c for c in cols if c in df.columns and c not in helper.ID_FIELDS]


def lightgbm_params(kind: str) -> dict:
    base = {
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if kind == "accuracy_plain":
        return {
            **base,
            "n_estimators": 260,
            "learning_rate": 0.045,
            "num_leaves": 31,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "min_child_samples": 25,
            "reg_lambda": 1.0,
        }
    if kind == "accuracy_regularized":
        return {
            **base,
            "n_estimators": 220,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "max_depth": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "min_child_samples": 45,
            "reg_lambda": 5.0,
            "reg_alpha": 1.0,
        }
    if kind == "balanced":
        return {
            **base,
            "n_estimators": 320,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "class_weight": "balanced",
        }
    raise ValueError(kind)


def build_specs(df: pd.DataFrame) -> list[ModelSpec]:
    safe_all = helper.production_columns(df)
    safe_compact = helper.compact_columns(df)
    leak = leakage_columns(df)
    return [
        ModelSpec(
            "majority_baseline",
            True,
            "none",
            safe_all,
            {},
            note="Dummy baseline: predicts the majority class. This is the minimum accuracy to beat.",
        ),
        ModelSpec(
            "safe_all_accuracy_plain",
            True,
            "none",
            safe_all,
            lightgbm_params("accuracy_plain"),
            note="Production-safe features, LightGBM optimized for accuracy without class balancing.",
        ),
        ModelSpec(
            "safe_all_accuracy_regularized",
            True,
            "none",
            safe_all,
            lightgbm_params("accuracy_regularized"),
            note="Production-safe features with stronger regularization to reduce overfitting.",
        ),
        ModelSpec(
            "safe_compact_accuracy_plain",
            True,
            "none",
            safe_compact,
            lightgbm_params("accuracy_plain"),
            note="Reduced feature set to test whether fewer features improve accuracy/resource use.",
        ),
        ModelSpec(
            "safe_all_top20_accuracy",
            True,
            "none",
            safe_all,
            lightgbm_params("accuracy_plain"),
            top_k=20,
            note="Top 20 encoded features selected by LightGBM importance.",
        ),
        ModelSpec(
            "safe_all_top40_accuracy",
            True,
            "none",
            safe_all,
            lightgbm_params("accuracy_plain"),
            top_k=40,
            note="Top 40 encoded features selected by LightGBM importance.",
        ),
        ModelSpec(
            "safe_all_top80_accuracy",
            True,
            "none",
            safe_all,
            lightgbm_params("accuracy_plain"),
            top_k=80,
            note="Top 80 encoded features selected by LightGBM importance.",
        ),
        ModelSpec(
            "safe_all_balanced_reference",
            True,
            "none",
            safe_all,
            lightgbm_params("balanced"),
            note="Balanced reference for recall/cost comparison, not expected to maximize accuracy.",
        ),
        ModelSpec(
            "leakage_ceiling_post_event",
            False,
            "post_event_target_leakage",
            leak,
            lightgbm_params("accuracy_plain"),
            note="Diagnostic only: includes return/refund/status/risk fields unavailable at order time.",
        ),
    ]


def split_indices(y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y.iloc[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray, metric: str) -> tuple[float, dict]:
    best_threshold = 0.50
    best_score = -1.0
    best_stats: dict = {}
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        acc = accuracy_score(y_true, pred)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        cost = int(fn * COST_FN + fp * COST_FP)
        score = acc if metric == "accuracy" else f1
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_stats = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "expected_cost_thb": cost,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
    return best_threshold, best_stats


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "expected_cost_thb": int(fn * COST_FN + fp * COST_FP),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_lgbm(X_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def select_top_k(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    selector = train_lgbm(X_train, y_train, params)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    selected = importances.sort_values(ascending=False).head(k).index.tolist()
    return X_train[selected], X_val[selected], X_test[selected], selected


def evaluate_spec(df: pd.DataFrame, spec: ModelSpec, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    y = df["is_returned"].astype(int).reset_index(drop=True)
    X_all = helper.prepare_matrix(df, spec.columns)
    X_train = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    X_test = X_all.iloc[test_idx]
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    selected_features = list(X_train.columns)
    if spec.name == "majority_baseline":
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_val_proba = dummy.predict_proba(X_val)[:, 1]
        y_test_proba = dummy.predict_proba(X_test)[:, 1]
    else:
        if spec.top_k is not None:
            X_train, X_val, X_test, selected_features = select_top_k(
                X_train,
                y_train,
                X_val,
                X_test,
                spec.model_params,
                spec.top_k,
            )
        model = train_lgbm(X_train, y_train, spec.model_params)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

    acc_threshold, val_acc_stats = threshold_search(y_val, y_val_proba, "accuracy")
    f1_threshold, val_f1_stats = threshold_search(y_val, y_val_proba, "f1")
    test_at_acc = evaluate_predictions(y_test, y_test_proba, acc_threshold)
    test_at_f1 = evaluate_predictions(y_test, y_test_proba, f1_threshold)
    test_default = evaluate_predictions(y_test, y_test_proba, 0.50)

    return {
        "variant": spec.name,
        "production_safe": spec.production_safe,
        "leakage_level": spec.leakage_level,
        "note": spec.note,
        "original_feature_count": len(spec.columns),
        "encoded_feature_count": X_all.shape[1],
        "selected_encoded_feature_count": len(selected_features),
        "accuracy_threshold_from_val": acc_threshold,
        "f1_threshold_from_val": f1_threshold,
        "val_best_accuracy": val_acc_stats["accuracy"],
        "val_best_f1": val_f1_stats["f1_score"],
        "test_accuracy_at_acc_threshold": test_at_acc["accuracy"],
        "test_precision_at_acc_threshold": test_at_acc["precision"],
        "test_recall_at_acc_threshold": test_at_acc["recall"],
        "test_f1_at_acc_threshold": test_at_acc["f1_score"],
        "test_auc_roc": test_at_acc["auc_roc"],
        "test_cost_at_acc_threshold": test_at_acc["expected_cost_thb"],
        "test_accuracy_at_f1_threshold": test_at_f1["accuracy"],
        "test_precision_at_f1_threshold": test_at_f1["precision"],
        "test_recall_at_f1_threshold": test_at_f1["recall"],
        "test_f1_at_f1_threshold": test_at_f1["f1_score"],
        "test_cost_at_f1_threshold": test_at_f1["expected_cost_thb"],
        "test_accuracy_default_050": test_default["accuracy"],
        "test_recall_default_050": test_default["recall"],
        "test_f1_default_050": test_default["f1_score"],
        "test_cost_default_050": test_default["expected_cost_thb"],
        "selected_features": ", ".join(selected_features),
    }


def save_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values("test_accuracy_at_acc_threshold", ascending=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(plot_df))
    width = 0.22
    bars = [
        ("test_accuracy_at_acc_threshold", "Accuracy-max threshold", "#1f77b4"),
        ("test_recall_at_acc_threshold", "Recall at accuracy threshold", "#d62728"),
        ("test_f1_at_acc_threshold", "F1 at accuracy threshold", "#2ca02c"),
    ]
    for i, (col, label, color) in enumerate(bars):
        ax.bar(x + (i - 1) * width, plot_df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant"], rotation=22, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy Maximization: Safe Features vs Leakage Ceiling")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(i - 0.28, row.test_accuracy_at_acc_threshold + 0.015, f"{row.test_accuracy_at_acc_threshold:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_maximization_results.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy().fillna("")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def write_report(results: pd.DataFrame) -> None:
    safe = results[results["production_safe"]].sort_values("test_accuracy_at_acc_threshold", ascending=False)
    leakage = results[~results["production_safe"]].sort_values("test_accuracy_at_acc_threshold", ascending=False)
    best_safe = safe.iloc[0]
    best_leak = leakage.iloc[0]
    columns = [
        "variant",
        "production_safe",
        "test_accuracy_at_acc_threshold",
        "test_precision_at_acc_threshold",
        "test_recall_at_acc_threshold",
        "test_f1_at_acc_threshold",
        "test_cost_at_acc_threshold",
        "test_accuracy_at_f1_threshold",
        "test_recall_at_f1_threshold",
        "test_f1_at_f1_threshold",
        "selected_encoded_feature_count",
        "note",
    ]
    content = f"""# Accuracy Maximization Report

รอบนี้ทดลองเพิ่ม/ลด feature และเลือก threshold เพื่อ maximize Accuracy โดยใช้ train/validation/test split แยกกัน: train ใช้สอน model, validation ใช้เลือก threshold, test ใช้รายงานผลจริง

## Best Production-safe Accuracy

ตัวที่ดีที่สุดแบบไม่ใช้ข้อมูลหลังเหตุการณ์คือ `{best_safe["variant"]}` ได้ Accuracy `{best_safe["test_accuracy_at_acc_threshold"]:.4f}` ที่ threshold `{best_safe["accuracy_threshold_from_val"]:.2f}` ใช้ encoded feature `{int(best_safe["selected_encoded_feature_count"])}` ตัว และมี Recall `{best_safe["test_recall_at_acc_threshold"]:.4f}`

## Leakage Ceiling

ตัว diagnostic ที่รวมข้อมูลหลังเหตุการณ์คือ `{best_leak["variant"]}` ได้ Accuracy `{best_leak["test_accuracy_at_acc_threshold"]:.4f}` ซึ่งแสดงว่า Accuracy สูงมากสามารถเกิดจาก leakage ได้ แต่ไม่ควรนำไปใช้จริง เพราะใช้ field เช่น return/refund/status/risk ที่ยังไม่รู้ตอน order เข้า

## Results

{markdown_table(results.sort_values("test_accuracy_at_acc_threshold", ascending=False), columns)}

## Decision

- ถ้าต้องการ Accuracy อย่างเดียว production-safe model ทำได้ใกล้ majority baseline เพราะข้อมูล positive/negative ซ้อนกันสูง และ return rate มี imbalance
- การลด feature ด้วย top-k ช่วยลด resource ได้ แต่ไม่ได้ทำให้ Accuracy กระโดดถึง 80-90%
- การใส่ leakage feature ทำให้ Accuracy สูงผิดธรรมชาติ จึงใช้เป็นหลักฐานว่าเราต้องเพิ่ม proxy feature ที่รู้ล่วงหน้าแทน เช่น SKU defect history, courier delay rate, complaint history, click/add-to-cart behavior
"""
    DOC_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_feature_frame()
    train_idx, val_idx, test_idx = split_indices(df["is_returned"])

    rows = []
    feature_sets: dict[str, list[str]] = {}
    for spec in build_specs(df):
        row = evaluate_spec(df, spec, train_idx, val_idx, test_idx)
        rows.append(row)
        feature_sets[spec.name] = spec.columns

    results = pd.DataFrame(rows).sort_values("test_accuracy_at_acc_threshold", ascending=False)
    results.to_csv(OUT_DIR / "accuracy_maximization_results.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "accuracy_maximization_feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plot(results)
    write_report(results)

    print("[OK] Accuracy maximization complete.")
    print(
        results[
            [
                "variant",
                "production_safe",
                "test_accuracy_at_acc_threshold",
                "test_precision_at_acc_threshold",
                "test_recall_at_acc_threshold",
                "test_f1_at_acc_threshold",
                "test_cost_at_acc_threshold",
                "selected_encoded_feature_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
