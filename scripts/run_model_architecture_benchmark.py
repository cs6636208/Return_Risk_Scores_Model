from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "run_high_accuracy_feature_search.py"
OUT_DIR = ROOT / "reports" / "model_experiments"
DOC_PATH = ROOT / "docs" / "analysis" / "model_architecture_benchmark_report.md"

RANDOM_STATE = 42
COST_FN = 150
COST_FP = 50


def load_helper():
    spec = importlib.util.spec_from_file_location("high_accuracy_helper_benchmark", HELPER_PATH)
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
    family: str
    accuracy_first: bool
    feature_set: str
    model: object
    note: str


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_plus = helper.add_point_in_time_features(helper.load_dataset()).reset_index(drop=True)
    engineered = pd.read_csv(ROOT / "data" / "features" / "df_engineered.csv", low_memory=False).reset_index(drop=True)
    engineered["is_returned"] = clean_plus["is_returned"].values
    engineered["order_id"] = clean_plus["order_id"].values
    engineered["customer_id"] = clean_plus["customer_id"].values
    return clean_plus, engineered


def clean_plus_columns(df: pd.DataFrame) -> list[str]:
    return helper.production_columns(df)


def compact_columns(df: pd.DataFrame) -> list[str]:
    return helper.compact_columns(df)


def engineered_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c
        not in {
            "is_returned",
            "order_id",
            "customer_id",
            "customer_name",
            "customer_phone",
            "product_id",
            "product_name",
            "return_id",
            "return_date",
            "return_reason",
            "refund_amount",
            "risk_score",
            "risk_tier",
            "shap_values",
        }
    ]


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return helper.prepare_matrix(df, columns)


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


def model_specs(scale_pos_weight: float) -> list[tuple[str, object, str, bool]]:
    return [
        (
            "LogisticRegression_balanced",
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
            ),
            "linear",
            False,
        ),
        (
            "RandomForest_accuracy",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "bagging_tree",
            True,
        ),
        (
            "RandomForest_balanced",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "bagging_tree",
            False,
        ),
        (
            "ExtraTrees_accuracy",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "bagging_tree",
            True,
        ),
        (
            "ExtraTrees_balanced",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "bagging_tree",
            False,
        ),
        (
            "HistGradientBoosting_accuracy",
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.5,
                random_state=RANDOM_STATE,
            ),
            "boosting_tree",
            True,
        ),
        (
            "LightGBM_accuracy",
            LGBMClassifier(
                n_estimators=360,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.90,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "boosting_tree",
            True,
        ),
        (
            "LightGBM_balanced",
            LGBMClassifier(
                n_estimators=360,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.90,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "boosting_tree",
            False,
        ),
        (
            "XGBoost_accuracy",
            XGBClassifier(
                n_estimators=360,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=0.90,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "boosting_tree",
            True,
        ),
        (
            "XGBoost_balanced",
            XGBClassifier(
                n_estimators=360,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=0.90,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "boosting_tree",
            False,
        ),
        (
            "MLP_balanced",
            make_pipeline(
                StandardScaler(with_mean=False),
                MLPClassifier(
                    hidden_layer_sizes=(96, 32),
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=350,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
            "neural_net",
            False,
        ),
        (
            "GaussianNB_reference",
            GaussianNB(),
            "bayes",
            False,
        ),
    ]


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray, metric: str) -> tuple[float, dict]:
    best_threshold = 0.50
    best_score = -1.0
    best = {}
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
            best = {
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
    return best_threshold, best


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
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


def performance_rating(accuracy: float, f1: float, recall: float) -> str:
    if accuracy >= 0.80 and f1 >= 0.60 and recall >= 0.60:
        return "A"
    if accuracy >= 0.70 and f1 >= 0.50 and recall >= 0.50:
        return "B"
    if accuracy >= 0.65 and f1 >= 0.40:
        return "C"
    if accuracy >= 0.60:
        return "D"
    return "E"


def run_benchmark() -> pd.DataFrame:
    clean_plus, engineered = load_frames()
    y = clean_plus["is_returned"].astype(int).reset_index(drop=True)
    train_idx, val_idx, test_idx = split_indices(y)
    scale_pos = (y.iloc[train_idx].eq(0).sum() / max(y.iloc[train_idx].eq(1).sum(), 1))

    feature_sets = {
        "clean_plus_safe": (clean_plus, clean_plus_columns(clean_plus)),
        "clean_plus_compact": (clean_plus, compact_columns(clean_plus)),
        "engineered_existing": (engineered, engineered_columns(engineered)),
    }

    rows = []
    for feature_set_name, (df, cols) in feature_sets.items():
        X_all = prepare_matrix(df, cols)
        X_train = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        for model_name, model, family, accuracy_first in model_specs(scale_pos):
            start = time.perf_counter()
            fit_X_train = X_train
            fit_X_val = X_val
            fit_X_test = X_test
            if isinstance(model, GaussianNB):
                fit_X_train = X_train.to_numpy()
                fit_X_val = X_val.to_numpy()
                fit_X_test = X_test.to_numpy()
            model.fit(fit_X_train, y_train)
            y_val_proba = model.predict_proba(fit_X_val)[:, 1]
            y_test_proba = model.predict_proba(fit_X_test)[:, 1]
            train_seconds = time.perf_counter() - start

            acc_t, val_acc = threshold_search(y_val, y_val_proba, "accuracy")
            f1_t, val_f1 = threshold_search(y_val, y_val_proba, "f1")
            test_acc = evaluate_at_threshold(y_test, y_test_proba, acc_t)
            test_f1 = evaluate_at_threshold(y_test, y_test_proba, f1_t)
            default = evaluate_at_threshold(y_test, y_test_proba, 0.50)

            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "model_family": family,
                    "accuracy_first_model": accuracy_first,
                    "input_feature_count": len(cols),
                    "encoded_feature_count": X_all.shape[1],
                    "train_seconds": train_seconds,
                    "accuracy_threshold_from_val": acc_t,
                    "f1_threshold_from_val": f1_t,
                    "val_best_accuracy": val_acc["accuracy"],
                    "val_best_f1": val_f1["f1_score"],
                    "test_accuracy_at_acc_threshold": test_acc["accuracy"],
                    "test_precision_at_acc_threshold": test_acc["precision"],
                    "test_recall_at_acc_threshold": test_acc["recall"],
                    "test_f1_at_acc_threshold": test_acc["f1_score"],
                    "test_cost_at_acc_threshold": test_acc["expected_cost_thb"],
                    "test_auc_roc": test_acc["auc_roc"],
                    "test_avg_precision": test_acc["avg_precision"],
                    "test_accuracy_at_f1_threshold": test_f1["accuracy"],
                    "test_precision_at_f1_threshold": test_f1["precision"],
                    "test_recall_at_f1_threshold": test_f1["recall"],
                    "test_f1_at_f1_threshold": test_f1["f1_score"],
                    "test_cost_at_f1_threshold": test_f1["expected_cost_thb"],
                    "default_accuracy_050": default["accuracy"],
                    "default_precision_050": default["precision"],
                    "default_recall_050": default["recall"],
                    "default_f1_050": default["f1_score"],
                    "default_cost_050": default["expected_cost_thb"],
                    "performance_rating_at_f1_threshold": performance_rating(
                        test_f1["accuracy"], test_f1["f1_score"], test_f1["recall"]
                    ),
                    "performance_rating_at_acc_threshold": performance_rating(
                        test_acc["accuracy"], test_acc["f1_score"], test_acc["recall"]
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["test_accuracy_at_acc_threshold", "test_auc_roc"], ascending=False
    )


def save_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values("test_accuracy_at_f1_threshold", ascending=False).head(12)
    x = np.arange(len(plot_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, (col, label, color) in enumerate(
        [
            ("test_accuracy_at_f1_threshold", "Accuracy at best-F1 threshold", "#1f77b4"),
            ("test_recall_at_f1_threshold", "Recall", "#d62728"),
            ("test_f1_at_f1_threshold", "F1", "#2ca02c"),
        ]
    ):
        ax.bar(x + (i - 1) * width, plot_df[col], width, label=label, color=color)
    labels = plot_df["model_name"] + "\n" + plot_df["feature_set"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Model Architecture Benchmark: Best Practical Tradeoffs")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_architecture_benchmark.png", dpi=160)
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
    best_accuracy = results.iloc[0]
    practical = results.sort_values(
        ["performance_rating_at_f1_threshold", "test_f1_at_f1_threshold", "test_cost_at_f1_threshold"],
        ascending=[True, False, True],
    )
    # Explicit practical sort with ratings is awkward lexicographically; rank manually.
    rating_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    practical = results.assign(_rank=results["performance_rating_at_f1_threshold"].map(rating_rank)).sort_values(
        ["_rank", "test_f1_at_f1_threshold", "test_cost_at_f1_threshold"],
        ascending=[True, False, True],
    )
    best_practical = practical.iloc[0]
    top_cols = [
        "feature_set",
        "model_name",
        "test_accuracy_at_acc_threshold",
        "test_recall_at_acc_threshold",
        "test_f1_at_acc_threshold",
        "test_accuracy_at_f1_threshold",
        "test_recall_at_f1_threshold",
        "test_f1_at_f1_threshold",
        "test_auc_roc",
        "test_cost_at_f1_threshold",
        "performance_rating_at_f1_threshold",
    ]
    content = f"""# Model Architecture Benchmark Report

รอบนี้ใช้ `clean_dataset.csv` และ `df_engineered.csv` โดยสร้าง feature set 3 ชุด แล้วลองหลาย model architecture: Logistic Regression, RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost, MLP และ GaussianNB

## Highest Accuracy

Accuracy สูงสุดแบบ production-safe คือ `{best_accuracy["model_name"]}` บน feature set `{best_accuracy["feature_set"]}` ได้ Accuracy `{best_accuracy["test_accuracy_at_acc_threshold"]:.4f}` แต่ Recall ที่ threshold นั้นคือ `{best_accuracy["test_recall_at_acc_threshold"]:.4f}` และ F1 `{best_accuracy["test_f1_at_acc_threshold"]:.4f}`

## Best Practical Performance

ถ้าดู Accuracy + Recall + F1 + Cost พร้อมกัน ตัวที่น่าใช้ที่สุดคือ `{best_practical["model_name"]}` บน feature set `{best_practical["feature_set"]}` ที่ best-F1 threshold `{best_practical["f1_threshold_from_val"]:.2f}` ได้ Accuracy `{best_practical["test_accuracy_at_f1_threshold"]:.4f}`, Recall `{best_practical["test_recall_at_f1_threshold"]:.4f}`, F1 `{best_practical["test_f1_at_f1_threshold"]:.4f}`, AUC `{best_practical["test_auc_roc"]:.4f}`, Cost `{int(best_practical["test_cost_at_f1_threshold"]):,}` THB และ Rating `{best_practical["performance_rating_at_f1_threshold"]}`

## Top Results

{markdown_table(results.head(15), top_cols)}

## Recommendation

- ถ้าต้องเอาไปใช้จริง ไม่ควรเลือกตัวที่ Accuracy สูงสุดอย่างเดียว เพราะหลายตัวได้ Accuracy สูงจากการทาย `ไม่คืน` เป็นหลัก
- ให้เลือกจาก best-F1 threshold หรือ cost-aware threshold เพราะโจทย์นี้ต้องจับ return จริงให้ได้ ไม่ใช่แค่ accuracy ดูดี
- ถ้าต้องการ Accuracy 80-90% พร้อม Recall/F1 ดี ยังต้องเพิ่ม data signal ใหม่ ไม่ใช่เปลี่ยน model อย่างเดียว
"""
    DOC_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark()
    results.to_csv(OUT_DIR / "model_architecture_benchmark.csv", index=False, encoding="utf-8-sig")
    save_plot(results)
    write_report(results)
    print("[OK] Model architecture benchmark complete.")
    print(
        results[
            [
                "feature_set",
                "model_name",
                "test_accuracy_at_acc_threshold",
                "test_recall_at_acc_threshold",
                "test_f1_at_acc_threshold",
                "test_accuracy_at_f1_threshold",
                "test_recall_at_f1_threshold",
                "test_f1_at_f1_threshold",
                "test_auc_roc",
                "test_cost_at_f1_threshold",
                "performance_rating_at_f1_threshold",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
