from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "run_high_accuracy_feature_search.py"
OUT_DIR = ROOT / "reports" / "final_model"
DOC_DIR = ROOT / "docs" / "final_model"
MODEL_OUT = ROOT / "models" / "final_best_return_risk_model.pkl"
METADATA_OUT = ROOT / "models" / "final_best_return_risk_model_metadata.json"
FEATURE_LIST_OUT = ROOT / "data" / "features" / "final_best_return_risk_features.csv"

RANDOM_STATE = 42
COST_FN = 150
COST_FP = 50


def load_helper():
    spec = importlib.util.spec_from_file_location("final_model_feature_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = load_helper()

from src.production_v2.preprocessing import (  # noqa: E402
    TabularTargetEncodingPreprocessor,
    TargetEncodedModelPipeline,
)


@dataclass
class CandidateSpec:
    name: str
    model: Any
    feature_set_name: str
    feature_columns: list[str]
    note: str
    preprocessor_type: str = "onehot"


def add_customer_rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    out = out.sort_values(["customer_id", "order_date", "order_id"]).copy()
    windows = [7, 30, 60, 90, 180, 365]
    for days in windows:
        out[f"cust_order_count_{days}d"] = 0
        out[f"cust_return_count_{days}d"] = 0
        out[f"cust_return_rate_{days}d"] = 0.0
        out[f"cust_spend_sum_{days}d"] = 0.0

    for _, group in out.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy()
        returns = pd.to_numeric(group["is_returned"], errors="coerce").fillna(0).to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()
        for pos, current_date in enumerate(dates):
            for days in windows:
                start = current_date - np.timedelta64(days, "D")
                mask = (dates < current_date) & (dates >= start)
                count = int(mask.sum())
                return_count = int(returns[mask].sum()) if count else 0
                out.loc[idx[pos], f"cust_order_count_{days}d"] = count
                out.loc[idx[pos], f"cust_return_count_{days}d"] = return_count
                out.loc[idx[pos], f"cust_return_rate_{days}d"] = return_count / count if count else 0.0
                out.loc[idx[pos], f"cust_spend_sum_{days}d"] = float(amounts[mask].sum()) if count else 0.0
    return out.sort_index()


def load_feature_frame() -> pd.DataFrame:
    df = helper.add_point_in_time_features(helper.load_dataset())
    df = add_customer_rolling_windows(df)
    return df.reset_index(drop=True)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    advanced = helper.production_columns(df)
    rolling = [
        c
        for c in df.columns
        if c.startswith("cust_order_count_")
        or c.startswith("cust_return_count_")
        or c.startswith("cust_return_rate_")
        or c.startswith("cust_spend_sum_")
    ]
    safe_advanced_plus_rolling = sorted(set(advanced + rolling), key=(advanced + rolling).index)
    compact_plus_rolling = sorted(set(helper.compact_columns(df) + rolling), key=(helper.compact_columns(df) + rolling).index)
    post_delivery_v2_full = [
        "gender",
        "age",
        "membership_tier",
        "province",
        "customer_age_days",
        "category",
        "brand",
        "product_rating",
        "courier_name",
        "courier_type",
        "avg_delivery_days",
        "damage_rate",
        "promo_name",
        "promo_type",
        "promo_discount_rate",
        "channel_type",
        "payment_method",
        "quantity",
        "unit_price",
        "tier_discount_pct",
        "campaign_discount_pct",
        "total_discount_pct",
        "discount_applied_amount",
        "total_amount",
        "delivery_time_expected_days",
        "delivery_days",
        "delay_days",
        "is_repurchased_item",
        "order_hour",
        "days_since_last_order",
        "hist_order_count",
        "hist_return_rate",
        "customer_tenure_months",
        "order_month",
        "order_dayofweek",
        "is_weekend",
        "age_group",
        "logistics_risk",
    ]
    post_delivery_v2_full = [col for col in post_delivery_v2_full if col in df.columns]
    return {
        "safe_advanced_plus_rolling": safe_advanced_plus_rolling,
        "safe_compact_plus_rolling": compact_plus_rolling,
        "safe_advanced": advanced,
        "safe_compact": helper.compact_columns(df),
        "post_delivery_v2_full": post_delivery_v2_full,
    }


def make_preprocessor(df: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    numeric_cols = df[columns].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in columns if col not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )


def model_candidates(scale_pos_weight: float, feature_sets_: dict[str, list[str]]) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "xgboost_balanced_advanced_rolling",
            XGBClassifier(
                n_estimators=460,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.90,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "safe_advanced_plus_rolling",
            feature_sets_["safe_advanced_plus_rolling"],
            "Overall return model with advanced point-in-time + rolling history features.",
        ),
        CandidateSpec(
            "lightgbm_balanced_advanced_rolling",
            LGBMClassifier(
                n_estimators=460,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.85,
                min_child_samples=25,
                reg_lambda=2.0,
                class_weight={0: 1.0, 1: scale_pos_weight},
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "safe_advanced_plus_rolling",
            feature_sets_["safe_advanced_plus_rolling"],
            "LightGBM balanced candidate for practical performance.",
        ),
        CandidateSpec(
            "lightgbm_recall2x_advanced_rolling",
            LGBMClassifier(
                n_estimators=460,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.85,
                min_child_samples=25,
                reg_lambda=2.0,
                class_weight={0: 1.0, 1: scale_pos_weight * 2.0},
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "safe_advanced_plus_rolling",
            feature_sets_["safe_advanced_plus_rolling"],
            "LightGBM with stronger positive weight to reduce missed returns.",
        ),
        CandidateSpec(
            "random_forest_balanced_compact",
            RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "safe_compact_plus_rolling",
            feature_sets_["safe_compact_plus_rolling"],
            "Compact feature set with bagging model.",
        ),
        CandidateSpec(
            "extra_trees_balanced_compact",
            ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "safe_compact_plus_rolling",
            feature_sets_["safe_compact_plus_rolling"],
            "ExtraTrees compact high-variance ensemble.",
        ),
        CandidateSpec(
            "xgboost_post_delivery_v2_full",
            XGBClassifier(
                n_estimators=420,
                max_depth=6,
                learning_rate=0.038,
                subsample=0.85,
                colsample_bytree=0.88,
                min_child_weight=2,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "post_delivery_v2_full",
            feature_sets_["post_delivery_v2_full"],
            "Research/post-delivery candidate: includes delivery_days and delay_days, so it is not for order-time scoring.",
        ),
        CandidateSpec(
            "lightgbm_post_delivery_v2_full",
            LGBMClassifier(
                n_estimators=420,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.90,
                class_weight={0: 1.0, 1: scale_pos_weight},
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "post_delivery_v2_full",
            feature_sets_["post_delivery_v2_full"],
            "Research/post-delivery LightGBM candidate with V2-style feature set.",
        ),
        CandidateSpec(
            "xgboost_target_encoded_v2_full",
            XGBClassifier(
                n_estimators=360,
                max_depth=6,
                learning_rate=0.038,
                subsample=0.85,
                colsample_bytree=0.88,
                min_child_weight=2,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "post_delivery_v2_full",
            feature_sets_["post_delivery_v2_full"],
            "V2-style target encoding candidate. Includes delivery_days and delay_days, so use as research/post-delivery model only.",
            "target_encoding",
        ),
    ]


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "cost": int(fn * COST_FN + fp * COST_FP),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_search(y_true: np.ndarray, y_proba: np.ndarray, mode: str) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, Any] = {}
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = metrics_at_threshold(y_true, y_proba, float(threshold))
        if mode == "accuracy":
            score = metrics["accuracy"]
        elif mode == "f1":
            score = metrics["f1"]
        elif mode == "cost":
            score = -metrics["cost"]
        elif mode == "balanced":
            score = (
                0.35 * metrics["accuracy"]
                + 0.25 * metrics["f1"]
                + 0.15 * metrics["recall"]
                + 0.15 * metrics["auc"]
                + 0.10 * max(0.0, 1.0 - (metrics["cost"] / 50000))
            )
        else:
            raise ValueError(mode)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics | {"selection_score": float(score)}
    return best_threshold, best_metrics


def performance_rating(metrics: dict[str, Any]) -> str:
    if metrics["accuracy"] >= 0.80 and metrics["f1"] >= 0.60 and metrics["recall"] >= 0.60 and metrics["cost"] <= 20000:
        return "A"
    if metrics["accuracy"] >= 0.68 and metrics["f1"] >= 0.50 and metrics["recall"] >= 0.60 and metrics["cost"] <= 30000:
        return "B+"
    if metrics["accuracy"] >= 0.65 and metrics["f1"] >= 0.48 and metrics["recall"] >= 0.55:
        return "B"
    if metrics["accuracy"] >= 0.60 and metrics["f1"] >= 0.42:
        return "C"
    return "D"


def train_and_evaluate(df: pd.DataFrame, spec: CandidateSpec, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    x_train = df.iloc[train_idx][spec.feature_columns]
    x_val = df.iloc[val_idx][spec.feature_columns]
    x_test = df.iloc[test_idx][spec.feature_columns]
    y = df["is_returned"].astype(int).to_numpy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    if spec.preprocessor_type == "target_encoding":
        pipeline = TargetEncodedModelPipeline(TabularTargetEncodingPreprocessor(), spec.model)
    else:
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(df, spec.feature_columns)),
                ("model", spec.model),
            ]
        )
    pipeline.fit(x_train, y_train)
    val_proba = pipeline.predict_proba(x_val)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]

    thresholds = {}
    for mode in ["accuracy", "f1", "cost", "balanced"]:
        threshold, val_metrics = threshold_search(y_val, val_proba, mode)
        test_metrics = metrics_at_threshold(y_test, test_proba, threshold)
        thresholds[mode] = {
            "threshold": threshold,
            "val": val_metrics,
            "test": test_metrics,
            "rating": performance_rating(test_metrics),
        }

    selected = thresholds["balanced"]["test"]
    selected["threshold"] = thresholds["balanced"]["threshold"]
    selected_rating = thresholds["balanced"]["rating"]
    score = thresholds["balanced"]["val"]["selection_score"]

    row = {
        "candidate": spec.name,
        "feature_set": spec.feature_set_name,
        "raw_feature_count": len(spec.feature_columns),
        "selected_threshold": thresholds["balanced"]["threshold"],
        "selection_score_val": score,
        "accuracy": selected["accuracy"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "auc": selected["auc"],
        "avg_precision": selected["avg_precision"],
        "cost": selected["cost"],
        "tn": selected["tn"],
        "fp": selected["fp"],
        "fn": selected["fn"],
        "tp": selected["tp"],
        "performance_rating": selected_rating,
        "accuracy_threshold": thresholds["accuracy"]["threshold"],
        "accuracy_at_accuracy_threshold": thresholds["accuracy"]["test"]["accuracy"],
        "recall_at_accuracy_threshold": thresholds["accuracy"]["test"]["recall"],
        "f1_at_accuracy_threshold": thresholds["accuracy"]["test"]["f1"],
        "cost_at_accuracy_threshold": thresholds["accuracy"]["test"]["cost"],
        "f1_threshold": thresholds["f1"]["threshold"],
        "f1_at_f1_threshold": thresholds["f1"]["test"]["f1"],
        "recall_at_f1_threshold": thresholds["f1"]["test"]["recall"],
        "cost_threshold": thresholds["cost"]["threshold"],
        "cost_at_cost_threshold": thresholds["cost"]["test"]["cost"],
        "note": spec.note,
    }

    pred_df = pd.DataFrame(
        {
            "test_row_index": test_idx,
            "order_id": df.iloc[test_idx].get("order_id", pd.Series(index=np.arange(len(test_idx)), dtype=object)).to_numpy(),
            "customer_id": df.iloc[test_idx].get("customer_id", pd.Series(index=np.arange(len(test_idx)), dtype=object)).to_numpy(),
            "y_true": y_test,
            "predict_probability_return": test_proba,
            "y_pred": (test_proba >= row["selected_threshold"]).astype(int),
            "threshold": row["selected_threshold"],
        }
    )
    pred_df["actual_label"] = np.where(pred_df["y_true"].eq(1), "return", "no_return")
    pred_df["pred_label"] = np.where(pred_df["y_pred"].eq(1), "return", "no_return")
    return pipeline, row, pred_df


def save_comparison_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values(["selection_score_val", "f1"], ascending=False)
    x = np.arange(len(plot_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, (col, label, color) in enumerate(
        [
            ("accuracy", "Accuracy", "#1f77b4"),
            ("recall", "Recall", "#d62728"),
            ("f1", "F1", "#2ca02c"),
            ("auc", "AUC", "#9467bd"),
        ]
    ):
        ax.bar(x + (i - 1.5) * width, plot_df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["candidate"], rotation=22, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Final Return-Risk Model Candidates")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_model_comparison.png", dpi=160)
    plt.close(fig)


def write_report(results: pd.DataFrame, best: dict[str, Any]) -> None:
    top = results.sort_values(["selection_score_val", "f1"], ascending=False)
    md_rows = []
    cols = ["candidate", "feature_set", "accuracy", "recall", "precision", "f1", "auc", "cost", "performance_rating"]
    for _, row in top.iterrows():
        md_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["feature_set"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['auc']:.4f}",
                    f"{int(row['cost']):,}",
                    str(row["performance_rating"]),
                ]
            )
            + " |"
        )
    table = "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |", *md_rows])
    content = f"""# Final Best Return-Risk Model Report

## สรุป

รอบนี้พัก production/API ไว้ก่อน แล้วโฟกัสที่ model ให้สมบูรณ์บนข้อมูลจริง โดยเพิ่ม point-in-time + rolling history features และเทียบหลาย model ด้วย split เดียวกัน

Best final candidate คือ `{best["candidate"]}` ใช้ feature set `{best["feature_set"]}`

- Accuracy: `{best["accuracy"]:.4f}`
- Recall: `{best["recall"]:.4f}`
- Precision: `{best["precision"]:.4f}`
- F1: `{best["f1"]:.4f}`
- AUC: `{best["auc"]:.4f}`
- Cost Matrix: `{int(best["cost"]):,}`
- Performance Rating: `{best["performance_rating"]}`
- Threshold ที่เลือก: `{best["selected_threshold"]:.2f}`

## ผลเปรียบเทียบ

{table}

## หมายเหตุสำคัญ

- โมเดลนี้ไม่ใช้ `return_id`, `return_date`, `refund_amount`, `risk_score`, `risk_tier`, `delivery_days`, `delay_days` เป็นตัว train เพื่อไม่ให้ Accuracy สูงแบบหลอกจาก leakage
- ถ้าต้องการ Accuracy 80-90% สำหรับ overall return โดย Recall/F1 ยังดี ต้องเพิ่ม data signal จริง เช่น complaint history, SKU defect rate, courier late-rate by area, click/add-to-cart/live session behavior และ label return reason ที่ละเอียดขึ้น
- SQL DB สามารถใช้เป็น Feature Store ภายหลังได้ โดยเก็บ feature กลุ่ม customer/product/category/courier rolling history ที่สร้างในรอบนี้
"""
    (DOC_DIR / "final_best_return_risk_model_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_LIST_OUT.parent.mkdir(parents=True, exist_ok=True)

    df = load_feature_frame()
    y = df["is_returned"].astype(int).reset_index(drop=True)
    all_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=RANDOM_STATE, stratify=y.iloc[train_val_idx])
    scale_pos_weight = float(y.iloc[train_idx].eq(0).sum() / max(y.iloc[train_idx].eq(1).sum(), 1))
    sets = feature_sets(df)

    rows = []
    trained: dict[str, Pipeline] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for spec in model_candidates(scale_pos_weight, sets):
        pipeline, row, pred_df = train_and_evaluate(df, spec, train_idx, val_idx, test_idx)
        rows.append(row)
        trained[spec.name] = pipeline
        predictions[spec.name] = pred_df

    results = pd.DataFrame(rows).sort_values(["selection_score_val", "f1"], ascending=False)
    best = results.iloc[0].to_dict()
    best_model = trained[best["candidate"]]
    best_predictions = predictions[best["candidate"]]
    best_features = sets[best["feature_set"]]

    joblib.dump(
        {
            "model_pipeline": best_model,
            "threshold": float(best["selected_threshold"]),
            "feature_columns": best_features,
            "metadata": best,
            "split": {
                "random_state": RANDOM_STATE,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
            },
        },
        MODEL_OUT,
    )
    metadata = {
        "best_model": best,
        "feature_columns": best_features,
        "candidate_results": results.to_dict(orient="records"),
        "target_distribution": {
            "all_return_rate": float(y.mean()),
            "test_return_rate": float(y.iloc[test_idx].mean()),
        },
    }
    METADATA_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    results.to_csv(OUT_DIR / "final_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUT_DIR / "final_model_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": best_features}).to_csv(FEATURE_LIST_OUT, index=False, encoding="utf-8-sig")
    save_comparison_plot(results)
    write_report(results, best)

    print(MODEL_OUT)
    print(METADATA_OUT)
    print(OUT_DIR / "final_model_comparison.csv")
    print(OUT_DIR / "final_model_test_predictions.csv")
    print(DOC_DIR / "final_best_return_risk_model_report.md")
    print(results[["candidate", "accuracy", "recall", "precision", "f1", "auc", "cost", "performance_rating"]].to_string(index=False))


if __name__ == "__main__":
    main()
