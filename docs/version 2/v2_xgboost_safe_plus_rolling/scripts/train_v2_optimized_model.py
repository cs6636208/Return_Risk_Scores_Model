from __future__ import annotations

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.production_v2.preprocessing import (  # noqa: E402
    TabularTargetEncodingPreprocessor,
    TargetEncodedModelPipeline,
)


RANDOM_STATE = 42
COST_FN = 150
COST_FP = 50

DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
OUT_DIR = ROOT / "reports" / "model_evaluation_v2_optimized"
DOC_DIR = ROOT / "docs" / "version 2" / "optimized_model"
MODEL_OUT = ROOT / "models" / "best_model_v2_optimized.pkl"
METADATA_OUT = ROOT / "models" / "best_model_v2_optimized_metadata.json"
FEATURE_SET_OUT = ROOT / "data" / "features" / "v2_optimized_used_features.csv"
TRAIN_TEST_OUT = ROOT / "data" / "features" / "train_test_sets_v2_optimized.pkl"


V2_ORIGINAL_FEATURES = [
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


@dataclass
class CandidateSpec:
    name: str
    model: Any
    feature_columns: list[str]
    feature_policy: str
    production_safe: bool


def load_v2_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df = df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)

    df["customer_tenure_months"] = ((df["order_date"] - df["registration_date"]).dt.days / 30).fillna(0)
    df["order_month"] = df["order_date"].dt.month.fillna(0).astype(int)
    df["order_dayofweek"] = df["order_date"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    df["age_group"] = pd.cut(
        pd.to_numeric(df["age"], errors="coerce").fillna(0),
        bins=[0, 20, 30, 40, 50, 120],
        labels=["<20", "20-30", "30-40", "40-50", ">50"],
        include_lowest=True,
    ).astype(str)
    df["is_fragile"] = pd.to_numeric(df.get("is_fragile", 0), errors="coerce").fillna(0).astype(int)
    df["damage_rate"] = pd.to_numeric(df.get("damage_rate", 0), errors="coerce").fillna(0)
    df["logistics_risk"] = df["damage_rate"] * df["is_fragile"]

    df["is_cod"] = df["payment_method"].eq("COD").astype(int)
    df["is_high_discount"] = pd.to_numeric(df["total_discount_pct"], errors="coerce").fillna(0).gt(0.20).astype(int)
    df["low_rating_alert"] = pd.to_numeric(df["product_rating"], errors="coerce").fillna(5).lt(4.0).astype(int)
    df["discount_amount_ratio"] = (
        pd.to_numeric(df["discount_applied_amount"], errors="coerce").fillna(0)
        / (pd.to_numeric(df["unit_price"], errors="coerce").fillna(0) * pd.to_numeric(df["quantity"], errors="coerce").fillna(1)).replace(0, np.nan)
    ).fillna(0)
    df["category_payment"] = df["category"].astype(str) + "_" + df["payment_method"].astype(str)
    df["category_channel"] = df["category"].astype(str) + "_" + df["channel_type"].astype(str)
    df["province_payment"] = df["province"].astype(str) + "_" + df["payment_method"].astype(str)

    for days in [30, 60, 90, 180, 365]:
        df[f"hist_order_count_{days}d"] = 0
        df[f"hist_return_count_{days}d"] = 0
        df[f"hist_return_rate_{days}d"] = 0.0
        df[f"hist_spend_sum_{days}d"] = 0.0

    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy()
        returns = pd.to_numeric(group["is_returned"], errors="coerce").fillna(0).to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()
        for pos, current_date in enumerate(dates):
            for days in [30, 60, 90, 180, 365]:
                start = current_date - np.timedelta64(days, "D")
                mask = (dates < current_date) & (dates >= start)
                order_count = int(mask.sum())
                return_count = int(returns[mask].sum()) if order_count else 0
                df.loc[idx[pos], f"hist_order_count_{days}d"] = order_count
                df.loc[idx[pos], f"hist_return_count_{days}d"] = return_count
                df.loc[idx[pos], f"hist_return_rate_{days}d"] = return_count / order_count if order_count else 0.0
                df.loc[idx[pos], f"hist_spend_sum_{days}d"] = float(amounts[mask].sum()) if order_count else 0.0

    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    v2_full = existing(df, V2_ORIGINAL_FEATURES)
    order_time_safe = [c for c in v2_full if c not in {"delivery_days", "delay_days"}]
    compact_core = existing(
        df,
        [
            "hist_return_rate",
            "hist_order_count",
            "days_since_last_order",
            "is_repurchased_item",
            "category",
            "brand",
            "province",
            "payment_method",
            "channel_type",
            "product_rating",
            "total_amount",
            "total_discount_pct",
            "delivery_time_expected_days",
            "order_hour",
            "customer_age_days",
            "age",
            "logistics_risk",
            "is_cod",
            "is_high_discount",
            "low_rating_alert",
        ],
    )
    rolling = [
        col
        for col in df.columns
        if col.startswith("hist_order_count_")
        or col.startswith("hist_return_count_")
        or col.startswith("hist_return_rate_")
        or col.startswith("hist_spend_sum_")
    ]
    interactions = existing(df, ["discount_amount_ratio", "category_payment", "category_channel", "province_payment"])
    return {
        "v2_full_38": v2_full,
        "v2_order_time_safe_36": order_time_safe,
        "v2_compact_core": compact_core,
        "v2_full_plus_rolling": existing(df, v2_full + rolling + interactions),
        "v2_safe_plus_rolling": existing(df, order_time_safe + rolling + interactions),
    }


def target_encode_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]


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
    best = {}
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
                0.40 * metrics["accuracy"]
                + 0.25 * metrics["f1"]
                + 0.15 * metrics["recall"]
                + 0.10 * metrics["auc"]
                + 0.10 * max(0.0, 1.0 - (metrics["cost"] / 50000))
            )
        else:
            raise ValueError(mode)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best = metrics | {"selection_score": float(score)}
    return best_threshold, best


def performance_rating(metrics: dict[str, Any]) -> str:
    if metrics["accuracy"] >= 0.80 and metrics["f1"] >= 0.60 and metrics["recall"] >= 0.60 and metrics["cost"] <= 20000:
        return "A"
    if metrics["accuracy"] >= 0.68 and metrics["f1"] >= 0.52 and metrics["recall"] >= 0.60 and metrics["cost"] <= 28000:
        return "B+"
    if metrics["accuracy"] >= 0.65 and metrics["f1"] >= 0.48 and metrics["recall"] >= 0.55:
        return "B"
    if metrics["accuracy"] >= 0.60 and metrics["f1"] >= 0.42:
        return "C"
    return "D"


def candidate_specs(feature_sets: dict[str, list[str]], scale_pos_weight: float) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "v2_xgboost_full_38",
            XGBClassifier(n_estimators=360, max_depth=6, learning_rate=0.038, subsample=0.85, colsample_bytree=0.88, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0),
            feature_sets["v2_full_38"],
            "V2 original 38 features with target encoding; includes delivery_days/delay_days.",
            False,
        ),
        CandidateSpec(
            "v2_lightgbm_full_38",
            LGBMClassifier(n_estimators=360, learning_rate=0.035, num_leaves=31, subsample=0.90, colsample_bytree=0.90, class_weight={0: 1.0, 1: scale_pos_weight}, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
            feature_sets["v2_full_38"],
            "V2 original 38 features with LightGBM.",
            False,
        ),
        CandidateSpec(
            "v2_random_forest_full_38",
            RandomForestClassifier(n_estimators=650, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            feature_sets["v2_full_38"],
            "V2 original 38 features with RandomForest.",
            False,
        ),
        CandidateSpec(
            "v2_xgboost_order_time_safe_36",
            XGBClassifier(n_estimators=360, max_depth=5, learning_rate=0.038, subsample=0.85, colsample_bytree=0.88, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0),
            feature_sets["v2_order_time_safe_36"],
            "V2 features excluding delivery_days/delay_days for order-time use.",
            True,
        ),
        CandidateSpec(
            "v2_xgboost_full_plus_rolling",
            XGBClassifier(n_estimators=420, max_depth=5, learning_rate=0.03, subsample=0.90, colsample_bytree=0.85, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0),
            feature_sets["v2_full_plus_rolling"],
            "V2 original features plus customer rolling history windows.",
            False,
        ),
        CandidateSpec(
            "v2_xgboost_safe_plus_rolling",
            XGBClassifier(n_estimators=420, max_depth=5, learning_rate=0.03, subsample=0.90, colsample_bytree=0.85, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0),
            feature_sets["v2_safe_plus_rolling"],
            "V2 order-time-safe features plus customer rolling history windows.",
            True,
        ),
        CandidateSpec(
            "v2_extra_trees_compact_core",
            ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            feature_sets["v2_compact_core"],
            "Compact V2 core features to reduce resource cost.",
            True,
        ),
    ]


def train_candidate(df: pd.DataFrame, spec: CandidateSpec, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[TargetEncodedModelPipeline, dict[str, Any], pd.DataFrame]:
    y = df["is_returned"].astype(int).to_numpy()
    x_train = df.iloc[train_idx][spec.feature_columns]
    x_val = df.iloc[val_idx][spec.feature_columns]
    x_test = df.iloc[test_idx][spec.feature_columns]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    preprocessor = TabularTargetEncodingPreprocessor(target_encode_columns=target_encode_columns(df, spec.feature_columns))
    pipeline = TargetEncodedModelPipeline(preprocessor, spec.model)
    pipeline.fit(x_train, y_train)

    val_proba = pipeline.predict_proba(x_val)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    modes = {}
    for mode in ["accuracy", "f1", "cost", "balanced"]:
        threshold, val_metrics = threshold_search(y_val, val_proba, mode)
        test_metrics = metrics_at_threshold(y_test, test_proba, threshold)
        modes[mode] = {"threshold": threshold, "val": val_metrics, "test": test_metrics}

    selected = modes["balanced"]["test"]
    selected_threshold = modes["balanced"]["threshold"]
    row = {
        "candidate": spec.name,
        "production_safe": spec.production_safe,
        "raw_feature_count": len(spec.feature_columns),
        "encoded_feature_count": len(pipeline.preprocessor.feature_columns),
        "selected_threshold": selected_threshold,
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
        "performance_rating": performance_rating(selected),
        "selection_score_val": modes["balanced"]["val"]["selection_score"],
        "accuracy_threshold": modes["accuracy"]["threshold"],
        "accuracy_at_accuracy_threshold": modes["accuracy"]["test"]["accuracy"],
        "recall_at_accuracy_threshold": modes["accuracy"]["test"]["recall"],
        "f1_at_accuracy_threshold": modes["accuracy"]["test"]["f1"],
        "cost_at_accuracy_threshold": modes["accuracy"]["test"]["cost"],
        "f1_threshold": modes["f1"]["threshold"],
        "f1_at_f1_threshold": modes["f1"]["test"]["f1"],
        "recall_at_f1_threshold": modes["f1"]["test"]["recall"],
        "cost_threshold": modes["cost"]["threshold"],
        "cost_at_cost_threshold": modes["cost"]["test"]["cost"],
        "feature_policy": spec.feature_policy,
    }
    predictions = pd.DataFrame(
        {
            "test_row_index": test_idx,
            "order_id": df.iloc[test_idx]["order_id"].to_numpy(),
            "customer_id": df.iloc[test_idx]["customer_id"].to_numpy(),
            "y_true": y_test,
            "actual_label": np.where(y_test == 1, "return", "no_return"),
            "predict_probability_return": test_proba,
            "selected_threshold": selected_threshold,
            "y_pred": (test_proba >= selected_threshold).astype(int),
        }
    )
    predictions["pred_label"] = np.where(predictions["y_pred"].eq(1), "return", "no_return")
    return pipeline, row, predictions


def save_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values(["selection_score_val", "f1"], ascending=False)
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(plot_df))
    width = 0.18
    for i, (col, label, color) in enumerate(
        [("accuracy", "Accuracy", "#1f77b4"), ("recall", "Recall", "#d62728"), ("f1", "F1", "#2ca02c"), ("auc", "AUC", "#9467bd")]
    ):
        ax.bar(x + (i - 1.5) * width, plot_df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["candidate"], rotation=24, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("V2 Optimized Model Candidates")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v2_optimized_model_comparison.png", dpi=160)
    plt.close(fig)


def write_report(results: pd.DataFrame, best: dict[str, Any]) -> None:
    rows = []
    for _, row in results.sort_values(["selection_score_val", "f1"], ascending=False).iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    "Yes" if row["production_safe"] else "No",
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
    table = "\n".join(
        [
            "| Candidate | Order-time safe | Accuracy | Recall | Precision | F1 | AUC | Cost | Rating |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
        ]
    )
    content = f"""# V2 Optimized Model Report

## สรุป

รอบนี้โฟกัสเฉพาะ **Model V2** โดยใช้ `clean_dataset.csv` เป็นฐาน, target encoding แบบ V2, และทดลองเพิ่ม/ลด feature ภายในกรอบ V2

Best V2 optimized model คือ `{best["candidate"]}`

- Accuracy: `{best["accuracy"]:.4f}`
- Recall: `{best["recall"]:.4f}`
- Precision: `{best["precision"]:.4f}`
- F1: `{best["f1"]:.4f}`
- AUC: `{best["auc"]:.4f}`
- Cost Matrix: `{int(best["cost"]):,}`
- Performance Rating: `{best["performance_rating"]}`
- Threshold: `{best["selected_threshold"]:.2f}`
- Order-time safe: `{"Yes" if best["production_safe"] else "No"}`

## Comparison

{table}

## หมายเหตุ

- Candidate ที่ `Order-time safe = No` ใช้ `delivery_days` / `delay_days` เหมือน V2 เดิม จึงเหมาะกับ project experiment หรือ post-delivery scoring มากกว่า real-time order scoring
- Candidate ที่ `Order-time safe = Yes` ตัด field หลังเหตุการณ์ออก เหมาะกว่าเมื่อจะเอา SQL DB ไปทำ Feature Store ภายหลัง
- ถ้าต้องการ Accuracy 80-90% พร้อม Recall/F1 ดี ต้องเพิ่ม data signal ใหม่ ไม่ใช่แค่จูน model เช่น complaint history, SKU defect rate, courier late-rate by area, click/add-to-cart/session behavior
"""
    (DOC_DIR / "v2_optimized_model_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    for path in [OUT_DIR, DOC_DIR, MODEL_OUT.parent, FEATURE_SET_OUT.parent]:
        path.mkdir(parents=True, exist_ok=True)

    df = load_v2_frame()
    y = df["is_returned"].astype(int).reset_index(drop=True)
    all_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=RANDOM_STATE, stratify=y.iloc[train_val_idx])
    scale_pos_weight = float(y.iloc[train_idx].eq(0).sum() / max(y.iloc[train_idx].eq(1).sum(), 1))
    sets = build_feature_sets(df)

    trained: dict[str, TargetEncodedModelPipeline] = {}
    predictions: dict[str, pd.DataFrame] = {}
    rows = []
    for spec in candidate_specs(sets, scale_pos_weight):
        pipeline, row, pred = train_candidate(df, spec, train_idx, val_idx, test_idx)
        rows.append(row)
        trained[spec.name] = pipeline
        predictions[spec.name] = pred

    results = pd.DataFrame(rows).sort_values(["selection_score_val", "f1"], ascending=False)
    best = results.iloc[0].to_dict()
    best_pipeline = trained[best["candidate"]]
    best_features = candidate_specs(sets, scale_pos_weight)[[s.name for s in candidate_specs(sets, scale_pos_weight)].index(best["candidate"])].feature_columns
    best_predictions = predictions[best["candidate"]]

    joblib.dump(
        {
            "model_pipeline": best_pipeline,
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
    results.to_csv(OUT_DIR / "v2_optimized_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUT_DIR / "v2_optimized_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": best_features}).to_csv(FEATURE_SET_OUT, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "X_test_raw": df.iloc[test_idx][best_features],
            "y_test": y.iloc[test_idx].to_numpy(),
            "feature_names": best_features,
            "test_index": test_idx,
        },
        TRAIN_TEST_OUT,
    )
    save_plot(results)
    write_report(results, best)

    # Mirror key files into docs/version 2 for project submission packaging.
    results.to_csv(DOC_DIR / "v2_optimized_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(DOC_DIR / "v2_optimized_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": best_features}).to_csv(DOC_DIR / "v2_optimized_used_features.csv", index=False, encoding="utf-8-sig")

    print(MODEL_OUT)
    print(METADATA_OUT)
    print(OUT_DIR / "v2_optimized_model_comparison.csv")
    print(OUT_DIR / "v2_optimized_test_predictions.csv")
    print(DOC_DIR / "v2_optimized_model_report.md")
    print(results[["candidate", "production_safe", "accuracy", "recall", "precision", "f1", "auc", "cost", "performance_rating"]].to_string(index=False))


if __name__ == "__main__":
    main()
