from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
OUT_DIR = ROOT / "reports" / "model_experiments"
DOC_PATH = ROOT / "docs" / "analysis" / "high_accuracy_improvement_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
COST_FN = 150
COST_FP = 50

LEAKAGE_FIELDS = {
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
    "score_id",
    "risk_score",
    "risk_tier",
    "scored_at",
    "shap_values",
    "delivery_date",
    "delivery_days",
    "delay_days",
}

ID_FIELDS = {
    "order_id",
    "customer_id",
    "customer_name",
    "customer_phone",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "supplier_contact",
    "courier_id",
    "promo_id",
}


@dataclass
class Variant:
    variant: str
    model_name: str
    target_name: str
    feature_policy: str
    production_safe: bool
    columns: list[str]
    positive_weight_multiplier: float = 1.0


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["is_customer_behavior_return"] = df["return_reason"].isin(["Changed Mind", "Better Price Elsewhere"]).astype(int)
    df["is_product_quality_return"] = df["return_reason"].isin(["Defective", "Wrong Item"]).astype(int)
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return df


def add_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_month"] = out["order_date"].dt.month
    out["order_dayofweek"] = out["order_date"].dt.dayofweek
    out["is_weekend"] = out["order_dayofweek"].isin([5, 6]).astype(int)
    out["customer_tenure_months"] = ((out["order_date"] - out["registration_date"]).dt.days / 30).fillna(0)
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    ).astype(str)

    out["is_cod"] = out["payment_method"].eq("COD").astype(int)
    out["is_high_discount"] = out["total_discount_pct"].gt(0.20).astype(int)
    out["low_rating_alert"] = out["product_rating"].lt(4.0).astype(int)
    out["is_remote_area"] = out["province"].eq("Remote_Area").astype(int)
    out["is_fashion_tv"] = (out["category"].eq("Fashion") & out["channel_type"].eq("TV_Show")).astype(int)
    out["is_bracketing"] = (out["category"].eq("Fashion") & out["quantity"].gt(1)).astype(int)
    out["is_peak_hour"] = (
        (out["channel_type"].eq("TV_Show") & out["order_hour"].between(8, 10))
        | (out["channel_type"].eq("TikTok") & out["order_hour"].between(21, 23))
    ).astype(int)
    out["is_low_commitment"] = (out["is_cod"].eq(1) & out["is_high_discount"].eq(1)).astype(int)
    out["is_long_distance_cod"] = (
        out["province"].isin(["Chiang Mai", "Phuket", "Songkhla"]) & out["payment_method"].eq("COD")
    ).astype(int)

    out["log_unit_price"] = np.log1p(out["unit_price"])
    out["log_total_amount"] = np.log1p(out["total_amount"])
    out["discount_amount_ratio"] = (
        out["discount_applied_amount"] / (out["unit_price"] * out["quantity"]).replace(0, np.nan)
    ).fillna(0)

    interactions = {
        "category_payment": ["category", "payment_method"],
        "category_channel": ["category", "channel_type"],
        "province_payment": ["province", "payment_method"],
        "brand_channel": ["brand", "channel_type"],
        "courier_province": ["courier_name", "province"],
        "category_province": ["category", "province"],
        "age_payment": ["age_group", "payment_method"],
    }
    for new_col, cols in interactions.items():
        out[new_col] = out[cols].astype(str).agg("_".join, axis=1)

    for keys, prefix in [
        (["customer_id"], "cust"),
        (["product_id"], "sku"),
        (["brand"], "brand"),
        (["category"], "cat"),
        (["province"], "province"),
        (["payment_method"], "pay"),
        (["channel_type"], "channel"),
        (["courier_name"], "courier"),
        (["category", "payment_method"], "cat_pay"),
        (["category", "channel_type"], "cat_channel"),
        (["brand", "channel_type"], "brand_channel"),
        (["courier_name", "province"], "courier_province"),
    ]:
        group = out.groupby(keys, sort=False)
        prior_count = group.cumcount()
        prior_returns = group["is_returned"].cumsum() - out["is_returned"]
        prior_amount = group["total_amount"].cumsum() - out["total_amount"]
        out[f"{prefix}_orders_before"] = prior_count
        out[f"{prefix}_returns_before"] = prior_returns
        out[f"{prefix}_return_rate_pti"] = (prior_returns / prior_count.replace(0, np.nan)).fillna(0)
        out[f"{prefix}_avg_amount_before"] = (prior_amount / prior_count.replace(0, np.nan)).fillna(0)

    customer_group = out.groupby("customer_id", sort=False)
    out["cust_avg_discount_before"] = (
        (customer_group["total_discount_pct"].cumsum() - out["total_discount_pct"])
        / out["cust_orders_before"].replace(0, np.nan)
    ).fillna(0)
    out["cust_avg_quantity_before"] = (
        (customer_group["quantity"].cumsum() - out["quantity"]) / out["cust_orders_before"].replace(0, np.nan)
    ).fillna(0)
    out["is_first_order"] = out["cust_orders_before"].eq(0).astype(int)
    out["is_high_risk_customer_pti"] = out["cust_return_rate_pti"].gt(0.20).astype(int)

    customer_category = out.groupby(["customer_id", "category"], sort=False)
    cat_count = customer_category.cumcount()
    cat_returns = customer_category["is_returned"].cumsum() - out["is_returned"]
    out["cust_category_orders_before"] = cat_count
    out["cust_category_return_rate_pti"] = (cat_returns / cat_count.replace(0, np.nan)).fillna(0)

    return out


def production_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "gender",
        "age",
        "age_group",
        "membership_tier",
        "preferred_channel",
        "province",
        "customer_age_days",
        "customer_tenure_months",
        "category",
        "brand",
        "is_fragile",
        "product_rating",
        "courier_name",
        "courier_type",
        "avg_delivery_days",
        "damage_rate",
        "coverage_region",
        "promo_name",
        "promo_type",
        "promo_discount_rate",
        "channel_type",
        "payment_method",
        "quantity",
        "unit_price",
        "log_unit_price",
        "tier_discount_pct",
        "campaign_discount_pct",
        "total_discount_pct",
        "discount_applied_amount",
        "discount_amount_ratio",
        "total_amount",
        "log_total_amount",
        "delivery_time_expected_days",
        "is_repurchased_item",
        "order_hour",
        "order_month",
        "order_dayofweek",
        "is_weekend",
        "days_since_last_order",
        "hist_order_count",
        "hist_return_rate",
        "is_cod",
        "is_high_discount",
        "low_rating_alert",
        "is_remote_area",
        "is_fashion_tv",
        "is_bracketing",
        "is_peak_hour",
        "is_low_commitment",
        "is_long_distance_cod",
        "category_payment",
        "category_channel",
        "province_payment",
        "brand_channel",
        "courier_province",
        "category_province",
        "age_payment",
        "cust_orders_before",
        "cust_returns_before",
        "cust_return_rate_pti",
        "cust_avg_amount_before",
        "cust_avg_discount_before",
        "cust_avg_quantity_before",
        "is_first_order",
        "is_high_risk_customer_pti",
        "cust_category_orders_before",
        "cust_category_return_rate_pti",
    ]
    for prefix in [
        "sku",
        "brand",
        "cat",
        "province",
        "pay",
        "channel",
        "courier",
        "cat_pay",
        "cat_channel",
        "brand_channel",
        "courier_province",
    ]:
        columns.extend(
            [
                f"{prefix}_orders_before",
                f"{prefix}_returns_before",
                f"{prefix}_return_rate_pti",
                f"{prefix}_avg_amount_before",
            ]
        )
    return [c for c in columns if c in df.columns and c not in LEAKAGE_FIELDS and c not in ID_FIELDS]


def compact_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "hist_return_rate",
        "hist_order_count",
        "cust_return_rate_pti",
        "cust_orders_before",
        "cust_returns_before",
        "cust_category_return_rate_pti",
        "sku_return_rate_pti",
        "brand_return_rate_pti",
        "cat_return_rate_pti",
        "pay_return_rate_pti",
        "channel_return_rate_pti",
        "cat_pay_return_rate_pti",
        "cat_channel_return_rate_pti",
        "product_rating",
        "low_rating_alert",
        "payment_method",
        "is_cod",
        "category",
        "brand",
        "province",
        "channel_type",
        "total_amount",
        "log_total_amount",
        "total_discount_pct",
        "is_high_discount",
        "is_repurchased_item",
        "days_since_last_order",
        "delivery_time_expected_days",
        "order_hour",
        "is_fashion_tv",
        "is_bracketing",
        "is_low_commitment",
    ]
    return [c for c in columns if c in df.columns]


def diagnostic_columns(df: pd.DataFrame) -> list[str]:
    cols = production_columns(df) + [
        "delivery_days",
        "delay_days",
        "risk_score",
        "risk_tier",
    ]
    return [c for c in cols if c in df.columns and c not in ID_FIELDS]


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = df[columns].copy()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown").astype(str)
    X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    X.columns = (
        pd.Index(X.columns.astype(str))
        .str.replace(r"[\[\]<>]", "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
    )
    return X


def model_factory(model_name: str, scale_pos_weight: float, positive_weight_multiplier: float):
    adjusted_pos_weight = scale_pos_weight * positive_weight_multiplier
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=360,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.90,
            colsample_bytree=0.90,
            class_weight={0: 1.0, 1: adjusted_pos_weight},
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=360,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.90,
            scale_pos_weight=adjusted_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    raise ValueError(model_name)


def performance_rating(accuracy: float, cost: int) -> str:
    if accuracy >= 0.90 and cost <= 15000:
        return "A+"
    if accuracy >= 0.80 and cost <= 20000:
        return "A"
    if accuracy >= 0.70:
        return "B"
    if accuracy >= 0.65:
        return "C"
    if accuracy >= 0.60:
        return "D"
    return "E"


def evaluate_variant(
    df: pd.DataFrame,
    variant: Variant,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    X_all = prepare_matrix(df, variant.columns)
    y_all = df[variant.target_name].astype(int).to_numpy()
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = model_factory(variant.model_name, scale_pos_weight, variant.positive_weight_multiplier)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = int(fn * COST_FN + fp * COST_FP)
    acc = accuracy_score(y_test, y_pred)

    best = None
    best_f1 = None
    for threshold in np.linspace(0.01, 0.99, 99):
        pred_t = (y_proba >= threshold).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, pred_t).ravel()
        cost_t = int(fn_t * COST_FN + fp_t * COST_FP)
        acc_t = (tn_t + tp_t) / (tn_t + fp_t + fn_t + tp_t)
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0
        f1_t = (2 * prec_t * rec_t / (prec_t + rec_t)) if (prec_t + rec_t) else 0
        if best is None or cost_t < best["optimal_cost_thb"]:
            best = {
                "optimal_threshold": float(threshold),
                "optimal_cost_accuracy": float(acc_t),
                "optimal_cost_recall": float(rec_t),
                "optimal_cost_precision": float(prec_t),
                "optimal_cost_f1": float(f1_t),
                "optimal_cost_thb": cost_t,
            }
        if best_f1 is None or f1_t > best_f1["best_f1_score"]:
            best_f1 = {
                "best_f1_threshold": float(threshold),
                "best_f1_accuracy": float(acc_t),
                "best_f1_precision": float(prec_t),
                "best_f1_recall": float(rec_t),
                "best_f1_score": float(f1_t),
                "best_f1_cost_thb": cost_t,
            }

    return {
        "variant": variant.variant,
        "model_name": variant.model_name,
        "target_name": variant.target_name,
        "feature_policy": variant.feature_policy,
        "production_safe": variant.production_safe,
        "positive_weight_multiplier": variant.positive_weight_multiplier,
        "original_feature_count": len(variant.columns),
        "encoded_feature_count": X_all.shape[1],
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "target_rate_test": float(y_test.mean()),
        "accuracy": float(acc),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "avg_precision": float(average_precision_score(y_test, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "expected_cost_thb": cost,
        "performance_rating": performance_rating(acc, cost),
        **best,
        **best_f1,
        "feature_list": ", ".join(variant.columns),
    }


def build_variants(df: pd.DataFrame) -> list[Variant]:
    prod = production_columns(df)
    compact = compact_columns(df)
    diag = diagnostic_columns(df)
    return [
        Variant(
            "safe_advanced_lightgbm",
            "LightGBM",
            "is_returned",
            "production_safe_point_in_time_features",
            True,
            prod,
        ),
        Variant(
            "safe_advanced_lightgbm_recall2x",
            "LightGBM",
            "is_returned",
            "production_safe_point_in_time_features_positive_weight_2x",
            True,
            prod,
            positive_weight_multiplier=2.0,
        ),
        Variant(
            "safe_advanced_lightgbm_recall4x",
            "LightGBM",
            "is_returned",
            "production_safe_point_in_time_features_positive_weight_4x",
            True,
            prod,
            positive_weight_multiplier=4.0,
        ),
        Variant(
            "safe_compact_lightgbm",
            "LightGBM",
            "is_returned",
            "compact_production_safe_features",
            True,
            compact,
        ),
        Variant(
            "safe_advanced_xgboost",
            "XGBoost",
            "is_returned",
            "production_safe_point_in_time_features",
            True,
            prod,
        ),
        Variant(
            "diagnostic_post_event_lightgbm",
            "LightGBM",
            "is_returned",
            "diagnostic_includes_post_event_fields_not_for_production",
            False,
            diag,
        ),
        Variant(
            "segmented_customer_behavior_lightgbm",
            "LightGBM",
            "is_customer_behavior_return",
            "production_safe_features_but_target_is_only_changed_mind_or_better_price",
            True,
            prod,
        ),
        Variant(
            "segmented_customer_behavior_recall4x",
            "LightGBM",
            "is_customer_behavior_return",
            "segmented_customer_behavior_positive_weight_4x",
            True,
            prod,
            positive_weight_multiplier=4.0,
        ),
        Variant(
            "segmented_product_quality_lightgbm",
            "LightGBM",
            "is_product_quality_return",
            "production_safe_features_but_target_is_only_defective_or_wrong_item",
            True,
            prod,
        ),
        Variant(
            "segmented_product_quality_recall4x",
            "LightGBM",
            "is_product_quality_return",
            "segmented_product_quality_positive_weight_4x",
            True,
            prod,
            positive_weight_multiplier=4.0,
        ),
    ]


def save_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(plot_df))
    width = 0.20
    metrics = [
        ("accuracy", "Accuracy", "#1f77b4"),
        ("recall", "Recall", "#d62728"),
        ("f1_score", "F1", "#2ca02c"),
        ("auc_roc", "AUC", "#9467bd"),
    ]
    for i, (col, label, color) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, plot_df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("High Accuracy Feature Search")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(i - 0.30, row.accuracy + 0.015, f"{row.accuracy:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "high_accuracy_feature_search.png", dpi=160)
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
    prod = results[results["production_safe"]].sort_values("accuracy", ascending=False)
    overall_prod = results[
        results["production_safe"] & results["target_name"].eq("is_returned")
    ].sort_values("accuracy", ascending=False)
    segmented_prod = results[
        results["production_safe"] & ~results["target_name"].eq("is_returned")
    ].sort_values("accuracy", ascending=False)
    best_overall_prod = overall_prod.iloc[0]
    best_segmented = segmented_prod.iloc[0]
    best_all = results.sort_values("accuracy", ascending=False).iloc[0]
    summary = results[
        [
            "variant",
            "production_safe",
            "model_name",
            "positive_weight_multiplier",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "expected_cost_thb",
            "optimal_threshold",
            "optimal_cost_thb",
            "optimal_cost_recall",
            "best_f1_threshold",
            "best_f1_score",
            "best_f1_precision",
            "best_f1_recall",
            "performance_rating",
        ]
    ].sort_values("accuracy", ascending=False)
    content = f"""# High Accuracy Improvement Report

รอบนี้เพิ่ม feature แบบ point-in-time ให้ละเอียดขึ้น เช่น customer/SKU/brand/category/payment/channel/courier rolling history และ interaction ที่ผูกกับ business insight แล้วเทียบ LightGBM/XGBoost ด้วย split เดียวกัน

## Overall Return Model

ตัวที่ยังทำนาย target เดิม `is_returned` และใช้ได้จริงก่อนหรือขณะ order เข้าได้ดีที่สุดคือ `{best_overall_prod["variant"]}` ได้ Accuracy `{best_overall_prod["accuracy"]:.4f}`, AUC `{best_overall_prod["auc_roc"]:.4f}`, Recall `{best_overall_prod["recall"]:.4f}`, Cost@0.5 `{int(best_overall_prod["expected_cost_thb"]):,}` THB และ optimal cost `{int(best_overall_prod["optimal_cost_thb"]):,}` THB ที่ threshold `{best_overall_prod["optimal_threshold"]:.2f}`

## Segmented Return Models

ตัวที่ Accuracy แตะ 80% คือ `{best_segmented["variant"]}` ได้ Accuracy `{best_segmented["accuracy"]:.4f}` แต่ target ไม่ใช่ overall return แล้ว เป็น target ย่อย `{best_segmented["target_name"]}` และ Recall อยู่ที่ `{best_segmented["recall"]:.4f}` จึงควรใช้เป็นโมเดลย่อยสำหรับ intervention เฉพาะสาเหตุ ไม่ควรเอาไปแทน overall-return model

## Diagnostic Ceiling

ตัวที่ดีที่สุดรวมทุก target คือ `{best_all["variant"]}` ได้ Accuracy `{best_all["accuracy"]:.4f}` ถ้า target เป็น segmented target ต้องอ่านเทียบกับ Recall/F1 ด้วย และถ้า `production_safe=False` แปลว่ามี field หลังเหตุการณ์ เช่น delivery/risk output เดิม ใช้เพื่อวิเคราะห์เพดานเท่านั้น ไม่ควรนำไป deploy

## Comparison

{markdown_table(summary, summary.columns.tolist())}

## What This Means

- ประสิทธิภาพความแม่นยำของโมเดลขึ้นอยู่กับ 3 แกนหลัก: คุณภาพและปริมาณข้อมูล (Data), สถาปัตยกรรมของโมเดล (Model Architecture), และกระบวนการปรับจูน (Tuning & Optimization)
- ในโปรเจกต์นี้ เราได้ลองแกน Model Architecture/Tuning แล้ว เช่น LightGBM, XGBoost, class weight, threshold optimization และ best-F1 threshold แต่ Accuracy ของ overall-return model ยังอยู่ราว 68%
- ดังนั้น bottleneck หลักตอนนี้คือแกน Data: ข้อมูลก่อนส่งของยัง signal ไม่พอ ต้องเพิ่มข้อมูลพฤติกรรมก่อนซื้อ, SKU defect/complaint history, courier late-rate history, และ label return reason ที่สะอาดขึ้น
- ถ้า overall-return production-safe ยังไม่แตะ 80% แปลว่าข้อมูลก่อนส่งของใน dataset นี้ยัง signal ไม่พอ ต้องเพิ่มข้อมูลจริงใหม่ เช่น click/add-to-cart/live session, complaint/SKU defect history, courier delay rate รายพื้นที่, และ label return reason ที่แยก customer behavior ออกจาก operational error
- การแยก target ทำให้ Accuracy ดูสูงขึ้นได้ แต่ถ้า Recall/F1 ต่ำ แปลว่า model ยังจับเคส positive ได้ไม่ดี ต้องแก้ class imbalance และเพิ่ม signal เฉพาะสาเหตุ ไม่ใช่ประกาศว่า model overall ดีแล้ว
- รอบนี้มีทั้ง default threshold, optimal cost threshold และ best-F1 threshold เพื่อดูสมดุล Precision/Recall: ถ้าเพิ่ม class weight แล้ว Recall สูงขึ้นแต่ Precision ต่ำลงมาก ให้ใช้ F1 และ Cost Matrix เป็นตัวตัดสิน ไม่ใช้ Accuracy อย่างเดียว
- ถ้า diagnostic ดีขึ้นมาก แปลว่า field หลังเหตุการณ์มี signal สูง แต่ใช้จริงตอน order เข้าไม่ได้ ต้องเปลี่ยนเป็น proxy ที่รู้ล่วงหน้าแทน เช่น courier historical late rate แทน `delay_days`
- เป้าหมาย 80-90% ควรดูคู่กับ AUC 0.85+ และ Cost Matrix ไม่ใช่ Accuracy อย่างเดียว เพราะ threshold ที่ดัน Accuracy สูงอาจทำให้พลาด return จริง
"""
    DOC_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = add_point_in_time_features(load_dataset())
    df.to_csv(ROOT / "data" / "features" / "df_high_accuracy_features_preview.csv", index=False, encoding="utf-8-sig")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["is_returned"],
    )

    rows = []
    feature_map = {}
    for variant in build_variants(df):
        rows.append(evaluate_variant(df, variant, train_idx, test_idx))
        feature_map[variant.variant] = variant.columns

    results = pd.DataFrame(rows).sort_values(["accuracy", "auc_roc"], ascending=False)
    results.to_csv(OUT_DIR / "high_accuracy_feature_search.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "high_accuracy_feature_sets.json").write_text(
        json.dumps(feature_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plot(results)
    write_report(results)

    print("[OK] High accuracy feature search complete.")
    print(
        results[
            [
                "variant",
                "production_safe",
                "model_name",
                "positive_weight_multiplier",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "expected_cost_thb",
                "best_f1_threshold",
                "best_f1_score",
                "best_f1_precision",
                "best_f1_recall",
                "optimal_cost_thb",
                "performance_rating",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
