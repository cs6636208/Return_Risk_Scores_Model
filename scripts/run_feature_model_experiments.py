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


ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
ENGINEERED_PATH = ROOT / "data" / "features" / "df_engineered.csv"
DOCS_ANALYSIS = ROOT / "docs" / "analysis"
FEATURE_AUDIT_DIR = ROOT / "reports" / "feature_audit"
EXPERIMENT_DIR = ROOT / "reports" / "model_experiments"

RANDOM_STATE = 42
TEST_SIZE = 0.20
COST_FN = 150
COST_FP = 50

POST_EVENT_OR_LEAKAGE_FIELDS = {
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
    "is_returned",
    "delivery_date",
    "delivery_days",
    "delay_days",
}

IDENTIFIER_FIELDS = {
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
class ExperimentSpec:
    version: str
    title: str
    description: str
    source: str
    columns: list[str]


def ensure_dirs() -> None:
    DOCS_ANALYSIS.mkdir(parents=True, exist_ok=True)
    FEATURE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    engineered = pd.read_csv(ENGINEERED_PATH, low_memory=False)
    if len(clean) != len(engineered):
        raise ValueError(
            f"Row mismatch: clean_dataset has {len(clean)} rows, df_engineered has {len(engineered)} rows"
        )
    engineered = engineered.copy()
    engineered["order_id"] = clean["order_id"].values
    engineered["customer_id"] = clean["customer_id"].values
    engineered["is_returned"] = clean["is_returned"].values
    return clean, engineered


def add_clean_time_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    df["customer_tenure_months"] = ((df["order_date"] - df["registration_date"]).dt.days / 30).fillna(0)
    return df


def safe_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    return [c for c in requested if c in df.columns and c not in POST_EVENT_OR_LEAKAGE_FIELDS]


def build_specs(clean: pd.DataFrame, engineered: pd.DataFrame) -> list[ExperimentSpec]:
    clean_pre_order = [
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
        "customer_age_days",
        "customer_tenure_months",
        "order_month",
        "order_dayofweek",
        "is_weekend",
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
        "tier_discount_pct",
        "campaign_discount_pct",
        "total_discount_pct",
        "discount_applied_amount",
        "total_amount",
        "delivery_time_expected_days",
        "is_repurchased_item",
        "order_hour",
        "days_since_last_order",
        "hist_order_count",
        "hist_return_rate",
    ]

    engineered_full = [
        c
        for c in engineered.columns
        if c not in POST_EVENT_OR_LEAKAGE_FIELDS
        and c not in IDENTIFIER_FIELDS
        and c not in {"order_id", "customer_id"}
    ]

    history_focus = [
        "is_repurchased_item",
        "days_since_last_order",
        "customer_age_days",
        "age",
        "is_high_risk_customer",
        "is_first_order",
        "total_orders_before",
        "total_returns_before",
        "customer_return_ratio",
        "days_since_last_return",
        "hist_spend_sum_30d",
        "hist_order_count_30d",
        "hist_return_rate_30d",
        "hist_spend_sum_60d",
        "hist_order_count_60d",
        "hist_return_rate_60d",
        "hist_spend_sum_180d",
        "hist_order_count_180d",
        "hist_return_rate_180d",
        "hist_return_rate",
        "hist_order_count",
    ]

    business_interactions = [
        "channel_type",
        "payment_method",
        "membership_tier",
        "province",
        "category",
        "brand",
        "gender_province",
        "unit_price",
        "log_unit_price",
        "promo_discount_pct",
        "total_amount",
        "log_total_amount",
        "product_rating",
        "is_peak_hour",
        "is_fashion_tv",
        "is_remote_area",
        "low_rating_alert",
        "is_cod",
        "is_high_discount",
        "return_rate_by_category",
        "delivery_time_expected_days",
        "is_long_distance_cod",
        "is_impulse_buy",
        "is_low_commitment",
        "category_payment",
        "category_channel",
        "province_payment",
    ]

    leakage_safe_reduced = [
        "hist_return_rate",
        "hist_order_count",
        "days_since_last_order",
        "is_repurchased_item",
        "category",
        "brand",
        "province",
        "channel_type",
        "payment_method",
        "membership_tier",
        "product_rating",
        "total_discount_pct",
        "total_amount",
        "delivery_time_expected_days",
        "order_hour",
        "customer_age_days",
        "age",
        "promo_discount_pct",
        "return_rate_by_category",
        "is_cod",
        "is_high_discount",
        "low_rating_alert",
        "category_payment",
        "category_channel",
    ]

    compact_best = [
        "hist_return_rate",
        "hist_order_count",
        "days_since_last_order",
        "is_repurchased_item",
        "category",
        "brand",
        "province",
        "payment_method",
        "product_rating",
        "total_amount",
        "total_discount_pct",
        "delivery_time_expected_days",
        "order_hour",
        "return_rate_by_category",
        "category_payment",
        "category_channel",
        "is_cod",
        "low_rating_alert",
        "is_high_discount",
        "customer_age_days",
    ]

    return [
        ExperimentSpec(
            "v0_clean_baseline",
            "Clean baseline",
            "ใช้เฉพาะ feature จาก clean_dataset ที่รู้ได้ก่อนหรือขณะ order เข้า และตัด field หลังเกิด return/refund ออก",
            "clean",
            safe_columns(clean, clean_pre_order),
        ),
        ExperimentSpec(
            "v1_engineered_full",
            "Engineered full",
            "ใช้ df_engineered.csv ชุดเต็ม เพื่อวัดผลของ engineered feature ทั้งหมด",
            "engineered",
            safe_columns(engineered, engineered_full),
        ),
        ExperimentSpec(
            "v2_history_focus",
            "History focus",
            "เน้นประวัติลูกค้าและ rolling/expanding history เช่น return ratio, order count, days since last return",
            "engineered",
            safe_columns(engineered, history_focus),
        ),
        ExperimentSpec(
            "v3_business_interactions",
            "Business interactions",
            "เน้น feature ที่มาจาก business insight เช่น category-payment, category-channel, COD, discount, rating, province",
            "engineered",
            safe_columns(engineered, business_interactions),
        ),
        ExperimentSpec(
            "v4_leakage_safe_reduced",
            "Leakage-safe reduced",
            "ตัด feature อนาคต/ซ้ำซ้อนออก เหลือชุดผสมที่ปลอด leakage และยังผูกกับ insight หลัก",
            "engineered",
            safe_columns(engineered, leakage_safe_reduced),
        ),
        ExperimentSpec(
            "v5_compact_best",
            "Compact candidate",
            "ชุด compact สำหรับ production candidate ลดจำนวน feature เพื่อให้ query/serve เร็วขึ้น",
            "engineered",
            safe_columns(engineered, compact_best),
        ),
    ]


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = df[columns].copy()
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64") // 10**9
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown").astype(str)
    return pd.get_dummies(X, columns=categorical_cols, dummy_na=False)


def rate_performance(acc: float) -> str:
    if acc >= 0.75:
        return "A"
    if acc >= 0.70:
        return "B"
    if acc >= 0.65:
        return "C"
    if acc >= 0.60:
        return "D"
    return "E"


def evaluate_specs(clean: pd.DataFrame, engineered: pd.DataFrame, specs: list[ExperimentSpec]) -> pd.DataFrame:
    train_idx, test_idx = train_test_split(
        np.arange(len(clean)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean["is_returned"],
    )
    rows = []
    feature_details = {}
    for spec in specs:
        source_df = clean if spec.source == "clean" else engineered
        X_all = prepare_matrix(source_df, spec.columns)
        y = clean["is_returned"].astype(int).to_numpy()
        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = int(fn * COST_FN + fp * COST_FP)
        acc = accuracy_score(y_test, y_pred)

        rows.append(
            {
                "version": spec.version,
                "title": spec.title,
                "source": spec.source,
                "description": spec.description,
                "input_rows": len(source_df),
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "original_feature_count": len(spec.columns),
                "encoded_feature_count": X_all.shape[1],
                "accuracy": acc,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y_test, y_proba),
                "avg_precision": average_precision_score(y_test, y_proba),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "expected_cost_thb": cost,
                "performance_rating": rate_performance(acc),
                "feature_list": ", ".join(spec.columns),
            }
        )
        feature_details[spec.version] = spec.columns

    results = pd.DataFrame(rows).sort_values(["accuracy", "auc_roc"], ascending=False)
    (EXPERIMENT_DIR / "lightgbm_feature_version_features.json").write_text(
        json.dumps(feature_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def create_manual_check(clean: pd.DataFrame) -> pd.DataFrame:
    customer_stats = (
        clean.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            returned_orders=("is_returned", "sum"),
            observed_return_rate=("is_returned", "mean"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )
    eligible = customer_stats[customer_stats["total_orders"] >= 8].copy()
    if len(eligible) < 10:
        eligible = customer_stats.sort_values("total_orders", ascending=False).head(10)
    else:
        eligible = eligible.sample(10, random_state=RANDOM_STATE)

    order_rows = []
    for customer_id in eligible["customer_id"]:
        rows = clean.loc[clean["customer_id"].eq(customer_id)].copy()
        rows = rows.sort_values(["order_date", "order_id"]).reset_index(drop=True)
        rows["order_sequence"] = np.arange(1, len(rows) + 1)
        rows["manual_formula"] = (
            rows["is_returned"].cumsum().astype(str)
            + " / "
            + rows["order_sequence"].astype(str)
        )
        rows["manual_running_return_rate_including_current"] = rows["is_returned"].expanding().mean().values
        rows["point_in_time_expected_hist_return_rate"] = (
            rows["is_returned"].shift().expanding().mean().fillna(0).values
        )
        rows["hist_return_rate_abs_error"] = (
            rows["hist_return_rate"].astype(float) - rows["point_in_time_expected_hist_return_rate"]
        ).abs()
        order_rows.append(
            rows[
                [
                    "customer_id",
                    "order_id",
                    "order_sequence",
                    "order_date",
                    "category",
                    "payment_method",
                    "is_returned",
                    "manual_formula",
                    "manual_running_return_rate_including_current",
                    "hist_order_count",
                    "hist_return_rate",
                    "point_in_time_expected_hist_return_rate",
                    "hist_return_rate_abs_error",
                ]
            ]
        )
    return pd.concat(order_rows, ignore_index=True)


def create_feature_audit(clean: pd.DataFrame, engineered: pd.DataFrame, specs: list[ExperimentSpec]) -> pd.DataFrame:
    clean_features = set(clean.columns)
    engineered_features = set(engineered.columns)
    used_by_version = {}
    for spec in specs:
        for col in spec.columns:
            used_by_version.setdefault(col, []).append(spec.version)

    all_features = sorted((clean_features | engineered_features) - {"is_returned"})
    rows = []
    for col in all_features:
        if col in POST_EVENT_OR_LEAKAGE_FIELDS:
            status = "dropped_leakage_or_post_event"
            reason = "เกิดหลัง order/return หรือเป็นผลลัพธ์จาก model จึงไม่ควรเข้า train"
        elif col in IDENTIFIER_FIELDS or col in {"order_id", "customer_id"}:
            status = "dropped_identifier"
            reason = "ใช้สำหรับ audit/join/search แต่ไม่ควรให้ model จำรหัส"
        elif col in used_by_version:
            status = "used"
            reason = "ใช้ใน experiment version: " + ", ".join(used_by_version[col])
        elif col in engineered_features and col not in clean_features:
            status = "available_engineered_not_used"
            reason = "มีใน engineered dataset แต่ไม่อยู่ในชุด experiment compact/current"
        else:
            status = "available_clean_not_used"
            reason = "มีใน clean dataset แต่ไม่ถูกใช้เป็น feature หลัก"
        rows.append(
            {
                "feature": col,
                "in_clean_dataset": col in clean_features,
                "in_df_engineered": col in engineered_features,
                "used_in_versions": ", ".join(used_by_version.get(col, [])),
                "audit_status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def save_comparison_plot(results: pd.DataFrame) -> None:
    plot_df = results.sort_values("version")
    x = np.arange(len(plot_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 7))
    metrics = [
        ("accuracy", "Accuracy", "#1f77b4"),
        ("recall", "Recall", "#d62728"),
        ("f1_score", "F1", "#2ca02c"),
        ("auc_roc", "AUC", "#9467bd"),
    ]
    for i, (col, label, color) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, plot_df[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["version"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("LightGBM Feature Version Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for idx, row in plot_df.iterrows():
        xpos = list(plot_df.index).index(idx)
        ax.text(xpos - 0.27, row["accuracy"] + 0.015, f"{row['accuracy']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(EXPERIMENT_DIR / "lightgbm_feature_version_comparison.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    view = view.fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_business_insight_report() -> None:
    content = """# Business Insight Feature Summary

โปรเจกต์นี้เลือก insight จากกราฟ EDA จำนวนมากโดยยึดหลักว่า insight ต้องแปลงเป็น feature ได้จริง และต้องช่วยอธิบายการตัดสินใจของ model ให้ทีม Operations/Call Center เข้าใจได้

| Insight | กราฟหลัก | ข้อมูลบอกอะไร | เหตุผลที่เลือก | Feature ที่ตามมา | Business action |
| --- | --- | --- | --- | --- | --- |
| Customer return history | `reports/business_insights/charts/05_feature_importance.png`, `reports/Graph Item/eda_customer_history.png` | ประวัติการคืนของลูกค้าเป็น signal สำคัญที่สุด เช่น return ratio, return count, days since last return | สอดคล้องกับโจทย์ที่ต้องดู order ก่อนหน้า เช่น order ที่ 3 ต้องใช้ 2 order แรกเท่านั้น | `hist_return_rate`, `customer_return_ratio`, `total_returns_before`, `hist_return_rate_30d/60d/180d`, `days_since_last_return` | ถ้าประวัติคืนสูง ให้ call center โทรยืนยันก่อนส่ง |
| Category x channel | `reports/Graph Item/cross_chart_category_channel.png`, `reports/eda_full/04_interaction_cat_channel.png` | หมวดสินค้าและช่องทางขายบางคู่มี return rate สูง เช่น Fashion ผ่าน TV/TikTok | เป็นพฤติกรรม expectation gap จาก home shopping ที่ใช้เล่า business ได้ดี | `category_channel`, `is_fashion_tv`, `channel_type`, `category` | เพิ่ม confirmation script เฉพาะหมวด/ช่องทางเสี่ยง |
| Payment/COD risk | `reports/Graph Item/cross_chart_category_payment.png`, `reports/Graph Relation Feature/relation_payment_risk.png` | COD สะท้อน commitment ต่ำกว่า prepaid และมักเสี่ยง return/refuse รับ | เป็น field ที่รู้ตั้งแต่ order เข้า ใช้ deploy ได้จริง | `payment_method`, `is_cod`, `category_payment`, `province_payment`, `is_low_commitment` | กระตุ้น prepaid หรือยืนยัน COD high-risk |
| Province/logistics friction | `reports/Graph Item/cross_chart_province_payment.png`, `reports/Graph Relation Feature/relation_province_gap.png` | บางพื้นที่มี delivery friction หรือ return pattern สูงกว่า | ช่วยแยก risk จากพื้นที่และขนส่ง ไม่ใช่ดูเฉพาะลูกค้า | `province`, `is_remote_area`, `is_long_distance_cod`, `damage_rate`, `avg_delivery_days` | เลือก courier/packaging หรือ alert พื้นที่เสี่ยง |
| Product rating and quality | `reports/Graph Item/eda_rating_threshold.png`, `reports/Graph Relation Feature/v2_rating_threshold.png` | สินค้า rating ต่ำมีโอกาสคืนสูงกว่า | เชื่อมกับสาเหตุสินค้าไม่ตรงปก/คุณภาพไม่ถึง expectation | `product_rating`, `low_rating_alert`, `brand`, `return_rate_by_category` | audit SKU/brand rating ต่ำก่อนทำ campaign |
| Discount and impulse buying | `reports/Graph Relation Feature/relation_discount_hunter.png`, `reports/Graph Relation Feature/v2_promotion_pattern.png` | ส่วนลดสูงอาจกระตุ้น impulse buy และ return ภายหลัง | เป็น feature ที่รู้ทันทีตอน order และผูกกับ policy promotion | `total_discount_pct`, `promo_discount_pct`, `is_high_discount`, `promo_type` | ตรวจ campaign ที่สร้าง return cost สูง |
| Repurchase behavior | `reports/Graph Relation Feature/relation_repurchase_risk.png` | ลูกค้าซื้อซ้ำสินค้าเดิมมักมี uncertainty ต่ำกว่า | ช่วยลด false alarm เพราะไม่ใช่ทุก high spender จะ high risk | `is_repurchased_item`, `days_since_last_order`, `hist_order_count` | ให้ผ่าน flow ปกติถ้าเป็น repurchase ที่มีประวัติดี |
| Time/order behavior | `reports/Graph Item/eda_hour_trend.png`, `reports/Graph Relation Feature/relation_hour_channel.png` | บางช่วงเวลาหรือ peak hour มี pattern การคืนต่างกัน | เป็น behavioral signal ราคาถูกต่อการ query และรู้ทันที | `order_hour`, `is_peak_hour`, `order_dayofweek`, `is_weekend` | ใช้จัดลำดับ review queue ในช่วง order peak |

## สรุปการแปลง Insight เป็น Feature

ชุด feature ที่ควรเป็นแกนหลักคือ `hist_return_rate`, `customer_return_ratio`, `category_channel`, `category_payment`, `payment_method/is_cod`, `product_rating/low_rating_alert`, `total_discount_pct/is_high_discount`, `province`, และ `is_repurchased_item` เพราะทั้งหมดอธิบายได้เชิงธุรกิจและรู้ได้ก่อนหรือขณะ order เข้า
"""
    (DOCS_ANALYSIS / "business_insight_feature_summary.md").write_text(content, encoding="utf-8")


def write_feature_audit_report(audit: pd.DataFrame, manual: pd.DataFrame, specs: list[ExperimentSpec]) -> None:
    status_counts = audit["audit_status"].value_counts().reset_index()
    status_counts.columns = ["audit_status", "count"]
    manual_summary = (
        manual.groupby("customer_id")
        .agg(
            checked_orders=("order_id", "count"),
            returned_orders=("is_returned", "sum"),
            final_manual_rate=("manual_running_return_rate_including_current", "last"),
            max_hist_abs_error=("hist_return_rate_abs_error", "max"),
        )
        .reset_index()
    )
    spec_rows = pd.DataFrame(
        [
            {
                "version": s.version,
                "source": s.source,
                "feature_count": len(s.columns),
                "description": s.description,
            }
            for s in specs
        ]
    )
    content = f"""# Feature Audit And Manual Validation

เอกสารนี้ตรวจว่า feature จาก `clean_dataset.csv` และ `df_engineered.csv` ถูกใช้/ถูกตัดอย่างไร และมีการสุ่มเช็ค return rate รายลูกค้าเพื่อยืนยัน logic แบบ manual

## Feature Audit Summary

{markdown_table(status_counts, ["audit_status", "count"])}

## Version Feature Sets

{markdown_table(spec_rows, ["version", "source", "feature_count", "description"])}

## Manual Cross-check รายลูกค้า

ไฟล์ละเอียดอยู่ที่ `reports/feature_audit/customer_return_rate_manual_check.csv` โดยคำนวณทั้งแบบ manual running rate และ point-in-time history rate

ตัวอย่างสูตร: ถ้าลูกค้าคนหนึ่งมี 8 orders และ return 4 orders ค่า return rate แบบรวมปัจจุบันคือ `4 / 8 = 0.5` หรือ `50%`

{markdown_table(manual_summary, ["customer_id", "checked_orders", "returned_orders", "final_manual_rate", "max_hist_abs_error"])}

## Leakage Notes

field ที่ถูกตัดออกจาก model ทุก version ได้แก่ `return_id`, `return_date`, `refund_amount`, `return_reason`, `risk_score`, `risk_tier`, `shap_values`, `delivery_date`, `delivery_days`, และ `delay_days` เพราะเป็นข้อมูลหลังเหตุการณ์หรือผลลัพธ์จาก model เดิม

`order_id`, `customer_id`, ชื่อลูกค้า เบอร์โทร และรหัสสินค้า/ขนส่ง ใช้สำหรับ audit/join/search เท่านั้น ไม่ใช้ train model เพื่อป้องกัน model จำ identity
"""
    (DOCS_ANALYSIS / "feature_audit_and_validation.md").write_text(content, encoding="utf-8")


def write_model_report(results: pd.DataFrame) -> None:
    best = results.iloc[0]
    compact = results.loc[results["version"].eq("v5_compact_best")].iloc[0]
    deltas = results[["version", "accuracy", "recall", "f1_score", "auc_roc", "expected_cost_thb", "original_feature_count", "performance_rating"]]
    content = f"""# Model Feature Version Report

ใช้ LightGBM เป็น model กลาง และใช้ train/test split เดียวกันทุก version เพื่อให้เปรียบเทียบผลจาก feature ได้ยุติธรรม โดยเลือก Accuracy เป็น metric หลักตามโจทย์ แต่ยังรายงาน Recall, F1, AUC และ Cost ประกอบ

## Best Version

version ที่ Accuracy สูงสุดคือ `{best["version"]}` ({best["title"]}) ได้ Accuracy `{best["accuracy"]:.4f}`, Recall `{best["recall"]:.4f}`, F1 `{best["f1_score"]:.4f}`, AUC `{best["auc_roc"]:.4f}`, Cost `{int(best["expected_cost_thb"]):,}` THB และใช้ feature ตั้งต้น `{int(best["original_feature_count"])}` ตัว

## Compact Candidate

`v5_compact_best` ได้ Accuracy `{compact["accuracy"]:.4f}` จาก feature ตั้งต้น `{int(compact["original_feature_count"])}` ตัว เหมาะเป็น candidate ถ้าต้องลด query/resource ตอน deploy เพราะจำนวน feature ต่ำกว่า full engineered set มาก

## Version Comparison

{markdown_table(deltas, ["version", "accuracy", "recall", "f1_score", "auc_roc", "expected_cost_thb", "original_feature_count", "performance_rating"])}

## Feature Decision

- ควรเก็บ feature กลุ่มประวัติลูกค้า เช่น `hist_return_rate`, `customer_return_ratio`, `hist_return_rate_30d/60d/180d` เพราะให้สัญญาณแรงและอธิบายโจทย์รายลูกค้าได้ตรงที่สุด
- ควรเก็บ business interaction เช่น `category_payment`, `category_channel`, `payment_method`, `is_cod`, `product_rating`, `total_discount_pct` เพราะผูกกับ insight และ action ได้
- ควรตัด field หลังเหตุการณ์ทั้งหมดออกจาก training แม้บางตัวจะเพิ่ม Accuracy ได้ เพราะใช้จริงตอน order เข้าไม่ได้
- ถ้า feature version ที่ใหญ่กว่าให้ Accuracy ใกล้เคียง compact version ควรเลือก compact เพื่อให้ query เร็วและลด resource

ไฟล์ผลเต็มอยู่ที่ `reports/model_experiments/lightgbm_feature_version_comparison.csv` และกราฟอยู่ที่ `reports/model_experiments/lightgbm_feature_version_comparison.png`
"""
    (DOCS_ANALYSIS / "model_feature_version_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    clean_raw, engineered = load_data()
    clean = add_clean_time_features(clean_raw)
    specs = build_specs(clean, engineered)

    manual = create_manual_check(clean)
    manual.to_csv(FEATURE_AUDIT_DIR / "customer_return_rate_manual_check.csv", index=False, encoding="utf-8-sig")

    audit = create_feature_audit(clean, engineered, specs)
    audit.to_csv(FEATURE_AUDIT_DIR / "feature_audit_table.csv", index=False, encoding="utf-8-sig")

    results = evaluate_specs(clean, engineered, specs)
    results.to_csv(EXPERIMENT_DIR / "lightgbm_feature_version_comparison.csv", index=False, encoding="utf-8-sig")
    save_comparison_plot(results)

    write_business_insight_report()
    write_feature_audit_report(audit, manual, specs)
    write_model_report(results)

    print("[OK] Generated feature audit, model experiments, and markdown reports.")
    print(results[["version", "accuracy", "recall", "f1_score", "auc_roc", "expected_cost_thb", "performance_rating"]])


if __name__ == "__main__":
    main()
