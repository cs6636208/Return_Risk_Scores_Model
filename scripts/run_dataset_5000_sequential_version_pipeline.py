from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".ml_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.compose import ColumnTransformer
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


DATASET_SIZE = 5000
DATASET_ROOT = ROOT / "docs" / "test" / "dataset_5000"
INPUT_PATH = DATASET_ROOT / "input" / "clean_dataset_5000_full_test.csv"
OUT_ROOT = DATASET_ROOT / "sequential_pipeline"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "is_returned"


def configure_dataset(dataset_size: int) -> None:
    global DATASET_SIZE, DATASET_ROOT, INPUT_PATH, OUT_ROOT
    DATASET_SIZE = dataset_size
    DATASET_ROOT = ROOT / "docs" / "test" / f"dataset_{dataset_size}"
    input_dir = DATASET_ROOT / "input"
    preferred = input_dir / f"clean_dataset_{dataset_size}_full_test.csv"
    if dataset_size == 50000:
        preferred = input_dir / "clean_dataset_generated_50000_full_test.csv"
    if preferred.exists():
        INPUT_PATH = preferred
    else:
        candidates = sorted(input_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No input CSV found under {input_dir}")
        INPUT_PATH = candidates[0]
    OUT_ROOT = DATASET_ROOT / "sequential_pipeline"

LEAKAGE_COLUMNS = {
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

IDENTITY_COLUMNS = {
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
    "promo_name",
}

BASE_FEATURES = [
    "gender",
    "age",
    "membership_tier",
    "preferred_channel",
    "province",
    "customer_age_days",
    "category",
    "brand",
    "is_fragile",
    "product_rating",
    "courier_name",
    "courier_type",
    "avg_delivery_days",
    "damage_rate",
    "coverage_region",
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

V2_FEATURES = [
    "customer_tenure_months",
    "order_month",
    "order_dayofweek",
    "is_weekend",
    "age_group",
    "total_orders_before",
    "total_returns_before",
    "customer_return_ratio",
    "days_since_last_return",
    "hist_spend_sum_30d",
    "hist_order_count_30d",
    "hist_return_count_30d",
    "hist_return_rate_30d",
    "hist_spend_sum_60d",
    "hist_order_count_60d",
    "hist_return_count_60d",
    "hist_return_rate_60d",
    "hist_spend_sum_180d",
    "hist_order_count_180d",
    "hist_return_count_180d",
    "hist_return_rate_180d",
    "hist_spend_sum_365d",
    "hist_order_count_365d",
    "hist_return_count_365d",
    "hist_return_rate_365d",
]

V3_FEATURES = [
    "is_cod",
    "is_high_discount",
    "low_rating_alert",
    "is_first_order",
    "discount_amount_ratio",
    "amount_per_item",
    "log_unit_price",
    "log_total_amount",
    "category_payment",
    "category_channel",
    "province_payment",
    "tier_payment",
    "category_return_rate_pti",
    "product_return_rate_pti",
    "courier_return_rate_pti",
]

V4_FEATURES = [
    "price_band",
    "discount_band",
    "rating_band",
    "is_high_value_order",
    "is_fragile_cod",
    "is_remote_cod",
    "is_fashion_cod",
    "logistics_risk",
    "category_province",
    "brand_channel",
    "province_category_return_rate_pti",
    "brand_return_rate_pti",
    "payment_return_rate_pti",
    "channel_return_rate_pti",
    "courier_type_return_rate_pti",
]

V5_COMPACT_KEEP = [
    "age",
    "membership_tier",
    "province",
    "category",
    "brand",
    "is_fragile",
    "product_rating",
    "courier_type",
    "avg_delivery_days",
    "damage_rate",
    "promo_type",
    "promo_discount_rate",
    "channel_type",
    "payment_method",
    "quantity",
    "unit_price",
    "total_discount_pct",
    "discount_applied_amount",
    "total_amount",
    "delivery_time_expected_days",
    "is_repurchased_item",
    "order_hour",
    "days_since_last_order",
    "hist_order_count",
    "hist_return_rate",
    "customer_tenure_months",
    "order_month",
    "order_dayofweek",
    "is_weekend",
    "total_orders_before",
    "total_returns_before",
    "customer_return_ratio",
    "days_since_last_return",
    "hist_order_count_30d",
    "hist_return_count_30d",
    "hist_return_rate_30d",
    "hist_spend_sum_30d",
    "hist_order_count_60d",
    "hist_return_rate_60d",
    "hist_order_count_180d",
    "hist_return_rate_180d",
    "hist_order_count_365d",
    "hist_return_rate_365d",
    "is_cod",
    "is_high_discount",
    "low_rating_alert",
    "discount_amount_ratio",
    "amount_per_item",
    "log_total_amount",
    "category_payment",
    "category_channel",
    "province_payment",
    "category_return_rate_pti",
    "product_return_rate_pti",
    "courier_return_rate_pti",
    "price_band",
    "discount_band",
    "rating_band",
    "is_high_value_order",
    "is_fragile_cod",
    "is_remote_cod",
    "logistics_risk",
    "province_category_return_rate_pti",
    "brand_return_rate_pti",
    "payment_return_rate_pti",
    "channel_return_rate_pti",
    "courier_type_return_rate_pti",
]


@dataclass
class VersionResult:
    version: str
    input_dataset: str
    output_dataset: str
    df_featured: str
    model_path: str
    train_test_path: str
    feature_count: int
    dropped_from_previous: int
    accuracy: float
    recall: float
    precision: float
    f1: float
    auc: float
    avg_precision: float
    cost: int
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        for sub in ["data", "models", "reports"]:
            (OUT_ROOT / f"version_{version}" / sub).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates("order_id", keep="last").reset_index(drop=True)

    date_columns = [
        "order_date",
        "expected_delivery_date",
        "delivery_date",
        "registration_date",
        "promo_start_date",
        "promo_end_date",
        "return_date",
        "scored_at",
    ]
    for col in date_columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col].replace({"Not Returned": pd.NA}), errors="coerce")

    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        out[col] = out[col].fillna("Unknown")

    numeric_cols = [
        "age",
        "customer_age_days",
        "product_rating",
        "avg_delivery_days",
        "damage_rate",
        "promo_discount_rate",
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
    for col in numeric_cols:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            fill = values.median()
            out[col] = values.fillna(0 if pd.isna(fill) else fill)

    if "is_fragile" in out.columns:
        out["is_fragile"] = out["is_fragile"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

    out = out.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return out


def pct_feature_list(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def add_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["promo_type"] = out.get("promo_type", "No Promotion").fillna("No Promotion").astype(str)
    if "order_date" in out.columns:
        out["order_month"] = out["order_date"].dt.month.fillna(0).astype(int)
        out["order_dayofweek"] = out["order_date"].dt.dayofweek.fillna(0).astype(int)
        out["is_weekend"] = out["order_dayofweek"].isin([5, 6]).astype(int)
    return out


def add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["customer_id", "order_date", "order_id"]).copy()
    default_values: dict[str, object] = {
        "total_orders_before": 0,
        "total_returns_before": 0,
        "customer_return_ratio": 0.0,
        "days_since_last_return": -1,
        "customer_tenure_months": 0.0,
        "age_group": "Unknown",
    }
    for col, default_value in default_values.items():
        if col not in out.columns:
            out[col] = default_value
    out["customer_return_ratio"] = pd.to_numeric(out["customer_return_ratio"], errors="coerce").fillna(0.0).astype(float)
    out["customer_tenure_months"] = pd.to_numeric(out["customer_tenure_months"], errors="coerce").fillna(0.0).astype(float)

    if "registration_date" in out.columns:
        out["customer_tenure_months"] = ((out["order_date"] - out["registration_date"]).dt.days / 30).fillna(0).clip(lower=0)
    out["age_group"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce").fillna(0),
        bins=[0, 20, 30, 40, 50, 120],
        labels=["<20", "20-30", "30-40", "40-50", ">50"],
        include_lowest=True,
    ).astype(str)

    for days in [30, 60, 180, 365]:
        out[f"hist_spend_sum_{days}d"] = 0.0
        out[f"hist_order_count_{days}d"] = 0
        out[f"hist_return_count_{days}d"] = 0
        out[f"hist_return_rate_{days}d"] = 0.0

    for _, group in out.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy(dtype="datetime64[ns]")
        returns = group[TARGET].to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()

        return_dates = pd.to_datetime(group.get("return_date", pd.Series(pd.NaT, index=group.index)), errors="coerce").to_numpy(dtype="datetime64[ns]")

        for pos, current_date in enumerate(dates):
            prior_mask = dates < current_date
            order_count = int(prior_mask.sum())
            return_count = int(returns[prior_mask].sum()) if order_count else 0
            out.loc[idx[pos], "total_orders_before"] = order_count
            out.loc[idx[pos], "total_returns_before"] = return_count
            out.loc[idx[pos], "customer_return_ratio"] = return_count / order_count if order_count else 0.0

            prior_return_mask = prior_mask & (returns == 1) & ~pd.isna(return_dates)
            if prior_return_mask.any():
                last_return_date = return_dates[prior_return_mask].max()
                out.loc[idx[pos], "days_since_last_return"] = int((current_date - last_return_date) / np.timedelta64(1, "D"))
            else:
                out.loc[idx[pos], "days_since_last_return"] = -1

            for days in [30, 60, 180, 365]:
                start = current_date - np.timedelta64(days, "D")
                window_mask = prior_mask & (dates >= start)
                window_count = int(window_mask.sum())
                window_return_count = int(returns[window_mask].sum()) if window_count else 0
                out.loc[idx[pos], f"hist_spend_sum_{days}d"] = float(amounts[window_mask].sum()) if window_count else 0.0
                out.loc[idx[pos], f"hist_order_count_{days}d"] = window_count
                out.loc[idx[pos], f"hist_return_count_{days}d"] = window_return_count
                out.loc[idx[pos], f"hist_return_rate_{days}d"] = window_return_count / window_count if window_count else 0.0

    return out.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def add_group_return_rate_pti(df: pd.DataFrame, group_cols: list[str], output_col: str) -> pd.DataFrame:
    out = df.sort_values(["order_date", "order_id"]).copy()
    out[output_col] = 0.0
    totals: dict[tuple, list[int]] = {}
    global_orders = 0
    global_returns = 0
    for idx, row in out.iterrows():
        key = tuple(row[col] for col in group_cols)
        orders, returns = totals.get(key, [0, 0])
        fallback = global_returns / global_orders if global_orders else 0.0
        out.at[idx, output_col] = returns / orders if orders else fallback
        totals[key] = [orders + 1, returns + int(row[TARGET])]
        global_orders += 1
        global_returns += int(row[TARGET])
    return out


def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_cod"] = out["payment_method"].eq("COD").astype(int)
    out["is_high_discount"] = pd.to_numeric(out["total_discount_pct"], errors="coerce").fillna(0).gt(0.20).astype(int)
    out["low_rating_alert"] = pd.to_numeric(out["product_rating"], errors="coerce").fillna(5).lt(4.0).astype(int)
    out["is_first_order"] = out["total_orders_before"].eq(0).astype(int)
    gross = (
        pd.to_numeric(out["unit_price"], errors="coerce").fillna(0)
        * pd.to_numeric(out["quantity"], errors="coerce").fillna(1)
    ).replace(0, np.nan)
    out["discount_amount_ratio"] = (pd.to_numeric(out["discount_applied_amount"], errors="coerce").fillna(0) / gross).fillna(0)
    out["amount_per_item"] = (
        pd.to_numeric(out["total_amount"], errors="coerce").fillna(0)
        / pd.to_numeric(out["quantity"], errors="coerce").fillna(1).replace(0, 1)
    )
    out["log_unit_price"] = np.log1p(pd.to_numeric(out["unit_price"], errors="coerce").fillna(0))
    out["log_total_amount"] = np.log1p(pd.to_numeric(out["total_amount"], errors="coerce").fillna(0))
    out["category_payment"] = out["category"].astype(str) + "_" + out["payment_method"].astype(str)
    out["category_channel"] = out["category"].astype(str) + "_" + out["channel_type"].astype(str)
    out["province_payment"] = out["province"].astype(str) + "_" + out["payment_method"].astype(str)
    out["tier_payment"] = out["membership_tier"].astype(str) + "_" + out["payment_method"].astype(str)

    for group_cols, output_col in [
        (["category"], "category_return_rate_pti"),
        (["product_id"], "product_return_rate_pti"),
        (["courier_id"], "courier_return_rate_pti"),
    ]:
        out = add_group_return_rate_pti(out, group_cols, output_col)
    return out


def add_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_band"] = pd.qcut(
        pd.to_numeric(out["total_amount"], errors="coerce").fillna(0).rank(method="first"),
        q=5,
        labels=["price_q1", "price_q2", "price_q3", "price_q4", "price_q5"],
    ).astype(str)
    out["discount_band"] = pd.cut(
        pd.to_numeric(out["total_discount_pct"], errors="coerce").fillna(0),
        bins=[-0.001, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["<=5%", "5-10%", "10-15%", "15-25%", ">25%"],
        include_lowest=True,
    ).astype(str)
    out["rating_band"] = pd.cut(
        pd.to_numeric(out["product_rating"], errors="coerce").fillna(0),
        bins=[0, 3.8, 4.2, 4.6, 5.0],
        labels=["<=3.8", "3.8-4.2", "4.2-4.6", ">4.6"],
        include_lowest=True,
    ).astype(str)
    out["is_high_value_order"] = pd.to_numeric(out["total_amount"], errors="coerce").fillna(0).gt(out["total_amount"].median()).astype(int)
    out["is_fragile_cod"] = (out["is_fragile"].astype(int).eq(1) & out["payment_method"].eq("COD")).astype(int)
    out["is_remote_cod"] = (out["province"].isin(["Remote_Area", "Phuket", "Songkhla"]) & out["payment_method"].eq("COD")).astype(int)
    out["is_fashion_cod"] = (out["category"].eq("Fashion") & out["payment_method"].eq("COD")).astype(int)
    out["logistics_risk"] = pd.to_numeric(out["damage_rate"], errors="coerce").fillna(0) * out["is_fragile"].astype(int)
    out["category_province"] = out["category"].astype(str) + "_" + out["province"].astype(str)
    out["brand_channel"] = out["brand"].astype(str) + "_" + out["channel_type"].astype(str)

    for group_cols, output_col in [
        (["province", "category"], "province_category_return_rate_pti"),
        (["brand"], "brand_return_rate_pti"),
        (["payment_method"], "payment_return_rate_pti"),
        (["channel_type"], "channel_return_rate_pti"),
        (["courier_type"], "courier_type_return_rate_pti"),
    ]:
        out = add_group_return_rate_pti(out, group_cols, output_col)
    return out


def selected_features(version: int, df: pd.DataFrame) -> list[str]:
    if version == 1:
        features = BASE_FEATURES
    elif version == 2:
        features = BASE_FEATURES + V2_FEATURES
    elif version == 3:
        features = BASE_FEATURES + V2_FEATURES + V3_FEATURES
    elif version == 4:
        features = BASE_FEATURES + V2_FEATURES + V3_FEATURES + V4_FEATURES
    elif version == 5:
        features = V5_COMPACT_KEEP
    else:
        raise ValueError(version)
    return pct_feature_list(features, df)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder()),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float | int] = {}
    for threshold in np.linspace(0.25, 0.75, 51):
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        accuracy = accuracy_score(y_true, pred)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        cost = int(fn * 500 + fp * 50)
        score = accuracy * 0.55 + f1 * 0.25 + recall * 0.15 - (cost / 500000)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": float(accuracy),
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
                "cost": cost,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
    return best_threshold, best_metrics


def train_version_model(version: int, df: pd.DataFrame, features: list[str], input_path: Path, previous_features: list[str]) -> VersionResult:
    version_dir = OUT_ROOT / f"version_{version}"
    data_dir = version_dir / "data"
    model_dir = version_dir / "models"
    report_dir = version_dir / "reports"

    dataset_path = data_dir / f"dataset_version_{version}.csv"
    featured_path = data_dir / f"df_featured_version_{version}.csv"
    used_path = data_dir / f"used_features_version_{version}.csv"
    dropped_path = data_dir / f"dropped_features_version_{version}.csv"
    model_path = model_dir / f"model_version_{version}_xgboost.pkl"
    train_test_path = data_dir / f"train_test_sets_version_{version}.pkl"
    metrics_path = report_dir / f"metrics_version_{version}.csv"
    predictions_path = report_dir / f"test_predictions_version_{version}.csv"

    x = df[features].copy()
    y = df[TARGET].astype(int).copy()
    df_featured = pd.concat([df[["order_id", "customer_id", "order_date"]].copy(), x, y.rename(TARGET)], axis=1)

    train_idx, test_idx = train_test_split(
        np.arange(len(df_featured)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=420,
        max_depth=4,
        learning_rate=0.045,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    threshold, _ = choose_threshold(y_train.to_numpy(), train_proba)
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    metrics = {
        "version": f"V{version}",
        "model": "XGBoost",
        "rows": len(df_featured),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "feature_count": len(features),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, test_proba)),
        "avg_precision": float(average_precision_score(y_test, test_proba)),
        "cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "input_dataset": str(input_path.relative_to(ROOT)),
        "output_dataset": str(dataset_path.relative_to(ROOT)),
        "df_featured": str(featured_path.relative_to(ROOT)),
        "train_test_path": str(train_test_path.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
    }

    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    df_featured.to_csv(featured_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(used_path, index=False, encoding="utf-8-sig")
    dropped = [feature for feature in previous_features if feature not in features]
    pd.DataFrame({"dropped_feature": dropped}).to_csv(dropped_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_index": train_idx,
            "test_index": test_idx,
            "feature_names": features,
            "threshold": threshold,
        },
        train_test_path,
    )
    joblib.dump(pipeline, model_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "order_id": df.iloc[test_idx]["order_id"].to_numpy(),
            "customer_id": df.iloc[test_idx]["customer_id"].to_numpy(),
            "actual_is_returned": y_test.to_numpy(),
            "predict_probability_return": test_proba,
            "predicted_is_returned": test_pred,
            "threshold": threshold,
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return VersionResult(
        version=f"V{version}",
        input_dataset=metrics["input_dataset"],
        output_dataset=metrics["output_dataset"],
        df_featured=metrics["df_featured"],
        model_path=metrics["model_path"],
        train_test_path=metrics["train_test_path"],
        feature_count=len(features),
        dropped_from_previous=len(dropped),
        accuracy=metrics["accuracy"],
        recall=metrics["recall"],
        precision=metrics["precision"],
        f1=metrics["f1"],
        auc=metrics["auc"],
        avg_precision=metrics["avg_precision"],
        cost=metrics["cost"],
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def write_version_readme(version: int, result: VersionResult, features: list[str], previous_path: Path) -> None:
    version_dir = OUT_ROOT / f"version_{version}"
    added_note = {
        1: f"V1 เริ่มจาก clean_dataset_{DATASET_SIZE} และใช้ raw/order-time features เป็น baseline",
        2: "V2 รับ dataset จาก V1 แล้วเพิ่ม customer history และ rolling history แบบ point-in-time",
        3: "V3 รับ dataset จาก V2 แล้วเพิ่ม business interaction และ group return-rate features",
        4: "V4 รับ dataset จาก V3 แล้วเพิ่ม segment/operation risk features",
        5: "V5 รับ dataset จาก V4 แล้วลด feature ให้เป็น compact feature set เพื่อดูว่าตัด feature แล้ว performance ยังดีไหม",
    }[version]
    lines = [
        f"# Sequential Dataset {DATASET_SIZE} - Version {version}",
        "",
        added_note,
        "",
        f"- Input dataset: `{previous_path.relative_to(ROOT)}`",
        f"- Output dataset: `{result.output_dataset}`",
        f"- Featured dataset: `{result.df_featured}`",
        f"- Model: `{result.model_path}`",
        f"- Train/test artifact: `{result.train_test_path}`",
        f"- Feature count: `{result.feature_count}`",
        f"- Accuracy: `{result.accuracy * 100:.2f}%`",
        f"- Recall: `{result.recall * 100:.2f}%`",
        f"- Precision: `{result.precision * 100:.2f}%`",
        f"- F1: `{result.f1 * 100:.2f}%`",
        f"- AUC: `{result.auc * 100:.2f}%`",
        f"- Cost: `{result.cost:,}`",
        "",
        "## Used Features",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in features)
    (version_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(summary: pd.DataFrame) -> None:
    image_dir = OUT_ROOT / "images"
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]
    labels = summary["version"].tolist()

    def draw_axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, max_value: float = 100.0) -> None:
        draw.line((x0, y1, x1, y1), fill="#263238", width=2)
        draw.line((x0, y0, x0, y1), fill="#263238", width=2)
        for tick in [0, 25, 50, 75, 100]:
            yy = int(y1 - tick / max_value * (y1 - y0))
            draw.line((x0 - 6, yy, x1, yy), fill="#ECEFF1", width=1)
            draw.text((x0 - 12, yy), str(tick), font=font(18), fill="#455A64", anchor="rm")

    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((width // 2, 50), f"Dataset {DATASET_SIZE} Sequential V1-V5 Accuracy", font=font(40), fill="#111111", anchor="ma")
    x0, y0, x1, y1 = 120, 150, 1500, 720
    draw_axes(draw, x0, y0, x1, y1)
    bar_space = (x1 - x0 - 80) // len(summary)
    for i, (_, row) in enumerate(summary.iterrows()):
        value = float(row["accuracy"]) * 100
        bx = x0 + 55 + i * bar_space
        bw = 120
        by = int(y1 - value / 100 * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=7, fill=colors[i])
        draw.text((bx + bw // 2, by - 10), f"{value:.2f}%", font=font(24), fill="#111111", anchor="ms")
        draw.text((bx + bw // 2, y1 + 20), str(row["version"]), font=font(25), fill="#111111", anchor="ma")
    img.save(image_dir / f"dataset_{DATASET_SIZE}_sequential_accuracy_v1_to_v5.png")

    img = Image.new("RGB", (1800, 1000), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((900, 50), f"Dataset {DATASET_SIZE} Sequential V1-V5 Performance Metrics", font=font(38), fill="#111111", anchor="ma")
    x0, y0, x1, y1 = 120, 160, 1700, 780
    draw_axes(draw, x0, y0, x1, y1)
    metric_specs = [
        ("accuracy", "Accuracy", "#1f77b4"),
        ("recall", "Recall", "#d62728"),
        ("f1", "F1", "#2ca02c"),
        ("auc", "AUC", "#9467bd"),
    ]
    group_space = (x1 - x0 - 80) // len(summary)
    bw = 38
    for i, (_, row) in enumerate(summary.iterrows()):
        center = x0 + 90 + i * group_space
        for j, (col, _, color) in enumerate(metric_specs):
            value = float(row[col]) * 100
            bx = center + (j - 1.5) * (bw + 8)
            by = int(y1 - value / 100 * (y1 - y0))
            draw.rectangle((bx, by, bx + bw, y1), fill=color)
        draw.text((center + bw, y1 + 20), str(row["version"]), font=font(23), fill="#111111", anchor="ma")
    legend_x = 1220
    for j, (_, label, color) in enumerate(metric_specs):
        yy = 840 + j * 36
        draw.rectangle((legend_x, yy, legend_x + 24, yy + 24), fill=color)
        draw.text((legend_x + 34, yy - 2), label, font=font(22), fill="#111111")
    img.save(image_dir / f"dataset_{DATASET_SIZE}_sequential_metrics_v1_to_v5.png")

    img = Image.new("RGB", (1600, 900), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((800, 50), f"Dataset {DATASET_SIZE} Sequential Feature Count and Cost", font=font(38), fill="#111111", anchor="ma")
    x0, y0, x1, y1 = 120, 150, 1500, 720
    max_features = max(float(summary["feature_count"].max()), 1.0)
    max_cost = max(float(summary["cost"].max()), 1.0)
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    draw.text((80, 130), "Features", font=font(18), fill="#455A64", anchor="ma")
    bar_space = (x1 - x0 - 80) // len(summary)
    cost_points: list[tuple[int, int]] = []
    for i, (_, row) in enumerate(summary.iterrows()):
        feature_value = float(row["feature_count"])
        cost_value = float(row["cost"])
        bx = x0 + 55 + i * bar_space
        bw = 110
        by = int(y1 - feature_value / max_features * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=7, fill="#607D8B")
        draw.text((bx + bw // 2, by - 8), str(int(feature_value)), font=font(21), fill="#111111", anchor="ms")
        draw.text((bx + bw // 2, y1 + 20), str(row["version"]), font=font(24), fill="#111111", anchor="ma")
        cy = int(y1 - cost_value / max_cost * (y1 - y0))
        cost_points.append((bx + bw // 2, cy))
    for start, end in zip(cost_points, cost_points[1:]):
        draw.line((start, end), fill="#C62828", width=4)
    for x, y in cost_points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#C62828")
    draw.text((1250, 812), "Bar = feature count", font=font(21), fill="#607D8B")
    draw.text((1250, 846), "Line = cost", font=font(21), fill="#C62828")
    img.save(image_dir / f"dataset_{DATASET_SIZE}_sequential_feature_count_cost.png")


def write_root_readme(summary: pd.DataFrame) -> None:
    best = summary.sort_values(["accuracy", "f1", "auc"], ascending=False).iloc[0]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {row['accuracy'] * 100:.2f}% | "
            f"{row['recall'] * 100:.2f}% | {row['precision'] * 100:.2f}% | {row['f1'] * 100:.2f}% | "
            f"{row['auc'] * 100:.2f}% | {int(row['cost']):,} |"
        )
    table = "\n".join(
        [
            "| Version | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )
    content = f"""# Dataset {DATASET_SIZE} Sequential V1 to V5 Pipeline

งานนี้ทำตามแนวคิดที่ dataset ไหลต่อกันเป็น version chain:

`clean_dataset_{DATASET_SIZE}` -> `dataset_version_1` -> `dataset_version_2` -> `dataset_version_3` -> `dataset_version_4` -> `dataset_version_5`

ทุก version train/evaluate ด้วย XGBoost และใช้ train/test split เดียวกัน (`random_state=42`, test size 20%) เพื่อให้เห็นผลของ feature engineering ที่เพิ่มหรือลดในแต่ละ version

## Summary

Best by Accuracy: `{best['version']}` = `{best['accuracy'] * 100:.2f}%`

{table}

## Version Logic

- V1: baseline จาก clean dataset ใช้ raw/order-time features
- V2: เพิ่ม customer history และ rolling history เพื่อจับประวัติการคืนย้อนหลัง
- V3: เพิ่ม business interaction เช่น category/payment, category/channel, province/payment และ group return-rate แบบ point-in-time
- V4: เพิ่ม segment/operation risk เช่น province-category, courier-type, price/discount/rating bands
- V5: ลด feature จาก V4 ให้เป็น compact feature set เพื่อทดสอบว่าตัด feature แล้ว performance ยังดีพอไหม

## Outputs

- `sequential_pipeline_summary.csv`
- `images/dataset_{DATASET_SIZE}_sequential_accuracy_v1_to_v5.png`
- `images/dataset_{DATASET_SIZE}_sequential_metrics_v1_to_v5.png`
- `images/dataset_{DATASET_SIZE}_sequential_feature_count_cost.png`
- `version_1` ถึง `version_5` แต่ละ folder มี dataset, df_featured, model, train_test artifact, metrics, predictions
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def run_pipeline() -> None:
    ensure_dirs()
    source = clean_dataset(pd.read_csv(INPUT_PATH, low_memory=False))

    current = add_v1_features(source)
    current_path = INPUT_PATH
    previous_features: list[str] = []
    results: list[VersionResult] = []

    for version in range(1, 6):
        if version == 1:
            current = add_v1_features(current)
        elif version == 2:
            current = add_customer_history(current)
        elif version == 3:
            current = add_v3_features(current)
        elif version == 4:
            current = add_v4_features(current)
        elif version == 5:
            # V5 intentionally reuses V4's data columns but trains on a reduced compact feature set.
            current = current.copy()

        features = selected_features(version, current)
        result = train_version_model(version, current, features, current_path, previous_features)
        results.append(result)
        write_version_readme(version, result, features, current_path)

        current_path = OUT_ROOT / f"version_{version}" / "data" / f"dataset_version_{version}.csv"
        previous_features = features

    summary = pd.DataFrame([result.__dict__ for result in results])
    summary.to_csv(OUT_ROOT / "sequential_pipeline_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "sequential_pipeline_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_results(summary)
    write_root_readme(summary)
    print(summary[["version", "feature_count", "accuracy", "recall", "precision", "f1", "auc", "cost"]].to_string(index=False))
    print(f"Saved sequential pipeline to {OUT_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential V1-V5 feature/model pipeline for a dataset folder.")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5000,
        choices=[5000, 50000],
        help="Dataset folder size under docs/test, for example 5000 or 50000.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_dataset(args.dataset_size)
    run_pipeline()
