from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"

SOURCE_PATH = DATA_DIR / "clean_dataset_v2_high_signal.csv"
ENGINEERED_PATH = DATA_DIR / "df_engineered_v2_HIGH_ACCURACY.csv"
USED_FEATURES_PATH = DATA_DIR / "used_features_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv"

LOOKBACK_WINDOWS = [30, 60, 90, 180, 365]

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

IDENTIFIER_COLUMNS = {
    "customer_name",
    "customer_phone",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "supplier_contact",
    "courier_id",
    "courier_name",
    "promo_id",
    "promo_name",
}


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator_arr, dtype=float)
    np.divide(numerator_arr, denominator_arr, out=out, where=denominator_arr != 0)
    return out


def add_customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time history so the current order never counts itself."""
    df = df.sort_values(["customer_id", "order_date", "order_id"]).copy()

    history_frames = []
    for _, group in df.groupby("customer_id", sort=False):
        g = group.copy()
        dates = g["order_date"].to_numpy(dtype="datetime64[ns]")
        returns = g["is_returned"].astype(float).to_numpy()
        spend = g["total_amount"].astype(float).to_numpy()
        order_index = np.arange(len(g))

        cum_returns_before = np.cumsum(returns) - returns
        g["hist_order_count"] = order_index
        g["hist_return_rate"] = safe_divide(cum_returns_before, order_index)

        prev_order_date = g["order_date"].shift(1)
        g["days_since_last_order"] = (g["order_date"] - prev_order_date).dt.days.fillna(999).clip(lower=0)

        prior_return_dates = g["order_date"].where(g["is_returned"].astype(int).eq(1)).shift(1).ffill()
        g["days_since_last_return"] = (g["order_date"] - prior_return_dates).dt.days.fillna(999).clip(lower=0)

        prefix_returns = np.r_[0.0, np.cumsum(returns)]
        prefix_spend = np.r_[0.0, np.cumsum(spend)]
        for days in LOOKBACK_WINDOWS:
            window_start = dates - np.timedelta64(days, "D")
            left_idx = np.searchsorted(dates, window_start, side="left")
            right_idx = np.arange(len(g))

            order_count = right_idx - left_idx
            return_count = prefix_returns[right_idx] - prefix_returns[left_idx]
            spend_sum = prefix_spend[right_idx] - prefix_spend[left_idx]

            g[f"hist_order_count_{days}d"] = order_count
            g[f"hist_return_count_{days}d"] = return_count
            g[f"hist_return_rate_{days}d"] = safe_divide(return_count, order_count)
            g[f"hist_spend_sum_{days}d"] = spend_sum

        history_frames.append(g)

    return pd.concat(history_frames).sort_index()


def build_engineered_dataset(source_path: Path = SOURCE_PATH) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    for col in ["order_date", "expected_delivery_date", "registration_date", "promo_start_date", "promo_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["customer_tenure_months"] = ((df["order_date"] - df["registration_date"]).dt.days / 30.44).fillna(0).clip(lower=0)
    df["discount_amount_ratio"] = safe_divide(df["discount_applied_amount"], (df["unit_price"] * df["quantity"]).replace(0, np.nan))
    df["amount_per_item"] = safe_divide(df["total_amount"], df["quantity"].replace(0, np.nan))
    df["log_total_amount"] = np.log1p(df["total_amount"].clip(lower=0))
    df["high_value_order"] = (df["total_amount"] >= df["total_amount"].quantile(0.75)).astype(int)

    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["under_25", "25_34", "35_44", "45_54", "55_plus"],
        include_lowest=True,
    ).astype(str)

    payment = df["payment_method"].astype(str).str.lower()
    df["is_cod"] = payment.str.contains("cod|cash").astype(int)
    df["is_bank_transfer"] = payment.str.contains("bank|transfer").astype(int)
    df["is_credit_card"] = payment.str.contains("credit|card").astype(int)

    df["is_high_discount"] = (df["total_discount_pct"] >= 0.20).astype(int)
    df["low_rating_alert"] = (df["product_rating"] < 3.5).astype(int)
    df["logistics_risk"] = (
        (df["damage_rate"] >= df["damage_rate"].quantile(0.75))
        | (df["avg_delivery_days"] >= df["avg_delivery_days"].quantile(0.75))
    ).astype(int)

    df["category_payment"] = df["category"].astype(str) + "__" + df["payment_method"].astype(str)
    df["category_channel"] = df["category"].astype(str) + "__" + df["channel_type"].astype(str)
    df["province_payment"] = df["province"].astype(str) + "__" + df["payment_method"].astype(str)
    df["tier_payment"] = df["membership_tier"].astype(str) + "__" + df["payment_method"].astype(str)

    df = add_customer_history_features(df)

    keep_columns = [
        "order_id",
        "customer_id",
        "order_date",
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
        "customer_age_days",
        "customer_tenure_months",
        "category",
        "brand",
        "is_fragile",
        "product_rating",
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
        "discount_amount_ratio",
        "total_amount",
        "amount_per_item",
        "log_total_amount",
        "high_value_order",
        "delivery_time_expected_days",
        "is_repurchased_item",
        "order_hour",
        "days_since_last_order",
        "days_since_last_return",
        "hist_order_count",
        "hist_return_rate",
        "order_month",
        "order_dayofweek",
        "is_weekend",
        "age_group",
        "is_cod",
        "is_bank_transfer",
        "is_credit_card",
        "is_high_discount",
        "low_rating_alert",
        "logistics_risk",
        "category_payment",
        "category_channel",
        "province_payment",
        "tier_payment",
    ]

    for days in LOOKBACK_WINDOWS:
        keep_columns.extend(
            [
                f"hist_order_count_{days}d",
                f"hist_return_count_{days}d",
                f"hist_return_rate_{days}d",
                f"hist_spend_sum_{days}d",
            ]
        )

    keep_columns.extend(["is_returned"])
    engineered = df[keep_columns].copy()

    train_idx, test_idx = train_test_split(
        engineered.index,
        test_size=0.20,
        random_state=42,
        stratify=engineered["is_returned"],
    )
    engineered["dataset_split"] = "train"
    engineered.loc[test_idx, "dataset_split"] = "test"
    return engineered


def main() -> None:
    engineered = build_engineered_dataset(SOURCE_PATH)
    engineered.to_csv(ENGINEERED_PATH, index=False, encoding="utf-8-sig")

    used_features = [c for c in engineered.columns if c not in {"order_id", "customer_id", "order_date", "is_returned", "dataset_split"}]
    pd.DataFrame({"feature": used_features}).to_csv(USED_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved engineered dataset: {ENGINEERED_PATH}")
    print(f"Rows: {len(engineered):,}, Columns: {len(engineered.columns):,}, Used features: {len(used_features):,}")


if __name__ == "__main__":
    main()
