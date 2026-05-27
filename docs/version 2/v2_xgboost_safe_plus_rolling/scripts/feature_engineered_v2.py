from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CLEAN_DATA = ROOT / "data" / "processed" / "clean_dataset.csv"

OUTPUT_DF_FEATURED = ROOT / "docs" / "version 2" / "data" / "features" / "df_featured.csv"
OUTPUT_SELECTED_PACKAGE = ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "data" / "df_featured.csv"
OUTPUT_USED_FEATURES = ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "data" / "v2_xgboost_safe_plus_rolling_used_features.csv"
OUTPUT_SUMMARY = ROOT / "docs" / "version 2" / "feature_documentation" / "feature_engineered_v2_summary.csv"

ROLLING_WINDOWS = [30, 60, 90, 180, 365]
TARGET_COLUMN = "is_returned"

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


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def load_clean_dataset(path: Path = INPUT_CLEAN_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)


def add_basic_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["customer_tenure_months"] = ((out["order_date"] - out["registration_date"]).dt.days / 30).fillna(0)
    out["order_month"] = out["order_date"].dt.month.fillna(0).astype(int)
    out["order_dayofweek"] = out["order_date"].dt.dayofweek.fillna(0).astype(int)
    out["is_weekend"] = out["order_dayofweek"].isin([5, 6]).astype(int)
    out["age_group"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce").fillna(0),
        bins=[0, 20, 30, 40, 50, 120],
        labels=["<20", "20-30", "30-40", "40-50", ">50"],
        include_lowest=True,
    ).astype(str)

    out["is_fragile"] = pd.to_numeric(out.get("is_fragile", 0), errors="coerce").fillna(0).astype(int)
    out["damage_rate"] = pd.to_numeric(out.get("damage_rate", 0), errors="coerce").fillna(0)
    out["logistics_risk"] = out["damage_rate"] * out["is_fragile"]
    return out


def add_business_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_cod"] = out["payment_method"].eq("COD").astype(int)
    out["is_high_discount"] = pd.to_numeric(out["total_discount_pct"], errors="coerce").fillna(0).gt(0.20).astype(int)
    out["low_rating_alert"] = pd.to_numeric(out["product_rating"], errors="coerce").fillna(5).lt(4.0).astype(int)
    gross_amount = (
        pd.to_numeric(out["unit_price"], errors="coerce").fillna(0)
        * pd.to_numeric(out["quantity"], errors="coerce").fillna(1)
    ).replace(0, np.nan)
    out["discount_amount_ratio"] = (
        pd.to_numeric(out["discount_applied_amount"], errors="coerce").fillna(0) / gross_amount
    ).fillna(0)
    out["category_payment"] = out["category"].astype(str) + "_" + out["payment_method"].astype(str)
    out["category_channel"] = out["category"].astype(str) + "_" + out["channel_type"].astype(str)
    out["province_payment"] = out["province"].astype(str) + "_" + out["payment_method"].astype(str)
    return out


def add_customer_rolling_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for days in ROLLING_WINDOWS:
        out[f"hist_order_count_{days}d"] = 0
        out[f"hist_return_count_{days}d"] = 0
        out[f"hist_return_rate_{days}d"] = 0.0
        out[f"hist_spend_sum_{days}d"] = 0.0

    for _, group in out.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy()
        returns = pd.to_numeric(group[TARGET_COLUMN], errors="coerce").fillna(0).to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()

        for position, current_date in enumerate(dates):
            if pd.isna(current_date):
                continue
            for days in ROLLING_WINDOWS:
                start_date = current_date - np.timedelta64(days, "D")
                historical_mask = (dates < current_date) & (dates >= start_date)
                order_count = int(historical_mask.sum())
                return_count = int(returns[historical_mask].sum()) if order_count else 0

                out.loc[idx[position], f"hist_order_count_{days}d"] = order_count
                out.loc[idx[position], f"hist_return_count_{days}d"] = return_count
                out.loc[idx[position], f"hist_return_rate_{days}d"] = (
                    return_count / order_count if order_count else 0.0
                )
                out.loc[idx[position], f"hist_spend_sum_{days}d"] = (
                    float(amounts[historical_mask].sum()) if order_count else 0.0
                )
    return out


def selected_v2_xgboost_safe_plus_rolling_features(df: pd.DataFrame) -> list[str]:
    v2_full = existing_columns(df, V2_ORIGINAL_FEATURES)
    order_time_safe = [column for column in v2_full if column not in {"delivery_days", "delay_days"}]
    rolling_features = [
        column
        for column in df.columns
        if column.startswith("hist_order_count_")
        or column.startswith("hist_return_count_")
        or column.startswith("hist_return_rate_")
        or column.startswith("hist_spend_sum_")
    ]
    interaction_features = existing_columns(
        df,
        ["discount_amount_ratio", "category_payment", "category_channel", "province_payment"],
    )
    return existing_columns(df, order_time_safe + rolling_features + interaction_features)


def build_feature_engineered_v2() -> tuple[pd.DataFrame, list[str]]:
    df = load_clean_dataset()
    df = add_basic_v2_features(df)
    df = add_business_interaction_features(df)
    df = add_customer_rolling_history(df)

    selected_features = selected_v2_xgboost_safe_plus_rolling_features(df)
    df_featured = df[selected_features + [TARGET_COLUMN]].copy()
    df_featured["dataset_split"] = "full_feature_frame"
    return df_featured, selected_features


def export_feature_engineered_v2() -> None:
    for output in [OUTPUT_DF_FEATURED, OUTPUT_SELECTED_PACKAGE, OUTPUT_USED_FEATURES, OUTPUT_SUMMARY]:
        output.parent.mkdir(parents=True, exist_ok=True)

    df_featured, selected_features = build_feature_engineered_v2()
    df_featured.to_csv(OUTPUT_DF_FEATURED, index=False, encoding="utf-8-sig")
    df_featured.to_csv(OUTPUT_SELECTED_PACKAGE, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": selected_features}).to_csv(OUTPUT_USED_FEATURES, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "source_data": str(INPUT_CLEAN_DATA.relative_to(ROOT)),
                "output_df_featured": str(OUTPUT_DF_FEATURED.relative_to(ROOT)),
                "selected_package_output": str(OUTPUT_SELECTED_PACKAGE.relative_to(ROOT)),
                "row_count": len(df_featured),
                "feature_count": len(selected_features),
                "target": TARGET_COLUMN,
                "rolling_windows": ", ".join(f"{days}d" for days in ROLLING_WINDOWS),
                "order_time_safe_drop": "delivery_days, delay_days",
                "key_logic": "for each customer/order, use only historical orders where order_date < current_order_date",
            }
        ]
    ).to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("Exported V2 safe plus rolling feature engineering outputs")
    print(OUTPUT_DF_FEATURED)
    print(OUTPUT_SELECTED_PACKAGE)
    print(OUTPUT_USED_FEATURES)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    export_feature_engineered_v2()
