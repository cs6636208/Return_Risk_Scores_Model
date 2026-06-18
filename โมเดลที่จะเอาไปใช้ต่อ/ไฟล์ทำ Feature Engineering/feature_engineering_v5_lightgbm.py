"""
Feature Engineering for LightGBM V5.

Purpose:
- Convert clean/raw order data into the 64-feature format required by LightGBM V5.
- Support both training data and future company data.
- Use point-in-time logic: only orders before the current order are used for history features.

Important:
- This script builds a production-style feature set.
- It does not use post-event leakage fields such as return_date, refund_amount, return_reason.
- If a future company dataset has no target column `is_returned`, the script still creates features
  and leaves the target column out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = BASE_DIR / "ข้อมูล clean และ real" / "clean_dataset_s1.csv"
DEFAULT_HISTORY_CSV = None
DEFAULT_FEATURE_LIST = BASE_DIR / "โมเดล" / "features" / "used_features_lgbm_s1_v5.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "ไฟล์ทำ Feature Engineering" / "outputs"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "df_featured_v5_from_input.csv"

ID_COLS = ["order_id", "customer_id", "order_date"]
TARGET_COL = "is_returned"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def load_required_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"Feature list must contain column 'feature': {path}")
    return df["feature"].astype(str).tolist()


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def safe_divide(numerator, denominator):
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (pd.Series(numerator) / denominator).fillna(0)


def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "age": 0,
        "membership_tier": "Unknown",
        "province": "Unknown",
        "category": "Unknown",
        "brand": "Unknown",
        "is_fragile": 0,
        "product_rating": 0.0,
        "damage_rate": 0.0,
        "courier_type": "Unknown",
        "promo_type": "No Promotion",
        "promo_discount_rate": 0.0,
        "channel_type": "Unknown",
        "payment_method": "Unknown",
        "quantity": 1,
        "unit_price": 0.0,
        "total_discount_pct": 0.0,
        "total_amount": 0.0,
        "delivery_time_expected_days": 0,
        "registration_date": pd.NaT,
        "product_id": "UNKNOWN_PRODUCT",
        "courier_id": "UNKNOWN_COURIER",
        "supplier_id": "UNKNOWN_SUPPLIER",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def add_customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "order_date", "_source_order"]).copy()

    is_returned = df[TARGET_COL].fillna(0).astype(int) if TARGET_COL in df.columns else pd.Series(0, index=df.index)

    df["total_orders_before"] = df.groupby("customer_id").cumcount()
    df["total_returns_before"] = (
        is_returned.groupby(df["customer_id"]).cumsum().groupby(df["customer_id"]).shift(1).fillna(0)
    )
    df["customer_return_ratio"] = safe_divide(df["total_returns_before"], df["total_orders_before"])

    df["customer_avg_spend_before"] = (
        df.groupby("customer_id")["total_amount"]
        .expanding()
        .mean()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    is_cod = df["payment_method"].astype(str).str.upper().eq("COD").astype(int)
    df["customer_cod_rate_before"] = (
        is_cod.groupby(df["customer_id"])
        .expanding()
        .mean()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    if "registration_date" in df.columns:
        registration_date = safe_to_datetime(df["registration_date"])
        tenure_days = (df["order_date"] - registration_date).dt.days
        df["customer_tenure_months"] = (tenure_days.fillna(0).clip(lower=0) / 30.0).round(2)
    elif "customer_age_days" in df.columns:
        df["customer_tenure_months"] = (pd.to_numeric(df["customer_age_days"], errors="coerce").fillna(0) / 30.0).round(2)
    else:
        df["customer_tenure_months"] = 0.0

    days_since_last_return = []
    for _, group in df.groupby("customer_id", sort=False):
        last_return_date = None
        for idx, row in group.iterrows():
            if last_return_date is None:
                days_since_last_return.append((idx, -1))
            else:
                days_since_last_return.append((idx, int((row["order_date"] - last_return_date).days)))
            if int(row.get(TARGET_COL, 0) or 0) == 1:
                last_return_date = row["order_date"]

    days_series = pd.Series({idx: value for idx, value in days_since_last_return})
    df["days_since_last_return"] = days_series.reindex(df.index).fillna(-1).astype(int)

    return df


def add_rolling_history_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [7, 30, 90, 365]
    df = df.sort_values(["customer_id", "order_date", "_source_order"]).copy()
    is_returned = df[TARGET_COL].fillna(0).astype(int) if TARGET_COL in df.columns else pd.Series(0, index=df.index)

    for window in windows:
        order_counts = pd.Series(0, index=df.index, dtype="int64")
        return_counts = pd.Series(0, index=df.index, dtype="int64")

        for _, group in df.groupby("customer_id", sort=False):
            group = group.sort_values(["order_date", "_source_order"])
            dates = group["order_date"]
            returns = is_returned.loc[group.index]
            for idx, current_date in dates.items():
                start_date = current_date - pd.Timedelta(days=window)
                prior_mask = (dates < current_date) & (dates >= start_date)
                prior_idx = group.index[prior_mask]
                order_counts.loc[idx] = len(prior_idx)
                return_counts.loc[idx] = int(returns.loc[prior_idx].sum()) if len(prior_idx) else 0

        df[f"hist_order_count_{window}d"] = order_counts
        df[f"hist_return_rate_{window}d"] = safe_divide(return_counts, order_counts)

    return df


def add_group_return_rate_pti(df: pd.DataFrame, group_col: str, output_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        df[output_col] = 0.0
        return df

    df = df.sort_values(["order_date", "_source_order"]).copy()
    is_returned = df[TARGET_COL].fillna(0).astype(int) if TARGET_COL in df.columns else pd.Series(0, index=df.index)

    group_key = df[group_col].fillna("Unknown").astype(str)
    prior_count = df.groupby(group_key).cumcount()
    prior_return = is_returned.groupby(group_key).cumsum().groupby(group_key).shift(1).fillna(0)
    df[output_col] = safe_divide(prior_return, prior_count)
    return df


def add_interaction_and_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_cod"] = df["payment_method"].astype(str).str.upper().eq("COD").astype(int)
    df["is_high_discount"] = (pd.to_numeric(df["total_discount_pct"], errors="coerce").fillna(0) >= 0.20).astype(int)
    df["low_rating_alert"] = (pd.to_numeric(df["product_rating"], errors="coerce").fillna(0) < 3.5).astype(int)

    df["discount_amount_ratio"] = safe_divide(
        pd.to_numeric(df.get("discount_applied_amount", 0), errors="coerce").fillna(0),
        pd.to_numeric(df["total_amount"], errors="coerce").fillna(0),
    )
    df["amount_per_item"] = safe_divide(
        pd.to_numeric(df["total_amount"], errors="coerce").fillna(0),
        pd.to_numeric(df["quantity"], errors="coerce").fillna(1),
    )
    df["log_total_amount"] = np.log1p(pd.to_numeric(df["total_amount"], errors="coerce").fillna(0).clip(lower=0))

    df["category_payment"] = df["category"].astype(str) + "_" + df["payment_method"].astype(str)
    df["category_channel"] = df["category"].astype(str) + "_" + df["channel_type"].astype(str)
    df["province_payment"] = df["province"].astype(str) + "_" + df["payment_method"].astype(str)
    df["category_province"] = df["category"].astype(str) + "_" + df["province"].astype(str)

    df["is_fragile_cod"] = ((pd.to_numeric(df["is_fragile"], errors="coerce").fillna(0) == 1) & (df["is_cod"] == 1)).astype(int)
    df["high_discount_cod"] = ((df["is_high_discount"] == 1) & (df["is_cod"] == 1)).astype(int)
    df["low_rating_high_discount"] = ((df["low_rating_alert"] == 1) & (df["is_high_discount"] == 1)).astype(int)

    rating = pd.to_numeric(df["product_rating"], errors="coerce").fillna(0)
    damage = pd.to_numeric(df["damage_rate"], errors="coerce").fillna(0)
    fragile = pd.to_numeric(df["is_fragile"], errors="coerce").fillna(0)
    total_amount = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)

    category_avg_rating = df.groupby("category")["product_rating"].transform(lambda s: pd.to_numeric(s, errors="coerce").mean()).fillna(rating.mean())
    category_avg_price = df.groupby("category")["total_amount"].transform(lambda s: pd.to_numeric(s, errors="coerce").mean()).replace(0, np.nan)

    df["product_quality_score"] = (rating / 5.0 - damage).clip(lower=0)
    df["product_rating_gap"] = (rating - category_avg_rating).fillna(0)
    df["damage_rating_gap"] = (damage * (5 - rating)).fillna(0)
    df["fragile_damage_risk"] = (fragile * damage).fillna(0)
    df["logistics_risk_score"] = (damage + (fragile * 0.15)).clip(lower=0)
    df["remote_logistics_risk"] = (
        df["province"].astype(str).str.contains("Remote|Rural|Nakhon|Khon", case=False, regex=True).astype(int)
        * df["logistics_risk_score"]
    )
    df["product_price_index"] = safe_divide(total_amount, category_avg_price).replace([np.inf, -np.inf], 0).fillna(0)

    df["price_band"] = pd.cut(
        total_amount,
        bins=[-np.inf, 1000, 5000, 15000, np.inf],
        labels=["low", "medium", "high", "premium"],
    ).astype(str)
    df["discount_band"] = pd.cut(
        pd.to_numeric(df["total_discount_pct"], errors="coerce").fillna(0),
        bins=[-np.inf, 0.0, 0.10, 0.20, np.inf],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    df["rating_band"] = pd.cut(
        rating,
        bins=[-np.inf, 2.5, 3.5, 4.5, np.inf],
        labels=["low", "fair", "good", "excellent"],
    ).astype(str)

    order_hour = pd.to_numeric(df.get("order_hour", df["order_date"].dt.hour), errors="coerce").fillna(df["order_date"].dt.hour)
    df["order_time_bucket"] = pd.cut(
        order_hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)

    return df


def build_features(input_df: pd.DataFrame, history_df: pd.DataFrame | None, required_features: list[str]) -> pd.DataFrame:
    input_df = input_df.copy()
    input_df["_is_scoring_input"] = 1
    input_df["_source_order"] = np.arange(len(input_df))

    if history_df is not None:
        history_df = history_df.copy()
        history_df["_is_scoring_input"] = 0
        history_df["_source_order"] = np.arange(len(history_df)) - len(history_df)
        combined = pd.concat([history_df, input_df], ignore_index=True, sort=False)
    else:
        combined = input_df

    combined = ensure_base_columns(combined)
    combined["order_date"] = safe_to_datetime(combined["order_date"])
    combined["registration_date"] = safe_to_datetime(combined.get("registration_date", pd.NaT))
    combined[TARGET_COL] = pd.to_numeric(combined[TARGET_COL], errors="coerce") if TARGET_COL in combined.columns else np.nan

    combined = combined.sort_values(["order_date", "_source_order"]).reset_index(drop=True)

    combined = add_customer_history_features(combined)
    combined = add_rolling_history_features(combined, windows=[7, 30, 90, 365])

    group_specs = [
        ("category", "category_return_rate_pti"),
        ("product_id", "product_return_rate_pti"),
        ("brand", "brand_return_rate_pti"),
        ("courier_id", "courier_return_rate_pti"),
        ("payment_method", "payment_return_rate_pti"),
        ("channel_type", "channel_return_rate_pti"),
        ("supplier_id", "supplier_return_rate_pti"),
    ]
    for group_col, output_col in group_specs:
        combined = add_group_return_rate_pti(combined, group_col, output_col)

    combined = add_interaction_and_risk_features(combined)

    output = combined[combined["_is_scoring_input"] == 1].copy()
    output = output.sort_values("_source_order").reset_index(drop=True)

    keep_cols = [col for col in ID_COLS if col in output.columns] + required_features
    if TARGET_COL in output.columns:
        keep_cols.append(TARGET_COL)

    for col in required_features:
        if col not in output.columns:
            output[col] = 0

    output = output[keep_cols]

    for col in output.columns:
        if output[col].dtype == "object":
            output[col] = output[col].fillna("Unknown")
        else:
            output[col] = output[col].replace([np.inf, -np.inf], 0).fillna(0)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightGBM V5 feature dataset from clean/raw CSV.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Clean/raw CSV to transform.")
    parser.add_argument(
        "--history-csv",
        default=DEFAULT_HISTORY_CSV,
        help="Optional prior history CSV. Use this when transforming future/company data.",
    )
    parser.add_argument("--feature-list", default=str(DEFAULT_FEATURE_LIST), help="V5 required feature list.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output featured CSV.")
    parser.add_argument("--metadata-json", default=None, help="Optional metadata JSON output path.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    history_path = Path(args.history_csv) if args.history_csv else None
    feature_list_path = Path(args.feature_list)
    output_path = Path(args.output_csv)
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_path.with_suffix(".metadata.json")

    required_features = load_required_features(feature_list_path)
    input_df = read_csv(input_path)
    history_df = read_csv(history_path) if history_path else None

    featured = build_features(input_df, history_df, required_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False, encoding="utf-8-sig")

    metadata = {
        "input_csv": str(input_path),
        "history_csv": str(history_path) if history_path else None,
        "feature_list": str(feature_list_path),
        "output_csv": str(output_path),
        "rows": int(len(featured)),
        "columns": int(len(featured.columns)),
        "feature_count": len(required_features),
        "has_target": TARGET_COL in featured.columns,
        "mode": "history_plus_input_point_in_time" if history_df is not None else "input_only_point_in_time",
        "note": "Feature values are built with point-in-time logic and may not exactly match old saved experimental df if that df was generated by a different pipeline.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("LightGBM V5 Feature Engineering")
    print("=" * 72)
    print(f"Input rows     : {len(input_df):,}")
    print(f"History rows   : {0 if history_df is None else len(history_df):,}")
    print(f"Output rows    : {len(featured):,}")
    print(f"Output columns : {len(featured.columns):,}")
    print(f"Feature count  : {len(required_features):,}")
    print(f"Has target     : {TARGET_COL in featured.columns}")
    print(f"Saved CSV      : {output_path}")
    print(f"Saved metadata : {metadata_path}")


if __name__ == "__main__":
    main()
