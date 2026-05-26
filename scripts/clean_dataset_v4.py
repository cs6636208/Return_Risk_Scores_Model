from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "clean_dataset_v4.csv"
AUDIT_JSON_PATH = ROOT / "reports" / "data_cleaning_v4" / "clean_dataset_v4_audit.json"
AUDIT_CSV_PATH = ROOT / "reports" / "data_cleaning_v4" / "clean_dataset_v4_column_quality.csv"


DATE_COLUMNS = [
    "order_date",
    "expected_delivery_date",
    "delivery_date",
    "registration_date",
    "promo_start_date",
    "promo_end_date",
    "return_date",
    "scored_at",
]

NUMERIC_RULES = {
    "age": (18, 90),
    "customer_age_days": (0, None),
    "product_rating": (1.0, 5.0),
    "avg_delivery_days": (0.1, 30.0),
    "damage_rate": (0.0, 1.0),
    "promo_discount_rate": (0.0, 1.0),
    "quantity": (1, None),
    "unit_price": (0.01, None),
    "tier_discount_pct": (0.0, 1.0),
    "campaign_discount_pct": (0.0, 1.0),
    "total_discount_pct": (0.0, 1.0),
    "discount_applied_amount": (0.0, None),
    "total_amount": (0.0, None),
    "delivery_time_expected_days": (1, None),
    "delivery_days": (0, None),
    "delay_days": (None, None),
    "is_repurchased_item": (0, 1),
    "order_hour": (0, 23),
    "days_since_last_order": (0, None),
    "hist_order_count": (0, None),
    "hist_return_rate": (0.0, 1.0),
    "refund_amount": (0.0, None),
    "risk_score": (0.0, 1.0),
    "is_returned": (0, 1),
}

RETURN_FIELDS = [
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
]

TEXT_SENTINELS = {
    "promo_id": "NO_PROMO",
    "promo_name": "No_Promotion",
    "promo_type": "No_Promotion",
    "return_id": "NO_RETURN",
    "return_reason": "No_Return",
    "return_scenario": "No_Return",
    "item_condition": "No_Return",
    "return_status": "No_Return",
    "risk_tier": "Not_Scored",
    "score_id": "NO_SCORE",
    "shap_values": "[]",
}


def record(audit: list[dict], step: str, detail: str, rows: int | None = None) -> None:
    audit.append({"step": step, "detail": detail, "rows_affected": rows})


def strip_text_columns(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        before_missing = int(df[col].isna().sum())
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        after_missing = int(df[col].isna().sum())
        if after_missing != before_missing:
            record(audit, "strip_text", f"normalized blank strings in {col}", after_missing - before_missing)
    return df


def parse_dates(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue
        missing_before = int(df[col].isna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce")
        missing_after = int(df[col].isna().sum())
        if missing_after > missing_before:
            record(audit, "parse_dates", f"coerced invalid dates in {col} to missing", missing_after - missing_before)
    return df


def clean_numeric(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    for col, (lower, upper) in NUMERIC_RULES.items():
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        coerced = int(raw.isna().sum() - df[col].isna().sum())
        if coerced > 0:
            record(audit, "numeric_type", f"coerced non-numeric values in {col}", coerced)
        clipped = raw.copy()
        if lower is not None:
            mask = clipped.lt(lower) & clipped.notna()
            if mask.any():
                record(audit, "numeric_clip", f"clipped {col} below {lower}", int(mask.sum()))
            clipped = clipped.clip(lower=lower)
        if upper is not None:
            mask = clipped.gt(upper) & clipped.notna()
            if mask.any():
                record(audit, "numeric_clip", f"clipped {col} above {upper}", int(mask.sum()))
            clipped = clipped.clip(upper=upper)
        df[col] = clipped
    for col in ["quantity", "delivery_time_expected_days", "delivery_days", "is_repurchased_item", "order_hour", "hist_order_count", "is_returned"]:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    return df


def fix_business_consistency(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    subtotal = (df["unit_price"] * df["quantity"]).replace(0, np.nan)
    expected_discount_pct = (
        df["tier_discount_pct"].fillna(0) + df["campaign_discount_pct"].fillna(0)
    ).clip(0, 1)
    discount_gap = (df["total_discount_pct"] - expected_discount_pct).abs()
    mask = discount_gap.gt(0.0001)
    if mask.any():
        df.loc[mask, "total_discount_pct"] = expected_discount_pct.loc[mask]
        record(audit, "discount_consistency", "recalculated total_discount_pct", int(mask.sum()))

    expected_discount_amount = (subtotal * df["total_discount_pct"]).fillna(0)
    amount_gap = (df["discount_applied_amount"] - expected_discount_amount).abs()
    mask = amount_gap.gt(0.01)
    if mask.any():
        df.loc[mask, "discount_applied_amount"] = expected_discount_amount.loc[mask]
        record(audit, "discount_consistency", "recalculated discount_applied_amount", int(mask.sum()))

    expected_total = (subtotal - df["discount_applied_amount"]).clip(lower=0)
    total_gap = (df["total_amount"] - expected_total).abs()
    mask = total_gap.gt(0.01)
    if mask.any():
        df.loc[mask, "total_amount"] = expected_total.loc[mask]
        record(audit, "amount_consistency", "recalculated total_amount", int(mask.sum()))

    expected_delay = (df["delivery_date"] - df["expected_delivery_date"]).dt.days
    mask = expected_delay.notna() & df["delay_days"].notna() & (df["delay_days"] != expected_delay)
    if mask.any():
        df.loc[mask, "delay_days"] = expected_delay.loc[mask]
        record(audit, "delivery_consistency", "recalculated delay_days from delivery dates", int(mask.sum()))
    return df


def fill_contextual_missing(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    for col, sentinel in TEXT_SENTINELS.items():
        if col in df.columns:
            missing = int(df[col].isna().sum())
            if missing:
                df[col] = df[col].fillna(sentinel)
                record(audit, "missing_text", f"filled missing {col} with {sentinel}", missing)

    if "promo_discount_rate" in df.columns:
        missing = int(df["promo_discount_rate"].isna().sum())
        if missing:
            df["promo_discount_rate"] = df["promo_discount_rate"].fillna(0)
            record(audit, "missing_numeric", "filled missing promo_discount_rate with 0", missing)

    no_return = df["is_returned"].eq(0)
    if "return_date" in df.columns:
        expected_missing = int((no_return & df["return_date"].isna()).sum())
        if expected_missing:
            record(
                audit,
                "accepted_missing",
                "kept return_date missing for non-return rows because it is a post-event field excluded from training",
                expected_missing,
            )
    if "refund_amount" in df.columns:
        affected = int((no_return & df["refund_amount"].isna()).sum())
        df.loc[no_return, "refund_amount"] = df.loc[no_return, "refund_amount"].fillna(0)
        if affected:
            record(audit, "return_consistency", "filled refund_amount for non-return rows with 0", affected)

    for col in RETURN_FIELDS:
        if col not in df.columns or col == "refund_amount":
            continue
        sentinel = TEXT_SENTINELS.get(col)
        if sentinel is None:
            continue
        affected = int((no_return & df[col].isna()).sum())
        df.loc[no_return, col] = df.loc[no_return, col].fillna(sentinel)
        if affected:
            record(audit, "return_consistency", f"filled {col} for non-return rows", affected)

    numeric_cols = df.select_dtypes(include=[np.number, "Int64", "Float64"]).columns
    for col in numeric_cols:
        missing = int(df[col].isna().sum())
        if missing and col != "refund_amount":
            median = df[col].median()
            df[col] = df[col].fillna(median if pd.notna(median) else 0)
            record(audit, "missing_numeric", f"filled missing {col} with median", missing)

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        missing = int(df[col].isna().sum())
        if missing and col not in DATE_COLUMNS:
            df[col] = df[col].fillna("Unknown")
            record(audit, "missing_text", f"filled remaining missing {col} with Unknown", missing)
    return df


def remove_duplicates(df: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed_rows = before - len(df)
    if removed_rows:
        record(audit, "duplicates", "dropped exact duplicate rows", removed_rows)

    before = len(df)
    df = df.sort_values(["order_date", "order_id"]).drop_duplicates("order_id", keep="last")
    removed_orders = before - len(df)
    if removed_orders:
        record(audit, "duplicates", "dropped duplicate order_id rows", removed_orders)
    return df


def write_quality_report(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    rows = []
    for col in cleaned.columns:
        rows.append(
            {
                "column": col,
                "dtype_before": str(original[col].dtype) if col in original.columns else "",
                "dtype_after": str(cleaned[col].dtype),
                "missing_before": int(original[col].isna().sum()) if col in original.columns else None,
                "missing_after": int(cleaned[col].isna().sum()),
                "unique_after": int(cleaned[col].nunique(dropna=False)),
            }
        )
    pd.DataFrame(rows).to_csv(AUDIT_CSV_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit: list[dict] = []

    original = pd.read_csv(INPUT_PATH, low_memory=False)
    df = original.copy()
    record(audit, "input", f"loaded {INPUT_PATH.name}", len(df))

    df = strip_text_columns(df, audit)
    df = parse_dates(df, audit)
    df = clean_numeric(df, audit)
    df = fix_business_consistency(df, audit)
    df = fill_contextual_missing(df, audit)
    df = remove_duplicates(df, audit)

    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").replace("NaT", "")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    write_quality_report(original, df)

    target_distribution = {
        str(key): int(value)
        for key, value in df["is_returned"].value_counts(dropna=False).to_dict().items()
    }
    summary = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_rows": int(len(original)),
        "output_rows": int(len(df)),
        "input_columns": int(original.shape[1]),
        "output_columns": int(df.shape[1]),
        "target_distribution": target_distribution,
        "audit_steps": audit,
    }
    AUDIT_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {OUTPUT_PATH}")
    print(f"[OK] Wrote {AUDIT_JSON_PATH}")
    print(f"[OK] Wrote {AUDIT_CSV_PATH}")


if __name__ == "__main__":
    main()
