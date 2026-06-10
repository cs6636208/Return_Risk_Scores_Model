from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "POSTGRES_RAW"
OUT_CSV = OUT_DIR / "real_dataset_postgres_raw.csv"
OUT_VALIDATION = OUT_DIR / "real_dataset_postgres_raw_validation.csv"
OUT_README = OUT_DIR / "README.md"

DB_CONTAINER = "oshopping_postgres"
DB_USER = "admin"
DB_NAME = "gmm_oshopping_db"
SOURCE_TABLE = 'public.order_history_rawdata'

MODEL_SCHEMA_COLUMNS = [
    "order_id",
    "order_date",
    "expected_delivery_date",
    "delivery_date",
    "customer_id",
    "customer_name",
    "customer_phone",
    "gender",
    "age",
    "membership_tier",
    "preferred_channel",
    "province",
    "registration_date",
    "customer_age_days",
    "product_id",
    "product_name",
    "category",
    "brand",
    "is_fragile",
    "product_rating",
    "supplier_id",
    "supplier_name",
    "supplier_contact",
    "courier_id",
    "courier_name",
    "courier_type",
    "avg_delivery_days",
    "damage_rate",
    "coverage_region",
    "promo_id",
    "promo_name",
    "promo_type",
    "promo_discount_rate",
    "promo_start_date",
    "promo_end_date",
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
]


def export_raw_table() -> pd.DataFrame:
    sql = f"""
COPY (
    SELECT *
    FROM {SOURCE_TABLE}
    ORDER BY order_date, order_id
) TO STDOUT WITH CSV HEADER
"""
    cmd = [
        "docker",
        "exec",
        DB_CONTAINER,
        "psql",
        "-U",
        DB_USER,
        "-d",
        DB_NAME,
        "-c",
        sql,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    return pd.read_csv(io.BytesIO(result.stdout))


def align_to_model_schema(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    aligned = pd.DataFrame()

    direct_map = {
        "supplier_contact": "contact",
        "promo_discount_rate": "discount_rate",
        "promo_start_date": "start_date",
        "promo_end_date": "end_date",
        "unit_price": "unit_price_x",
    }

    for col in MODEL_SCHEMA_COLUMNS:
        source_col = direct_map.get(col, col)
        if source_col in df.columns:
            aligned[col] = df[source_col]
        else:
            aligned[col] = pd.NA

    aligned["delay_days"] = (
        pd.to_numeric(aligned["delivery_days"], errors="coerce").fillna(0)
        - pd.to_numeric(aligned["delivery_time_expected_days"], errors="coerce").fillna(0)
    ).clip(lower=0)

    aligned["is_returned"] = pd.to_numeric(aligned["is_returned"], errors="coerce").fillna(0).astype(int)
    aligned["is_fragile"] = aligned["is_fragile"].astype(str).str.lower().isin(["true", "t", "1", "yes"]).astype(int)

    not_returned = aligned["is_returned"].eq(0)
    aligned.loc[not_returned, "return_id"] = aligned.loc[not_returned, "return_id"].fillna("NO_RETURN")
    aligned.loc[not_returned, "return_date"] = aligned.loc[not_returned, "return_date"].fillna("Not Returned")
    aligned.loc[not_returned, "return_reason"] = aligned.loc[not_returned, "return_reason"].fillna("Not Returned")
    aligned.loc[not_returned, "return_scenario"] = aligned.loc[not_returned, "return_scenario"].fillna("Not Returned")
    aligned.loc[not_returned, "item_condition"] = aligned.loc[not_returned, "item_condition"].fillna("Not Returned")
    aligned.loc[not_returned, "return_status"] = aligned.loc[not_returned, "return_status"].fillna("Not Returned")
    aligned.loc[not_returned, "refund_amount"] = aligned.loc[not_returned, "refund_amount"].fillna(0)

    text_cols = aligned.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        aligned[col] = (
            aligned[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
            .fillna("Unknown")
        )

    numeric_cols = aligned.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        values = pd.to_numeric(aligned[col], errors="coerce")
        fill = values.median()
        aligned[col] = values.fillna(0 if pd.isna(fill) else fill)

    aligned = aligned.drop_duplicates("order_id", keep="last")
    aligned = aligned.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return aligned[MODEL_SCHEMA_COLUMNS]


def build_validation(df: pd.DataFrame) -> dict[str, object]:
    return {
        "source_table": SOURCE_TABLE,
        "output_file": str(OUT_CSV.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()),
        "distinct_order_id": int(df["order_id"].nunique()),
        "distinct_customer_id": int(df["customer_id"].nunique()),
        "min_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").min()),
        "max_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").max()),
        "returned_count": int(df["is_returned"].eq(1).sum()),
        "not_returned_count": int(df["is_returned"].eq(0).sum()),
        "return_rate": float(df["is_returned"].mean()),
        "note": "This is exported from PostgreSQL raw/history table, not generated from clean_dataset_s1/s2.",
    }


def write_readme(validation: dict[str, object]) -> None:
    content = f"""# PostgreSQL Raw Real Dataset

This folder contains the real/raw-style external test dataset exported from PostgreSQL.

Important distinction:

- `real_dataset_postgres_raw.csv` comes from `{SOURCE_TABLE}` in PostgreSQL.
- It is **not generated from** `clean_dataset_s1.csv` or `clean_dataset_s2.csv`.
- Older files named `real_dataset_s1.csv` / `real_dataset_s2.csv` in the parent folder are synthetic/benchmark datasets because their generation scripts use clean dataset files as distribution sources.

## Validation

- Rows: `{int(validation["rows"]):,}`
- Columns: `{int(validation["columns"])}`
- Missing/null cells after alignment: `{int(validation["missing_or_null_cells"])}`
- Duplicate order_id: `{int(validation["duplicate_order_id"])}`
- Distinct customers: `{int(validation["distinct_customer_id"]):,}`
- Returned: `{int(validation["returned_count"]):,}`
- Not Returned: `{int(validation["not_returned_count"]):,}`
- Return rate: `{float(validation["return_rate"]) * 100:.2f}%`
- Date range: `{validation["min_order_date"]}` to `{validation["max_order_date"]}`

## Usage

Use this dataset when you want to explain a real external test source. Apply the same feature engineering version before prediction, but do not split this file into train/test again.
"""
    OUT_README.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = export_raw_table()
    aligned = align_to_model_schema(raw)
    aligned.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    validation = build_validation(aligned)
    pd.DataFrame([validation]).to_csv(OUT_VALIDATION, index=False, encoding="utf-8-sig")
    (OUT_DIR / "real_dataset_postgres_raw_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
