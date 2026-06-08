from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import generate_clean_dataset_s2 as s2_generator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s2.csv"
OUT_DIR = ROOT / "docs" / "XGBoost" / "SETB" / "real_data" / "S4"
OUT_CSV = OUT_DIR / "real_data_s2.csv"
OUT_VALIDATION = OUT_DIR / "real_data_s2_validation_summary.csv"
OUT_META = OUT_DIR / "real_data_s2_metadata.json"
OUT_DIST = OUT_DIR / "real_data_s2_distribution_comparison.csv"
OUT_README = OUT_DIR / "README.md"

RANDOM_STATE = 20260608
ROW_COUNT = 50_000
ORDER_START = 55_001
ORDER_END = 105_000
TARGET = "is_returned"

DATE_COLUMNS = [
    "order_date",
    "expected_delivery_date",
    "delivery_date",
    "registration_date",
    "promo_start_date",
    "promo_end_date",
    "scored_at",
]


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            (df[col].astype(str).str.strip() == "").sum()
            for col in df.select_dtypes(include=["object", "string"]).columns
        )
    )


def normalize_text_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = (
            out[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "<NA>": "Unknown"})
        )
    for column in out.select_dtypes(include=[np.number]).columns:
        median = out[column].median()
        out[column] = out[column].fillna(0 if pd.isna(median) else median)
    return out


def remap_group_id(series: pd.Series, prefix: str, keep_values: set[str] | None = None) -> pd.Series:
    keep_values = keep_values or set()
    values = pd.Series(series.astype(str).unique()).sort_values().tolist()
    mapping = {
        value: value if value in keep_values else f"{prefix}_{pos:05d}"
        for pos, value in enumerate(values, start=1)
    }
    return series.astype(str).map(mapping)


def create_real_data_s2() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = pd.read_csv(SOURCE)
    if len(source) != ROW_COUNT:
        raise ValueError(f"Expected source to have {ROW_COUNT:,} rows, got {len(source):,}")
    if int(source.isna().sum().sum()) != 0:
        raise ValueError("clean_dataset_s2.csv still has missing/null values")

    # Reuse the S2 generation process with a different seed so this file has
    # the same schema/distribution family, but is not the exact S2 training file.
    s2_generator.RANDOM_STATE = RANDOM_STATE
    generated = s2_generator.generate_s2().sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    if len(generated) != ROW_COUNT:
        raise ValueError(f"Generated row count mismatch: {len(generated):,}")

    numbers = np.arange(ORDER_START, ORDER_END + 1)
    generated["order_id"] = [f"ORD_REAL_S2_{num:06d}" for num in numbers]
    generated["score_id"] = [f"SCR_REAL_S2_{num:06d}" for num in numbers]

    old_customers = pd.Series(generated["customer_id"].astype(str).unique()).sort_values().tolist()
    customer_map = {
        old_customer: f"C_REAL_S2_{idx:05d}"
        for idx, old_customer in enumerate(old_customers, start=1)
    }
    generated["customer_id"] = generated["customer_id"].astype(str).map(customer_map)
    generated["customer_name"] = "Real S2 Customer " + generated["customer_id"].astype(str)
    generated["customer_phone"] = [f"08{80000000 + int(num):08d}" for num in numbers]

    if "product_id" in generated.columns:
        generated["product_id"] = remap_group_id(generated["product_id"], "P_REAL_S2")
    if "courier_id" in generated.columns:
        generated["courier_id"] = remap_group_id(generated["courier_id"], "COURIER_REAL_S2")
    if "supplier_id" in generated.columns:
        generated["supplier_id"] = remap_group_id(generated["supplier_id"], "SUP_REAL_S2")
    if "promo_id" in generated.columns:
        generated["promo_id"] = remap_group_id(generated["promo_id"], "PROMO_REAL_S2", keep_values={"PROMO_NONE"})

    returned_mask = generated[TARGET].astype(int).eq(1)
    generated["return_id"] = "NO_RETURN"
    generated.loc[returned_mask, "return_id"] = [
        f"RET_REAL_S2_{num:06d}" for num in numbers[returned_mask.to_numpy()]
    ]

    for column in DATE_COLUMNS:
        if column in generated.columns:
            generated[column] = pd.to_datetime(generated[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    generated = normalize_text_and_numeric(generated[source.columns.tolist()])
    return generated


def distribution_comparison(source: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categorical_cols = ["category", "payment_method", "channel_type", "province", "membership_tier", "courier_type"]
    for column in categorical_cols:
        if column not in source.columns or column not in generated.columns:
            continue
        source_dist = source[column].astype(str).value_counts(normalize=True)
        generated_dist = generated[column].astype(str).value_counts(normalize=True)
        values = sorted(set(source_dist.index).union(set(generated_dist.index)))
        for value in values:
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "clean_dataset_s2_share": float(source_dist.get(value, 0.0)),
                    "real_data_s2_share": float(generated_dist.get(value, 0.0)),
                    "share_gap": float(generated_dist.get(value, 0.0) - source_dist.get(value, 0.0)),
                }
            )

    numeric_cols = [
        "hist_order_count",
        "hist_return_rate",
        "delay_days",
        "risk_score",
        "total_amount",
        "product_rating",
        "is_returned",
    ]
    for column in numeric_cols:
        if column not in source.columns or column not in generated.columns:
            continue
        source_values = pd.to_numeric(source[column], errors="coerce")
        generated_values = pd.to_numeric(generated[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "value": "mean",
                "clean_dataset_s2_share": float(source_values.mean()),
                "real_data_s2_share": float(generated_values.mean()),
                "share_gap": float(generated_values.mean() - source_values.mean()),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "file": str(OUT_CSV.relative_to(ROOT)),
        "source_distribution_file": str(SOURCE.relative_to(ROOT)),
        "generation_type": "s2_unseen_like_representative_test_data",
        "rows": len(df),
        "columns": len(df.columns),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "blank_text_cells": blank_text_count(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()),
        "distinct_order_id": int(df["order_id"].nunique()),
        "distinct_customer_id": int(df["customer_id"].nunique()),
        "first_order_id": str(df["order_id"].iloc[0]),
        "last_order_id": str(df["order_id"].iloc[-1]),
        "min_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").min()),
        "max_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").max()),
        "not_returned_count": int((df[TARGET] == 0).sum()),
        "returned_count": int((df[TARGET] == 1).sum()),
        "return_rate": float(pd.to_numeric(df[TARGET], errors="coerce").mean()),
        "negative_amount_rows": int((pd.to_numeric(df["total_amount"], errors="coerce") < 0).sum()),
        "invalid_discount_rows": int((~pd.to_numeric(df["total_discount_pct"], errors="coerce").between(0, 1)).sum()),
        "invalid_rating_rows": int((~pd.to_numeric(df["product_rating"], errors="coerce").between(1, 5)).sum()),
        "invalid_quantity_rows": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
        "note": (
            "Generated unseen-like test data from the S2 distribution family. "
            "It is representative synthetic real-like data for model testing, not actual company production data."
        ),
    }
    return pd.DataFrame([checks])


def write_readme(summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    content = f"""# SETB Real Data S2 - S4 Test Data

File: `real_data_s2.csv`

Purpose: external test dataset for model evaluation. This file is generated as unseen-like, representative test data using the S2 distribution family, then assigned new order/customer/product/courier identifiers.

Important: this is synthetic real-like data, not actual company production data. It is designed to act like a final exam dataset because the model files in SETA were not trained on these exact rows or IDs.

## Dataset Summary

- Rows: `{int(row["rows"]):,}`
- Columns: `{int(row["columns"])}`
- Order range: `{row["first_order_id"]}` to `{row["last_order_id"]}`
- Date range: `{row["min_order_date"]}` to `{row["max_order_date"]}`
- Returned: `{int(row["returned_count"]):,}`
- Not Returned: `{int(row["not_returned_count"]):,}`
- Return rate: `{float(row["return_rate"]) * 100:.2f}%`
- Missing/null cells: `{int(row["missing_or_null_cells"])}`
- Duplicate order_id: `{int(row["duplicate_order_id"])}`

## Generation Logic

- Source distribution: `docs/XGBoost/SETA/clean_data/clean_dataset_s2.csv`
- Row count: 50,000 rows
- Order ID range: `ORD_REAL_S2_055001` to `ORD_REAL_S2_105000`
- Recreates order/customer timeline with a different random seed from S2
- Recomputes point-in-time history features such as `hist_order_count`, `hist_return_rate`, and `days_since_last_order`
- Keeps `is_returned` as ground truth for checking model predictions
- Keeps schema compatible with the 65-column clean dataset family

## How To Use

Use this file as full external test input. Before prediction, run the same feature engineering version as the model version being tested, then compare model prediction with `is_returned`.
"""
    OUT_README.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    data = create_real_data_s2()
    summary = validation_summary(data)
    dist = distribution_comparison(source, data)

    data.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_VALIDATION, index=False, encoding="utf-8-sig")
    dist.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")
    OUT_META.write_text(json.dumps(summary.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)

    print(summary.to_string(index=False))
    print(f"Created: {OUT_CSV}")


if __name__ == "__main__":
    main()
