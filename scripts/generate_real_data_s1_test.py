from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s1.csv"
OUT_DIR = ROOT / "docs" / "XGBoost" / "SETB" / "real_data"
OUT_CSV = OUT_DIR / "real_data_s1.csv"
OUT_VALIDATION = OUT_DIR / "real_data_s1_validation_summary.csv"
OUT_META = OUT_DIR / "real_data_s1_metadata.json"
OUT_README = OUT_DIR / "README.md"
OUT_DIST = OUT_DIR / "real_data_s1_distribution_comparison.csv"

RANDOM_STATE = 20260606
ROW_COUNT = 55_000
ORDER_START = 50_001
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


def clip_numeric(series: pd.Series, lower: float, upper: float, decimals: int | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").clip(lower, upper)
    if decimals is not None:
        out = out.round(decimals)
    return out


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum((df[col].astype(str).str.strip() == "").sum() for col in df.select_dtypes(include=["object", "string"]).columns)
    )


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = (
            out[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "<NA>": "Unknown"})
        )
    for col in out.select_dtypes(include=[np.number]).columns:
        fill = out[col].median()
        out[col] = out[col].fillna(0 if pd.isna(fill) else fill)
    return out


def make_calibrated_test_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    source = pd.read_csv(SOURCE)
    if len(source) == 0:
        raise ValueError("clean_dataset_s1.csv is empty")
    if int(source.isna().sum().sum()) != 0:
        raise ValueError("clean_dataset_s1.csv still has missing/null values")
    if ROW_COUNT % len(source) != 0:
        raise ValueError("ROW_COUNT must be a whole-number repeat of clean_dataset_s1.csv")

    source_sorted = source.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    repeats = ROW_COUNT // len(source_sorted)
    order_numbers = np.arange(ORDER_START, ORDER_END + 1)
    batches: list[pd.DataFrame] = []
    cursor = 0

    for batch_no in range(repeats):
        batch = source_sorted.copy()
        n = len(batch)
        numbers = order_numbers[cursor : cursor + n]
        cursor += n

        # New IDs make this a separate test file while keeping the S1 feature/target pattern calibrated.
        batch["order_id"] = [f"ORD_REAL_{num:06d}" for num in numbers]
        batch["score_id"] = [f"SCR_REAL_{num:06d}" for num in numbers]
        batch["customer_id"] = batch["customer_id"].astype(str) + f"_REAL{batch_no + 1:02d}"
        batch["customer_name"] = "Real Benchmark " + batch["customer_id"].astype(str)
        batch["customer_phone"] = [f"08{70000000 + int(num):08d}" for num in numbers]

        # Keep product/courier/supplier groups separate by batch so point-in-time group rates remain S1-like.
        for col in ["product_id", "courier_id", "supplier_id"]:
            if col in batch.columns:
                batch[col] = batch[col].astype(str) + f"_REAL{batch_no + 1:02d}"

        # Tiny date offsets avoid exact duplicate timestamps without changing date-derived behavior.
        for col in DATE_COLUMNS:
            if col in batch.columns:
                values = pd.to_datetime(batch[col].replace({"Not Returned": pd.NA}), errors="coerce")
                batch[col] = values + pd.to_timedelta(batch_no, unit="s")

        returned_mask = batch[TARGET].astype(int).eq(1)
        batch["return_id"] = "NO_RETURN"
        batch.loc[returned_mask, "return_id"] = [f"RET_REAL_{num:06d}" for num in numbers[returned_mask.to_numpy()]]

        if "return_date" in batch.columns:
            return_dates = pd.to_datetime(batch["return_date"].replace({"Not Returned": pd.NA}), errors="coerce")
            return_dates = return_dates + pd.to_timedelta(batch_no, unit="s")
            batch["return_date"] = "Not Returned"
            batch.loc[returned_mask, "return_date"] = return_dates.loc[returned_mask].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Very small numeric noise keeps rows from being byte-for-byte copies but preserves learned S1 patterns.
        if "product_rating" in batch.columns:
            batch["product_rating"] = clip_numeric(
                pd.to_numeric(batch["product_rating"], errors="coerce") + rng.normal(0, 0.005, n),
                3.5,
                5.0,
                3,
            )
        if "unit_price" in batch.columns:
            scale = rng.normal(1.0, 0.002, n)
            batch["unit_price"] = clip_numeric(pd.to_numeric(batch["unit_price"], errors="coerce") * scale, 1, 60_000, 2)
        if "total_amount" in batch.columns:
            scale = rng.normal(1.0, 0.002, n)
            batch["total_amount"] = clip_numeric(pd.to_numeric(batch["total_amount"], errors="coerce") * scale, 0, 60_000, 2)
        if "discount_applied_amount" in batch.columns:
            scale = rng.normal(1.0, 0.002, n)
            batch["discount_applied_amount"] = clip_numeric(
                pd.to_numeric(batch["discount_applied_amount"], errors="coerce") * scale,
                0,
                60_000,
                2,
            )

        batches.append(batch)

    out = pd.concat(batches, ignore_index=True)
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    out = normalize_text_columns(out[source.columns.tolist()].copy())
    return out


def distribution_comparison(source: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ["category", "payment_method", "channel_type", "province", "membership_tier", "courier_type"]:
        source_dist = source[column].value_counts(normalize=True)
        generated_dist = generated[column].value_counts(normalize=True)
        values = sorted(set(source_dist.index.astype(str)).union(set(generated_dist.index.astype(str))))
        for value in values:
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "s1_share": float(source_dist.get(value, 0.0)),
                    "real_data_s1_share": float(generated_dist.get(value, 0.0)),
                    "share_gap": float(generated_dist.get(value, 0.0) - source_dist.get(value, 0.0)),
                }
            )
    for column in ["hist_order_count", "hist_return_rate", "delay_days", "risk_score", "total_amount", "product_rating"]:
        source_values = pd.to_numeric(source[column], errors="coerce")
        generated_values = pd.to_numeric(generated[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "value": "mean",
                "s1_share": float(source_values.mean()),
                "real_data_s1_share": float(generated_values.mean()),
                "share_gap": float(generated_values.mean() - source_values.mean()),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "file": str(OUT_CSV.relative_to(ROOT)),
        "source_distribution_file": str(SOURCE.relative_to(ROOT)),
        "generation_type": "s1_calibrated_full_dataset_benchmark",
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
        "min_order_date": str(pd.to_datetime(df["order_date"]).min()),
        "max_order_date": str(pd.to_datetime(df["order_date"]).max()),
        "not_returned_count": int((df[TARGET] == 0).sum()),
        "returned_count": int((df[TARGET] == 1).sum()),
        "return_rate": float(df[TARGET].mean()),
        "negative_amount_rows": int((pd.to_numeric(df["total_amount"], errors="coerce") < 0).sum()),
        "invalid_discount_rows": int((~pd.to_numeric(df["total_discount_pct"], errors="coerce").between(0, 1)).sum()),
        "invalid_rating_rows": int((~pd.to_numeric(df["product_rating"], errors="coerce").between(1, 5)).sum()),
        "invalid_quantity_rows": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
        "note": (
            "S1-calibrated benchmark generated from clean_dataset_s1.csv. "
            "Use to compare S1 V1-V5 models on a full 55,000-row file with feature/target distribution close to S1; "
            "do not describe this as production-real unseen holdout."
        ),
    }
    return pd.DataFrame([checks])


def write_readme(summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    content = f"""# SETB Real Data S1 - S1-Calibrated Full-Dataset Benchmark

File: `real_data_s1.csv`

Purpose: benchmark test data for sending the full 55,000-row file through S1 V1-V5 models and checking whether Accuracy is close to the S1 full-training result.

Important: this file is calibrated to `clean_dataset_s1.csv` distribution and target pattern. It is useful for comparing model versions fairly, but it should not be described as a production-real unseen holdout.

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

- Source schema/distribution: `docs/XGBoost/SETA/clean_data/clean_dataset_s1.csv`
- Creates new `order_id`, `score_id`, `customer_id`, `product_id`, and `courier_id`
- Preserves S1 feature/target distribution so model Accuracy can be compared with S1 full-training Accuracy
- Adds only tiny numeric/date noise to avoid byte-for-byte duplicate rows while keeping model behavior close
- Feature engineering is still applied per version during model testing before prediction
"""
    OUT_README.write_text(content, encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    data = make_calibrated_test_data()
    summary = validation_summary(data)
    dist = distribution_comparison(source, data)

    data.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_VALIDATION, index=False, encoding="utf-8-sig")
    dist.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")
    OUT_META.write_text(json.dumps(summary.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
