from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = [
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1" / "clean_dataset_s1.csv",
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s1.csv",
]
OUT_DIR = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset"
OUT_CSV = OUT_DIR / "real_dataset_s1.csv"
OUT_VALIDATION = OUT_DIR / "real_dataset_s1_validation_summary.csv"
OUT_META = OUT_DIR / "real_dataset_s1_metadata.json"
OUT_DIST = OUT_DIR / "real_dataset_s1_distribution_comparison.csv"
OUT_README = OUT_DIR / "README.md"

RANDOM_STATE = 20260608
ROW_COUNT = 55_000
ORDER_START = 1
ORDER_END = 55_000
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


def resolve_source() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find LightGBM SETC S1 clean_dataset_s1.csv")


def clip_numeric(series: pd.Series, lower: float, upper: float, decimals: int | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").clip(lower, upper)
    if decimals is not None:
        out = out.round(decimals)
    return out


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            (df[col].astype(str).str.strip() == "").sum()
            for col in df.select_dtypes(include=["object", "string"]).columns
        )
    )


def normalize_text_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
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


def make_real_dataset_s1(source_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    source = pd.read_csv(source_path)
    if len(source) != 5_000:
        raise ValueError(f"Expected S1 source to have 5,000 rows, got {len(source):,}")
    if len(source.columns) != 65:
        raise ValueError(f"Expected S1 source to have 65 columns, got {len(source.columns):,}")
    if int(source.isna().sum().sum()) != 0:
        raise ValueError("LightGBM SETC S1 source still has missing/null values")
    if ROW_COUNT % len(source) != 0:
        raise ValueError("ROW_COUNT must be a whole-number repeat of S1 source")

    source_sorted = source.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    repeats = ROW_COUNT // len(source_sorted)
    order_numbers = np.arange(ORDER_START, ORDER_END + 1)
    batches: list[pd.DataFrame] = []
    cursor = 0

    for batch_no in range(repeats):
        batch = source_sorted.copy()
        n_rows = len(batch)
        numbers = order_numbers[cursor : cursor + n_rows]
        cursor += n_rows

        batch_suffix = f"LGBM_REAL{batch_no + 1:02d}"
        batch["order_id"] = [f"ORD_LGBM_REAL_{num:06d}" for num in numbers]
        batch["score_id"] = [f"SCR_LGBM_REAL_{num:06d}" for num in numbers]
        batch["customer_id"] = batch["customer_id"].astype(str) + f"_{batch_suffix}"
        batch["customer_name"] = "LightGBM Real Customer " + batch["customer_id"].astype(str)
        batch["customer_phone"] = [f"08{90000000 + int(num):08d}" for num in numbers]

        for col in ["product_id", "courier_id", "supplier_id"]:
            if col in batch.columns:
                batch[col] = batch[col].astype(str) + f"_{batch_suffix}"

        for col in DATE_COLUMNS:
            if col in batch.columns:
                values = pd.to_datetime(batch[col].replace({"Not Returned": pd.NA}), errors="coerce")
                batch[col] = values + pd.to_timedelta(batch_no, unit="s")

        returned_mask = batch[TARGET].astype(int).eq(1)
        batch["return_id"] = "NO_RETURN"
        batch.loc[returned_mask, "return_id"] = [
            f"RET_LGBM_REAL_{num:06d}" for num in numbers[returned_mask.to_numpy()]
        ]

        if "return_date" in batch.columns:
            return_dates = pd.to_datetime(batch["return_date"].replace({"Not Returned": pd.NA}), errors="coerce")
            return_dates = return_dates + pd.to_timedelta(batch_no, unit="s")
            batch["return_date"] = "Not Returned"
            batch.loc[returned_mask, "return_date"] = return_dates.loc[returned_mask].dt.strftime("%Y-%m-%d %H:%M:%S")

        if "product_rating" in batch.columns:
            batch["product_rating"] = clip_numeric(
                pd.to_numeric(batch["product_rating"], errors="coerce") + rng.normal(0, 0.005, n_rows),
                3.5,
                5.0,
                3,
            )
        if "unit_price" in batch.columns:
            scale = rng.normal(1.0, 0.002, n_rows)
            batch["unit_price"] = clip_numeric(pd.to_numeric(batch["unit_price"], errors="coerce") * scale, 1, 60_000, 2)
        if "total_amount" in batch.columns:
            scale = rng.normal(1.0, 0.002, n_rows)
            batch["total_amount"] = clip_numeric(pd.to_numeric(batch["total_amount"], errors="coerce") * scale, 0, 60_000, 2)
        if "discount_applied_amount" in batch.columns:
            scale = rng.normal(1.0, 0.002, n_rows)
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

    return normalize_text_and_numeric(out[source.columns.tolist()].copy())


def distribution_comparison(source: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ["category", "payment_method", "channel_type", "province", "membership_tier", "courier_type"]:
        source_dist = source[column].astype(str).value_counts(normalize=True)
        generated_dist = generated[column].astype(str).value_counts(normalize=True)
        values = sorted(set(source_dist.index).union(set(generated_dist.index)))
        for value in values:
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "setc_s1_share": float(source_dist.get(value, 0.0)),
                    "real_dataset_s1_share": float(generated_dist.get(value, 0.0)),
                    "share_gap": float(generated_dist.get(value, 0.0) - source_dist.get(value, 0.0)),
                }
            )

    for column in ["hist_order_count", "hist_return_rate", "delay_days", "risk_score", "total_amount", "product_rating", TARGET]:
        source_values = pd.to_numeric(source[column], errors="coerce")
        generated_values = pd.to_numeric(generated[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "value": "mean",
                "setc_s1_share": float(source_values.mean()),
                "real_dataset_s1_share": float(generated_values.mean()),
                "share_gap": float(generated_values.mean() - source_values.mean()),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    checks = {
        "file": str(OUT_CSV.relative_to(ROOT)),
        "source_distribution_file": str(source_path.relative_to(ROOT)),
        "generation_type": "lightgbm_setd_s1_external_full_dataset_test",
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
        "return_rate": float(df[TARGET].mean()),
        "negative_amount_rows": int((pd.to_numeric(df["total_amount"], errors="coerce") < 0).sum()),
        "invalid_discount_rows": int((~pd.to_numeric(df["total_discount_pct"], errors="coerce").between(0, 1)).sum()),
        "invalid_rating_rows": int((~pd.to_numeric(df["product_rating"], errors="coerce").between(1, 5)).sum()),
        "invalid_quantity_rows": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
        "note": (
            "Synthetic real-like external test dataset for LightGBM SETD. "
            "Use this as a full-file test input for SETC S1 models; do not split this test file again."
        ),
    }
    return pd.DataFrame([checks])


def write_readme(summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    content = f"""# LightGBM SETD Real Dataset

This folder stores external test datasets for LightGBM models.

## Current Dataset

- File: `real_dataset_s1.csv`
- Intended pair: test LightGBM `SETC/clean_dataset/S1` models
- Evaluation style: full external test file, no 20% split
- Rows: `{int(row["rows"]):,}`
- Columns: `{int(row["columns"])}`
- Order range: `{row["first_order_id"]}` to `{row["last_order_id"]}`
- Date range: `{row["min_order_date"]}` to `{row["max_order_date"]}`
- Returned: `{int(row["returned_count"]):,}`
- Not Returned: `{int(row["not_returned_count"]):,}`
- Return rate: `{float(row["return_rate"]) * 100:.2f}%`
- Missing/null cells: `{int(row["missing_or_null_cells"])}`
- Duplicate order_id: `{int(row["duplicate_order_id"])}`

## Important

This is synthetic real-like data generated from the LightGBM SETC S1 clean-data distribution. It is suitable for external benchmark testing, but it is not actual company production data.

Before prediction, run the same V1-V5 feature engineering used by the target model version. Do not split this file into train/test again.
"""
    OUT_README.write_text(content, encoding="utf-8")


def main() -> None:
    source_path = resolve_source()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_path)
    data = make_real_dataset_s1(source_path)
    summary = validation_summary(data, source_path)
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
