from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S1_SOURCE_CANDIDATES = [
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s1.csv",
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1" / "clean_dataset_s1.csv",
]
S2_SOURCE_CANDIDATES = [
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s2.csv",
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2" / "clean_dataset_s2.csv",
]
OUT_DIR = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset"
OUT_CSV = OUT_DIR / "real_dataset.csv"
OUT_VALIDATION = OUT_DIR / "real_dataset_validation_summary.csv"
OUT_META = OUT_DIR / "real_dataset_metadata.json"
OUT_DIST = OUT_DIR / "real_dataset_distribution_comparison.csv"
OUT_README = OUT_DIR / "README.md"

RANDOM_STATE = 20260608
TARGET = "is_returned"
S1_ROWS = 55_000
S2_ROWS = 50_000
TOTAL_ROWS = 105_000
DATE_COLUMNS = [
    "order_date",
    "expected_delivery_date",
    "delivery_date",
    "registration_date",
    "promo_start_date",
    "promo_end_date",
    "scored_at",
]


def resolve_source(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {label} source file")


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            (df[col].astype(str).str.strip() == "").sum()
            for col in df.select_dtypes(include=["object", "string"]).columns
        )
    )


def clip_numeric(series: pd.Series, lower: float, upper: float, decimals: int | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").clip(lower, upper)
    if decimals is not None:
        out = out.round(decimals)
    return out


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


def remap_values(series: pd.Series, prefix: str, keep_values: set[str] | None = None) -> pd.Series:
    keep_values = keep_values or set()
    values = pd.Series(series.astype(str).unique()).sort_values().tolist()
    mapping = {
        value: value if value in keep_values else f"{prefix}_{pos:05d}"
        for pos, value in enumerate(values, start=1)
    }
    return series.astype(str).map(mapping)


def add_small_numeric_noise(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = batch.copy()
    n_rows = len(out)
    if "product_rating" in out.columns:
        out["product_rating"] = clip_numeric(
            pd.to_numeric(out["product_rating"], errors="coerce") + rng.normal(0, 0.005, n_rows),
            3.5,
            5.0,
            3,
        )
    for col in ["unit_price", "total_amount", "discount_applied_amount"]:
        if col in out.columns:
            scale = rng.normal(1.0, 0.002, n_rows)
            out[col] = clip_numeric(pd.to_numeric(out[col], errors="coerce") * scale, 0, 60_000, 2)
    return out


def build_s1_block(source: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if len(source) != 5_000:
        raise ValueError(f"Expected S1 source to have 5,000 rows, got {len(source):,}")
    if S1_ROWS % len(source) != 0:
        raise ValueError("S1_ROWS must repeat S1 source exactly")

    source_sorted = source.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    repeats = S1_ROWS // len(source_sorted)
    order_numbers = np.arange(1, S1_ROWS + 1)
    batches: list[pd.DataFrame] = []
    cursor = 0

    for batch_no in range(repeats):
        batch = source_sorted.copy()
        n_rows = len(batch)
        numbers = order_numbers[cursor : cursor + n_rows]
        cursor += n_rows
        suffix = f"LGBM_FULL_S1_{batch_no + 1:02d}"

        batch["order_id"] = [f"ORD_LGBM_REAL_{num:06d}" for num in numbers]
        batch["score_id"] = [f"SCR_LGBM_REAL_{num:06d}" for num in numbers]
        batch["customer_id"] = batch["customer_id"].astype(str) + f"_{suffix}"
        batch["customer_name"] = "LightGBM Real Customer " + batch["customer_id"].astype(str)
        batch["customer_phone"] = [f"08{90000000 + int(num):08d}" for num in numbers]

        for col in ["product_id", "courier_id", "supplier_id"]:
            if col in batch.columns:
                batch[col] = batch[col].astype(str) + f"_{suffix}"

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

        batches.append(add_small_numeric_noise(batch, rng))

    return pd.concat(batches, ignore_index=True)


def build_s2_block(source: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if len(source) != S2_ROWS:
        raise ValueError(f"Expected S2 source to have {S2_ROWS:,} rows, got {len(source):,}")

    batch = source.sort_values(["order_date", "customer_id", "order_id"]).reset_index(drop=True).copy()
    numbers = np.arange(S1_ROWS + 1, TOTAL_ROWS + 1)
    batch["order_id"] = [f"ORD_LGBM_REAL_{num:06d}" for num in numbers]
    batch["score_id"] = [f"SCR_LGBM_REAL_{num:06d}" for num in numbers]

    customer_map = {
        customer_id: f"C_LGBM_REAL_S2_{idx:05d}"
        for idx, customer_id in enumerate(sorted(batch["customer_id"].astype(str).unique()), start=1)
    }
    batch["customer_id"] = batch["customer_id"].astype(str).map(customer_map)
    batch["customer_name"] = "LightGBM Real Customer " + batch["customer_id"].astype(str)
    batch["customer_phone"] = [f"08{90000000 + int(num):08d}" for num in numbers]

    if "product_id" in batch.columns:
        batch["product_id"] = remap_values(batch["product_id"], "P_LGBM_REAL_S2")
    if "courier_id" in batch.columns:
        batch["courier_id"] = remap_values(batch["courier_id"], "COURIER_LGBM_REAL_S2")
    if "supplier_id" in batch.columns:
        batch["supplier_id"] = remap_values(batch["supplier_id"], "SUP_LGBM_REAL_S2")
    if "promo_id" in batch.columns:
        batch["promo_id"] = remap_values(batch["promo_id"], "PROMO_LGBM_REAL_S2", keep_values={"PROMO_NONE"})

    returned_mask = batch[TARGET].astype(int).eq(1)
    batch["return_id"] = "NO_RETURN"
    batch.loc[returned_mask, "return_id"] = [
        f"RET_LGBM_REAL_{num:06d}" for num in numbers[returned_mask.to_numpy()]
    ]
    for col in DATE_COLUMNS:
        if col in batch.columns:
            batch[col] = pd.to_datetime(batch[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    return add_small_numeric_noise(batch, rng)


def distribution_comparison(s1_source: pd.DataFrame, s2_source: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    combined_source = pd.concat([s1_source, s2_source], ignore_index=True)
    categorical_cols = ["category", "payment_method", "channel_type", "province", "membership_tier", "courier_type"]
    for column in categorical_cols:
        source_dist = combined_source[column].astype(str).value_counts(normalize=True)
        generated_dist = generated[column].astype(str).value_counts(normalize=True)
        values = sorted(set(source_dist.index).union(set(generated_dist.index)))
        for value in values:
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "setc_combined_share": float(source_dist.get(value, 0.0)),
                    "real_dataset_share": float(generated_dist.get(value, 0.0)),
                    "share_gap": float(generated_dist.get(value, 0.0) - source_dist.get(value, 0.0)),
                }
            )

    for column in ["hist_order_count", "hist_return_rate", "delay_days", "risk_score", "total_amount", "product_rating", TARGET]:
        source_values = pd.to_numeric(combined_source[column], errors="coerce")
        generated_values = pd.to_numeric(generated[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "value": "mean",
                "setc_combined_share": float(source_values.mean()),
                "real_dataset_share": float(generated_values.mean()),
                "share_gap": float(generated_values.mean() - source_values.mean()),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame, s1_source: Path, s2_source: Path) -> pd.DataFrame:
    order_numbers = df["order_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    checks = {
        "file": str(OUT_CSV.relative_to(ROOT)),
        "s1_source_distribution_file": str(s1_source.relative_to(ROOT)),
        "s2_source_distribution_file": str(s2_source.relative_to(ROOT)),
        "generation_type": "lightgbm_setd_full_external_test_dataset_1_to_105000",
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
        "s1_segment_rows": int(order_numbers.between(1, 55_000).sum()),
        "s2_segment_rows": int(order_numbers.between(55_001, 105_000).sum()),
        "negative_amount_rows": int((pd.to_numeric(df["total_amount"], errors="coerce") < 0).sum()),
        "invalid_discount_rows": int((~pd.to_numeric(df["total_discount_pct"], errors="coerce").between(0, 1)).sum()),
        "invalid_rating_rows": int((~pd.to_numeric(df["product_rating"], errors="coerce").between(1, 5)).sum()),
        "invalid_quantity_rows": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
        "note": (
            "Synthetic real-like external test dataset for LightGBM SETD. "
            "Use as full-file external test input; do not split this file into train/test."
        ),
    }
    return pd.DataFrame([checks])


def write_readme(summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    content = f"""# LightGBM SETD Real Dataset

This folder stores external test datasets for LightGBM models.

## Current Full Dataset

- File: `real_dataset.csv`
- Row range: `1` to `105000`
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

## Segments

- `S1_range_000001_055000`: `{int(row["s1_segment_rows"]):,}` rows, intended for testing SETC S1 models
- `S2_range_055001_105000`: `{int(row["s2_segment_rows"]):,}` rows, intended for testing SETC S2 models

## Important

This is synthetic real-like data generated from the LightGBM SETC clean-data distributions. It is suitable for external benchmark testing, but it is not actual company production data.

Before prediction, run the same V1-V5 feature engineering used by the target model version. Do not split this file into train/test again.
"""
    OUT_README.write_text(content, encoding="utf-8")


def main() -> None:
    s1_source_path = resolve_source(S1_SOURCE_CANDIDATES, "LightGBM SETC S1")
    s2_source_path = resolve_source(S2_SOURCE_CANDIDATES, "LightGBM SETC S2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_STATE)
    s1_source = pd.read_csv(s1_source_path)
    s2_source = pd.read_csv(s2_source_path)
    if int(s1_source.isna().sum().sum()) != 0:
        raise ValueError("LightGBM SETC S1 source still has missing/null values")
    if int(s2_source.isna().sum().sum()) != 0:
        raise ValueError("LightGBM SETC S2 source still has missing/null values")

    s1_block = build_s1_block(s1_source, rng)
    s2_block = build_s2_block(s2_source, rng)
    output_columns = s1_source.columns.tolist()
    data = pd.concat([s1_block[output_columns], s2_block[output_columns]], ignore_index=True)
    data = normalize_text_and_numeric(data)

    summary = validation_summary(data, s1_source_path, s2_source_path)
    dist = distribution_comparison(s1_source, s2_source, data)

    data.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_VALIDATION, index=False, encoding="utf-8-sig")
    dist.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")
    OUT_META.write_text(json.dumps(summary.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)

    print(summary.to_string(index=False))
    print(f"Created: {OUT_CSV}")


if __name__ == "__main__":
    main()
