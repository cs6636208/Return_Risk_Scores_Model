from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "clean_dataset.csv"
OUTPUT_DIR = ROOT / "SETA" / "S1" / "clean_data"
OUTPUT_CSV = OUTPUT_DIR / "clean_dataset_s1.csv"
SUMMARY_CSV = OUTPUT_DIR / "clean_dataset_s1_validation_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "clean_dataset_s1_validation_summary.json"

TARGET_COLUMN = "is_returned"
BINARY_COLUMNS = {"is_returned", "is_fragile", "is_repurchased_item"}
SENTINEL_COLUMNS = {"days_since_last_order"}

DOMAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "customer_age_days": (0, 3650),
    "product_rating": (1.0, 5.0),
    "avg_delivery_days": (0.0, 30.0),
    "damage_rate": (0.0, 1.0),
    "promo_discount_rate": (0.0, 1.0),
    "quantity": (1, 100),
    "unit_price": (0.0, 1_000_000.0),
    "tier_discount_pct": (0.0, 1.0),
    "campaign_discount_pct": (0.0, 1.0),
    "total_discount_pct": (0.0, 1.0),
    "discount_applied_amount": (0.0, 1_000_000.0),
    "total_amount": (0.0, 10_000_000.0),
    "delivery_time_expected_days": (0, 60),
    "delivery_days": (0, 90),
    "delay_days": (-30, 90),
    "order_hour": (0, 23),
    "hist_order_count": (0, 10_000),
    "hist_return_rate": (0.0, 1.0),
    "refund_amount": (0.0, 10_000_000.0),
    "risk_score": (0.0, 1.0),
}

IQR_CLIP_COLUMNS = {
    "customer_age_days",
    "unit_price",
    "discount_applied_amount",
    "total_amount",
    "delivery_days",
    "delay_days",
    "hist_order_count",
    "refund_amount",
}


def outlier_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    if clean.empty:
        return 0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    return int(((numeric < lower) | (numeric > upper)).sum())


def iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - (1.5 * iqr)), float(q3 + (1.5 * iqr))


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    no_promo_mask = (
        out["promo_type"].isna()
        & out["promo_id"].astype(str).eq("PROMO_NONE")
        & pd.to_numeric(out["promo_discount_rate"], errors="coerce").fillna(0).eq(0)
    )
    out.loc[no_promo_mask, "promo_type"] = "No Promotion"

    not_returned_mask = out[TARGET_COLUMN].astype(int).eq(0)
    out.loc[not_returned_mask & out["return_id"].isna(), "return_id"] = "NO_RETURN"
    out.loc[not_returned_mask & out["return_date"].isna(), "return_date"] = "Not Returned"

    object_columns = out.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        out[column] = out[column].fillna("Unknown")

    numeric_columns = out.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if out[column].isna().any():
            out[column] = out[column].fillna(out[column].median())

    bool_columns = out.select_dtypes(include=["bool"]).columns
    for column in bool_columns:
        if out[column].isna().any():
            out[column] = out[column].fillna(False)

    return out


def clean_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = df.copy()
    records: list[dict[str, object]] = []

    for column in out.select_dtypes(include=[np.number]).columns:
        if column in BINARY_COLUMNS or column in SENTINEL_COLUMNS:
            records.append(
                {
                    "column": column,
                    "method": "skipped_binary_or_sentinel",
                    "outliers_before": outlier_count(out[column]),
                    "outliers_after": outlier_count(out[column]),
                    "lower_bound": "",
                    "upper_bound": "",
                    "clipped_rows": 0,
                }
            )
            continue

        original = pd.to_numeric(out[column], errors="coerce")
        lower, upper = DOMAIN_BOUNDS.get(column, (float(original.min()), float(original.max())))

        method = "domain_bounds"
        if column in IQR_CLIP_COLUMNS:
            bounds = iqr_bounds(original)
            if bounds is not None:
                iqr_lower, iqr_upper = bounds
                lower = max(lower, iqr_lower)
                upper = min(upper, iqr_upper)
                method = "domain_plus_iqr_winsorization"

        before = outlier_count(original)
        clipped = original.clip(lower=lower, upper=upper)
        out[column] = clipped.astype(out[column].dtype, errors="ignore")
        after = outlier_count(out[column]) if column in IQR_CLIP_COLUMNS else 0
        records.append(
            {
                "column": column,
                "method": method,
                "outliers_before": before,
                "outliers_after": after,
                "lower_bound": lower,
                "upper_bound": upper,
                "clipped_rows": int((original != clipped).sum()),
            }
        )

    return out, records


def blank_text_count(df: pd.DataFrame) -> int:
    total = 0
    for column in df.select_dtypes(include=["object", "string"]).columns:
        total += int(df[column].astype(str).str.strip().eq("").sum())
    return total


def build_summary(
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
    outlier_records: list[dict[str, object]],
) -> pd.DataFrame:
    rows = [
        {"metric": "source_rows", "value": len(source)},
        {"metric": "source_columns", "value": len(source.columns)},
        {"metric": "source_missing_total", "value": int(source.isna().sum().sum())},
        {"metric": "source_duplicate_rows", "value": int(source.duplicated().sum())},
        {"metric": "output_rows", "value": len(cleaned)},
        {"metric": "output_columns", "value": len(cleaned.columns)},
        {"metric": "output_missing_total", "value": int(cleaned.isna().sum().sum())},
        {"metric": "output_blank_text_cells", "value": blank_text_count(cleaned)},
        {"metric": "output_duplicate_rows", "value": int(cleaned.duplicated().sum())},
    ]

    if TARGET_COLUMN in cleaned.columns:
        for label, count in cleaned[TARGET_COLUMN].value_counts(dropna=False).sort_index().items():
            rows.append({"metric": f"target_{TARGET_COLUMN}_{label}_count", "value": int(count)})

    for record in outlier_records:
        rows.append(
            {
                "metric": f"outlier_{record['column']}",
                "value": (
                    f"method={record['method']}; before={record['outliers_before']}; "
                    f"after={record['outliers_after']}; clipped_rows={record['clipped_rows']}; "
                    f"bounds=[{record['lower_bound']}, {record['upper_bound']}]"
                ),
            }
        )

    return pd.DataFrame(rows)


def validate(cleaned: pd.DataFrame) -> None:
    checks = {
        "row_count_is_5000": len(cleaned) == 5000,
        "column_count_is_65": len(cleaned.columns) == 65,
        "missing_total_is_zero": int(cleaned.isna().sum().sum()) == 0,
        "blank_text_cells_is_zero": blank_text_count(cleaned) == 0,
        "duplicate_rows_is_zero": int(cleaned.duplicated().sum()) == 0,
        "target_0_count_is_3545": int((cleaned[TARGET_COLUMN] == 0).sum()) == 3545,
        "target_1_count_is_1455": int((cleaned[TARGET_COLUMN] == 1).sum()) == 1455,
        "promo_type_missing_is_zero": int(cleaned["promo_type"].isna().sum()) == 0,
        "no_return_id_placeholder_ok": bool(
            cleaned.loc[cleaned[TARGET_COLUMN] == 0, "return_id"].astype(str).eq("NO_RETURN").all()
        ),
        "no_return_date_placeholder_ok": bool(
            cleaned.loc[cleaned[TARGET_COLUMN] == 0, "return_date"].astype(str).eq("Not Returned").all()
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(SOURCE)
    cleaned = clean_missing_values(source)
    cleaned, outlier_records = clean_outliers(cleaned)
    validate(cleaned)

    cleaned.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    summary = build_summary(source, cleaned, outlier_records)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
                "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                "rows": len(cleaned),
                "columns": len(cleaned.columns),
                "missing_total": int(cleaned.isna().sum().sum()),
                "blank_text_cells": blank_text_count(cleaned),
                "duplicate_rows": int(cleaned.duplicated().sum()),
                "target_distribution": {
                    str(k): int(v)
                    for k, v in cleaned[TARGET_COLUMN].value_counts(dropna=False).sort_index().items()
                },
                "outlier_records": outlier_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_CSV}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
