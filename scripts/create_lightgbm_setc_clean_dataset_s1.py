from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import create_clean_dataset_s1 as cleaner


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "clean_dataset.csv"
OUTPUT_DIR = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1"
OUTPUT_CSV = OUTPUT_DIR / "clean_dataset_s1.csv"
SUMMARY_CSV = OUTPUT_DIR / "clean_dataset_s1_validation_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "clean_dataset_s1_validation_summary.json"
README = OUTPUT_DIR / "README.md"


def write_readme(summary: pd.DataFrame) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    content = f"""# LightGBM SETC S1 Clean Dataset

This folder stores the clean 5,000-row dataset that will be used as the S1 source for LightGBM experiments.

## Source

- Input: `data/processed/clean_dataset.csv`
- Output: `docs/LightGBM/SETC/clean_dataset/S1/clean_dataset_s1.csv`

## Cleaning Rules

- Kept all 5,000 rows and all 65 source columns.
- Filled business missing values:
  - `promo_type` for `PROMO_NONE` -> `No Promotion`
  - not-returned rows -> `return_id = NO_RETURN`
  - not-returned rows -> `return_date = Not Returned`
- Filled remaining text missing values with `Unknown`.
- Filled numeric missing values with median.
- Removed duplicate rows by validation; no source duplicate rows were dropped.
- Clipped numeric outliers using domain bounds and IQR winsorization where appropriate.
- Kept sentinel values such as `days_since_last_order = -1` because they have business meaning.

## Validation

- Rows: `{metrics.get("output_rows")}`
- Columns: `{metrics.get("output_columns")}`
- Missing/null cells: `{metrics.get("output_missing_total")}`
- Blank text cells: `{metrics.get("output_blank_text_cells")}`
- Duplicate rows: `{metrics.get("output_duplicate_rows")}`
- Not Returned: `{metrics.get("target_is_returned_0_count")}`
- Returned: `{metrics.get("target_is_returned_1_count")}`

This file is a clean source dataset only. Feature engineering and LightGBM train/test split will be created in the next step.
"""
    README.write_text(content, encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(SOURCE)
    cleaned = cleaner.clean_missing_values(source)
    cleaned, outlier_records = cleaner.clean_outliers(cleaned)
    cleaner.validate(cleaned)

    cleaned.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    summary = cleaner.build_summary(source, cleaned, outlier_records)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "model_family": "LightGBM",
                "set": "SETC",
                "dataset_version": "S1",
                "source_csv": str(SOURCE.relative_to(ROOT)),
                "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
                "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                "rows": len(cleaned),
                "columns": len(cleaned.columns),
                "missing_total": int(cleaned.isna().sum().sum()),
                "blank_text_cells": cleaner.blank_text_count(cleaned),
                "duplicate_rows": int(cleaned.duplicated().sum()),
                "target_distribution": {
                    str(k): int(v)
                    for k, v in cleaned[cleaner.TARGET_COLUMN].value_counts(dropna=False).sort_index().items()
                },
                "outlier_records": outlier_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(summary)

    print(f"Created: {OUTPUT_CSV}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
