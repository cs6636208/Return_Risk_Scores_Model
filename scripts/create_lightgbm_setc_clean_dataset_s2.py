from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import generate_clean_dataset_s2 as s2_generator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1" / "clean_dataset_s1.csv"
OUTPUT_DIR = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2"
OUTPUT_CSV = OUTPUT_DIR / "clean_dataset_s2.csv"
SUMMARY_CSV = OUTPUT_DIR / "clean_dataset_s2_validation_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "clean_dataset_s2_validation_summary.json"
README = OUTPUT_DIR / "README.md"
PARENT_README = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "README.md"


def make_lightgbm_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_id"] = [f"ORD_LGBM_S2_{i:06d}" for i in range(1, len(out) + 1)]
    out["score_id"] = [f"SCR_LGBM_S2_{i:06d}" for i in range(1, len(out) + 1)]

    customer_map = {
        customer_id: f"C_LGBM_S2_{idx:05d}"
        for idx, customer_id in enumerate(sorted(out["customer_id"].astype(str).unique()), start=1)
    }
    out["customer_id"] = out["customer_id"].astype(str).map(customer_map)
    out["customer_name"] = "LightGBM S2 Customer " + out["customer_id"].astype(str)
    out["customer_phone"] = [f"07{i:08d}" for i in range(1, len(out) + 1)]

    returned_mask = out[s2_generator.TARGET].astype(int).eq(1)
    out.loc[~returned_mask, "return_id"] = "NO_RETURN"
    out.loc[returned_mask, "return_id"] = [
        f"RET_LGBM_S2_{i:06d}" for i in range(1, int(returned_mask.sum()) + 1)
    ]
    return out


def write_readme(summary: pd.DataFrame) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    content = f"""# LightGBM SETC S2 Clean Dataset

This folder stores the clean 50,000-row dataset that will be used as the S2 source for LightGBM experiments.

## Source

- Input: `docs/LightGBM/SETC/clean_dataset/S1/clean_dataset_s1.csv`
- Output: `docs/LightGBM/SETC/clean_dataset/S2/clean_dataset_s2.csv`

## Generation And Cleaning Rules

- Generated from the LightGBM SETC S1 clean dataset.
- Kept the same 65-column schema.
- Created 5,000 customers with 10 orders each.
- Recomputed customer history features such as `hist_order_count`, `hist_return_rate`, and `days_since_last_order`.
- Preserved target distribution close to S1:
  - Not Returned: 35,450
  - Returned: 14,550
- Filled return fields for non-returned rows with `NO_RETURN` and `Not Returned`.
- Kept missing/null cells at zero.
- Kept duplicate rows and duplicate `order_id` at zero.
- Clipped numeric values within the clean S1 source range so no new outlier is introduced.

## Validation

- Rows: `{metrics.get("output_rows")}`
- Columns: `{metrics.get("output_columns")}`
- Missing/null cells: `{metrics.get("missing_total")}`
- Blank text cells: `{metrics.get("blank_text_cells")}`
- Duplicate rows: `{metrics.get("duplicate_rows")}`
- Unique customers: `{metrics.get("unique_customer_id")}`
- Not Returned: `{metrics.get("is_returned_0_count")}`
- Returned: `{metrics.get("is_returned_1_count")}`
- Return rate: `{float(metrics.get("return_rate", 0)) * 100:.2f}%`

This file is a clean source dataset only. Feature engineering and LightGBM train/test split will be created in the next step.
"""
    README.write_text(content, encoding="utf-8")


def write_parent_readme() -> None:
    content = """# LightGBM SETC Clean Datasets

This folder stores clean source datasets for LightGBM experiments.

| Dataset | Rows | Purpose |
|---|---:|---|
| `S1/clean_dataset_s1.csv` | 5,000 | Clean source dataset generated from `data/processed/clean_dataset.csv` |
| `S2/clean_dataset_s2.csv` | 50,000 | Clean expanded dataset generated from S1 distribution |

These files are clean datasets only. Feature engineering and model training artifacts will be stored in separate version folders later.
"""
    PARENT_README.write_text(content, encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s2_generator.SOURCE = SOURCE
    source = pd.read_csv(SOURCE)
    generated = s2_generator.generate_s2()
    generated = make_lightgbm_ids(generated)
    summary, bounds = s2_generator.validate(generated, source)
    summary.loc[summary["metric"].eq("output_file"), "value"] = str(OUTPUT_CSV.relative_to(ROOT))

    generated.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "model_family": "LightGBM",
                "set": "SETC",
                "dataset_version": "S2",
                "source_csv": str(SOURCE.relative_to(ROOT)),
                "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
                "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                "summary": dict(zip(summary["metric"], summary["value"])),
                "numeric_bounds": bounds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(summary)
    write_parent_readme()

    print(f"Created: {OUTPUT_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
