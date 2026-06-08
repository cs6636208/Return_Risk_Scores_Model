from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "clean_dataset.csv"
OUT_ROOT = ROOT / "docs" / "test" / "datasets"
TEST_ROOT = ROOT / "docs" / "test"
TARGET = "is_returned"
RANDOM_STATE = 42


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(df):
        if n == len(df):
            return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        return stratified_bootstrap(df, n)

    counts = df[TARGET].value_counts().sort_index()
    allocations = (counts / len(df) * n).round().astype(int)
    diff = n - int(allocations.sum())
    if diff != 0:
        # Assign rounding remainder to the largest class to keep the sample size exact.
        largest_class = counts.idxmax()
        allocations.loc[largest_class] += diff

    parts = []
    for cls, cls_n in allocations.items():
        cls_df = df[df[TARGET] == cls]
        parts.append(cls_df.sample(n=int(cls_n), random_state=RANDOM_STATE + int(cls)))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def stratified_bootstrap(df: pd.DataFrame, n: int) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    allocations = (counts / len(df) * n).round().astype(int)
    diff = n - int(allocations.sum())
    if diff != 0:
        allocations.loc[counts.idxmax()] += diff

    parts = []
    for cls, cls_n in allocations.items():
        cls_df = df[df[TARGET] == cls]
        parts.append(cls_df.sample(n=int(cls_n), replace=True, random_state=RANDOM_STATE + int(cls)))

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return make_generated_rows_unique(out, n)


def make_generated_rows_unique(df: pd.DataFrame, dataset_size: int) -> pd.DataFrame:
    out = df.copy()
    prefix = f"TEST{dataset_size}"

    if "order_id" in out.columns:
        out["order_id"] = [f"{prefix}_O{i + 1:06d}" for i in range(len(out))]
    if "return_id" in out.columns:
        returned_mask = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int).eq(1)
        out["return_id"] = ""
        out.loc[returned_mask, "return_id"] = [f"{prefix}_R{i + 1:06d}" for i in range(returned_mask.sum())]
    if "score_id" in out.columns:
        out["score_id"] = [f"{prefix}_S{i + 1:06d}" for i in range(len(out))]

    rng = np.random.default_rng(RANDOM_STATE)
    offsets = rng.integers(0, 365, size=len(out))
    date_like_columns = [
        "order_date",
        "expected_delivery_date",
        "delivery_date",
        "registration_date",
        "promo_start_date",
        "promo_end_date",
        "scored_at",
    ]
    for col in date_like_columns:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        shifted = parsed + pd.to_timedelta(offsets, unit="D")
        out.loc[parsed.notna(), col] = shifted.loc[parsed.notna()].astype(str)

    if "return_date" in out.columns:
        parsed = pd.to_datetime(out["return_date"], errors="coerce")
        shifted = parsed + pd.to_timedelta(offsets, unit="D")
        out.loc[parsed.notna(), "return_date"] = shifted.loc[parsed.notna()].astype(str)
        out.loc[pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int).eq(0), "return_date"] = "Not Returned"

    return out


def distribution_summary(df: pd.DataFrame, dataset_size: int) -> pd.DataFrame:
    total = len(df)
    returned = int(df[TARGET].sum())
    not_returned = total - returned
    return pd.DataFrame(
        [
            {
                "dataset_size": dataset_size,
                "split": "full_test",
                "rows": total,
                "not_returned_count": not_returned,
                "returned_count": returned,
                "return_rate": returned / total if total else 0.0,
                "unique_customers": df["customer_id"].nunique() if "customer_id" in df.columns else None,
                "unique_orders": df["order_id"].nunique() if "order_id" in df.columns else None,
                "usage": "use all rows as external/new test data for every model version",
            }
        ]
    )


def export_dataset(df: pd.DataFrame, dataset_size: int) -> pd.DataFrame:
    name = f"clean_dataset_{dataset_size}" if dataset_size <= 5000 else f"clean_dataset_generated_{dataset_size}"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{name}_full_test.csv", index=False, encoding="utf-8-sig")

    summary = distribution_summary(df, dataset_size)
    summary.to_csv(out_dir / f"{name}_distribution_summary.csv", index=False, encoding="utf-8-sig")

    readme = f"""# {name} Test Dataset

Source: `data/processed/clean_dataset.csv`

Generation policy:

- 5,000 rows: original clean dataset sample/full frame from dataset 1.
- 50,000 rows: stratified bootstrap generated from `clean_dataset.csv`, with regenerated `order_id`, `return_id`, and `score_id` plus date jitter to avoid exact duplicate order records.

Test policy:

- No train/validation/test split is created here.
- Use all rows as a new external test set for every model version.
- This is designed to compare old model accuracy vs accuracy on newly generated test data.
- Stratified by `{TARGET}` so Return / Not Returned ratio stays close to the source dataset.

Files:

- `{name}_full_test.csv`
- `{name}_distribution_summary.csv`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE, low_memory=False)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    all_summaries = []
    for dataset_size in [5000, 50000]:
        sample = stratified_sample(df, dataset_size)
        all_summaries.append(export_dataset(sample, dataset_size))

    all_summary = pd.concat(all_summaries, ignore_index=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(OUT_ROOT / "clean_dataset_test_dataset_summary.csv", index=False, encoding="utf-8-sig")
    write_full_test_evaluation_plan(all_summary)
    print(f"Generated test datasets in {OUT_ROOT}")
    print(all_summary.to_string(index=False))


def write_full_test_evaluation_plan(summary: pd.DataFrame) -> None:
    comparison_path = ROOT / "docs" / "Comparison Version" / "version_1_to_4_selected_model_comparison.csv"
    if not comparison_path.exists():
        return

    comparison = pd.read_csv(comparison_path)
    test_files = {
        5000: OUT_ROOT / "clean_dataset_5000" / "clean_dataset_5000_full_test.csv",
        50000: OUT_ROOT / "clean_dataset_generated_50000" / "clean_dataset_generated_50000_full_test.csv",
    }
    rows = []
    for dataset_size, test_path in test_files.items():
        summary_row = summary[summary["dataset_size"].eq(dataset_size)].iloc[0].to_dict()
        for _, version in comparison.iterrows():
            rows.append(
                {
                    "dataset_size": dataset_size,
                    "test_file": str(test_path.relative_to(ROOT)),
                    "test_rows": int(summary_row["rows"]),
                    "test_return_rate": float(summary_row["return_rate"]),
                    "version": version["display_version"],
                    "version_id": version["version"],
                    "model": version["model"],
                    "original_accuracy": version["accuracy"],
                    "original_recall": version["recall"],
                    "original_f1": version["f1"],
                    "original_auc": version["auc"],
                    "new_test_accuracy": "",
                    "new_test_recall": "",
                    "new_test_f1": "",
                    "new_test_auc": "",
                    "accuracy_delta": "",
                    "status": "pending_inference_on_full_test_data",
                }
            )
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TEST_ROOT / "full_test_model_evaluation_plan.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
