from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s1.csv"
OUT_DIR = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data"
OUT_CSV = OUT_DIR / "clean_dataset_s2.csv"
SUMMARY_CSV = OUT_DIR / "clean_dataset_s2_validation_summary.csv"
SUMMARY_JSON = OUT_DIR / "clean_dataset_s2_validation_summary.json"
README = OUT_DIR / "README.md"

RANDOM_STATE = 20260605
N_ROWS = 50_000
N_CUSTOMERS = 5_000
ORDERS_PER_CUSTOMER = 10
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

RETURN_REASON_VALUES = ["Defective", "Changed Mind", "Better Price Elsewhere", "Wrong Item"]
ITEM_CONDITION_VALUES = ["Damaged Packaging", "Unopened", "Defective", "Used"]


def as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.replace({"Not Returned": pd.NA, "Unknown": pd.NA}), errors="coerce")


def blank_text_count(df: pd.DataFrame) -> int:
    total = 0
    for column in df.select_dtypes(include=["object", "string"]).columns:
        total += int(df[column].astype(str).str.strip().eq("").sum())
    return total


def weighted_sample(values: pd.Series, rng: np.random.Generator, size: int) -> np.ndarray:
    counts = values.value_counts(normalize=True)
    return rng.choice(counts.index.to_numpy(), size=size, p=counts.to_numpy())


def make_customer_profiles(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    profile_cols = [
        "customer_id",
        "customer_name",
        "customer_phone",
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
    ]
    profiles = base[profile_cols].drop_duplicates("customer_id").reset_index(drop=True)
    sampled = profiles.iloc[rng.choice(len(profiles), size=N_CUSTOMERS, replace=True)].reset_index(drop=True)
    sampled["customer_id"] = [f"C_S2_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
    sampled["customer_name"] = [f"Customer S2 {i:05d}" for i in range(1, N_CUSTOMERS + 1)]
    sampled["customer_phone"] = [f"09{i:08d}" for i in range(1, N_CUSTOMERS + 1)]
    return sampled


def make_order_schedule(customers: pd.DataFrame, base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    min_dt = pd.to_datetime(base["order_date"]).min().floor("D")
    max_dt = pd.to_datetime(base["order_date"]).max().floor("D")
    order_hours = weighted_sample(base["order_hour"], rng, N_ROWS).astype(int)

    rows: list[dict[str, object]] = []
    hour_pos = 0
    for _, customer in customers.iterrows():
        intervals = rng.integers(7, 45, size=ORDERS_PER_CUSTOMER - 1)
        span_days = int(intervals.sum())
        latest_start = max_dt - pd.Timedelta(days=span_days + 1)
        latest_start = max(latest_start, min_dt)
        max_offset = max((latest_start - min_dt).days, 1)
        start = min_dt + pd.Timedelta(days=int(rng.integers(0, max_offset + 1)))
        order_dates = [start]
        for gap in intervals:
            order_dates.append(order_dates[-1] + pd.Timedelta(days=int(gap)))

        first_order = order_dates[0]
        registration_date = first_order - pd.Timedelta(days=int(rng.integers(540, 2100)))

        for order_seq, order_date in enumerate(order_dates, start=1):
            hour = int(order_hours[hour_pos])
            hour_pos += 1
            timestamp = pd.Timestamp(order_date.date()) + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "customer_id": customer["customer_id"],
                    "customer_name": customer["customer_name"],
                    "customer_phone": customer["customer_phone"],
                    "gender": customer["gender"],
                    "age": customer["age"],
                    "membership_tier": customer["membership_tier"],
                    "preferred_channel": customer["preferred_channel"],
                    "province": customer["province"],
                    "registration_date": registration_date,
                    "order_date": timestamp,
                    "customer_order_seq": order_seq,
                }
            )
    return pd.DataFrame(rows)


def build_template_rows(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    returned_count = int(round(N_ROWS * float(base[TARGET].mean())))
    non_returned_count = N_ROWS - returned_count
    targets = np.array([0] * non_returned_count + [1] * returned_count)
    rng.shuffle(targets)

    returned = base[base[TARGET].eq(1)].reset_index(drop=True)
    non_returned = base[base[TARGET].eq(0)].reset_index(drop=True)
    sampled = pd.DataFrame(index=np.arange(N_ROWS), columns=base.columns)

    returned_mask = targets == 1
    non_returned_mask = ~returned_mask
    returned_idx = rng.choice(len(returned), size=int(returned_mask.sum()), replace=True)
    non_returned_idx = rng.choice(len(non_returned), size=int(non_returned_mask.sum()), replace=True)
    sampled.loc[returned_mask, :] = returned.iloc[returned_idx].to_numpy()
    sampled.loc[non_returned_mask, :] = non_returned.iloc[non_returned_idx].to_numpy()
    sampled = sampled.reset_index(drop=True)
    sampled[TARGET] = targets
    return sampled


def recompute_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["customer_id", "order_date"]).copy()
    out["days_since_last_order"] = -1
    out["hist_order_count"] = 0
    out["hist_return_rate"] = 0.0

    for _, group in out.groupby("customer_id", sort=False):
        idx = group.index.to_list()
        dates = group["order_date"].tolist()
        returns = group[TARGET].astype(int).to_numpy()
        prior_returns = 0
        for pos, row_idx in enumerate(idx):
            out.at[row_idx, "hist_order_count"] = pos
            out.at[row_idx, "hist_return_rate"] = prior_returns / pos if pos else 0.0
            if pos == 0:
                out.at[row_idx, "days_since_last_order"] = -1
            else:
                out.at[row_idx, "days_since_last_order"] = int((dates[pos] - dates[pos - 1]).days)
            prior_returns += int(returns[pos])

    return out


def refresh_dates_and_amounts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    out["delivery_time_expected_days"] = pd.to_numeric(out["delivery_time_expected_days"], errors="coerce").fillna(2).clip(1, 3).astype(int)
    out["delivery_days"] = pd.to_numeric(out["delivery_days"], errors="coerce").fillna(4).clip(1, 6).astype(int)
    out["delay_days"] = out["delivery_days"] - out["delivery_time_expected_days"]
    out["expected_delivery_date"] = out["order_date"] + pd.to_timedelta(out["delivery_time_expected_days"], unit="D")
    out["delivery_date"] = out["order_date"] + pd.to_timedelta(out["delivery_days"], unit="D")
    out["customer_age_days"] = (out["order_date"] - out["registration_date"]).dt.days.clip(lower=0).astype(int)
    out["order_hour"] = out["order_date"].dt.hour.astype(int)

    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(1).clip(1, 2).round().astype(int)
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce").fillna(1200).clip(290, 3850).round(2)
    out["tier_discount_pct"] = pd.to_numeric(out["tier_discount_pct"], errors="coerce").fillna(0.10).clip(0.05, 0.20).round(2)
    out["campaign_discount_pct"] = pd.to_numeric(out["campaign_discount_pct"], errors="coerce").fillna(0.0).clip(0.0, 0.15).round(2)
    out["promo_discount_rate"] = pd.to_numeric(out["promo_discount_rate"], errors="coerce").fillna(0.0).clip(0.0, 0.15).round(2)
    out["total_discount_pct"] = (
        out["tier_discount_pct"] + out["campaign_discount_pct"]
    ).clip(0.05, 0.35).round(2)

    gross = out["quantity"] * out["unit_price"]
    out["discount_applied_amount"] = (gross * out["total_discount_pct"]).clip(14.5, 562.5).round(2)
    out["total_amount"] = (gross - out["discount_applied_amount"]).clip(217.5, 5031.75).round(2)

    no_promo = out["promo_id"].astype(str).eq("PROMO_NONE")
    out.loc[no_promo, "promo_name"] = "No Promotion"
    out.loc[no_promo, "promo_type"] = "No Promotion"
    out.loc[no_promo, "promo_discount_rate"] = 0.0
    out.loc[no_promo, "campaign_discount_pct"] = 0.0

    returned_mask = out[TARGET].astype(int).eq(1)
    return_counter = np.arange(1, int(returned_mask.sum()) + 1)
    out.loc[returned_mask, "return_id"] = [f"RET_S2_{i:06d}" for i in return_counter]
    return_offsets = rng.integers(1, 15, size=int(returned_mask.sum()))
    out.loc[returned_mask, "return_date"] = (
        out.loc[returned_mask, "delivery_date"].reset_index(drop=True)
        + pd.to_timedelta(return_offsets, unit="D")
    ).to_numpy()
    out.loc[returned_mask, "return_scenario"] = "Standard Return"
    out.loc[returned_mask, "return_status"] = "Completed"
    out.loc[returned_mask, "return_reason"] = rng.choice(RETURN_REASON_VALUES, size=int(returned_mask.sum()))
    out.loc[returned_mask, "item_condition"] = rng.choice(ITEM_CONDITION_VALUES, size=int(returned_mask.sum()))
    refund = out.loc[returned_mask, "total_amount"].to_numpy() * rng.uniform(0.60, 1.00, size=int(returned_mask.sum()))
    out.loc[returned_mask, "refund_amount"] = np.minimum(refund, 1668.75).round(2)

    not_returned_mask = ~returned_mask
    out.loc[not_returned_mask, "return_id"] = "NO_RETURN"
    out.loc[not_returned_mask, "return_date"] = "Not Returned"
    out.loc[not_returned_mask, "return_reason"] = "Not Returned"
    out.loc[not_returned_mask, "return_scenario"] = "Not Returned"
    out.loc[not_returned_mask, "item_condition"] = "Not Returned"
    out.loc[not_returned_mask, "return_status"] = "Not Returned"
    out.loc[not_returned_mask, "refund_amount"] = 0.0
    return out


def refresh_scores(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    category_effect = out["category"].map(
        {
            "Fashion": 0.08,
            "Electronics": 0.06,
            "Cosmetics": 0.05,
            "Home_Appliance": 0.04,
            "Supplement": 0.03,
        }
    ).fillna(0.04)
    payment_effect = out["payment_method"].astype(str).eq("COD").astype(float) * 0.05
    province_effect = out["province"].astype(str).isin(["Remote_Area", "Phuket", "Songkhla"]).astype(float) * 0.04
    history_effect = pd.to_numeric(out["hist_return_rate"], errors="coerce").fillna(0).astype(float) * 0.20
    rating_effect = (5 - pd.to_numeric(out["product_rating"], errors="coerce").fillna(4.3)).clip(0, 2) * 0.04
    target_effect = out[TARGET].astype(int) * 0.18
    noise = rng.normal(0, 0.035, size=len(out))

    score = 0.06 + category_effect + payment_effect + province_effect + history_effect + rating_effect + target_effect + noise
    out["risk_score"] = np.clip(score, 0.01, 0.69).round(2)
    out["risk_tier"] = pd.cut(
        out["risk_score"],
        bins=[-0.001, 0.25, 0.45, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    out["score_id"] = [f"SCR_S2_{i:06d}" for i in range(1, len(out) + 1)]
    out["scored_at"] = out["order_date"] + pd.to_timedelta(rng.integers(0, 6, size=len(out)), unit="h")
    out["shap_values"] = np.where(
        out[TARGET].astype(int).eq(1),
        "{'rating': 0.1, 'history': 0.2}",
        "{'rating': -0.05, 'history': -0.1}",
    )
    return out


def stringify_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in DATE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["return_date"] = out["return_date"].apply(
        lambda value: value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, (pd.Timestamp,))
        else str(value)
    )
    return out


def generate_s2() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    base = pd.read_csv(SOURCE)
    if len(base) != 5000:
        raise AssertionError(f"Expected S1 source to have 5000 rows, got {len(base)}")
    if int(base.isna().sum().sum()) != 0:
        raise AssertionError("S1 source still has missing/null values")

    for column in DATE_COLUMNS:
        base[column] = as_datetime(base[column])

    customers = make_customer_profiles(base, rng)
    schedule = make_order_schedule(customers, base, rng)
    templates = build_template_rows(base, rng)
    generated = templates.copy()

    customer_cols = [
        "customer_id",
        "customer_name",
        "customer_phone",
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
        "registration_date",
        "order_date",
    ]
    generated[customer_cols] = schedule[customer_cols].to_numpy()
    generated = refresh_dates_and_amounts(generated, rng)
    generated = recompute_history(generated)
    generated = refresh_scores(generated, rng)
    generated = generated.sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    generated["order_id"] = [f"ORD_S2_{i:06d}" for i in range(1, len(generated) + 1)]
    generated = stringify_dates(generated)

    for column in generated.select_dtypes(include=["object", "string"]).columns:
        generated[column] = generated[column].fillna("Unknown").astype(str).str.strip()
    for column in generated.select_dtypes(include=[np.number]).columns:
        generated[column] = generated[column].fillna(generated[column].median())

    return generated[base.columns.tolist()]


def numeric_bound_violations(df: pd.DataFrame, source: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric_cols = source.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        lower = float(source[column].min())
        upper = float(source[column].max())
        values = pd.to_numeric(df[column], errors="coerce")
        violations = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": column,
                "source_min": lower,
                "source_max": upper,
                "output_min": float(values.min()),
                "output_max": float(values.max()),
                "bound_violations": violations,
            }
        )
    return rows


def validate(df: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    bounds = numeric_bound_violations(df, source)
    checks = {
        "row_count_is_50000": len(df) == N_ROWS,
        "column_count_matches_source": len(df.columns) == len(source.columns),
        "missing_total_is_zero": int(df.isna().sum().sum()) == 0,
        "blank_text_cells_is_zero": blank_text_count(df) == 0,
        "duplicate_rows_is_zero": int(df.duplicated().sum()) == 0,
        "order_id_is_unique": df["order_id"].is_unique,
        "customer_count_is_5000": df["customer_id"].nunique() == N_CUSTOMERS,
        "target_0_count_is_35450": int((df[TARGET] == 0).sum()) == 35_450,
        "target_1_count_is_14550": int((df[TARGET] == 1).sum()) == 14_550,
        "numeric_bound_violations_is_zero": sum(row["bound_violations"] for row in bounds) == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")

    rows = [
        {"metric": "source_file", "value": str(SOURCE.relative_to(ROOT))},
        {"metric": "output_file", "value": str(OUT_CSV.relative_to(ROOT))},
        {"metric": "source_rows", "value": len(source)},
        {"metric": "output_rows", "value": len(df)},
        {"metric": "output_columns", "value": len(df.columns)},
        {"metric": "missing_total", "value": int(df.isna().sum().sum())},
        {"metric": "blank_text_cells", "value": blank_text_count(df)},
        {"metric": "duplicate_rows", "value": int(df.duplicated().sum())},
        {"metric": "unique_order_id", "value": int(df["order_id"].nunique())},
        {"metric": "unique_customer_id", "value": int(df["customer_id"].nunique())},
        {"metric": "is_returned_0_count", "value": int((df[TARGET] == 0).sum())},
        {"metric": "is_returned_1_count", "value": int((df[TARGET] == 1).sum())},
        {"metric": "return_rate", "value": round(float(df[TARGET].mean()), 4)},
        {"metric": "numeric_bound_violations", "value": sum(row["bound_violations"] for row in bounds)},
    ]
    for column in ["category", "payment_method", "channel_type", "province", "membership_tier", "courier_type"]:
        top = df[column].value_counts(normalize=True).head(8)
        rows.append({"metric": f"{column}_distribution_top", "value": json.dumps(top.round(4).to_dict(), ensure_ascii=False)})

    summary = pd.DataFrame(rows)
    return summary, bounds


def write_readme(summary: pd.DataFrame) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    content = f"""# SETA S2 Clean Dataset

`clean_dataset_s2.csv` ถูก generate จาก `docs/XGBoost/SETA/clean_data/clean_dataset_s1.csv` เพื่อขยายข้อมูลจาก 5,000 rows เป็น 50,000 rows

## Output

- Rows: `{metrics["output_rows"]}`
- Columns: `{metrics["output_columns"]}`
- Missing/null: `{metrics["missing_total"]}`
- Blank text cells: `{metrics["blank_text_cells"]}`
- Duplicate rows: `{metrics["duplicate_rows"]}`
- Unique customers: `{metrics["unique_customer_id"]}`
- Return rate: `{float(metrics["return_rate"]) * 100:.2f}%`

## Generation Logic

- ใช้ S1 เป็น clean source และ bootstrap แบบ stratified เพื่อรักษาสัดส่วน `is_returned`
- สร้าง customer ใหม่ 5,000 คน โดยอิง distribution ของ profile เดิม
- สร้าง order คนละ 10 order รวม 50,000 rows
- คำนวณ `hist_order_count`, `hist_return_rate`, `days_since_last_order` ใหม่แบบราย customer
- เติม return fields ให้ครบ: not returned ใช้ `NO_RETURN` และ `Not Returned`
- คุม numeric values ให้อยู่ใน min/max ของ S1 clean source เพื่อไม่สร้าง outlier ใหม่

ไฟล์นี้เป็น clean dataset สำหรับต่อยอดทำ S2 V1-V5 feature engineering/model training ไม่ใช่ holdout test set
"""
    README.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    generated = generate_s2()
    summary, bounds = validate(generated, source)

    generated.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "summary": dict(zip(summary["metric"], summary["value"])),
                "numeric_bounds": bounds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(summary)
    print(summary.to_string(index=False))
    print(f"Created: {OUT_CSV}")


if __name__ == "__main__":
    main()
