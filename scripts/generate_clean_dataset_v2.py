from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "clean_dataset_v2.csv"

N_ROWS = 50_000
N_CUSTOMERS = 6_000
RANDOM_SEED = 20260528


def weighted_choice(rng: np.random.Generator, values: pd.Series, size: int) -> np.ndarray:
    counts = values.value_counts(normalize=True)
    return rng.choice(counts.index.to_numpy(), size=size, p=counts.to_numpy())


def format_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d %H:%M:%S")


def format_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def make_phone_numbers(rng: np.random.Generator, n: int) -> list[str]:
    prefixes = np.array(["080", "081", "082", "083", "084", "085", "086", "087", "088", "089"])
    selected = rng.choice(prefixes, size=n)
    middle = rng.integers(100, 999, size=n)
    tail = rng.integers(1000, 9999, size=n)
    return [f"{p}-{m:03d}-{t:04d}" for p, m, t in zip(selected, middle, tail)]


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    source = pd.read_csv(SOURCE_PATH)

    product_cols = [
        "product_id",
        "product_name",
        "category",
        "brand",
        "is_fragile",
        "product_rating",
        "unit_price",
        "supplier_id",
        "supplier_name",
        "supplier_contact",
    ]
    courier_cols = [
        "courier_id",
        "courier_name",
        "courier_type",
        "avg_delivery_days",
        "damage_rate",
        "coverage_region",
    ]
    promo_cols = [
        "promo_id",
        "promo_name",
        "promo_type",
        "promo_discount_rate",
        "promo_start_date",
        "promo_end_date",
    ]
    customer_cols = [
        "customer_name",
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
        "registration_date",
    ]

    products = source[product_cols].drop_duplicates("product_id").reset_index(drop=True)
    couriers = source[courier_cols].drop_duplicates("courier_id").reset_index(drop=True)
    promos = source[promo_cols].drop_duplicates("promo_id").reset_index(drop=True)
    promos["promo_type"] = promos["promo_type"].fillna("No Promotion")
    customers_seed = source[customer_cols].drop_duplicates().reset_index(drop=True)

    product_probs = (
        source["product_id"]
        .value_counts(normalize=True)
        .reindex(products["product_id"])
        .fillna(0.0)
        .to_numpy()
    )
    product_probs = product_probs / product_probs.sum()

    courier_probs = (
        source["courier_id"]
        .value_counts(normalize=True)
        .reindex(couriers["courier_id"])
        .fillna(0.0)
        .to_numpy()
    )
    courier_probs = courier_probs / courier_probs.sum()

    sampled_customer_idx = rng.choice(customers_seed.index.to_numpy(), size=N_CUSTOMERS, replace=True)
    customer_profiles = customers_seed.loc[sampled_customer_idx].reset_index(drop=True).copy()
    customer_profiles["customer_id"] = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    customer_profiles["customer_phone"] = make_phone_numbers(rng, N_CUSTOMERS)
    customer_profiles["age"] = np.clip(
        customer_profiles["age"].to_numpy() + rng.integers(-5, 6, size=N_CUSTOMERS),
        18,
        75,
    )

    registration_dates = pd.to_datetime(customer_profiles["registration_date"]) + pd.to_timedelta(
        rng.integers(-365, 366, size=N_CUSTOMERS), unit="D"
    )
    registration_dates = registration_dates.clip(
        lower=pd.Timestamp("2018-01-01"),
        upper=pd.Timestamp("2024-12-31"),
    )
    customer_profiles["registration_date"] = registration_dates

    tier_bias = customer_profiles["membership_tier"].map(
        {"Bronze": 0.03, "Silver": 0.00, "Gold": -0.02, "Platinum": -0.04}
    ).fillna(0.0)
    province_bias = customer_profiles["province"].map(
        {"Remote_Area": 0.08, "Phuket": 0.03, "Songkhla": 0.03, "Bangkok": -0.02}
    ).fillna(0.0)
    customer_profiles["customer_return_tendency"] = np.clip(
        rng.beta(1.8, 4.8, size=N_CUSTOMERS) + tier_bias + province_bias,
        0.02,
        0.75,
    )

    start = pd.to_datetime(source["order_date"].min()).normalize()
    end = pd.to_datetime(source["order_date"].max()).normalize()
    date_offsets = rng.integers(0, (end - start).days + 1, size=N_ROWS)
    hours = weighted_choice(rng, source["order_hour"], N_ROWS).astype(int)
    order_dates = start + pd.to_timedelta(date_offsets, unit="D") + pd.to_timedelta(hours, unit="h")

    customer_weights = rng.gamma(shape=5.0, scale=1.0, size=N_CUSTOMERS)
    customer_weights = customer_weights / customer_weights.sum()
    customer_idx = rng.choice(customer_profiles.index.to_numpy(), size=N_ROWS, p=customer_weights)

    product_idx = rng.choice(products.index.to_numpy(), size=N_ROWS, p=product_probs)
    courier_idx = rng.choice(couriers.index.to_numpy(), size=N_ROWS, p=courier_probs)

    orders = pd.DataFrame(
        {
            "order_id": [f"ORDV2{i:06d}" for i in range(1, N_ROWS + 1)],
            "order_date": order_dates,
            "customer_profile_idx": customer_idx,
            "product_profile_idx": product_idx,
            "courier_profile_idx": courier_idx,
        }
    )
    orders = orders.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    orders["order_id"] = [f"ORDV2{i:06d}" for i in range(1, N_ROWS + 1)]

    orders = orders.join(customer_profiles.drop(columns=["customer_return_tendency"]), on="customer_profile_idx")
    orders = orders.join(products, on="product_profile_idx")
    orders = orders.join(couriers, on="courier_profile_idx")
    orders["customer_return_tendency"] = customer_profiles.loc[
        orders["customer_profile_idx"], "customer_return_tendency"
    ].to_numpy()

    promo_none = promos[promos["promo_id"].eq("PROMO_NONE")].iloc[0].to_dict()
    campaign_promos = promos[~promos["promo_id"].eq("PROMO_NONE")].copy()
    campaign_promos["promo_start_date_dt"] = pd.to_datetime(campaign_promos["promo_start_date"])
    campaign_promos["promo_end_date_dt"] = pd.to_datetime(campaign_promos["promo_end_date"])

    promo_rows: list[dict[str, object]] = []
    for order_date in orders["order_date"]:
        active = campaign_promos[
            (campaign_promos["promo_start_date_dt"] <= order_date)
            & (campaign_promos["promo_end_date_dt"] >= order_date)
        ]
        if len(active) and rng.random() < 0.48:
            promo_rows.append(active.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0].to_dict())
        else:
            promo_rows.append(promo_none)
    promo_df = pd.DataFrame(promo_rows)[promo_cols].reset_index(drop=True)
    orders = pd.concat([orders.reset_index(drop=True), promo_df], axis=1)

    orders["channel_type"] = weighted_choice(rng, source["channel_type"], N_ROWS)
    orders["payment_method"] = weighted_choice(rng, source["payment_method"], N_ROWS)
    orders["quantity"] = weighted_choice(rng, source["quantity"], N_ROWS).astype(int)
    orders["tier_discount_pct"] = orders["membership_tier"].map(
        {"Bronze": 0.05, "Silver": 0.10, "Gold": 0.15, "Platinum": 0.20}
    ).astype(float)
    orders["campaign_discount_pct"] = orders["promo_discount_rate"].astype(float)
    orders["total_discount_pct"] = (orders["tier_discount_pct"] + orders["campaign_discount_pct"]).clip(0, 0.35)
    gross_amount = orders["quantity"] * orders["unit_price"]
    orders["discount_applied_amount"] = (gross_amount * orders["total_discount_pct"]).round(2)
    orders["total_amount"] = (gross_amount - orders["discount_applied_amount"]).round(2)

    orders["delivery_time_expected_days"] = rng.choice([1, 2, 3], size=N_ROWS, p=[0.34, 0.33, 0.33])
    delivery_noise = rng.normal(loc=0.3, scale=1.10, size=N_ROWS)
    courier_base = orders["avg_delivery_days"].to_numpy()
    remote_penalty = np.where(orders["province"].eq("Remote_Area"), rng.choice([0, 1, 2], size=N_ROWS, p=[0.35, 0.50, 0.15]), 0)
    orders["delivery_days"] = np.clip(np.rint(courier_base + delivery_noise + remote_penalty), 1, 6).astype(int)
    orders["delay_days"] = orders["delivery_days"] - orders["delivery_time_expected_days"]
    orders["expected_delivery_date"] = orders["order_date"] + pd.to_timedelta(
        orders["delivery_time_expected_days"], unit="D"
    )
    orders["delivery_date"] = orders["order_date"] + pd.to_timedelta(orders["delivery_days"], unit="D")
    orders["order_hour"] = pd.to_datetime(orders["order_date"]).dt.hour
    orders["customer_age_days"] = (
        pd.to_datetime(orders["order_date"]).dt.normalize()
        - pd.to_datetime(orders["registration_date"]).dt.normalize()
    ).dt.days.clip(lower=0)

    orders["hist_order_count"] = 0
    orders["hist_return_rate"] = 0.0
    orders["days_since_last_order"] = -1
    orders["is_repurchased_item"] = 0
    orders["is_returned"] = 0
    orders["risk_score"] = 0.0
    orders["risk_tier"] = "Low"
    orders["return_id"] = "NO_RETURN"
    orders["return_date"] = "Not Returned"
    orders["return_reason"] = "Not Returned"
    orders["return_scenario"] = "Not Returned"
    orders["item_condition"] = "Not Returned"
    orders["return_status"] = "Not Returned"
    orders["refund_amount"] = 0.0
    orders["shap_values"] = "{'rating': 0.0, 'history': 0.0}"

    return_reasons = source.loc[source["return_reason"].ne("Not Returned"), "return_reason"]
    item_conditions = source.loc[source["item_condition"].ne("Not Returned"), "item_condition"]
    reason_values = return_reasons.value_counts(normalize=True)
    condition_values = item_conditions.value_counts(normalize=True)

    customer_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"orders": 0, "returns": 0, "last_order_date": None, "products": set()}
    )
    return_counter = 1

    for idx, row in orders.iterrows():
        customer_id = row["customer_id"]
        stats = customer_stats[customer_id]
        prior_orders = int(stats["orders"])
        prior_returns = int(stats["returns"])
        hist_return_rate = prior_returns / prior_orders if prior_orders else 0.0
        last_order_date = stats["last_order_date"]
        days_since_last = -1 if last_order_date is None else int((row["order_date"] - last_order_date).days)
        is_repurchased = int(row["product_id"] in stats["products"])

        orders.at[idx, "hist_order_count"] = prior_orders
        orders.at[idx, "hist_return_rate"] = round(hist_return_rate, 4)
        orders.at[idx, "days_since_last_order"] = days_since_last
        orders.at[idx, "is_repurchased_item"] = is_repurchased

        low_rating = max(0.0, 4.4 - float(row["product_rating"])) * 0.35
        high_discount = 0.45 if float(row["total_discount_pct"]) >= 0.15 else 0.0
        cod_risk = 0.25 if row["payment_method"] == "COD" else 0.0
        fragile_risk = 0.18 if bool(row["is_fragile"]) else 0.0
        logistics_risk = float(row["damage_rate"]) * 4.0 + max(0, int(row["delay_days"])) * 0.22
        remote_risk = 0.18 if row["province"] == "Remote_Area" else 0.0
        category_risk = {
            "Fashion": 0.22,
            "Electronics": 0.18,
            "Home_Appliance": 0.12,
            "Cosmetics": 0.08,
            "Supplement": -0.02,
        }.get(row["category"], 0.0)
        repeat_discount = -0.12 if is_repurchased else 0.0
        recent_order_risk = 0.10 if 0 <= days_since_last <= 7 else 0.0

        score_logit = (
            -2.69
            + 1.30 * float(row["customer_return_tendency"])
            + 1.10 * hist_return_rate
            + min(prior_orders, 20) * 0.008
            + low_rating
            + high_discount
            + cod_risk
            + fragile_risk
            + logistics_risk
            + remote_risk
            + category_risk
            + repeat_discount
            + recent_order_risk
            + rng.normal(0, 0.18)
        )
        return_probability = logistic(score_logit)
        risk_score = float(np.clip(return_probability + rng.normal(0, 0.055), 0.01, 0.98))
        is_returned = int(rng.random() < return_probability)

        orders.at[idx, "is_returned"] = is_returned
        orders.at[idx, "risk_score"] = round(risk_score, 2)
        orders.at[idx, "shap_values"] = (
            "{"
            f"'rating': {round(low_rating, 3)}, "
            f"'history': {round(hist_return_rate, 3)}, "
            f"'discount': {round(high_discount, 3)}, "
            f"'logistics': {round(logistics_risk, 3)}"
            "}"
        )

        if is_returned:
            return_delay = int(rng.integers(2, 15))
            refund_ratio = rng.choice([1.0, 0.9, 0.8, 0.7], p=[0.72, 0.12, 0.10, 0.06])
            orders.at[idx, "return_id"] = f"RETV2{return_counter:06d}"
            orders.at[idx, "return_date"] = (row["delivery_date"] + pd.Timedelta(days=return_delay)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            orders.at[idx, "return_reason"] = rng.choice(reason_values.index.to_numpy(), p=reason_values.to_numpy())
            orders.at[idx, "return_scenario"] = "Standard Return"
            orders.at[idx, "item_condition"] = rng.choice(
                condition_values.index.to_numpy(), p=condition_values.to_numpy()
            )
            orders.at[idx, "return_status"] = "Completed"
            orders.at[idx, "refund_amount"] = round(float(row["total_amount"]) * float(refund_ratio), 2)
            return_counter += 1
            stats["returns"] = prior_returns + 1

        stats["orders"] = prior_orders + 1
        stats["last_order_date"] = row["order_date"]
        stats["products"].add(row["product_id"])

    risk_distribution = source["risk_tier"].value_counts(normalize=True)
    low_cut = float(risk_distribution.get("Low", 0.388))
    medium_cut = low_cut + float(risk_distribution.get("Medium", 0.2696))
    risk_signal_rank = orders["risk_score"].rank(method="first", pct=True)
    orders["risk_tier"] = np.select(
        [risk_signal_rank <= low_cut, risk_signal_rank <= medium_cut],
        ["Low", "Medium"],
        default="High",
    )

    risk_ranges = source.groupby("risk_tier")["risk_score"].agg(["min", "max"])
    calibrated_scores = pd.Series(index=orders.index, dtype=float)
    for tier in ["Low", "Medium", "High"]:
        mask = orders["risk_tier"].eq(tier)
        tier_rank = orders.loc[mask, "risk_score"].rank(method="first", pct=True)
        lower = float(risk_ranges.loc[tier, "min"])
        upper = float(risk_ranges.loc[tier, "max"])
        tier_scores = lower + tier_rank * (upper - lower)
        tier_scores = tier_scores + rng.normal(0, 0.008, size=mask.sum())
        calibrated_scores.loc[mask] = np.clip(tier_scores, lower, upper)
    orders["risk_score"] = calibrated_scores.round(2)

    orders["score_id"] = [f"SCRV2{i:06d}" for i in range(1, N_ROWS + 1)]
    orders["scored_at"] = orders["order_date"]

    final_cols = source.columns.tolist()
    final = orders[final_cols].copy()
    final["order_date"] = format_datetime(final["order_date"])
    final["expected_delivery_date"] = format_datetime(final["expected_delivery_date"])
    final["delivery_date"] = format_datetime(final["delivery_date"])
    final["registration_date"] = format_date(final["registration_date"])
    final["promo_start_date"] = format_date(final["promo_start_date"])
    final["promo_end_date"] = format_date(final["promo_end_date"])
    final["scored_at"] = format_datetime(final["scored_at"])

    final = final.fillna(
        {
            "promo_type": "No Promotion",
            "return_id": "NO_RETURN",
            "return_date": "Not Returned",
            "return_reason": "Not Returned",
            "return_scenario": "Not Returned",
            "item_condition": "Not Returned",
            "return_status": "Not Returned",
        }
    )

    bool_cols = ["is_fragile"]
    int_cols = [
        "age",
        "customer_age_days",
        "quantity",
        "delivery_time_expected_days",
        "delivery_days",
        "delay_days",
        "is_repurchased_item",
        "order_hour",
        "days_since_last_order",
        "hist_order_count",
        "is_returned",
    ]
    float_cols = [
        "product_rating",
        "avg_delivery_days",
        "damage_rate",
        "promo_discount_rate",
        "unit_price",
        "tier_discount_pct",
        "campaign_discount_pct",
        "total_discount_pct",
        "discount_applied_amount",
        "total_amount",
        "hist_return_rate",
        "refund_amount",
        "risk_score",
    ]
    for col in bool_cols:
        final[col] = final[col].astype(bool)
    for col in int_cols:
        final[col] = final[col].astype(int)
    for col in float_cols:
        final[col] = final[col].astype(float).round(4)

    if final.shape != (N_ROWS, len(source.columns)):
        raise ValueError(f"Unexpected output shape: {final.shape}")
    if final.isna().sum().sum() != 0:
        missing = final.isna().sum()
        raise ValueError(f"Generated data still contains missing values: {missing[missing > 0].to_dict()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"source={SOURCE_PATH}")
    print(f"output={OUTPUT_PATH}")
    print(f"shape={final.shape}")
    print(f"is_returned_rate={final['is_returned'].mean():.4f}")
    print(f"date_range={final['order_date'].min()} to {final['order_date'].max()}")
    print(f"missing_values={int(final.isna().sum().sum())}")


if __name__ == "__main__":
    main()
