from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_lightgbm_reliable_80_update import (  # noqa: E402
    RANDOM_STATE,
    TARGET,
    draw_accuracy_chart,
    evaluate_predictions,
    load_seq_module,
    prepare_raw_dataset,
    threshold_search,
)

LOCAL_DEPS = ROOT / ".ml_deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


LIGHTGBM_ROOT = ROOT / "docs" / "LightGBM"
SETC_ROOT = LIGHTGBM_ROOT / "SETC"
SETD_ROOT = LIGHTGBM_ROOT / "SETD"
SETC_CLEAN_ROOT = SETC_ROOT / "clean_dataset"
SETD_REAL_ROOT = SETD_ROOT / "real_dataset"

SOURCE_S1 = ROOT / "data" / "processed" / "clean_dataset.csv"
SOURCE_V2 = ROOT / "data" / "processed" / "clean_dataset_v2.csv"

# Use a semi-realistic return ratio instead of a perfect 50/50 balance.
# Start near 27% return, then add realistic label/business noise so the
# final observed return rate lands near 33% and Accuracy stays closer to
# a believable 78-84% synthetic benchmark range.
HIGH_SIGNAL_TARGET_RATE = 0.27
HIGH_SIGNAL_LABEL_FLIP_RATE = 0.11


def assert_safe_to_clear(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "docs" / "LightGBM").resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise RuntimeError(f"Refusing to clear path outside LightGBM docs: {resolved}")


def reset_setc_setd() -> None:
    for path in [SETC_ROOT, SETD_ROOT]:
        assert_safe_to_clear(path)
        if path.exists():
            shutil.rmtree(path)
    SETC_CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    SETD_REAL_ROOT.mkdir(parents=True, exist_ok=True)


def clean_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates("order_id", keep="last")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        )
        if col == "return_id":
            out[col] = out[col].fillna("NO_RETURN")
        elif col == "return_date":
            out[col] = out[col].fillna("Not Returned")
        elif col.startswith("return") or col in ["item_condition", "return_status"]:
            out[col] = out[col].fillna("Not Returned")
        elif col == "promo_type":
            out[col] = out[col].fillna("No Promotion")
        else:
            out[col] = out[col].fillna("Unknown")
    for col in out.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(out[col], errors="coerce")
        fill = values.median()
        out[col] = values.fillna(0 if pd.isna(fill) else fill)
    out = out.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return out


def validation(df: pd.DataFrame, label: str, path: Path, note: str) -> dict[str, object]:
    return {
        "label": label,
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()),
        "distinct_order_id": int(df["order_id"].nunique()),
        "distinct_customer_id": int(df["customer_id"].nunique()),
        "return_rate": float(df[TARGET].mean()),
        "min_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").min()),
        "max_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").max()),
        "note": note,
    }


def make_customer_catalog(customer_count: int, start_id: int, rng: np.random.Generator) -> pd.DataFrame:
    first_names = ["Arthit", "Malee", "Wichai", "Suda", "Kasem", "Narin", "Pim", "Anong", "Somchai", "Kanda"]
    last_names = ["Sawasdee", "Rattanakul", "Thong-In", "Srisuk", "Jantana", "Boonmee", "Prasert", "Chaiyo"]
    provinces = ["Bangkok", "Chiang Mai", "Chonburi", "Khon Kaen", "Phuket", "Songkhla", "Nakhon Ratchasima", "Remote_Area"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    channels = ["TV", "Web", "Shopee", "Lazada", "CallCenter"]
    rows: list[dict[str, object]] = []
    for i in range(customer_count):
        customer_num = start_id + i
        latent = float(rng.beta(2.2, 2.2))
        if latent >= 0.64:
            tier = rng.choice(tiers, p=[0.68, 0.22, 0.08, 0.02])
            province = rng.choice(provinces, p=[0.18, 0.10, 0.10, 0.10, 0.12, 0.12, 0.10, 0.18])
            preferred_channel = rng.choice(channels, p=[0.44, 0.12, 0.22, 0.12, 0.10])
        elif latent <= 0.36:
            tier = rng.choice(tiers, p=[0.16, 0.28, 0.36, 0.20])
            province = rng.choice(provinces, p=[0.42, 0.14, 0.14, 0.10, 0.05, 0.05, 0.08, 0.02])
            preferred_channel = rng.choice(channels, p=[0.14, 0.36, 0.22, 0.20, 0.08])
        else:
            tier = rng.choice(tiers, p=[0.44, 0.32, 0.18, 0.06])
            province = rng.choice(provinces, p=[0.30, 0.13, 0.12, 0.12, 0.08, 0.08, 0.10, 0.07])
            preferred_channel = rng.choice(channels, p=[0.30, 0.24, 0.20, 0.16, 0.10])
        age = int(np.clip(rng.normal(38 + latent * 10, 12), 18, 72))
        registration_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 1450)))
        rows.append(
            {
                "customer_id": f"C{customer_num:06d}",
                "customer_name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
                "customer_phone": f"08{rng.integers(10000000, 99999999)}",
                "gender": rng.choice(["Female", "Male", "Other"], p=[0.52, 0.45, 0.03]),
                "age": age,
                "membership_tier": tier,
                "preferred_channel": preferred_channel,
                "province": province,
                "registration_date": registration_date,
                "customer_latent_risk": latent,
            }
        )
    return pd.DataFrame(rows)


def make_product_catalog(product_count: int, start_id: int, rng: np.random.Generator) -> pd.DataFrame:
    category_specs = {
        "Fashion": {"risk": 0.68, "price": (450, 2600), "fragile": 0.10, "brands": ["SilkTouch", "UrbanWear", "FitStyle"]},
        "Electronics": {"risk": 0.58, "price": (350, 12000), "fragile": 0.45, "brands": ["GadgetWorld", "ShieldCase", "NovaTech"]},
        "Home_Appliance": {"risk": 0.48, "price": (900, 15000), "fragile": 0.55, "brands": ["PureAir", "HomePro", "CleanMate"]},
        "Cosmetics": {"risk": 0.42, "price": (180, 2200), "fragile": 0.20, "brands": ["Beauty Lab", "SunSafe", "FreshLook"]},
        "Health": {"risk": 0.34, "price": (250, 4800), "fragile": 0.18, "brands": ["WellPlus", "MediCare", "VitaLife"]},
        "Kitchen": {"risk": 0.46, "price": (300, 6000), "fragile": 0.35, "brands": ["CookEasy", "ChefHome", "KitchenMax"]},
    }
    categories = list(category_specs)
    rows: list[dict[str, object]] = []
    for i in range(product_count):
        product_num = start_id + i
        category = str(rng.choice(categories, p=[0.22, 0.18, 0.17, 0.16, 0.12, 0.15]))
        spec = category_specs[category]
        quality_noise = float(rng.normal(0, 0.09))
        latent = float(np.clip(spec["risk"] + quality_noise, 0.08, 0.92))
        rating = float(np.clip(5.0 - latent * 2.05 + rng.normal(0, 0.06), 2.45, 5.0))
        damage_rate = float(np.clip(0.004 + latent * 0.155 + rng.normal(0, 0.004), 0.002, 0.18))
        is_fragile = bool(rng.random() < spec["fragile"])
        unit_price = float(np.round(rng.uniform(*spec["price"]) / 10) * 10)
        brand = str(rng.choice(spec["brands"]))
        rows.append(
            {
                "product_id": f"PRD{product_num:06d}",
                "product_name": f"{brand} {category} Item {product_num:04d}",
                "category": category,
                "brand": brand,
                "is_fragile": is_fragile,
                "product_rating": round(rating, 2),
                "supplier_id": f"SUP{rng.integers(1, 60):03d}",
                "supplier_name": f"{brand} Supplier",
                "supplier_contact": f"02-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}",
                "unit_price": unit_price,
                "product_latent_risk": latent,
                "damage_rate": round(damage_rate, 4),
            }
        )
    return pd.DataFrame(rows)


def make_courier_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"courier_id": "COUR01", "courier_name": "FastShip", "courier_type": "Express", "avg_delivery_days": 1.6, "coverage_region": "Nationwide", "courier_risk": 0.22},
            {"courier_id": "COUR02", "courier_name": "SafeLogistics", "courier_type": "Standard", "avg_delivery_days": 3.0, "coverage_region": "Nationwide", "courier_risk": 0.32},
            {"courier_id": "COUR03", "courier_name": "EcoDelivery", "courier_type": "Eco", "avg_delivery_days": 5.0, "coverage_region": "Bangkok Only", "courier_risk": 0.48},
            {"courier_id": "COUR04", "courier_name": "RegionalPost", "courier_type": "Regional", "avg_delivery_days": 4.2, "coverage_region": "Regional", "courier_risk": 0.42},
        ]
    )


def high_signal_source_frame(
    n_rows: int,
    order_prefix: str,
    score_prefix: str,
    return_prefix: str,
    order_start: int,
    date_start: str,
    date_end: str,
    customer_catalog: pd.DataFrame,
    product_catalog: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = customer_catalog.reset_index(drop=True)
    products = product_catalog.reset_index(drop=True)
    couriers = make_courier_catalog()
    order_dates = pd.date_range(date_start, date_end, periods=n_rows)
    customer_weights = 0.35 + customers["customer_latent_risk"].to_numpy()
    customer_weights = customer_weights / customer_weights.sum()
    product_weights = 0.45 + products["product_latent_risk"].to_numpy()
    product_weights = product_weights / product_weights.sum()
    customer_idx = rng.choice(len(customers), size=n_rows, p=customer_weights)
    product_idx = rng.choice(len(products), size=n_rows, p=product_weights)
    courier_idx = rng.choice(len(couriers), size=n_rows, p=[0.30, 0.34, 0.20, 0.16])

    rows: list[dict[str, object]] = []
    base_scores: list[float] = []
    for pos in range(n_rows):
        order_date = pd.Timestamp(order_dates[pos]).round("h")
        c = customers.iloc[int(customer_idx[pos])]
        p = products.iloc[int(product_idx[pos])]
        courier = couriers.iloc[int(courier_idx[pos])]
        cust_risk = float(c["customer_latent_risk"])
        prod_risk = float(p["product_latent_risk"])
        courier_risk = float(courier["courier_risk"])

        is_cod = rng.random() < np.clip(0.28 + cust_risk * 0.30 + prod_risk * 0.12, 0.10, 0.80)
        payment_method = "COD" if is_cod else str(rng.choice(["Credit_Card", "PromptPay", "Bank_Transfer"], p=[0.48, 0.32, 0.20]))
        high_discount = rng.random() < np.clip(0.18 + prod_risk * 0.36 + cust_risk * 0.16, 0.08, 0.72)
        promo_discount_rate = float(rng.choice([0.15, 0.20, 0.25], p=[0.45, 0.35, 0.20]) if high_discount else rng.choice([0.0, 0.05, 0.10], p=[0.48, 0.34, 0.18]))
        promo_id = "PROMO_NONE" if promo_discount_rate == 0 else f"PROMO_{int(promo_discount_rate * 100):03d}"
        promo_type = "No Promotion" if promo_discount_rate == 0 else ("Campaign" if promo_discount_rate < 0.20 else "Clearance")
        promo_name = "No Promotion" if promo_discount_rate == 0 else f"{promo_type} {int(promo_discount_rate * 100)}%"
        channel_type = str(rng.choice(["TV_Show", "Web", "Shopee", "Lazada", "CallCenter"], p=[0.25, 0.22, 0.22, 0.18, 0.13]))
        quantity = int(rng.choice([1, 2, 3, 4], p=[0.56, 0.28, 0.12, 0.04]))
        tier_discount_pct = {"Bronze": 0.03, "Silver": 0.05, "Gold": 0.08, "Platinum": 0.10}.get(str(c["membership_tier"]), 0.03)
        campaign_discount_pct = promo_discount_rate
        total_discount_pct = float(np.clip(tier_discount_pct + campaign_discount_pct, 0, 0.35))
        unit_price = float(p["unit_price"]) * float(np.clip(rng.normal(1.0, 0.04), 0.88, 1.14))
        gross = unit_price * quantity
        discount_applied = gross * total_discount_pct
        total_amount = max(gross - discount_applied, 50.0)
        delivery_expected = int({"Express": 1, "Standard": 3, "Eco": 5, "Regional": 4}[str(courier["courier_type"])])
        delay_base = courier_risk + float(p["damage_rate"]) * 5 + (0.18 if str(c["province"]) in ["Remote_Area", "Phuket", "Songkhla"] else 0.0)
        delay_days = int(max(0, rng.poisson(np.clip(delay_base, 0.05, 2.5))))
        delivery_days = int(delivery_expected + delay_days)
        expected_delivery_date = order_date + pd.Timedelta(days=delivery_expected)
        delivery_date = order_date + pd.Timedelta(days=delivery_days)
        is_repurchased = int(rng.random() < np.clip(0.12 + (1 - cust_risk) * 0.26, 0.05, 0.45))
        order_hour = int(order_date.hour)

        remote = 1 if str(c["province"]) in ["Remote_Area", "Phuket", "Songkhla"] else 0
        rating_risk = float(np.clip((4.85 - float(p["product_rating"])) / 2.35, 0, 1))
        damage_signal = float(np.clip(float(p["damage_rate"]) / 0.18, 0, 1))
        category_signal = {
            "Fashion": 0.78,
            "Electronics": 0.68,
            "Home_Appliance": 0.58,
            "Kitchen": 0.50,
            "Cosmetics": 0.40,
            "Health": 0.30,
        }.get(str(p["category"]), 0.45)
        tier_signal = {"Bronze": 0.68, "Silver": 0.48, "Gold": 0.28, "Platinum": 0.18}.get(str(c["membership_tier"]), 0.45)
        channel_signal = {"TV_Show": 0.72, "CallCenter": 0.62, "Shopee": 0.54, "Lazada": 0.45, "Web": 0.30}.get(channel_type, 0.45)
        amount_risk = min(1.0, np.log1p(total_amount) / np.log1p(15000))
        base_score = (
            0.27 * rating_risk
            + 0.21 * damage_signal
            + 0.13 * category_signal
            + 0.10 * float(is_cod)
            + 0.08 * float(high_discount)
            + 0.07 * remote
            + 0.06 * courier_risk
            + 0.05 * tier_signal
            + 0.04 * channel_signal
            + 0.04 * amount_risk
            + rng.normal(0, 0.032)
        )
        base_scores.append(float(base_score))

        rows.append(
            {
                "order_id": f"{order_prefix}{order_start + pos:07d}",
                "order_date": order_date,
                "expected_delivery_date": expected_delivery_date,
                "delivery_date": delivery_date,
                "customer_id": c["customer_id"],
                "customer_name": c["customer_name"],
                "customer_phone": c["customer_phone"],
                "gender": c["gender"],
                "age": int(c["age"]),
                "membership_tier": c["membership_tier"],
                "preferred_channel": c["preferred_channel"],
                "province": c["province"],
                "registration_date": c["registration_date"],
                "customer_age_days": int(max((order_date - pd.Timestamp(c["registration_date"])).days, 0)),
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "brand": p["brand"],
                "is_fragile": bool(p["is_fragile"]),
                "product_rating": float(p["product_rating"]),
                "supplier_id": p["supplier_id"],
                "supplier_name": p["supplier_name"],
                "supplier_contact": p["supplier_contact"],
                "courier_id": courier["courier_id"],
                "courier_name": courier["courier_name"],
                "courier_type": courier["courier_type"],
                "avg_delivery_days": float(courier["avg_delivery_days"]),
                "damage_rate": float(p["damage_rate"]),
                "coverage_region": courier["coverage_region"],
                "promo_id": promo_id,
                "promo_name": promo_name,
                "promo_type": promo_type,
                "promo_discount_rate": promo_discount_rate,
                "promo_start_date": pd.Timestamp("2025-01-01"),
                "promo_end_date": pd.Timestamp("2026-12-31"),
                "channel_type": channel_type,
                "payment_method": payment_method,
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "tier_discount_pct": tier_discount_pct,
                "campaign_discount_pct": campaign_discount_pct,
                "total_discount_pct": total_discount_pct,
                "discount_applied_amount": round(discount_applied, 2),
                "total_amount": round(total_amount, 2),
                "delivery_time_expected_days": delivery_expected,
                "delivery_days": delivery_days,
                "delay_days": delay_days,
                "is_repurchased_item": is_repurchased,
                "order_hour": order_hour,
                "days_since_last_order": -1,
                "hist_order_count": 0,
                "hist_return_rate": 0.0,
                "return_id": "NO_RETURN",
                "return_date": "Not Returned",
                "return_reason": "Not Returned",
                "return_scenario": "Not Returned",
                "item_condition": "Not Returned",
                "return_status": "Not Returned",
                "refund_amount": 0.0,
                "score_id": f"{score_prefix}{order_start + pos:07d}",
                "risk_score": 0.0,
                "risk_tier": "Low",
                "scored_at": order_date,
                "shap_values": "{}",
                TARGET: 0,
            }
        )

    out = pd.DataFrame(rows)
    risk = pd.Series(base_scores)
    cutoff = float(risk.quantile(1 - HIGH_SIGNAL_TARGET_RATE))
    labels = (risk >= cutoff).astype(int).to_numpy().copy()
    flip_count = int(round(len(labels) * HIGH_SIGNAL_LABEL_FLIP_RATE))
    if flip_count:
        flip_idx = rng.choice(len(labels), size=flip_count, replace=False)
        labels[flip_idx] = 1 - labels[flip_idx]
    out[TARGET] = labels.astype(int)

    out = recompute_history_and_return_fields(out, risk.to_numpy(), return_prefix, rng)
    return clean_source_frame(out)


def recompute_history_and_return_fields(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    return_prefix: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    out["__base_risk"] = risk_scores
    out = out.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return_counter = 1
    reasons = ["Damaged Packaging", "Size Issue", "Not as Expected", "Changed Mind", "Late Delivery"]
    conditions = ["Damaged Packaging", "Opened Box", "Wrong Size", "Good Condition", "Defective"]

    for _, group in out.groupby("customer_id", sort=False):
        idxs = group.sort_values(["order_date", "order_id"]).index.to_list()
        prior_returns = 0
        prior_orders = 0
        last_order_date: pd.Timestamp | None = None
        seen_products: set[str] = set()
        for idx in idxs:
            current_date = pd.Timestamp(out.at[idx, "order_date"])
            out.at[idx, "hist_order_count"] = prior_orders
            out.at[idx, "hist_return_rate"] = prior_returns / prior_orders if prior_orders else 0.0
            out.at[idx, "days_since_last_order"] = int((current_date - last_order_date).days) if last_order_date is not None else -1
            product_id = str(out.at[idx, "product_id"])
            out.at[idx, "is_repurchased_item"] = int(product_id in seen_products)
            seen_products.add(product_id)
            prior_orders += 1
            prior_returns += int(out.at[idx, TARGET])
            last_order_date = current_date

    for idx, row in out.iterrows():
        score = float(np.clip(row["__base_risk"], 0, 1))
        out.at[idx, "risk_score"] = round(score, 4)
        out.at[idx, "risk_tier"] = "High" if score >= 0.62 else ("Medium" if score >= 0.46 else "Low")
        out.at[idx, "shap_values"] = (
            "{"
            f"'product': {round(float(row['product_rating']) - 4.2, 3)}, "
            f"'history': {round(float(row['hist_return_rate']), 3)}, "
            f"'discount': {round(float(row['total_discount_pct']), 3)}"
            "}"
        )
        if int(row[TARGET]) == 1:
            return_date = pd.Timestamp(row["delivery_date"]) + pd.Timedelta(days=int(rng.integers(1, 12)))
            out.at[idx, "return_id"] = f"{return_prefix}{return_counter:07d}"
            out.at[idx, "return_date"] = return_date.strftime("%Y-%m-%d %H:%M:%S")
            out.at[idx, "return_reason"] = str(rng.choice(reasons, p=[0.30, 0.22, 0.20, 0.14, 0.14]))
            out.at[idx, "return_scenario"] = "Standard Return"
            out.at[idx, "item_condition"] = str(rng.choice(conditions, p=[0.26, 0.18, 0.20, 0.16, 0.20]))
            out.at[idx, "return_status"] = "Completed"
            out.at[idx, "refund_amount"] = round(float(row["total_amount"]) * float(rng.uniform(0.75, 1.0)), 2)
            return_counter += 1
        else:
            out.at[idx, "return_id"] = "NO_RETURN"
            out.at[idx, "return_date"] = "Not Returned"
            out.at[idx, "return_reason"] = "Not Returned"
            out.at[idx, "return_scenario"] = "Not Returned"
            out.at[idx, "item_condition"] = "Not Returned"
            out.at[idx, "return_status"] = "Not Returned"
            out.at[idx, "refund_amount"] = 0.0
    out = out.drop(columns=["__base_risk"])
    return out


def build_high_signal_lightgbm_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng_s1 = np.random.default_rng(421)
    rng_s2 = np.random.default_rng(422)
    s1_customers = make_customer_catalog(1_200, 1, rng_s1)
    s1_products = make_product_catalog(360, 1, rng_s1)
    s2_customers = make_customer_catalog(8_000, 10_001, rng_s2)
    s2_products = make_product_catalog(1_200, 10_001, rng_s2)

    s1 = high_signal_source_frame(
        5_000,
        "LGS1ORD",
        "LGS1SCR",
        "LGS1RET",
        1,
        "2025-01-01 00:00:00",
        "2026-01-31 23:00:00",
        s1_customers,
        s1_products,
        501,
    )
    s2 = high_signal_source_frame(
        40_000,
        "LGS2ORD",
        "LGS2SCR",
        "LGS2RET",
        1,
        "2025-01-01 00:00:00",
        "2026-01-31 23:00:00",
        s2_customers,
        s2_products,
        502,
    )
    s3 = high_signal_source_frame(
        10_000,
        "LGS3ORD",
        "LGS3SCR",
        "LGS3RET",
        50_001,
        "2026-02-01 00:00:00",
        "2026-05-08 23:00:00",
        s1_customers,
        s1_products,
        503,
    )
    s4 = high_signal_source_frame(
        10_000,
        "LGS4ORD",
        "LGS4SCR",
        "LGS4RET",
        55_001,
        "2026-02-01 00:00:00",
        "2026-05-08 23:00:00",
        s2_customers,
        s2_products,
        504,
    )
    return s1, s2, s3, s4


def build_clean_and_external_sources() -> tuple[Path, Path, Path, Path]:
    s1, s2_train, s3_external, s4_external = build_high_signal_lightgbm_sources()

    s1_path = SETC_CLEAN_ROOT / "clean_dataset_s1.csv"
    s2_path = SETC_CLEAN_ROOT / "clean_dataset_s2.csv"
    s3_path = SETD_REAL_ROOT / "S3" / "real_dataset_s3_unseen_future_from_v2.csv"
    s4_path = SETD_REAL_ROOT / "S4" / "real_dataset_s4_unseen_future_from_v2.csv"
    s3_path.parent.mkdir(parents=True, exist_ok=True)
    s4_path.parent.mkdir(parents=True, exist_ok=True)

    s1.to_csv(s1_path, index=False, encoding="utf-8-sig")
    s2_train.to_csv(s2_path, index=False, encoding="utf-8-sig")
    s3_external.to_csv(s3_path, index=False, encoding="utf-8-sig")
    s4_external.to_csv(s4_path, index=False, encoding="utf-8-sig")

    validations = [
        validation(s1, "SETC_S1_train_source", s1_path, "High-signal synthetic clean dataset generated from the project schema; semi-realistic return ratio near 33%."),
        validation(s2_train, "SETC_S2_train_source", s2_path, "High-signal synthetic clean dataset with more customers/products/orders; semi-realistic return ratio near 33%."),
        validation(s3_external, "SETD_S3_external_source", s3_path, "High-signal synthetic future test dataset for SETC/S1; semi-realistic return ratio near 33%."),
        validation(s4_external, "SETD_S4_external_source", s4_path, "High-signal synthetic future test dataset for SETC/S2; semi-realistic return ratio near 33%."),
    ]
    pd.DataFrame(validations).to_csv(LIGHTGBM_ROOT / "lightgbm_rebuild_dataset_validation.csv", index=False, encoding="utf-8-sig")
    return s1_path, s2_path, s3_path, s4_path


def lgbm_grid() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 360,
            "learning_rate": 0.045,
            "num_leaves": 19,
            "max_depth": 4,
            "min_child_samples": 90,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_lambda": 8.0,
            "reg_alpha": 0.8,
        },
        {
            "n_estimators": 480,
            "learning_rate": 0.035,
            "num_leaves": 27,
            "max_depth": 5,
            "min_child_samples": 120,
            "subsample": 0.86,
            "colsample_bytree": 0.86,
            "reg_lambda": 10.0,
            "reg_alpha": 1.0,
        },
        {
            "n_estimators": 620,
            "learning_rate": 0.028,
            "num_leaves": 35,
            "max_depth": 6,
            "min_child_samples": 150,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 12.0,
            "reg_alpha": 1.2,
        },
    ]


FEATURE_VERSION_DESCRIPTIONS: dict[int, str] = {
    1: "V1 base order-time features: customer profile, product, price, channel, payment, promotion, and logistics expectation.",
    2: "V2 customer temporal history: V1 plus point-in-time customer return/spend/order behavior and rolling windows.",
    3: "V3 product and logistics risk: V2 plus product/category/brand/courier point-in-time risk and quality/logistics scores.",
    4: "V4 business interactions: V3 plus category-payment-channel-province interactions, bands, and risk flags.",
    5: "V5 compact selected best: reduced feature set selected from V2-V4 to keep performance high while reducing noise/resource use.",
}

V1_BASE_FEATURES = [
    "gender",
    "age",
    "membership_tier",
    "preferred_channel",
    "province",
    "customer_age_days",
    "category",
    "brand",
    "is_fragile",
    "product_rating",
    "courier_type",
    "avg_delivery_days",
    "damage_rate",
    "coverage_region",
    "promo_type",
    "promo_discount_rate",
    "channel_type",
    "payment_method",
    "quantity",
    "unit_price",
    "total_discount_pct",
    "discount_applied_amount",
    "total_amount",
    "delivery_time_expected_days",
    "order_hour",
]

V2_HISTORY_FEATURES = [
    "order_month",
    "order_dayofweek",
    "is_weekend",
    "age_group",
    "customer_tenure_months",
    "total_orders_before",
    "total_returns_before",
    "customer_return_ratio",
    "days_since_last_return",
    "days_since_last_order_pti",
    "customer_avg_spend_before",
    "customer_max_discount_before",
    "customer_cod_rate_before",
    "customer_high_discount_rate_before",
    "customer_repurchase_rate_before",
    "hist_spend_sum_7d",
    "hist_order_count_7d",
    "hist_return_count_7d",
    "hist_return_rate_7d",
    "hist_spend_sum_30d",
    "hist_order_count_30d",
    "hist_return_count_30d",
    "hist_return_rate_30d",
    "hist_spend_sum_60d",
    "hist_order_count_60d",
    "hist_return_count_60d",
    "hist_return_rate_60d",
    "hist_spend_sum_90d",
    "hist_order_count_90d",
    "hist_return_count_90d",
    "hist_return_rate_90d",
    "hist_spend_sum_180d",
    "hist_order_count_180d",
    "hist_return_count_180d",
    "hist_return_rate_180d",
    "hist_spend_sum_365d",
    "hist_order_count_365d",
    "hist_return_count_365d",
    "hist_return_rate_365d",
]

V3_RISK_FEATURES = [
    "category_return_rate_pti",
    "product_return_rate_pti",
    "brand_return_rate_pti",
    "courier_return_rate_pti",
    "courier_type_return_rate_pti",
    "payment_return_rate_pti",
    "channel_return_rate_pti",
    "product_quality_score",
    "product_rating_gap",
    "damage_rating_gap",
    "fragile_damage_risk",
    "logistics_risk_score",
    "remote_logistics_risk",
    "product_price_index",
    "category_price_index",
    "category_avg_rating",
    "supplier_return_rate_pti",
]

V4_INTERACTION_FEATURES = [
    "is_cod",
    "is_high_discount",
    "low_rating_alert",
    "is_first_order",
    "discount_amount_ratio",
    "amount_per_item",
    "log_unit_price",
    "log_total_amount",
    "category_payment",
    "category_channel",
    "province_payment",
    "tier_payment",
    "category_province",
    "brand_channel",
    "province_category_return_rate_pti",
    "is_fragile_cod",
    "is_remote_cod",
    "is_fashion_cod",
    "high_discount_cod",
    "low_rating_high_discount",
    "remote_category_cod",
    "price_band",
    "discount_band",
    "rating_band",
    "order_time_bucket",
    "customer_value_band",
]

V5_COMPACT_FEATURES = [
    "age",
    "membership_tier",
    "province",
    "category",
    "brand",
    "is_fragile",
    "product_rating",
    "damage_rate",
    "courier_type",
    "promo_type",
    "promo_discount_rate",
    "channel_type",
    "payment_method",
    "quantity",
    "unit_price",
    "total_discount_pct",
    "total_amount",
    "delivery_time_expected_days",
    "customer_tenure_months",
    "total_orders_before",
    "total_returns_before",
    "customer_return_ratio",
    "days_since_last_return",
    "customer_avg_spend_before",
    "customer_cod_rate_before",
    "hist_order_count_7d",
    "hist_return_rate_7d",
    "hist_order_count_30d",
    "hist_return_rate_30d",
    "hist_order_count_90d",
    "hist_return_rate_90d",
    "hist_order_count_365d",
    "hist_return_rate_365d",
    "category_return_rate_pti",
    "product_return_rate_pti",
    "brand_return_rate_pti",
    "courier_return_rate_pti",
    "payment_return_rate_pti",
    "channel_return_rate_pti",
    "supplier_return_rate_pti",
    "product_quality_score",
    "product_rating_gap",
    "damage_rating_gap",
    "fragile_damage_risk",
    "logistics_risk_score",
    "remote_logistics_risk",
    "product_price_index",
    "is_cod",
    "is_high_discount",
    "low_rating_alert",
    "discount_amount_ratio",
    "amount_per_item",
    "log_total_amount",
    "category_payment",
    "category_channel",
    "province_payment",
    "category_province",
    "is_fragile_cod",
    "high_discount_cod",
    "low_rating_high_discount",
    "price_band",
    "discount_band",
    "rating_band",
    "order_time_bucket",
]


def existing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in df.columns]


def add_lightgbm_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["promo_type"] = out.get("promo_type", "No Promotion").fillna("No Promotion").astype(str)
    out["order_month"] = out["order_date"].dt.month.fillna(0).astype(int)
    out["order_dayofweek"] = out["order_date"].dt.dayofweek.fillna(0).astype(int)
    out["is_weekend"] = out["order_dayofweek"].isin([5, 6]).astype(int)
    out["order_hour"] = pd.to_numeric(out["order_hour"], errors="coerce").fillna(out["order_date"].dt.hour).astype(int)
    return out


def add_lightgbm_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["customer_id", "order_date", "order_id"]).copy()
    windows = [7, 30, 60, 90, 180, 365]
    for col in [
        "total_orders_before",
        "total_returns_before",
        "customer_return_ratio",
        "days_since_last_return",
        "days_since_last_order_pti",
        "customer_avg_spend_before",
        "customer_max_discount_before",
        "customer_cod_rate_before",
        "customer_high_discount_rate_before",
        "customer_repurchase_rate_before",
    ]:
        out[col] = 0.0
    out["days_since_last_return"] = -1
    out["days_since_last_order_pti"] = -1
    for days in windows:
        out[f"hist_spend_sum_{days}d"] = 0.0
        out[f"hist_order_count_{days}d"] = 0
        out[f"hist_return_count_{days}d"] = 0
        out[f"hist_return_rate_{days}d"] = 0.0

    if "registration_date" in out.columns:
        out["customer_tenure_months"] = ((out["order_date"] - out["registration_date"]).dt.days / 30).fillna(0).clip(lower=0)
    else:
        out["customer_tenure_months"] = 0.0
    out["age_group"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce").fillna(0),
        bins=[0, 20, 30, 40, 50, 120],
        labels=["<20", "20-30", "30-40", "40-50", ">50"],
        include_lowest=True,
    ).astype(str)

    for _, group in out.groupby("customer_id", sort=False):
        group = group.sort_values(["order_date", "order_id"])
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy(dtype="datetime64[ns]")
        returns = group[TARGET].astype(int).to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()
        discounts = pd.to_numeric(group["total_discount_pct"], errors="coerce").fillna(0).to_numpy()
        cod_flags = group["payment_method"].astype(str).eq("COD").astype(int).to_numpy()
        high_discount_flags = (discounts > 0.20).astype(int)
        repurchase_flags = pd.to_numeric(group["is_repurchased_item"], errors="coerce").fillna(0).astype(int).to_numpy()
        return_dates = pd.to_datetime(group.get("return_date", pd.Series(pd.NaT, index=group.index)).replace({"Not Returned": pd.NaT}), errors="coerce").to_numpy(dtype="datetime64[ns]")

        for pos, current_date in enumerate(dates):
            prior_mask = dates < current_date
            prior_count = int(prior_mask.sum())
            prior_return_count = int(returns[prior_mask].sum()) if prior_count else 0
            out.loc[idx[pos], "total_orders_before"] = prior_count
            out.loc[idx[pos], "total_returns_before"] = prior_return_count
            out.loc[idx[pos], "customer_return_ratio"] = prior_return_count / prior_count if prior_count else 0.0
            out.loc[idx[pos], "customer_avg_spend_before"] = float(amounts[prior_mask].mean()) if prior_count else 0.0
            out.loc[idx[pos], "customer_max_discount_before"] = float(discounts[prior_mask].max()) if prior_count else 0.0
            out.loc[idx[pos], "customer_cod_rate_before"] = float(cod_flags[prior_mask].mean()) if prior_count else 0.0
            out.loc[idx[pos], "customer_high_discount_rate_before"] = float(high_discount_flags[prior_mask].mean()) if prior_count else 0.0
            out.loc[idx[pos], "customer_repurchase_rate_before"] = float(repurchase_flags[prior_mask].mean()) if prior_count else 0.0
            if prior_count:
                out.loc[idx[pos], "days_since_last_order_pti"] = int((current_date - dates[prior_mask].max()) / np.timedelta64(1, "D"))

            prior_return_mask = prior_mask & (returns == 1) & ~pd.isna(return_dates)
            if prior_return_mask.any():
                last_return_date = return_dates[prior_return_mask].max()
                out.loc[idx[pos], "days_since_last_return"] = int((current_date - last_return_date) / np.timedelta64(1, "D"))
            else:
                out.loc[idx[pos], "days_since_last_return"] = -1

            for days in windows:
                start = current_date - np.timedelta64(days, "D")
                window_mask = prior_mask & (dates >= start)
                window_count = int(window_mask.sum())
                window_return_count = int(returns[window_mask].sum()) if window_count else 0
                out.loc[idx[pos], f"hist_spend_sum_{days}d"] = float(amounts[window_mask].sum()) if window_count else 0.0
                out.loc[idx[pos], f"hist_order_count_{days}d"] = window_count
                out.loc[idx[pos], f"hist_return_count_{days}d"] = window_return_count
                out.loc[idx[pos], f"hist_return_rate_{days}d"] = window_return_count / window_count if window_count else 0.0
    return out.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def add_pti_return_rate(df: pd.DataFrame, group_cols: list[str], output_col: str) -> pd.DataFrame:
    out = df.sort_values(["order_date", "order_id"]).copy()
    out[output_col] = 0.0
    totals: dict[tuple, list[int]] = {}
    global_orders = 0
    global_returns = 0
    for idx, row in out.iterrows():
        key = tuple(row[col] for col in group_cols)
        orders, returns = totals.get(key, [0, 0])
        fallback = global_returns / global_orders if global_orders else 0.0
        out.at[idx, output_col] = returns / orders if orders else fallback
        totals[key] = [orders + 1, returns + int(row[TARGET])]
        global_orders += 1
        global_returns += int(row[TARGET])
    return out


def add_lightgbm_product_logistics_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rating = pd.to_numeric(out["product_rating"], errors="coerce").fillna(4.0)
    damage = pd.to_numeric(out["damage_rate"], errors="coerce").fillna(0.02)
    amount = pd.to_numeric(out["total_amount"], errors="coerce").fillna(0)
    out["product_quality_score"] = (5.0 - rating).clip(lower=0) + damage * 8
    out["product_rating_gap"] = (4.5 - rating).clip(lower=0)
    out["damage_rating_gap"] = damage * out["product_rating_gap"]
    out["fragile_damage_risk"] = out["is_fragile"].astype(int) * damage
    out["logistics_risk_score"] = damage * out["is_fragile"].astype(int) + pd.to_numeric(out["avg_delivery_days"], errors="coerce").fillna(0) / 10
    out["remote_logistics_risk"] = out["province"].isin(["Remote_Area", "Phuket", "Songkhla"]).astype(int) * out["logistics_risk_score"]
    category_median_amount = out.groupby("category")["total_amount"].transform("median").replace(0, np.nan)
    out["product_price_index"] = (amount / category_median_amount).fillna(1.0).clip(0, 5)
    out["category_price_index"] = out.groupby("category")["product_price_index"].transform("mean").fillna(1.0)
    out["category_avg_rating"] = out.groupby("category")["product_rating"].transform("mean").fillna(rating.mean())
    for group_cols, output_col in [
        (["category"], "category_return_rate_pti"),
        (["product_id"], "product_return_rate_pti"),
        (["brand"], "brand_return_rate_pti"),
        (["courier_id"], "courier_return_rate_pti"),
        (["courier_type"], "courier_type_return_rate_pti"),
        (["payment_method"], "payment_return_rate_pti"),
        (["channel_type"], "channel_return_rate_pti"),
        (["supplier_id"], "supplier_return_rate_pti"),
    ]:
        out = add_pti_return_rate(out, group_cols, output_col)
    return out


def add_lightgbm_business_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total_discount = pd.to_numeric(out["total_discount_pct"], errors="coerce").fillna(0)
    unit_price = pd.to_numeric(out["unit_price"], errors="coerce").fillna(0)
    quantity = pd.to_numeric(out["quantity"], errors="coerce").fillna(1).replace(0, 1)
    gross = (unit_price * quantity).replace(0, np.nan)
    amount = pd.to_numeric(out["total_amount"], errors="coerce").fillna(0)
    rating = pd.to_numeric(out["product_rating"], errors="coerce").fillna(5)
    out["is_cod"] = out["payment_method"].astype(str).eq("COD").astype(int)
    out["is_high_discount"] = total_discount.gt(0.20).astype(int)
    out["low_rating_alert"] = rating.lt(4.0).astype(int)
    out["is_first_order"] = pd.to_numeric(out["total_orders_before"], errors="coerce").fillna(0).eq(0).astype(int)
    out["discount_amount_ratio"] = (pd.to_numeric(out["discount_applied_amount"], errors="coerce").fillna(0) / gross).fillna(0)
    out["amount_per_item"] = amount / quantity
    out["log_unit_price"] = np.log1p(unit_price)
    out["log_total_amount"] = np.log1p(amount)
    out["category_payment"] = out["category"].astype(str) + "_" + out["payment_method"].astype(str)
    out["category_channel"] = out["category"].astype(str) + "_" + out["channel_type"].astype(str)
    out["province_payment"] = out["province"].astype(str) + "_" + out["payment_method"].astype(str)
    out["tier_payment"] = out["membership_tier"].astype(str) + "_" + out["payment_method"].astype(str)
    out["category_province"] = out["category"].astype(str) + "_" + out["province"].astype(str)
    out["brand_channel"] = out["brand"].astype(str) + "_" + out["channel_type"].astype(str)
    out["is_fragile_cod"] = (out["is_fragile"].astype(int).eq(1) & out["payment_method"].astype(str).eq("COD")).astype(int)
    out["is_remote_cod"] = (out["province"].isin(["Remote_Area", "Phuket", "Songkhla"]) & out["payment_method"].astype(str).eq("COD")).astype(int)
    out["is_fashion_cod"] = (out["category"].astype(str).eq("Fashion") & out["payment_method"].astype(str).eq("COD")).astype(int)
    out["high_discount_cod"] = (out["is_high_discount"].eq(1) & out["is_cod"].eq(1)).astype(int)
    out["low_rating_high_discount"] = (out["low_rating_alert"].eq(1) & out["is_high_discount"].eq(1)).astype(int)
    out["remote_category_cod"] = out["is_remote_cod"].astype(str) + "_" + out["category"].astype(str)
    out["price_band"] = pd.qcut(amount.rank(method="first"), q=5, labels=["price_q1", "price_q2", "price_q3", "price_q4", "price_q5"]).astype(str)
    out["discount_band"] = pd.cut(total_discount, bins=[-0.001, 0.05, 0.10, 0.15, 0.25, 1.0], labels=["<=5%", "5-10%", "10-15%", "15-25%", ">25%"], include_lowest=True).astype(str)
    out["rating_band"] = pd.cut(rating, bins=[0, 3.6, 4.0, 4.4, 5.0], labels=["<=3.6", "3.6-4.0", "4.0-4.4", ">4.4"], include_lowest=True).astype(str)
    out["order_time_bucket"] = pd.cut(pd.to_numeric(out["order_hour"], errors="coerce").fillna(0), bins=[-1, 6, 12, 18, 24], labels=["night", "morning", "afternoon", "evening"]).astype(str)
    out["customer_value_band"] = pd.qcut(pd.to_numeric(out["customer_avg_spend_before"], errors="coerce").fillna(0).rank(method="first"), q=4, labels=["value_q1", "value_q2", "value_q3", "value_q4"]).astype(str)
    out = add_pti_return_rate(out, ["province", "category"], "province_category_return_rate_pti")
    return out


def custom_lightgbm_feature_versions(seq: Any, raw: pd.DataFrame) -> dict[int, pd.DataFrame]:
    clean = seq.clean_dataset(raw)
    v1 = add_lightgbm_v1_features(clean)
    v2 = add_lightgbm_customer_history(v1)
    v3 = add_lightgbm_product_logistics_risk(v2)
    v4 = add_lightgbm_business_interactions(v3)
    v5 = v4.copy()
    return {1: v1, 2: v2, 3: v3, 4: v4, 5: v5}


def selected_lightgbm_features(version: int, df: pd.DataFrame) -> list[str]:
    if version == 1:
        features = V1_BASE_FEATURES
    elif version == 2:
        features = V1_BASE_FEATURES + V2_HISTORY_FEATURES
    elif version == 3:
        features = V1_BASE_FEATURES + V2_HISTORY_FEATURES + V3_RISK_FEATURES
    elif version == 4:
        features = V1_BASE_FEATURES + V2_HISTORY_FEATURES + V3_RISK_FEATURES + V4_INTERACTION_FEATURES
    elif version == 5:
        features = V5_COMPACT_FEATURES
    else:
        raise ValueError(version)
    return existing_features(df, features)


def train_one_dataset(seq: Any, source_path: Path, out_root: Path, prefix: str, dataset_label: str) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    raw = prepare_raw_dataset(pd.read_csv(source_path))
    versions = custom_lightgbm_feature_versions(seq, raw)
    rows: list[dict[str, object]] = []

    for version, df_feat in versions.items():
        version_root = out_root / f"V{version}"
        feature_dir = version_root / "features"
        model_dir = version_root / "models"
        report_dir = version_root / "reports"
        feature_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        features = selected_lightgbm_features(version, df_feat)
        x = df_feat[features].copy()
        y = df_feat[TARGET].astype(int).copy()

        train_val_idx, holdout_idx = train_test_split(
            df_feat.index.to_numpy(),
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        fit_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.20,
            stratify=y.loc[train_val_idx],
            random_state=RANDOM_STATE,
        )

        x_fit, y_fit = x.loc[fit_idx], y.loc[fit_idx]
        x_val, y_val = x.loc[val_idx], y.loc[val_idx]
        x_hold, y_hold = x.loc[holdout_idx], y.loc[holdout_idx]
        scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

        best: dict[str, Any] | None = None
        for grid_id, params in enumerate(lgbm_grid(), start=1):
            full_params = {
                **params,
                "objective": "binary",
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "scale_pos_weight": scale_pos_weight,
            }
            model = Pipeline(
                steps=[
                    ("preprocessor", seq.build_preprocessor(x_fit)),
                    ("model", LGBMClassifier(**full_params)),
                ]
            )
            model.fit(x_fit, y_fit)
            val_proba = model.predict_proba(x_val)[:, 1]
            threshold, val_metrics = threshold_search(y_val.to_numpy(), val_proba)
            score = (
                float(val_metrics["accuracy"]) * 0.62
                + float(val_metrics["f1"]) * 0.22
                + float(val_metrics["auc"]) * 0.10
                + float(val_metrics["recall"]) * 0.06
            )
            candidate = {
                "grid_id": grid_id,
                "params": full_params,
                "threshold": threshold,
                "val_metrics": val_metrics,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = candidate

        assert best is not None
        final_model = Pipeline(
            steps=[
                ("preprocessor", seq.build_preprocessor(x.loc[train_val_idx])),
                ("model", LGBMClassifier(**best["params"])),
            ]
        )
        final_model.fit(x.loc[train_val_idx], y.loc[train_val_idx])
        hold_proba = final_model.predict_proba(x_hold)[:, 1]
        hold_metrics = evaluate_predictions(y_hold.to_numpy(), hold_proba, best["threshold"])

        model_path = model_dir / f"model_{prefix}_v{version}_lightgbm.pkl"
        joblib.dump(final_model, model_path)
        pd.DataFrame({"feature": features}).to_csv(feature_dir / f"used_features_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "order_id": df_feat["order_id"],
                "customer_id": df_feat["customer_id"],
                "order_date": df_feat["order_date"].astype(str),
                **{feature: df_feat[feature] for feature in features},
                TARGET: df_feat[TARGET],
            }
        ).to_csv(feature_dir / f"df_featured_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")
        joblib.dump(
            {
                "feature_names": features,
                "threshold": best["threshold"],
                "fit_indices": fit_idx,
                "validation_indices": val_idx,
                "holdout_indices": holdout_idx,
                "split_strategy": "64% fit / 16% validation / 20% holdout, stratified by is_returned",
            },
            feature_dir / f"train_validation_holdout_sets_{prefix}_v{version}.pkl",
        )

        pred = (hold_proba >= best["threshold"]).astype(int)
        pd.DataFrame(
            {
                "order_id": df_feat.loc[holdout_idx, "order_id"].astype(str).to_numpy(),
                "customer_id": df_feat.loc[holdout_idx, "customer_id"].astype(str).to_numpy(),
                "actual_is_returned": y_hold.to_numpy(),
                "predict_probability_return": hold_proba,
                "predicted_is_returned": pred,
                "threshold": best["threshold"],
                "correct_prediction": (pred == y_hold.to_numpy()).astype(int),
            }
        ).to_csv(report_dir / f"holdout_predictions_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")

        row = {
            "version": f"V{version}",
            "dataset": dataset_label,
            "model": "LightGBM",
            "feature_strategy": FEATURE_VERSION_DESCRIPTIONS[version],
            "evaluation_type": "fit_validation_holdout_tuning",
            "rows": int(len(df_feat)),
            "fit_rows": int(len(fit_idx)),
            "validation_rows": int(len(val_idx)),
            "holdout_rows": int(len(holdout_idx)),
            "feature_count": int(len(features)),
            "selected_grid_id": int(best["grid_id"]),
            "threshold": float(best["threshold"]),
            "validation_accuracy": float(best["val_metrics"]["accuracy"]),
            "validation_recall": float(best["val_metrics"]["recall"]),
            "validation_precision": float(best["val_metrics"]["precision"]),
            "validation_f1": float(best["val_metrics"]["f1"]),
            "validation_auc": float(best["val_metrics"]["auc"]),
            "holdout_accuracy": float(hold_metrics["accuracy"]),
            "holdout_recall": float(hold_metrics["recall"]),
            "holdout_precision": float(hold_metrics["precision"]),
            "holdout_f1": float(hold_metrics["f1"]),
            "holdout_auc": float(hold_metrics["auc"]),
            "holdout_cost": int(hold_metrics["cost"]),
            "tn": int(hold_metrics["tn"]),
            "fp": int(hold_metrics["fp"]),
            "fn": int(hold_metrics["fn"]),
            "tp": int(hold_metrics["tp"]),
            "source_dataset": str(source_path.relative_to(ROOT)),
            "model_path": str(model_path.relative_to(ROOT)),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(report_dir / f"metrics_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")
        (model_dir / f"model_{prefix}_v{version}_metadata.json").write_text(
            json.dumps({**row, "lightgbm_params": best["params"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (version_root / "README.md").write_text(
            f"""# {dataset_label} {row['version']}

- Source dataset: `{source_path.relative_to(ROOT)}`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `{FEATURE_VERSION_DESCRIPTIONS[version]}`
- Feature count: `{len(features)}`
- Holdout Accuracy: `{float(hold_metrics['accuracy']) * 100:.2f}%`
- Holdout Recall: `{float(hold_metrics['recall']) * 100:.2f}%`
- Holdout F1: `{float(hold_metrics['f1']) * 100:.2f}%`
- Holdout AUC: `{float(hold_metrics['auc']) * 100:.2f}%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
""",
            encoding="utf-8",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / f"{prefix}_v1_to_v5_holdout_summary.csv", index=False, encoding="utf-8-sig")
    (out_root / f"{prefix}_v1_to_v5_holdout_summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    draw_accuracy_chart(
        summary.rename(columns={"holdout_accuracy": "accuracy"}),
        out_root / "images" / f"{prefix}_holdout_accuracy_v1_to_v5.png",
        f"{dataset_label} Holdout Accuracy V1-V5",
        value_col="accuracy",
    )
    return summary


def build_external_versions_with_history_context(
    seq: Any,
    history_path: Path,
    external_path: Path,
) -> dict[int, pd.DataFrame]:
    history = pd.read_csv(history_path)
    external = pd.read_csv(external_path)
    history["__eval_split"] = "history"
    external["__eval_split"] = "external"

    # Build point-in-time features on historical context + external period, then
    # evaluate only external rows. This mirrors a feature-store workflow better
    # than computing external rows in isolation.
    combined = pd.concat([history, external], ignore_index=True, sort=False)
    versions = custom_lightgbm_feature_versions(seq, prepare_raw_dataset(combined))
    external_versions: dict[int, pd.DataFrame] = {}
    external_order_ids = set(external["order_id"].astype(str))
    for version, df_feat in versions.items():
        if "__eval_split" in df_feat.columns:
            external_rows = df_feat[df_feat["__eval_split"].astype(str).eq("external")].copy()
        else:
            external_rows = df_feat[df_feat["order_id"].astype(str).isin(external_order_ids)].copy()
        external_versions[version] = external_rows.reset_index(drop=True)
    return external_versions


def evaluate_external(
    seq: Any,
    model_root: Path,
    prefix: str,
    dataset_label: str,
    history_path: Path,
    external_path: Path,
    out_root: Path,
) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    versions = build_external_versions_with_history_context(seq, history_path, external_path)
    rows: list[dict[str, object]] = []

    for version, df_feat in versions.items():
        report_dir = out_root / f"V{version}" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(model_root / f"V{version}" / "features" / f"used_features_{prefix}_v{version}.csv")["feature"].astype(str).tolist()
        model = joblib.load(model_root / f"V{version}" / "models" / f"model_{prefix}_v{version}_lightgbm.pkl")
        holdout_row = pd.read_csv(model_root / f"V{version}" / "reports" / f"metrics_{prefix}_v{version}.csv").iloc[0].to_dict()
        threshold = float(holdout_row["threshold"])
        proba = model.predict_proba(df_feat[features])[:, 1]
        y_true = df_feat[TARGET].astype(int).to_numpy()
        metrics = evaluate_predictions(y_true, proba, threshold)
        pred = (proba >= threshold).astype(int)
        row = {
            "version": f"V{version}",
            "train_dataset": dataset_label,
            "test_dataset": external_path.parent.name,
            "rows": int(len(df_feat)),
            "feature_count": int(len(features)),
            "threshold": threshold,
            "holdout_accuracy": float(holdout_row["holdout_accuracy"]),
            "holdout_recall": float(holdout_row["holdout_recall"]),
            "holdout_f1": float(holdout_row["holdout_f1"]),
            "holdout_auc": float(holdout_row["holdout_auc"]),
            "external_accuracy": float(metrics["accuracy"]),
            "external_recall": float(metrics["recall"]),
            "external_precision": float(metrics["precision"]),
            "external_f1": float(metrics["f1"]),
            "external_auc": float(metrics["auc"]),
            "external_cost": int(metrics["cost"]),
            "model_path": str((model_root / f"V{version}" / "models" / f"model_{prefix}_v{version}_lightgbm.pkl").relative_to(ROOT)),
            "history_context_path": str(history_path.relative_to(ROOT)),
            "external_data_path": str(external_path.relative_to(ROOT)),
            "feature_context_mode": "history_plus_external_chronological_point_in_time",
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(report_dir / f"external_metrics_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "order_id": df_feat["order_id"].astype(str),
                "customer_id": df_feat["customer_id"].astype(str),
                "actual_is_returned": y_true,
                "predict_probability_return": proba,
                "predicted_is_returned": pred,
                "threshold": threshold,
                "correct_prediction": (pred == y_true).astype(int),
            }
        ).to_csv(report_dir / f"external_predictions_{prefix}_v{version}.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / f"{prefix}_v1_to_v5_external_summary.csv", index=False, encoding="utf-8-sig")
    (out_root / f"{prefix}_v1_to_v5_external_summary.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    draw_accuracy_chart(
        summary.rename(columns={"external_accuracy": "accuracy"}),
        out_root / "images" / f"{prefix}_external_accuracy_v1_to_v5.png",
        f"{dataset_label} External Accuracy V1-V5",
        value_col="accuracy",
    )
    return summary


def write_readmes(s1_summary: pd.DataFrame, s2_summary: pd.DataFrame, s3_summary: pd.DataFrame, s4_summary: pd.DataFrame) -> None:
    def best(df: pd.DataFrame, column: str) -> str:
        tie_breakers = [column]
        if column.startswith("holdout"):
            tie_breakers += ["holdout_f1", "holdout_auc"]
        elif column.startswith("external"):
            tie_breakers += ["external_f1", "external_auc"]
        available = [col for col in tie_breakers if col in df.columns]
        row = df.sort_values(available, ascending=[False] * len(available)).iloc[0]
        return f"{row['version']} = {float(row[column]) * 100:.2f}%"

    (SETC_ROOT / "README.md").write_text(
        f"""# LightGBM SETC Rebuild

SETC is the train/validation/holdout side of the rebuilt LightGBM workflow. The latest dataset is high-signal synthetic data with a semi-realistic return ratio near 33%.

## Sources

- S1: generated clean high-signal dataset -> 5,000 rows
- S2: generated clean high-signal dataset -> 40,000 rows
- SETD keeps separate generated future datasets for external testing.

## Best Holdout Accuracy

- S1 best: `{best(s1_summary, 'holdout_accuracy')}`
- S2 best: `{best(s2_summary, 'holdout_accuracy')}`

## Evaluation Rule

Each version uses `64% fit / 16% validation / 20% holdout`. Validation is used for parameter/threshold selection. Holdout is used for reporting.

## New Data Policy

When testing SETD, feature engineering is rebuilt with SETC as historical context first, then only SETD rows are evaluated. This is closer to production because new orders need customer/category/product history from the feature store.
""",
        encoding="utf-8",
    )
    (SETD_ROOT / "README.md").write_text(
        f"""# LightGBM SETD Rebuild

SETD is the external test side of the rebuilt LightGBM workflow.

## External Dataset

Both S3 and S4 use generated future high-signal datasets with the same row count as before.

- S3 tests SETC/S1 models.
- S4 tests SETC/S2 models.
- S3 feature engineering uses SETC/S1 as history context.
- S4 feature engineering uses SETC/S2 as history context.

This avoids the old S3 issue where test data was aligned too closely with train data and produced unrealistic 92% Accuracy. Current S3/S4 are still synthetic benchmarks, but they are separated as external full-dataset tests.

## Best External Accuracy

- S3 vs S1 best: `{best(s3_summary, 'external_accuracy')}`
- S4 vs S2 best: `{best(s4_summary, 'external_accuracy')}`
""",
        encoding="utf-8",
    )
    (LIGHTGBM_ROOT / "README.md").write_text(
        f"""# LightGBM Rebuild Summary

This folder was rebuilt from scratch with a semi-realistic high-signal synthetic dataset. Row counts are preserved, but the generated data now has more varied orders/products/customers and a return ratio near 33%.

## Main Results

| Area | Best Result |
| --- | --- |
| SETC/S1 holdout | {best(s1_summary, 'holdout_accuracy')} |
| SETC/S2 holdout | {best(s2_summary, 'holdout_accuracy')} |
| SETD/S3 external vs S1 | {best(s3_summary, 'external_accuracy')} |
| SETD/S4 external vs S2 | {best(s4_summary, 'external_accuracy')} |

## Important Interpretation

Use holdout and SETD external values as the current high-signal benchmark results. Do not describe these numbers as real production accuracy because the dataset is still generated/synthetic.

SETD external feature engineering is rebuilt with SETC history context before prediction. This means S3/S4 are no longer tested as isolated CSV rows with no prior customer/product/category history.
""",
        encoding="utf-8",
    )


def main() -> None:
    reset_setc_setd()
    s1_path, s2_path, s3_path, s4_path = build_clean_and_external_sources()
    seq = load_seq_module()

    s1_summary = train_one_dataset(
        seq,
        s1_path,
        SETC_CLEAN_ROOT / "S1",
        "lgbm_s1",
        "LightGBM_SETC_S1_REBUILD",
    )
    s2_summary = train_one_dataset(
        seq,
        s2_path,
        SETC_CLEAN_ROOT / "S2",
        "lgbm_s2",
        "LightGBM_SETC_S2_REBUILD",
    )
    s3_summary = evaluate_external(
        seq,
        SETC_CLEAN_ROOT / "S1",
        "lgbm_s1",
        "LightGBM_SETC_S1_REBUILD",
        s1_path,
        s3_path,
        SETD_REAL_ROOT / "S3",
    )
    s4_summary = evaluate_external(
        seq,
        SETC_CLEAN_ROOT / "S2",
        "lgbm_s2",
        "LightGBM_SETC_S2_REBUILD",
        s2_path,
        s4_path,
        SETD_REAL_ROOT / "S4",
    )
    write_readmes(s1_summary, s2_summary, s3_summary, s4_summary)

    print("\nSETC/S1 holdout")
    print(s1_summary[["version", "feature_count", "holdout_accuracy", "holdout_recall", "holdout_f1", "holdout_auc"]].to_string(index=False))
    print("\nSETC/S2 holdout")
    print(s2_summary[["version", "feature_count", "holdout_accuracy", "holdout_recall", "holdout_f1", "holdout_auc"]].to_string(index=False))
    print("\nSETD/S3 external vs S1")
    print(s3_summary[["version", "feature_count", "external_accuracy", "external_recall", "external_f1", "external_auc"]].to_string(index=False))
    print("\nSETD/S4 external vs S2")
    print(s4_summary[["version", "feature_count", "external_accuracy", "external_recall", "external_f1", "external_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
