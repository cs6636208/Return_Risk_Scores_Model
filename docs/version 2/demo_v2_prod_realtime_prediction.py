from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.production_v2.feature_builder import FeatureBuilderV2Production  # noqa: E402
from src.production_v2.predictor import V2ProdPredictor  # noqa: E402


DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
ARTIFACT_PATH = ROOT / "models" / "v2_prod_artifact.pkl"
TRAIN_TEST_PATH = ROOT / "data" / "features" / "train_test_sets_v2_prod.pkl"
REPORT_OUT = ROOT / "reports" / "model_evaluation_v2_prod" / "v2_prod_realtime_demo_prediction.json"
DOCS_OUT = ROOT / "docs" / "version 2" / "production" / "v2_prod_realtime_demo_prediction.json"


def first_existing(cols: list[str], frame: pd.DataFrame) -> list[str]:
    return [col for col in cols if col in frame.columns]


def main() -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["order_date", "registration_date"])
    split = joblib.load(TRAIN_TEST_PATH)
    sample_index = int(split["raw_test_index"][0])
    current = df.loc[sample_index].copy()

    customer_cols = first_existing(
        ["customer_id", "customer_name", "customer_phone", "gender", "age", "membership_tier", "preferred_channel", "province", "registration_date", "customer_age_days"],
        df,
    )
    product_cols = first_existing(["product_id", "product_name", "category", "brand", "supplier_id", "unit_price", "is_fragile", "product_rating"], df)
    courier_cols = first_existing(["courier_id", "courier_name", "courier_type", "avg_delivery_days", "damage_rate", "coverage_region"], df)
    promo_cols = first_existing(["promo_id", "promo_name", "promo_type", "promo_discount_rate", "discount_rate", "start_date", "end_date"], df)
    order_cols = first_existing(["order_id", "customer_id", "product_id", "order_date", "is_returned"], df)

    customers = df[customer_cols].drop_duplicates("customer_id")
    products = df[product_cols].drop_duplicates("product_id")
    couriers = df[courier_cols].drop_duplicates("courier_id")
    promotions = df[promo_cols].drop_duplicates("promo_id")
    orders = df[order_cols].copy()

    order_payload = {
        "order_id": current.get("order_id"),
        "customer_id": current["customer_id"],
        "product_id": current.get("product_id"),
        "courier_id": current.get("courier_id"),
        "promo_id": current.get("promo_id"),
        "order_date": current["order_date"].isoformat(),
        "channel_type": current.get("channel_type"),
        "payment_method": current.get("payment_method"),
        "quantity": int(current.get("quantity", 1)),
        "unit_price": float(current.get("unit_price", 0)),
        "tier_discount_pct": float(current.get("tier_discount_pct", 0)),
        "campaign_discount_pct": float(current.get("campaign_discount_pct", 0)),
        "total_discount_pct": float(current.get("total_discount_pct", 0)),
        "discount_applied_amount": float(current.get("discount_applied_amount", 0)),
        "total_amount": float(current.get("total_amount", 0)),
        "delivery_time_expected_days": int(current.get("delivery_time_expected_days", 3)),
        "order_hour": int(current.get("order_hour", pd.to_datetime(current["order_date"]).hour)),
    }

    builder = FeatureBuilderV2Production()
    predictor = V2ProdPredictor.from_artifact(ARTIFACT_PATH)
    realtime_snapshot = builder.build_from_dataframes(order_payload, customers, products, couriers, promotions, orders)
    realtime_prediction = predictor.predict_snapshot(realtime_snapshot)

    batch_snapshot = current.to_dict()
    batch_probability = float(predictor.model.predict_proba(predictor.preprocessor.transform(pd.DataFrame([batch_snapshot])))[:, 1][0])

    output = {
        "sample_index": sample_index,
        "order_id": order_payload["order_id"],
        "customer_id": order_payload["customer_id"],
        "actual_is_returned": int(current["is_returned"]),
        "realtime_probability": realtime_prediction["risk_probability"],
        "batch_probability_same_order": batch_probability,
        "absolute_probability_diff": abs(realtime_prediction["risk_probability"] - batch_probability),
        "risk_label": realtime_prediction["risk_label"],
        "threshold": realtime_prediction["threshold"],
        "feature_snapshot": realtime_prediction["feature_snapshot"],
    }

    REPORT_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    DOCS_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(REPORT_OUT)
    print(DOCS_OUT)
    print(json.dumps({k: output[k] for k in ["order_id", "customer_id", "actual_is_returned", "realtime_probability", "batch_probability_same_order", "absolute_probability_diff", "risk_label"]}, indent=2))


if __name__ == "__main__":
    main()
