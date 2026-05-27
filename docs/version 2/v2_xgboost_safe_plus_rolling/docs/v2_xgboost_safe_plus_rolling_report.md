# V2 XGBoost Safe Plus Rolling

## Summary

โฟลเดอร์นี้เก็บเฉพาะ candidate `v2_xgboost_safe_plus_rolling` จากกระบวนการ V2 Optimized

- Candidate: `v2_xgboost_safe_plus_rolling`
- Model: `XGBClassifier`
- Order-time Safe: `Yes`
- Feature count: `60`
- Threshold: `0.49`
- Performance Rating: `B`

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 71.07% |
| Recall | 56.88% |
| Precision | 50.20% |
| F1 | 53.33% |
| AUC | 71.47% |
| Cost Matrix | 20,250 |

## Confusion Matrix

|  | Pred no_return | Pred return |
| --- | ---: | ---: |
| Actual no_return | 409 | 123 |
| Actual return | 94 | 124 |

## Feature List

- `gender`
- `age`
- `membership_tier`
- `province`
- `customer_age_days`
- `category`
- `brand`
- `product_rating`
- `courier_name`
- `courier_type`
- `avg_delivery_days`
- `damage_rate`
- `promo_name`
- `promo_type`
- `promo_discount_rate`
- `channel_type`
- `payment_method`
- `quantity`
- `unit_price`
- `tier_discount_pct`
- `campaign_discount_pct`
- `total_discount_pct`
- `discount_applied_amount`
- `total_amount`
- `delivery_time_expected_days`
- `is_repurchased_item`
- `order_hour`
- `days_since_last_order`
- `hist_order_count`
- `hist_return_rate`
- `customer_tenure_months`
- `order_month`
- `order_dayofweek`
- `is_weekend`
- `age_group`
- `logistics_risk`
- `hist_order_count_30d`
- `hist_return_count_30d`
- `hist_return_rate_30d`
- `hist_spend_sum_30d`
- `hist_order_count_60d`
- `hist_return_count_60d`
- `hist_return_rate_60d`
- `hist_spend_sum_60d`
- `hist_order_count_90d`
- `hist_return_count_90d`
- `hist_return_rate_90d`
- `hist_spend_sum_90d`
- `hist_order_count_180d`
- `hist_return_count_180d`
- `hist_return_rate_180d`
- `hist_spend_sum_180d`
- `hist_order_count_365d`
- `hist_return_count_365d`
- `hist_return_rate_365d`
- `hist_spend_sum_365d`
- `discount_amount_ratio`
- `category_payment`
- `category_channel`
- `province_payment`

## Process

1. ใช้ `data/processed/clean_dataset.csv` เป็น input หลัก
2. สร้าง V2 order-time-safe features โดยตัด `delivery_days` และ `delay_days` ออก
3. เพิ่ม rolling customer history เช่น 30d, 60d, 90d, 180d, 365d
4. เพิ่ม interaction features เช่น discount ratio, category-payment, category-channel และ province-payment
5. ใช้ target encoding สำหรับ categorical features
6. แบ่ง train/validation/test ด้วย `random_state=42`
7. Train เฉพาะ `XGBClassifier` candidate นี้ และ export model, features, metrics, predictions, chart และ report เข้ามาในโฟลเดอร์นี้

## Production Note

ตัวนี้เหมาะกว่า `v2_random_forest_full_38` สำหรับ real-time หรือ order-time scoring เพราะ feature set ไม่ใช้ `delivery_days` และ `delay_days` ซึ่งเป็นข้อมูลหลังส่งสินค้า
