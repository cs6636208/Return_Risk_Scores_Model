# V2 NEW EDA Insight Summary

Source: `data/processed/clean_dataset_v2.csv`

## Dataset Profile

| metric | value |
| --- | --- |
| rows | 50000 |
| columns | 66 |
| missing_cells | 35381 |
| unique_orders | 50000 |
| unique_customers | 5962 |
| return_rate_pct | 29.24 |
| min_order_date | 2025-01-01 00:00:00 |
| max_order_date | 2026-05-08 23:00:00 |

## Insight to Feature Mapping

| insight_area | top_segment | segment_return_rate_pct | lift_vs_average | feature_action |
| --- | --- | --- | --- | --- |
| category | Electronics | 31.40 | 1.074 | Keep category/brand and create category_payment/category_channel interactions. |
| payment_method | COD | 32.21 | 1.101 | Create is_cod and payment interaction features. |
| channel_type | TikTok | 29.70 | 1.016 | Keep channel_type and create category_channel interaction. |
| province | Remote_Area | 37.14 | 1.270 | Keep province and create province_payment interaction. |
| discount_band | >25% | 36.77 | 1.258 | Create is_high_discount and discount_amount_ratio. |
| rating_band | <=3.8 | 34.22 | 1.170 | Create low_rating_alert and keep product_rating. |
| history_band | 75-100% | 49.37 | 1.688 | Create point-in-time rolling history windows 30/60/90/180/365 days. |
| membership_tier | Platinum | 37.90 | 1.296 | Keep membership_tier and customer_tenure_months. |

## Modeling Decision

- Use order-time-safe features only.
- Exclude post-event/leakage fields such as return/refund/risk score fields and actual delivery result fields.
- Use customer historical behavior and rolling windows as the main V2 signal.
- Use XGBoost as the V2 model because it handles non-linear interactions and mixed tabular patterns well.