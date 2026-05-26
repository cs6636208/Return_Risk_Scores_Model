# Data Dictionary V4 Generated

| column | dtype | missing_count | unique_count | description |
| --- | --- | --- | --- | --- |
| order_id | string | 0 | 9700 | Unique order identifier |
| order_date | datetime64[us] | 0 | 4262 | Project source field / engineered business context |
| expected_delivery_date | datetime64[us] | 0 | 4232 | Project source field / engineered business context |
| delivery_date | datetime64[us] | 0 | 4288 | Project source field / engineered business context |
| customer_id | string | 0 | 500 | Customer identifier |
| customer_name | string | 0 | 98 | Project source field / engineered business context |
| customer_phone | string | 0 | 500 | Project source field / engineered business context |
| gender | string | 0 | 3 | Project source field / engineered business context |
| age | float64 | 0 | 4637 | Project source field / engineered business context |
| membership_tier | string | 0 | 4 | Project source field / engineered business context |
| preferred_channel | string | 0 | 4 | Project source field / engineered business context |
| province | string | 0 | 8 | Project source field / engineered business context |
| registration_date | datetime64[us] | 0 | 442 | Project source field / engineered business context |
| customer_age_days | int64 | 0 | 442 | Project source field / engineered business context |
| product_id | string | 0 | 20 | Project source field / engineered business context |
| product_name | string | 0 | 20 | Project source field / engineered business context |
| category | string | 0 | 5 | Project source field / engineered business context |
| brand | string | 0 | 20 | Project source field / engineered business context |
| is_fragile | bool | 0 | 2 | Project source field / engineered business context |
| product_rating | float64 | 0 | 4670 | Project source field / engineered business context |
| supplier_id | string | 0 | 9 | Project source field / engineered business context |
| supplier_name | string | 0 | 9 | Project source field / engineered business context |
| supplier_contact | string | 0 | 9 | Project source field / engineered business context |
| courier_id | string | 0 | 3 | Project source field / engineered business context |
| courier_name | string | 0 | 3 | Project source field / engineered business context |
| courier_type | string | 0 | 3 | Project source field / engineered business context |
| avg_delivery_days | float64 | 0 | 3 | Project source field / engineered business context |
| damage_rate | float64 | 0 | 4676 | Project source field / engineered business context |
| coverage_region | string | 0 | 2 | Project source field / engineered business context |
| promo_id | string | 0 | 4 | Project source field / engineered business context |
| promo_name | string | 0 | 4 | Project source field / engineered business context |
| promo_type | string | 0 | 2 | Project source field / engineered business context |
| promo_discount_rate | float64 | 0 | 2540 | Project source field / engineered business context |
| promo_start_date | datetime64[us] | 0 | 4 | Project source field / engineered business context |
| promo_end_date | datetime64[us] | 0 | 4 | Project source field / engineered business context |
| channel_type | string | 0 | 4 | Project source field / engineered business context |
| payment_method | string | 0 | 3 | Project source field / engineered business context |
| quantity | int64 | 0 | 2 | Project source field / engineered business context |
| unit_price | float64 | 0 | 4717 | Project source field / engineered business context |
| tier_discount_pct | float64 | 0 | 4 | Project source field / engineered business context |
| campaign_discount_pct | float64 | 0 | 4 | Project source field / engineered business context |
| total_discount_pct | float64 | 0 | 9 | Total discount percentage |
| discount_applied_amount | float64 | 0 | 4818 | Project source field / engineered business context |
| total_amount | float64 | 0 | 4871 | Order amount after discount |
| delivery_time_expected_days | int64 | 0 | 3 | Project source field / engineered business context |
| delivery_days | int64 | 0 | 6 | Project source field / engineered business context |
| delay_days | int64 | 0 | 8 | Project source field / engineered business context |
| is_repurchased_item | int64 | 0 | 2 | Project source field / engineered business context |
| order_hour | int64 | 0 | 24 | Project source field / engineered business context |
| days_since_last_order | int64 | 0 | 235 | Project source field / engineered business context |
| hist_order_count | int64 | 0 | 24 | Project source field / engineered business context |
| hist_return_rate | float64 | 0 | 59 | Historical customer return rate available before the order |
| return_id | string | 0 | 1456 | Project source field / engineered business context |
| return_date | datetime64[ns] | 8245 | 1339 | Project source field / engineered business context |
| return_reason | string | 0 | 6 | Project source field / engineered business context |
| return_scenario | string | 0 | 3 | Project source field / engineered business context |
| item_condition | string | 0 | 6 | Project source field / engineered business context |
| return_status | string | 0 | 3 | Project source field / engineered business context |
| refund_amount | float64 | 0 | 149 | Project source field / engineered business context |
| score_id | string | 0 | 5000 | Project source field / engineered business context |
| risk_score | float64 | 0 | 29 | Project source field / engineered business context |
| risk_tier | string | 0 | 4 | Project source field / engineered business context |
| scored_at | datetime64[us] | 0 | 3901 | Project source field / engineered business context |
| shap_values | string | 0 | 3 | Project source field / engineered business context |
| is_returned | int64 | 0 | 2 | Target: 1 if order was returned, otherwise 0 |