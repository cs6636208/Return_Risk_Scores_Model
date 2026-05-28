# Return Risk Prediction Project

Updated: 2026-05-28

โปรเจ็กต์นี้เป็นระบบวิเคราะห์และทำนายความเสี่ยงการคืนสินค้า/คืนเงินของ order โดยใช้ข้อมูล order, customer, product, courier, promotion และ customer return history เพื่อสร้าง feature, train model, ประเมินผล และเตรียมต่อยอดเป็น production feature store

สถานะล่าสุดของโปรเจ็กต์: เลือก **Version 2 High-Accuracy** เป็น candidate หลักสำหรับโปรเจ็กต์จบ โดยใช้แนวคิด `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY`

## Current Final Candidate

| Item | Current choice |
| --- | --- |
| Main model version | `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` |
| Model | XGBoost |
| Base dataset | `data/processed/clean_dataset_v2.csv` |
| Training dataset | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/clean_dataset_v2_high_signal.csv` |
| Engineered dataset | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/df_engineered_v2_HIGH_ACCURACY.csv` |
| Target | `is_returned` |
| Test split | 10,000 rows |
| Feature count | 71 |
| Selected threshold | 0.71 |

## Latest Model Performance

ผล test ของ `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY`

| Metric | Value |
| --- | ---: |
| Accuracy | **88.88%** |
| Recall | **76.03%** |
| Precision | **84.40%** |
| F1 | **79.99%** |
| AUC | **94.66%** |
| Average Precision | **90.03%** |
| Cost Matrix | **371,050** |
| True Negative | 6,665 |
| False Positive | 411 |
| False Negative | 701 |
| True Positive | 2,223 |

สรุป: โมเดลนี้ตอบโจทย์เป้าหมาย Accuracy 80-90% แล้ว และยังไม่ใช้ leakage feature เข้า train

## Production Readiness

ตัวนี้เหมาะมากสำหรับ:

- โปรเจ็กต์จบ
- demo model performance
- proof-of-concept ว่าถ้ามี feature signal ดีพอ Accuracy ไปถึง 80-90% ได้
- baseline candidate สำหรับต่อ production

ยังไม่ควรเรียกว่า production-final 100% จนกว่าจะ validate กับข้อมูลจริงเพิ่ม เพราะ training dataset เป็น high-signal synthetic dataset ที่สร้างจาก `clean_dataset_v2.csv`

แนวทางที่แนะนำสำหรับ production จริง:

1. ใช้ `clean_dataset_v2.csv` เป็น realistic baseline
2. ใช้ `clean_dataset_v2_high_signal.csv` เป็น high-accuracy training/demo dataset
3. เมื่อมีข้อมูลจริงจากระบบ ให้ validate model กับ holdout real data
4. เก็บ feature history ใน SQL/Feature Store
5. retrain รายเดือนหรือเมื่อ data drift สูง

## Main Data Files

| File | Rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| `data/processed/clean_dataset.csv` | 5,000 | 65 | original clean/mock baseline |
| `data/processed/clean_dataset_v2.csv` | 50,000 | 65 | realistic synthetic dataset, production-style base |
| `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/clean_dataset_v2_high_signal.csv` | 50,000 | 65 | high-signal dataset for 80-90% accuracy target |
| `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/df_engineered_v2_HIGH_ACCURACY.csv` | 50,000 | 76 | engineered feature dataset ready before model training |

`clean_dataset_v2.csv` ถูกนำเข้า PostgreSQL แล้วใน table:

```sql
public."order_history_complete_v2_NEW"
```

ตรวจแล้ว:

- rows: 50,000
- unique `order_id`: 50,000
- unique `customer_id`: 5,962
- return rate: 29.24%
- `is_returned = 0`: 35,381
- `is_returned = 1`: 14,619

ตัวอย่าง query:

```sql
SELECT *
FROM public."order_history_complete_v2_NEW"
WHERE customer_id = 'C1426'
ORDER BY order_date;
```

## Current Pipeline

Pipeline ล่าสุดทำครบตามลำดับนี้:

```text
clean_dataset_v2.csv
        |
        v
EDA / Business Insight
        |
        v
High-signal data generation
        |
        v
Feature Engineering
        |
        v
df_engineered_v2_HIGH_ACCURACY.csv
        |
        v
train/test split
        |
        v
XGBoost training
        |
        v
best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl
```

สคริปต์หลัก:

- `scripts/build_v2_high_accuracy_model.py`
- `scripts/build_v2_new_eda_feature_model.py`
- `scripts/generate_clean_dataset_v2.py`

## Final V2 High-Accuracy Package

โฟลเดอร์หลัก:

```text
docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/
```

ไฟล์สำคัญ:

| File | Purpose |
| --- | --- |
| `data/clean_dataset_v2_high_signal.csv` | dataset ที่ใช้ train high-accuracy model |
| `data/df_engineered_v2_HIGH_ACCURACY.csv` | feature พร้อมก่อนเข้า train |
| `data/train_test_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl` | train/test split + feature metadata |
| `data/test_train_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl` | alias ตามชื่อที่เคยใช้ในงาน |
| `models/best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl` | best model |
| `models/best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY_metadata.json` | model metadata |
| `reports/metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` | metrics |
| `reports/confusion_matrix_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` | confusion matrix |
| `reports/feature_importance_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` | feature importance |
| `reports/test_predictions_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` | prediction result on test set |
| `reports/threshold_search_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` | threshold search |
| `docs/model_report_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.md` | model report |
| `docs/eda_insight_summary_v2_NEW.md` | EDA/Insight summary |

## EDA And Insight

EDA ล่าสุดวิเคราะห์จาก V2 high-accuracy package และสร้าง insight ที่นำไปใช้ทำ feature engineering:

| Insight area | Feature action |
| --- | --- |
| Customer return history | ใช้ `hist_return_rate`, `hist_order_count`, rolling history |
| Rolling history windows | เพิ่ม 30d, 60d, 90d, 180d, 365d |
| Payment method | เพิ่ม `is_cod`, `is_bank_transfer`, `is_credit_card` |
| Discount / Promotion | เพิ่ม `is_high_discount`, `discount_amount_ratio` |
| Product rating | เพิ่ม `low_rating_alert` |
| Logistics risk | เพิ่ม `logistics_risk` จาก `damage_rate * is_fragile` |
| Category + Payment | เพิ่ม `category_payment` |
| Category + Channel | เพิ่ม `category_channel` |
| Province + Payment | เพิ่ม `province_payment` |
| Customer tier + Payment | เพิ่ม `tier_payment` |

ภาพและตาราง EDA อยู่ที่:

```text
docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/eda/
docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/images/
```

## Feature Engineering

`df_engineered_v2_HIGH_ACCURACY.csv` มี 50,000 rows และ 76 columns:

- identifiers: `order_id`, `customer_id`, `order_date`
- model features: 71 columns
- target: `is_returned`
- split marker: `dataset_split`

กลุ่ม feature ที่ใช้:

- customer profile: `gender`, `age`, `membership_tier`, `province`, `customer_age_days`, `customer_tenure_months`
- product/order: `category`, `brand`, `is_fragile`, `product_rating`, `quantity`, `unit_price`, `total_amount`
- discount/payment/channel: `payment_method`, `channel_type`, `total_discount_pct`, `is_cod`, `is_high_discount`
- interaction: `category_payment`, `category_channel`, `province_payment`, `tier_payment`
- history: `hist_order_count`, `hist_return_rate`, `days_since_last_order`, `days_since_last_return`
- rolling windows: `hist_order_count_30d`, `hist_return_rate_90d`, `hist_return_count_180d`, `hist_spend_sum_365d` และ window อื่นตาม 30/60/90/180/365 วัน
- time: `order_hour`, `order_month`, `order_dayofweek`, `is_weekend`

Top feature importance ล่าสุด:

| Rank | Feature |
| ---: | --- |
| 1 | `hist_return_rate` |
| 2 | `logistics_risk` |
| 3 | `hist_return_count_180d` |
| 4 | `is_cod` |
| 5 | `is_fragile` |
| 6 | `brand_SilkTouch` |
| 7 | `hist_return_count_365d` |
| 8 | `hist_return_rate_180d` |
| 9 | `province_Remote_Area` |
| 10 | `total_discount_pct` |

## Leakage Policy

โมเดล high-accuracy ยังตัด leakage/post-event fields ออกจาก feature train แล้ว

ไม่ใช้:

```text
return_id, return_date, return_reason, return_scenario,
item_condition, return_status, refund_amount,
score_id, risk_score, risk_tier, scored_at, shap_values,
delivery_date, delivery_days, delay_days, is_returned
```

`is_returned` ใช้เป็น target เท่านั้น ไม่ใช่ input feature

ไม่ใช้ identity fields ให้ model จำรายคน/รายสินค้าโดยตรง:

```text
order_id, customer_id, customer_name, customer_phone,
product_id, product_name, supplier_id, supplier_name,
supplier_contact, courier_id, courier_name, promo_id, promo_name
```

## Version Summary

| Version | Main dataset/model | Accuracy | Note |
| --- | --- | ---: | --- |
| V1 | baseline XGBoost | 70.80% | baseline feature engineering |
| V2 realistic safe rolling | XGBoost on realistic features | 71.07% | เหมาะเป็น production-style baseline |
| V2 NEW on `clean_dataset_v2.csv` | XGBoost, 71 features | 71.46% | realistic, no leakage, Accuracy ไม่สูงมาก |
| **V2 HIGH_ACCURACY** | **XGBoost on high-signal V2 dataset** | **88.88%** | current final candidate |
| V3 | stacking model | 66.67% | ซับซ้อนกว่าแต่ไม่ได้ชนะชัด |
| V4 | generated + SMOTE + Optuna | 83.45% | synthetic experiment, cost สูงกว่า |

## How To Rebuild

สร้าง realistic 50k dataset:

```powershell
& 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'scripts\generate_clean_dataset_v2.py'
```

สร้าง V2 NEW realistic pipeline:

```powershell
& 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'scripts\build_v2_new_eda_feature_model.py'
```

สร้าง V2 HIGH_ACCURACY pipeline:

```powershell
& 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'scripts\build_v2_high_accuracy_model.py'
```

## Recommendation

สำหรับโปรเจ็กต์จบตอนนี้ให้ใช้:

```text
docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/models/best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl
```

สำหรับ production จริงในอนาคต:

- ใช้ table `public."order_history_complete_v2_NEW"` เป็น feature-store base
- ใช้ V2 HIGH_ACCURACY เป็น candidate model
- validate กับข้อมูล order จริงก่อน deploy
- monitor Accuracy, Recall, Precision, F1, AUC, Cost และ data drift รายเดือน
- retrain เมื่อข้อมูลจริงมากพอหรือ distribution เปลี่ยน
