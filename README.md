# Return Risk Prediction Project

อัปเดตล่าสุด: 2026-05-27

โปรเจ็กต์นี้เป็นระบบวิเคราะห์และทำนายความเสี่ยงการคืนสินค้า/คืนเงินของ order ในธุรกิจ e-commerce หรือ order operation โดยใช้ข้อมูล order, customer, product, courier, promotion และ return history เพื่อสร้าง feature, train model, ประเมินผลด้วย metric เชิงธุรกิจ และเตรียมแนวทางต่อยอดเป็น production feature store

จุดสำคัญของโปรเจ็กต์นี้ไม่ใช่ดู Accuracy อย่างเดียว แต่ต้องดูว่าโมเดลช่วยลด cost จากการพลาด order เสี่ยงคืนสินค้าได้แค่ไหน จึงมีการรายงาน Accuracy, Recall, Precision, F1, AUC และ Cost Matrix ควบคู่กัน

## Project Goal

เป้าหมายหลักคือสร้าง model ที่ตอบคำถามนี้:

> ถ้ามี order ใหม่เข้ามา ระบบควรประเมินได้ไหมว่า order นี้มีโอกาสถูกคืนสินค้าสูงหรือต่ำ โดยใช้ข้อมูลที่รู้ก่อนหรือระหว่าง order flow

Business use cases:

- ช่วย call center หรือ operation team ตรวจ order ที่มี return risk สูง
- ลดต้นทุนจากการส่งสินค้าแล้วถูกคืน
- ใช้ customer history เช่น `return_count / total_orders` เพื่อสร้าง risk signal
- เตรียมข้อมูลสำหรับ Feature Store ในอนาคต เช่น customer rolling history 30d, 60d, 90d, 180d, 365d
- อธิบายเหตุผลของ feature ที่เลือกจาก Business Insight และ EDA chart

ตัวอย่าง logic สำคัญ:

```text
ลูกค้ามี order ก่อนหน้า 2 order
- คืนสินค้า 1 order
- ไม่คืนสินค้า 1 order

hist_return_rate = return_count / total_orders
                 = 1 / 2
                 = 0.5 หรือ 50%
```

## Current Completion Status

| Work package | Status | Main outputs |
| --- | --- | --- |
| Data Collection & Understanding | Done | `data/processed/clean_dataset.csv`, V4 SQL/data dictionary |
| Data Cleansing | Done | clean dataset, generated clean V4 dataset |
| EDA & Business Insight | Done | selected charts, insight report PDF |
| Manual Validation | Done | customer return-rate Excel-ready CSV |
| Feature Engineering | Done by version | V1, V2, V2 safe rolling, V4 generated |
| Feature Audit | Done | clean vs engineered feature inventory |
| Model Training & Evaluation | Done by version | V1, V2, V2 XGBoost, V3, V4 |
| V3 Packaging | Done | structured V3 folder with model, data, reports, scripts |
| Production API | Deferred | production idea documented, current focus is model completeness |

## Project Structure

```text
return-risk-prediction/
|-- data/
|   |-- processed/
|   |   `-- clean_dataset.csv
|   |-- raw/
|   |   `-- mock_return_data.csv
|   `-- reference/
|       `-- input_materials/
|-- docs/
|   |-- analysis/
|   |   |-- business_insight_feature_summary.md
|   |   |-- business_insight_feature_summary.pdf
|   |   |-- feature_audit_and_validation.md
|   |   |-- feature_audit_and_validation.pdf
|   |   `-- model_versions_v1_to_v4_comparison.pdf
|   |-- version 1/
|   |-- version 2/
|   |-- version 3/
|   `-- version 4/
|-- reports/
|   |-- business_insights/
|   |-- feature_audit/
|   |-- Graph Item/
|   `-- Graph Relation Feature/
|-- scripts/
|   `-- complete_project_audit_pack.py
|-- src/
|   `-- production_v2/
|-- dashboard.py
`-- requirements.txt
```

## Main Reports To Open

| Report | Purpose |
| --- | --- |
| [Business Insight Feature Summary](docs/analysis/business_insight_feature_summary.pdf) | สรุปกราฟที่เลือก, insight, เหตุผลที่เลือก, feature ที่ตามมา, business action |
| [Feature Audit & Manual Validation](docs/analysis/feature_audit_and_validation.pdf) | สรุป clean vs engineered feature, manual return-rate check, model summary |
| [Model V1-V4 Comparison](docs/analysis/model_versions_v1_to_v4_comparison.pdf) | เปรียบเทียบภาพรวม version 1 ถึง version 4 |
| [V4 Generated End-to-End Report](docs/version%204/docs/v4_generated_end_to_end_report.pdf) | รายงาน pipeline V4 ตั้งแต่ generated data ถึง model |
| [V3 Stacking Process Report](docs/version%203/stacking_model_v3/docs/v3_stacking_model_process_report.pdf) | รายงาน process ของ V3 stacking model |

## Data Sources

### Main clean dataset

ไฟล์หลักของข้อมูลจริง/ข้อมูล mock ตาม schema:

- [data/processed/clean_dataset.csv](data/processed/clean_dataset.csv)

ข้อมูลนี้รวม field จากหลายส่วน เช่น:

- Order: `order_id`, `order_date`, `quantity`, `unit_price`, `total_amount`
- Customer: `customer_id`, `gender`, `age`, `membership_tier`, `province`, `registration_date`
- Product: `product_id`, `category`, `brand`, `is_fragile`, `product_rating`
- Courier: `courier_id`, `courier_name`, `courier_type`, `avg_delivery_days`, `damage_rate`
- Promotion: `promo_id`, `promo_name`, `promo_type`, `promo_discount_rate`
- Return/refund: `return_id`, `return_date`, `return_reason`, `refund_amount`
- Risk fields: `risk_score`, `risk_tier`, `shap_values`
- Target: `is_returned`

### Leakage warning

บาง field ใน `clean_dataset.csv` เป็นข้อมูลหลังเหตุการณ์ หรือเป็นผลลัพธ์จากระบบอื่น ไม่ควรใช้ train model แบบ real-time order-time scoring:

```text
return_id, return_date, return_reason, return_status,
refund_amount, risk_score, risk_tier, shap_values,
delivery_days, delay_days
```

ถ้าจะใช้ `delivery_days` หรือ `delay_days` ต้องระบุว่าเป็น post-delivery model ไม่ใช่ model ที่ใช้ตอน order เพิ่งเข้ามา

## Business Insight Summary

มีการคัดกราฟจากชุด EDA 40+ รูป เหลือ 10 insight ที่ผูกกับ feature และ business action ได้จริง

โฟลเดอร์รูปที่คัดแล้ว:

- [docs/analysis/business_insight_selected_charts](docs/analysis/business_insight_selected_charts)

Mapping file:

- [reports/feature_audit/business_insight_feature_mapping.csv](reports/feature_audit/business_insight_feature_mapping.csv)

Insight ที่เลือก:

| No | Insight | Feature examples |
| --- | --- | --- |
| 1 | Customer return history | `hist_order_count`, `hist_return_rate`, `customer_return_ratio`, rolling return rates |
| 2 | Category and payment interaction | `category`, `payment_method`, `category_payment`, `is_cod` |
| 3 | Category and channel interaction | `category`, `channel_type`, `category_channel` |
| 4 | Province and payment risk | `province`, `province_payment` |
| 5 | Product rating threshold | `product_rating`, `low_rating_alert` |
| 6 | Promotion and discount behavior | `promo_discount_rate`, `total_discount_pct`, `is_high_discount` |
| 7 | Price and basket amount | `unit_price`, `quantity`, `total_amount`, log amount features |
| 8 | Logistics expected risk | `courier_type`, `avg_delivery_days`, `delivery_time_expected_days`, `damage_rate` |
| 9 | Repurchase behavior | `is_repurchased_item`, `days_since_last_order` |
| 10 | Order time and weekend | `order_hour`, `order_dayofweek`, `is_weekend` |

## Manual Return-Rate Validation

เพิ่มไฟล์ cross-check รายลูกค้าเพื่อเปิดตรวจใน Excel แล้ว:

- [reports/feature_audit/customer_return_rate_manual_check.csv](reports/feature_audit/customer_return_rate_manual_check.csv)
- [reports/feature_audit/customer_return_rate_order_level_check.csv](reports/feature_audit/customer_return_rate_order_level_check.csv)

สิ่งที่ตรวจ:

- สุ่มลูกค้าหลายราย
- คำนวณ `return_count / total_orders`
- ตรวจแบบ order-level ว่า history ใช้เฉพาะ order ก่อนหน้า ไม่รวม order ปัจจุบัน
- ตรวจ `hist_order_count` เทียบกับ manual calculation

ผลตรวจล่าสุด:

- manual order-level rows: 104 rows
- `hist_order_count` match: 104/104
- mismatch: 0

ตัวอย่าง:

| customer_id | total_orders | return_count | formula | return rate |
| --- | ---: | ---: | --- | ---: |
| C0324 | 19 | 10 | `10/19` | 52.63% |
| C0270 | 10 | 5 | `5/10` | 50.00% |
| C0094 | 8 | 5 | `5/8` | 62.50% |
| C0052 | 6 | 4 | `4/6` | 66.67% |

## Feature Audit

Feature audit files:

- [reports/feature_audit/feature_inventory_by_version.csv](reports/feature_audit/feature_inventory_by_version.csv)
- [reports/feature_audit/feature_audit_summary_by_version.csv](reports/feature_audit/feature_audit_summary_by_version.csv)

Summary:

| Version | Feature count | Used feature count | Note |
| --- | ---: | ---: | --- |
| clean_dataset | 65 | - | Clean/raw usable dataset แต่ยังมี leakage/post-event fields |
| V1 engineered | 51 | 136 | มี feature engineering และ encoding ทำให้ model features เพิ่มเป็น 136 |
| V1 model used | 136 | 136 | Baseline encoded feature set |
| V2 engineered | 39 | 38 | ลด feature ให้ตีความง่ายขึ้น |
| V2 model used | 38 | 38 | V2 baseline แต่ยังมี `delivery_days` / `delay_days` |
| V2 XGBoost safe rolling | 60 | 60 | Order-time safe, เพิ่ม rolling history windows |
| V3 stacking | 38 | 38 | ใช้ feature set จาก V2 ไม่ได้สร้าง feature ใหม่เอง |
| V4 generated engineered | 123 | 180 | Synthetic/generated feature pipeline |
| V4 generated model used | 180 | 180 | ใช้กับ generated data และ SMOTE/Optuna |

## Model Versions

ผลล่าสุดจาก [reports/feature_audit/model_version_completion_summary.csv](reports/feature_audit/model_version_completion_summary.csv)

| Version | Model | Accuracy | Recall | Precision | F1 | AUC | Cost | Rating | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V1 | XGBClassifier | 70.80% | 26.80% | 49.68% | 34.82% | 68.82% | 35,900 | D | Baseline V1 model |
| V2 baseline | V2 tuned model | 67.87% | 65.60% | 46.28% | 54.27% | 73.66% | 19,550 | B | ข้อมูลจริง, recall/cost ดี |
| V2 XGBoost safe rolling | XGBClassifier | 71.07% | 56.88% | 50.20% | 53.33% | 71.47% | 20,250 | B | Order-time safe และเหมาะกับ Feature Store |
| V3 stacking | XGB+LGB+CAT Stacking | 66.67% | 63.76% | 44.84% | 52.65% | 71.90% | 20,400 | B | Reuse V2 features แล้วเปลี่ยน model architecture |
| V4 generated | XGBoost + SMOTE + Optuna | 83.45% | 46.39% | 45.00% | 45.69% | 85.38% | 31,650 | C | Synthetic/generated data, accuracy สูงแต่ cost สูง |

## Version Details

### Version 1 - Baseline Feature Engineering

Folder:

- [docs/version 1](<docs/version 1>)

สิ่งที่ทำ:

- สร้าง `df_engineered.csv`
- สร้าง train/test set
- Train baseline/tuned models
- ใช้ XGBoost เป็น best saved model ใน package นี้
- มี feature engineering จำนวนมาก เช่น customer return history, category/payment/channel interaction, discount, price, order time

Key files:

- [docs/version 1/feature_engineering.py](<docs/version 1/feature_engineering.py>)
- [docs/version 1/data/features/df_engineered.csv](<docs/version 1/data/features/df_engineered.csv>)
- [docs/version 1/data/features/train_test_sets.pkl](<docs/version 1/data/features/train_test_sets.pkl>)
- [docs/version 1/model/best_model.pkl](<docs/version 1/model/best_model.pkl>)
- [docs/version 1/model_training.py](<docs/version 1/model_training.py>)
- [docs/version 1/model_evaluation.py](<docs/version 1/model_evaluation.py>)

### Version 2 - Main Practical Baseline

Folder:

- [docs/version 2](<docs/version 2>)

สิ่งที่ทำ:

- ทำ feature engineering V2 ให้ตีความง่ายกว่า V1
- สร้าง `train_test_sets_v2.pkl`
- ประเมิน model ด้วย threshold scenarios, cost matrix, SHAP
- Export prediction result สำหรับ test set
- เป็น baseline ที่ดีที่สุดในมุมข้อมูลจริงและ cost/recall balance

Key files:

- [docs/version 2/feature_engineering_v2.py](<docs/version 2/feature_engineering_v2.py>)
- [docs/version 2/data/features/df_engineered_v2_preview.csv](<docs/version 2/data/features/df_engineered_v2_preview.csv>)
- [docs/version 2/data/features/train_test_sets_v2.pkl](<docs/version 2/data/features/train_test_sets_v2.pkl>)
- [docs/version 2/model_evaluation_v2/v2_test_prediction_summary.csv](<docs/version 2/model_evaluation_v2/v2_test_prediction_summary.csv>)
- [docs/version 2/model_evaluation_v2/v2_test_predictions.csv](<docs/version 2/model_evaluation_v2/v2_test_predictions.csv>)
- [docs/version 2/model_evaluation_v2/shap_summary_v2.png](<docs/version 2/model_evaluation_v2/shap_summary_v2.png>)

Important caveat:

- V2 baseline ใช้ `delivery_days` และ `delay_days`
- ดังนั้น V2 baseline ไม่ใช่ order-time safe แบบเต็ม 100%
- เหมาะเป็น baseline experiment หรือ post-delivery scoring มากกว่า real-time order scoring

### Version 2 XGBoost Safe Plus Rolling

Folder:

- [docs/version 2/v2_xgboost_safe_plus_rolling](<docs/version 2/v2_xgboost_safe_plus_rolling>)

นี่คือ candidate ที่เหมาะกับ production direction มากกว่า เพราะ:

- `Order-time Safe = Yes`
- ตัด `delivery_days` และ `delay_days` ออก
- เพิ่ม customer rolling history windows เช่น 30d, 60d, 90d, 180d, 365d
- ใช้ XGBoost
- เหมาะกับแนวคิด SQL DB หรือ Feature Store ในอนาคต

Key files:

- [best_model_v2_xgboost_safe_plus_rolling.pkl](<docs/version 2/v2_xgboost_safe_plus_rolling/models/best_model_v2_xgboost_safe_plus_rolling.pkl>)
- [v2_xgboost_safe_plus_rolling_used_features.csv](<docs/version 2/v2_xgboost_safe_plus_rolling/data/v2_xgboost_safe_plus_rolling_used_features.csv>)
- [v2_xgboost_safe_plus_rolling_metrics.csv](<docs/version 2/v2_xgboost_safe_plus_rolling/reports/v2_xgboost_safe_plus_rolling_metrics.csv>)
- [v2_xgboost_safe_plus_rolling_test_predictions.csv](<docs/version 2/v2_xgboost_safe_plus_rolling/reports/v2_xgboost_safe_plus_rolling_test_predictions.csv>)
- [v2_xgboost_safe_plus_rolling_report.md](<docs/version 2/v2_xgboost_safe_plus_rolling/docs/v2_xgboost_safe_plus_rolling_report.md>)

Selected metrics:

```text
Accuracy  : 71.07%
Recall    : 56.88%
Precision : 50.20%
F1        : 53.33%
AUC       : 71.47%
Cost      : 20,250
Rating    : B
```

### Version 3 - Stacking Model

Folder:

- [docs/version 3/stacking_model_v3](<docs/version 3/stacking_model_v3>)

สิ่งที่ทำ:

- ใช้ feature/split จาก V2
- Train StackingClassifier
- Base models: XGBoost, LightGBM, CatBoost
- Meta learner: LogisticRegression
- ประเมิน threshold, recall, cost matrix

สำคัญ:

- V3 เป็น model architecture experiment
- V3 ไม่ได้สร้าง feature engineering version ใหม่เอง
- ใช้ feature set จาก V2 จำนวน 38 ตัว

Key files:

- [README.md](<docs/version 3/stacking_model_v3/README.md>)
- [best_model_v3_stack.pkl](<docs/version 3/stacking_model_v3/models/best_model_v3_stack.pkl>)
- [v3_used_features.csv](<docs/version 3/stacking_model_v3/data/v3_used_features.csv>)
- [metrics_summary_v3.csv](<docs/version 3/stacking_model_v3/reports/metrics_summary_v3.csv>)
- [threshold_scenarios_v3.csv](<docs/version 3/stacking_model_v3/reports/threshold_scenarios_v3.csv>)
- [v3_stacking_model_process_report.pdf](<docs/version 3/stacking_model_v3/docs/v3_stacking_model_process_report.pdf>)

Runtime note:

- การโหลด V3 model ต้องมี `catboost` ติดตั้ง เพราะ model pickle มี CatBoostClassifier อยู่ข้างใน

### Version 4 - Generated Data End-to-End Pipeline

Folder:

- [docs/version 4](<docs/version 4>)

สิ่งที่ทำ:

- Generate synthetic order/return data
- Clean data ใหม่
- ทำ EDA
- ทำ feature engineering ขนาดใหญ่
- ใช้ SMOTE จัดการ imbalance
- Train LogisticRegression, RandomForest, XGBoost, LightGBM
- Tune ด้วย Optuna
- Evaluate ด้วย Cost Matrix, AUC, SHAP

Key files:

- [v4_synthetic_orders_returns.csv](<docs/version 4/data/generated/v4_synthetic_orders_returns.csv>)
- [clean_dataset_v4_generated.csv](<docs/version 4/data/processed/clean_dataset_v4_generated.csv>)
- [df_engineered_v4_generated.csv](<docs/version 4/data/features/df_engineered_v4_generated.csv>)
- [train_test_sets_v4_generated.pkl](<docs/version 4/data/features/train_test_sets_v4_generated.pkl>)
- [best_model_v4_generated.pkl](<docs/version 4/models/best_model_v4_generated.pkl>)
- [v4_generated_model_metrics.csv](<docs/version 4/reports/model_evaluation/v4_generated_model_metrics.csv>)
- [v4_generated_shap_summary.png](<docs/version 4/reports/model_evaluation/v4_generated_shap_summary.png>)
- [v4_generated_end_to_end_report.pdf](<docs/version 4/docs/v4_generated_end_to_end_report.pdf>)

Important caveat:

- V4 ใช้ generated/synthetic data
- Accuracy สูงกว่า version อื่น แต่ไม่ควรสรุปว่า production ดีกว่า V2 โดยตรง
- ใช้สำหรับแสดง end-to-end pipeline, imbalance handling, SMOTE, Optuna และ SHAP ได้ดี

## Recommended Model Positioning

| Use case | Recommended model/version | Reason |
| --- | --- | --- |
| Project presentation on real data | V2 baseline | Recall/cost balance ดีและอธิบายง่าย |
| Order-time / Feature Store direction | V2 XGBoost safe rolling | ไม่มี post-delivery leakage และมี rolling history |
| Model architecture experiment | V3 stacking | แสดง ensemble learning |
| End-to-end generated pipeline | V4 generated | ครบตั้งแต่ data generation ถึง SHAP |
| Accuracy showcase only | V4 generated | Accuracy 83.45% แต่เป็น synthetic data และ cost สูง |

ถ้าจะต่อ production จริง ควรใช้ V2 XGBoost safe rolling เป็นฐาน แล้วสร้าง SQL Feature Store สำหรับ rolling features เช่น:

```text
cust_order_count_30d
cust_return_count_30d
cust_return_rate_30d
cust_spend_sum_30d
cust_order_count_90d
cust_return_rate_90d
cust_order_count_365d
cust_return_rate_365d
```

## Order-Time Safe Definition

`Order-time Safe` หมายถึง feature ที่รู้ได้ตอน order เพิ่งเข้ามา โดยไม่ใช้ข้อมูลอนาคตของ order นั้น

ใช้ได้:

```text
customer profile
prior order history
prior return history
product/category/brand
payment method
promotion/discount
channel
province
expected delivery days
courier expected attributes
```

ไม่ควรใช้กับ real-time order-time model:

```text
delivery_days
delay_days
return_date
return_reason
refund_amount
risk_score
risk_tier
shap_values
is_returned ของ order ปัจจุบัน
```

## Regenerating Audit Reports

สคริปต์นี้ใช้สร้างรายงาน audit ล่าสุด:

```powershell
python scripts/complete_project_audit_pack.py
```

Outputs:

```text
docs/analysis/business_insight_feature_summary.md
docs/analysis/business_insight_feature_summary.pdf
docs/analysis/feature_audit_and_validation.md
docs/analysis/feature_audit_and_validation.pdf
reports/feature_audit/business_insight_feature_mapping.csv
reports/feature_audit/customer_return_rate_manual_check.csv
reports/feature_audit/customer_return_rate_order_level_check.csv
reports/feature_audit/feature_inventory_by_version.csv
reports/feature_audit/feature_audit_summary_by_version.csv
reports/feature_audit/model_version_completion_summary.csv
docs/version 3/stacking_model_v3/
```

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run dashboard:

```powershell
streamlit run dashboard.py
```

Run optional PostgreSQL stack:

```powershell
docker compose up -d
```

Regenerate audit package:

```powershell
python scripts/complete_project_audit_pack.py
```

## Notes For Final Project Presentation

ควรเล่าโปรเจ็กต์ตามลำดับนี้:

1. เริ่มจาก business problem: return/refund ทำให้เกิด cost
2. อธิบาย clean dataset และ leakage fields ที่ต้องระวัง
3. ใช้ Business Insight 10 ชุดเป็นเหตุผลเลือก feature
4. แสดง manual validation ว่า return rate รายลูกค้าคำนวณถูก
5. เปรียบเทียบ V1 ถึง V4
6. อธิบายว่า V2 baseline ดีในข้อมูลจริง ส่วน V2 XGBoost safe rolling เหมาะกับ production direction
7. อธิบายว่า V4 accuracy สูงเพราะ generated/synthetic data จึงใช้เป็น pipeline showcase ไม่ใช่ production winner
8. ปิดด้วยแนวทางต่อยอด SQL DB เป็น Feature Store สำหรับ real-time inference

## Known Limitations

- ข้อมูลหลักเป็น mock/simulated project data ไม่ใช่ production transaction จริง
- V2 baseline และ V3 ยังมี `delivery_days` / `delay_days` ใน feature set จึงไม่ใช่ order-time safe เต็มรูปแบบ
- V4 เป็น generated data จึงเทียบกับ V1-V3 แบบตรง ๆ ไม่ได้
- Accuracy สูงอย่างเดียวไม่พอ ต้องดู Recall และ Cost Matrix ด้วย
- Production API/real-time inference ยังถูก defer ไว้ก่อน เพราะรอบนี้โฟกัส model completeness และ project documentation

## Best Current Summary

- Best real-data practical baseline: V2 baseline
- Best order-time safe candidate: V2 XGBoost safe rolling
- Best architecture experiment: V3 stacking
- Best end-to-end pipeline showcase: V4 generated
- Most important business feature group: customer return history
- Most important validation file: `customer_return_rate_manual_check.csv`
