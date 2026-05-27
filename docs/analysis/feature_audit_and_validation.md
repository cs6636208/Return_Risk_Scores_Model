# Feature Audit & Manual Validation

รายงานนี้เติมส่วนที่ขาดจาก requirement: cross-check return rate รายลูกค้า, compare clean vs engineered features, และสรุป feature/model แต่ละ version

## Manual Customer Return Rate Check

- Summary CSV: `reports\feature_audit\customer_return_rate_manual_check.csv`
- Order-level point-in-time CSV: `reports\feature_audit\customer_return_rate_order_level_check.csv`
- สูตรที่ใช้: `return_count / total_orders` เช่น `4 / 8 = 0.5 = 50%`
- สำหรับ order-level history ใช้เฉพาะ order ก่อนหน้าเท่านั้น ไม่รวม order ปัจจุบัน

| customer_id | total_orders | return_count | manual_formula | manual_return_rate | manual_return_rate_pct |
| --- | --- | --- | --- | --- | --- |
| C0324 | 19 | 10 | 10/19 | 0.526316 | 52.63 |
| C0413 | 13 | 5 | 5/13 | 0.384615 | 38.46 |
| C0043 | 11 | 4 | 4/11 | 0.363636 | 36.36 |
| C0309 | 11 | 2 | 2/11 | 0.181818 | 18.18 |
| C0270 | 10 | 5 | 5/10 | 0.5 | 50.0 |
| C0086 | 10 | 1 | 1/10 | 0.1 | 10.0 |
| C0334 | 9 | 1 | 1/9 | 0.111111 | 11.11 |
| C0094 | 8 | 5 | 5/8 | 0.625 | 62.5 |
| C0163 | 7 | 3 | 3/7 | 0.428571 | 42.86 |
| C0052 | 6 | 4 | 4/6 | 0.666667 | 66.67 |

## Feature Summary by Version

| version | description | feature_count | used_feature_count | created_feature_count_vs_clean | retained_clean_feature_count | dropped_or_not_used_count | leakage_or_target_related_fields | post_delivery_fields | identifier_fields | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_dataset | clean/raw usable dataset | 65 |  | 0 | 65 | 0 | item_condition, refund_amount, return_date, return_id, return_reason, return_scenario, return_status, risk_score, risk_tier, score_id, scored_at, shap_values | delay_days, delivery_date, delivery_days | courier_id, customer_id, customer_name, customer_phone, order_id, product_id, promo_id, supplier_id | ฐานข้อมูล clean หลังรวม order/return/customer/product แต่ยังมี post-event/leakage fields ต้องตัดก่อน train |
| v1_engineered | V1 engineered before model encoding | 51 | 136 | 34 | 17 | 9 |  |  |  | มี feature engineering จำนวนมากและ one-hot/encoding ทำให้ final feature count สูง |
| v1_model_used | V1 encoded model feature set | 136 | 136 | 124 | 12 | 0 |  |  |  | ใช้ encoded features 136 ตัว เหมาะเป็น baseline experiment |
| v2_engineered | V2 engineered preview before training | 39 | 38 | 6 | 33 | 1 |  | delay_days, delivery_days |  | ลด feature ให้ตีความง่ายขึ้นและใช้ feature history/customer/product/logistics |
| v2_model_used | V2 model feature set | 38 | 38 | 6 | 32 | 0 |  | delay_days, delivery_days |  | V2 baseline 38 features แต่ยังมี delivery_days/delay_days จึงไม่ order-time safe |
| v2_xgboost_safe_plus_rolling | V2 order-time-safe XGBoost rolling feature set | 60 | 60 | 30 | 30 | 0 |  |  |  | ตัด post-delivery fields และเพิ่ม rolling history 30/60/90/180/365d เหมาะกับ Feature Store |
| v3_stacking | V3 stacking reuses V2 model feature set | 38 | 38 | 6 | 32 | 0 |  | delay_days, delivery_days |  | ไม่ได้สร้าง feature ใหม่เอง ใช้ V2 feature set แล้วเปลี่ยน model เป็น stacking ensemble |
| v4_generated_engineered | V4 generated engineered dataset | 123 | 180 | 123 | 0 | 1 |  |  |  | ใช้ synthetic/generated data สำหรับทดลอง imbalance/SMOTE/Optuna |
| v4_generated_model_used | V4 generated model feature set | 180 | 180 | 160 | 20 | 0 |  |  |  | feature set จาก generated pipeline ไม่ควรเทียบ production โดยตรงกับข้อมูลจริง |

## Model Summary

| version | model | accuracy | recall | precision | f1 | auc | cost | rating | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | XGBClassifier | 0.708 | 0.26804123711340205 | 0.4968152866242038 | 0.3482142857142857 | 0.6882352085847644 | 35900 | D | computed from packaged V1 model and test set |
| V2 baseline | V2 tuned model | 0.6786666666666666 | 0.6559633027522935 | 0.4627831715210356 | 0.5426944971537002 | 0.736583431054701 | 19550 | B | baseline V2 default threshold |
| V2 XGBoost safe rolling | XGBClassifier | 0.7106666666666667 | 0.5688073394495413 | 0.5020242914979757 | 0.5333333333333333 | 0.7146737255983997 | 20250 | B | order-time safe plus rolling customer history |
| V3 stacking | V3 Stacking (XGB+LGB+CAT) | 0.6666666666666666 | 0.6376146788990825 | 0.4483870967741935 | 0.5265151515151515 | 0.718984962406015 | 20400 | B | stacking model reusing V2 train/test features |
| V4 generated | XGBoost_SMOTE_Optuna | 0.8345360824742268 | 0.4639175257731959 | 0.45 | 0.4568527918781725 | 0.8538112237136325 | 31650 | C | synthetic/generated data with SMOTE/Optuna |

## Key Findings

- V2 baseline เป็น version ที่ครบที่สุดบนข้อมูลจริง ทั้ง feature engineering, evaluation, SHAP, threshold และ prediction output
- V2 XGBoost safe rolling เป็น candidate ที่เหมาะกว่าเมื่อจะใช้จริงแบบ order-time เพราะตัด `delivery_days` และ `delay_days` แล้ว
- V3 มี model process แล้ว แต่ feature engineering ไม่ได้แยกเอง ใช้ feature/split จาก V2
- V4 ครบ end-to-end มากที่สุดในเชิง pipeline แต่เป็น synthetic/generated data จึงควรระบุให้ชัดเวลานำเสนอ
- Feature ที่เป็น identifier เช่น `customer_id`, `order_id` ควรใช้สำหรับ query/audit ไม่ควรส่งเข้า model ให้จำ identity โดยตรง
- Feature หลังเหตุการณ์ เช่น return/refund/risk_score/shap_values ต้อง drop ก่อน train เพื่อกัน leakage