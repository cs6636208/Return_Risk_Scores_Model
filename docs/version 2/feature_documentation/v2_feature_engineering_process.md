# Version 2 - XGBoost Safe Plus Rolling HIGH_ACCURACY - Feature Engineering Process

## Process
- ทำ EDA จาก clean_dataset_v2/high-signal เพื่อดู pattern: history, category, channel, payment, discount, rating, province
- สร้าง customer historical features แบบ point-in-time โดยไม่ใช้ order ปัจจุบันเป็นประวัติของตัวเอง
- เพิ่ม rolling windows 30/60/90/180/365 days เช่น cust_return_rate_30d, cust_order_count_90d, cust_spend_sum_180d
- เพิ่ม business interaction features เช่น category x payment, channel x category, discount band, rating band, province risk
- ตัด leakage/post-event fields และ identifier fields ก่อน train
- train XGBoost พร้อม threshold tuning เพื่อ balance Accuracy, Recall, F1, AUC และ Cost Matrix

## Code / Artifact Reference
- `docs\version 2\v2_xgboost_safe_plus_rolling_HIGH_ACCURACY\scripts\feature_engineered_v2_HIGH_ACCURACY.py`
- `docs\version 2\v2_xgboost_safe_plus_rolling_HIGH_ACCURACY\data\train_test_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl`
- `docs\version 2\v2_xgboost_safe_plus_rolling_HIGH_ACCURACY\models\best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl`

## Decision
เลือกเป็น candidate หลัก เพราะ performance สูงสุดและ feature logic ตรงกับโจทย์ order-time prediction

Generated at: 2026-05-28T16:26:09