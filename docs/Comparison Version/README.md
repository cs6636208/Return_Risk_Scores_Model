# Comparison Version - Current State

โฟลเดอร์นี้เป็นเอกสารเปรียบเทียบ Version 1-4 ตามสถานะล่าสุดของโปรเจ็กต์

## Current Mapping

- V1 = `v1_baseline_xgboost`
- V2 = `v2_xgboost_safe_plus_rolling`
- V3 = `v3_stacking_from_v2`
- V4 = `v4_generated_xgboost_smote_optuna`
- V5 = `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` ถูกแยกออกจาก V2 แล้ว เพราะเป็น high-signal/generated-data experiment

## Important Note

ไฟล์ comparison รุ่นเก่าบางตัวเคยอ้าง V2 เป็น HIGH_ACCURACY ตอนนี้แก้ master CSV/PDF/graph ล่าสุดแล้ว ให้ใช้ไฟล์ root-level ต่อไปนี้เป็นแหล่งอ้างอิงหลัก:

- `version_1_to_4_selected_model_comparison.csv`
- `version_1_to_4_detailed_comparison.pdf`
- `images/version_1_to_4_performance_metrics.png`

## Current Metric Snapshot

- V1 `v1_baseline_xgboost`: model=XGBoost baseline, Accuracy=70.80%, Recall=26.80%, F1=34.82%, AUC=68.82%, Cost=35,900
- V2 `v2_xgboost_safe_plus_rolling`: model=XGBoost safe plus rolling, Accuracy=71.07%, Recall=56.88%, F1=53.33%, AUC=71.47%, Cost=20,250
- V3 `v3_stacking_from_v2`: model=Stacking XGBoost + LightGBM + CatBoost, Accuracy=66.67%, Recall=63.76%, F1=52.65%, AUC=71.90%, Cost=24,350
- V4 `v4_generated_xgboost_smote_optuna`: model=XGBoost_SMOTE_Optuna, Accuracy=83.45%, Recall=46.39%, F1=45.69%, AUC=85.38%, Cost=31,650
