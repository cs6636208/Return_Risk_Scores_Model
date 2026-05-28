# Version 4 - Generated Data + SMOTE + Optuna

## Role in Comparison
ทดลองเพิ่ม data และจัดการ imbalance ด้วย SMOTE พร้อม tuning ด้วย Optuna

## Result Summary
Accuracy สูงกว่า V1/V3 แต่ Recall/F1 ยังไม่ชนะ V2 และ feature volume สูงกว่า

## Metrics
- Model: `XGBoost + SMOTE + Optuna`
- Dataset: `clean_dataset_v4_generated.csv`
- Feature count: `180`
- Accuracy: `83.45%`
- Recall: `46.39%`
- Precision: `45.00%`
- F1: `45.69%`
- AUC: `85.38%`
- Cost: `31,650`
- Rating: `C`

## Folder Layout
- `01_feature_used_unused_audit_*.pdf` - feature ที่ใช้/ไม่ใช้/ตัดทิ้ง
- `02_feature_engineering_process_*.pdf` - ขั้นตอน feature engineering
- `csv/` - metric, used features, dropped features, feature importance
- `code/` - script หรือ code ที่เกี่ยวข้อง
- `images/` - กราฟหรือภาพประเมินผลของ version นั้น

Generated at: 2026-05-28T16:45:03