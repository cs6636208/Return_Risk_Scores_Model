# Version 1 - Baseline XGBoost

## Role in Comparison
ใช้เป็น baseline เพื่อวัดผล feature engineering ชุดแรกจาก clean_dataset.csv

## Result Summary
Accuracy ใช้ได้ระดับเริ่มต้น แต่ Recall ต่ำ จึงยังพลาด order ที่มีแนวโน้มคืนสินค้าเยอะ

## Metrics
- Model: `XGBoost baseline`
- Dataset: `clean_dataset.csv`
- Feature count: `136`
- Accuracy: `70.80%`
- Recall: `26.80%`
- Precision: `49.68%`
- F1: `34.82%`
- AUC: `68.82%`
- Cost: `35,900`
- Rating: `D`

## Folder Layout
- `01_feature_used_unused_audit_*.pdf` - feature ที่ใช้/ไม่ใช้/ตัดทิ้ง
- `02_feature_engineering_process_*.pdf` - ขั้นตอน feature engineering
- `csv/` - metric, used features, dropped features, feature importance
- `code/` - script หรือ code ที่เกี่ยวข้อง
- `images/` - กราฟหรือภาพประเมินผลของ version นั้น

Generated at: 2026-05-28T16:45:03