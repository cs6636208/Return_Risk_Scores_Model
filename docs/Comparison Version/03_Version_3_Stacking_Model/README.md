# Version 3 - Stacking Model

## Role in Comparison
ทดลองเพิ่มความซับซ้อนของ model โดยใช้ feature base ใกล้ V2 แล้ว ensemble หลายโมเดล

## Result Summary
Recall ดีขึ้นเมื่อเทียบกับ V1 แต่ Accuracy/F1/AUC ยังไม่ชนะ V2 และดูแล production ยากกว่า

## Metrics
- Model: `Stacking XGB+LGBM+CatBoost`
- Dataset: `V2 engineered feature set`
- Feature count: `38`
- Accuracy: `61.60%`
- Recall: `63.76%`
- Precision: `44.84%`
- F1: `52.65%`
- AUC: `71.90%`
- Cost: `19,500`
- Rating: `B`

## Folder Layout
- `01_feature_used_unused_audit_*.pdf` - feature ที่ใช้/ไม่ใช้/ตัดทิ้ง
- `02_feature_engineering_process_*.pdf` - ขั้นตอน feature engineering
- `csv/` - metric, used features, dropped features, feature importance
- `code/` - script หรือ code ที่เกี่ยวข้อง
- `images/` - กราฟหรือภาพประเมินผลของ version นั้น

Generated at: 2026-05-28T16:45:03