# Version 2 - XGBoost Safe Rolling HIGH_ACCURACY

## Role in Comparison
เลือกเป็น candidate หลัก เพราะใช้ rolling customer history, business insight features และตัด leakage/post-event fields

## Result Summary
Performance สูงสุดในชุดเปรียบเทียบ: Accuracy, Recall, F1 และ AUC ดีที่สุด

## Metrics
- Model: `XGBoost safe rolling`
- Dataset: `clean_dataset_v2_high_signal.csv`
- Feature count: `71`
- Accuracy: `88.88%`
- Recall: `76.03%`
- Precision: `84.40%`
- F1: `79.99%`
- AUC: `94.66%`
- Cost: `371,050`
- Rating: `A`

## Folder Layout
- `01_feature_used_unused_audit_*.pdf` - feature ที่ใช้/ไม่ใช้/ตัดทิ้ง
- `02_feature_engineering_process_*.pdf` - ขั้นตอน feature engineering
- `csv/` - metric, used features, dropped features, feature importance
- `code/` - script หรือ code ที่เกี่ยวข้อง
- `images/` - กราฟหรือภาพประเมินผลของ version นั้น

Generated at: 2026-05-28T15:24:46