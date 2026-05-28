# Version 4 - Generated Data + SMOTE + Optuna - Feature Engineering Process

## Process
- generate synthetic order/return data เพื่อทดลองเพิ่มปริมาณข้อมูล
- clean missing/outlier/duplicate และสร้าง clean_dataset_v4_generated.csv
- ทำ EDA category/channel/payment/correlation
- สร้าง feature จำนวนมาก รวม interaction/encoded features
- ใช้ SMOTE สำหรับ imbalance และ Optuna สำหรับ tuning
- ประเมิน XGBoost/LightGBM/RandomForest/Logistic และเลือก XGBoost_SMOTE_Optuna

## Code / Artifact Reference
- `docs\version 4\scripts\clean_dataset_v4.py`
- `docs\version 4\scripts\run_v4_generated_end_to_end_pipeline.py`

## Decision
ใช้เป็น benchmark ของ generated data แต่ยังไม่เลือก final เพราะ V2 ให้ balance ดีกว่า

Generated at: 2026-05-28T15:24:36