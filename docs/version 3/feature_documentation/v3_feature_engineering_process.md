# Version 3 - Stacking Model from V2 Features - Feature Engineering Process

## Process
- นำ feature base จาก V2 มาใช้ต่อเพื่อทดสอบผลของ model architecture
- สร้าง stacking ensemble จาก XGBoost, LightGBM, CatBoost และ meta learner
- ใช้ train/test split จาก V2 เพื่อให้เทียบผลจาก model ได้ยุติธรรม
- ประเมิน threshold/cost/recall trade-off

## Code / Artifact Reference
- `docs\version 3\stacking_model_v3\scripts\model_training_v3_stacking.py`
- `docs\version 3\stacking_model_v3\scripts\model_evaluation_v3.py`

## Decision
ไม่เลือกเป็น final เพราะซับซ้อนกว่า แต่ performance ไม่ชนะ V2

Generated at: 2026-05-28T15:24:36