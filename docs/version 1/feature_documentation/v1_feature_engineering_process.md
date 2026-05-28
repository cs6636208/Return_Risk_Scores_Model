# Version 1 - Baseline XGBoost - Feature Engineering Process

## Process
- อ่าน clean_dataset.csv เป็น source หลัก
- clean missing/outlier/duplicate จาก dataset ที่เตรียมไว้
- สร้าง feature พื้นฐาน เช่น order_hour, price/log price, customer/product/category fields และ encoded categorical columns
- drop target/post-event/identifier columns ก่อน train
- train XGBoost เป็น baseline เพื่อวัดว่าฟีเจอร์ชุดแรกให้ performance เท่าไร

## Code / Artifact Reference
- `docs\version 1\feature_engineering.py`
- `docs\version 1\model_training.py`
- `docs\version 1\model_evaluation.py`

## Decision
ใช้เป็น baseline เท่านั้น เพราะ recall ต่ำและพลาด return cases มาก

Generated at: 2026-05-28T16:26:09