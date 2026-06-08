# Version 2 - Current Production Candidate

Current V2 คือ `v2_xgboost_safe_plus_rolling`

- Dataset: `clean_dataset.csv`
- Model: XGBoost safe plus rolling
- Policy: order-time safe, ตัด `delivery_days` และ `delay_days`
- Rolling history: 30/60/90/180/365 วัน
- ใช้ไฟล์ metric จาก `docs/version 2/v2_xgboost_safe_plus_rolling/reports`

หมายเหตุ: `v2_xgboost_safe_plus_rolling_HIGH_ACCURACY` ไม่ใช่ V2 current แล้ว ถูกแยกไปเป็น Version 5 archive/experiment.
