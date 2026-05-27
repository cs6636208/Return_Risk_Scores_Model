# version 2 df_featured Export

## Source ก่อนทำ Feature

- `data/processed/clean_dataset.csv`

## Engineered Dataset ที่ใช้อ้างอิง

- `docs/version 2/data/features/df_engineered_v2_preview.csv + rolling features from train_v2_optimized_model.py`

## Output หลังทำ Feature

- `data/features/df_featured.csv`
- rows: `5000`
- model input features: `60`
- target: `is_returned`
- split column: `dataset_split`

## Files ในโฟลเดอร์นี้

- `source_data_summary.csv`: อธิบายว่า data มากจากไหนและ output คืออะไร
- `used_features.csv`: feature ที่ใช้เข้า model
- `dropped_or_not_used_features.csv`: feature ที่ตัดทิ้งหรือไม่ใช้ เทียบกับ clean/engineered
- `df_featured_schema.csv`: dtype, missing, unique, sample values ของ df_featured
- `df_featured_preview.csv`: ตัวอย่าง 30 rows แรก สำหรับเปิดดูเร็วใน Excel

## Note

V2 selected export เป็น order-time-safe raw feature frame 60 features ตัด delivery_days/delay_days และเพิ่ม rolling history 30/60/90/180/365d
