# version 4 df_featured Export

## Source ก่อนทำ Feature

- `docs/version 4/data/processed/clean_dataset_v4_generated.csv`

## Engineered Dataset ที่ใช้อ้างอิง

- `docs/version 4/data/features/df_engineered_v4_generated.csv`

## Output หลังทำ Feature

- `data/features/df_featured.csv`
- rows: `9700`
- model input features: `180`
- target: `is_returned`
- split column: `dataset_split`

## Files ในโฟลเดอร์นี้

- `source_data_summary.csv`: อธิบายว่า data มากจากไหนและ output คืออะไร
- `used_features.csv`: feature ที่ใช้เข้า model
- `dropped_or_not_used_features.csv`: feature ที่ตัดทิ้งหรือไม่ใช้ เทียบกับ clean/engineered
- `df_featured_schema.csv`: dtype, missing, unique, sample values ของ df_featured
- `df_featured_preview.csv`: ตัวอย่าง 30 rows แรก สำหรับเปิดดูเร็วใน Excel

## Note

V4 export เป็น generated/synthetic model feature set 180 features จาก train_test_sets_v4_generated.pkl ไม่ควรเทียบ production ตรงกับข้อมูลจริง
