# version 1 df_featured Export

## Source ก่อนทำ Feature

- `data/processed/clean_dataset.csv`

## Engineered Dataset ที่ใช้อ้างอิง

- `docs/version 1/data/features/df_engineered.csv`

## Output หลังทำ Feature

- `data/features/df_featured.csv`
- rows: `6672`
- model input features: `136`
- target: `is_returned`
- split column: `dataset_split`

## Files ในโฟลเดอร์นี้

- `source_data_summary.csv`: อธิบายว่า data มากจากไหนและ output คืออะไร
- `used_features.csv`: feature ที่ใช้เข้า model
- `dropped_or_not_used_features.csv`: feature ที่ตัดทิ้งหรือไม่ใช้ เทียบกับ clean/engineered
- `df_featured_schema.csv`: dtype, missing, unique, sample values ของ df_featured
- `df_featured_preview.csv`: ตัวอย่าง 30 rows แรก สำหรับเปิดดูเร็วใน Excel

## Note

V1 export เป็น encoded/scaled model feature set 136 features จาก train_test_sets.pkl จึงเป็น feature ที่เข้า model จริง
