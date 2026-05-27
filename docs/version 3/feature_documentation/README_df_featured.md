# version 3 df_featured Export

## Source ก่อนทำ Feature

- `docs/version 2/data/features/train_test_sets_v2.pkl`

## Engineered Dataset ที่ใช้อ้างอิง

- `docs/version 2/data/features/df_engineered_v2_preview.csv`

## Output หลังทำ Feature

- `data/features/df_featured.csv`
- rows: `5000`
- model input features: `38`
- target: `is_returned`
- split column: `dataset_split`

## Files ในโฟลเดอร์นี้

- `source_data_summary.csv`: อธิบายว่า data มากจากไหนและ output คืออะไร
- `used_features.csv`: feature ที่ใช้เข้า model
- `dropped_or_not_used_features.csv`: feature ที่ตัดทิ้งหรือไม่ใช้ เทียบกับ clean/engineered
- `df_featured_schema.csv`: dtype, missing, unique, sample values ของ df_featured
- `df_featured_preview.csv`: ตัวอย่าง 30 rows แรก สำหรับเปิดดูเร็วใน Excel

## Note

V3 ไม่ได้สร้าง feature ใหม่เอง แต่ reuse V2 baseline feature set 38 features เพื่อทดลอง model architecture แบบ stacking
