# Model Use - LightGBM V5

โฟลเดอร์นี้รวมไฟล์สำคัญสำหรับส่งต่อโมเดล **LightGBM V5** เพื่อใช้งานต่อ
โดยยึดชุดหลัก:

- Train/Clean Dataset: `SETC S1 clean_dataset_s1.csv` จำนวน 5,000 rows
- External/Real-like Test Dataset: `SETD S3 real_dataset_s1.csv` จำนวน 55,000 rows
- Selected Model: `LightGBM V5`

## 01_Model_Artifacts

ไฟล์ที่จำเป็นที่สุดสำหรับนำโมเดลไป predict:

- `models/model_lgbm_s1_v5_lightgbm.pkl` = ไฟล์โมเดลตัวจริง
- `models/model_lgbm_s1_v5_metadata.json` = ค่า threshold, parameter, metric และรายละเอียดโมเดล
- `features/used_features_lgbm_s1_v5.csv` = รายชื่อ feature 64 ตัวที่ต้องสร้างให้ตรงก่อน predict
- `features/train_validation_holdout_sets_lgbm_s1_v5.pkl` = schema/split artifact เดิม
- `features/df_featured_lgbm_s1_v5.csv` = dataset หลังทำ feature engineering

## 02_Datasets

ไฟล์ข้อมูลอ้างอิง:

- `clean_dataset_s1.csv` = clean dataset 5,000 rows ที่ใช้ train
- `real_dataset_s1.csv` = real-like dataset 55,000 rows ที่ใช้ทดสอบโมเดลทั้งก้อน

## 03_Evaluation_Reports

ไฟล์ผลการประเมิน:

- `holdout_reports/` = ผล test จาก holdout ของ SETC S1
- `external_test_reports/` = ผล test จาก SETD S3
- `lgbm_s1_v1_to_v5_external_summary.csv` = ตารางเปรียบเทียบ V1-V5
- `images/` = กราฟ Accuracy

## 04_User_Manual

คู่มือภาษาไทยสำหรับส่งต่องาน:

- `LightGBM_V5_User_Manual_TH.docx`

## 05_Environment

ไฟล์สำหรับเตรียม environment:

- `requirements.txt` = Python package ที่ต้องติดตั้ง
- `docker-compose.yml` = ใช้รัน PostgreSQL/service ที่เกี่ยวข้อง

หมายเหตุ: ไม่ได้คัดลอกไฟล์ `.env` มาไว้ในนี้ เพราะอาจมีรหัสผ่านหรือค่า config ส่วนตัว
ถ้าจะส่งต่อจริง ควรสร้าง `.env.example` แล้วให้ผู้ใช้งานกรอกค่าของเครื่องตัวเอง
