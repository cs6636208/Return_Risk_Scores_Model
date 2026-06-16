# Return Risk Prediction Project

Updated: 2026-06-16

โปรเจ็กต์นี้เป็นระบบทดลองสำหรับ **ทำนายความเสี่ยงการคืนสินค้า (Return Risk Prediction)** ของบริบทงานขายสินค้าแบบ O Shopping / Commerce โดยใช้ข้อมูล order, customer, product, courier, promotion, payment, channel และประวัติการคืนสินค้าของลูกค้า เพื่อสร้าง feature แล้วนำไป train model สำหรับช่วยประเมินว่า order ใหม่มีโอกาสคืนสินค้าสูงหรือต่ำ

สถานะล่าสุดของโปรเจ็กต์ตอนนี้เลือกโมเดลหลักสำหรับส่งต่องานเป็น:

```text
LightGBM V5
Train/Clean Dataset: SETC S1 clean_dataset_s1.csv จำนวน 5,000 rows
External/Real-like Test Dataset: SETD S3 real_dataset_s1.csv จำนวน 55,000 rows
```

## สรุปสถานะปัจจุบัน

| หัวข้อ | รายละเอียด |
| --- | --- |
| Selected model | LightGBM V5 |
| Main train dataset | `docs/LightGBM/SETC/clean_dataset/clean_dataset_s1.csv` |
| Train rows | 5,000 rows |
| External test dataset | `docs/LightGBM/SETD/real_dataset/S3/real_dataset_s1.csv` |
| External test rows | 55,000 rows |
| Feature count | 64 features |
| Target | `is_returned` |
| Main output folder for handoff | `Model Use/` |
| User manual | `Model Use/04_User_Manual/LightGBM_V5_User_Manual_TH.docx` |

## ทำไมเลือก LightGBM V5

LightGBM V5 ถูกเลือกเป็น candidate หลัก เพราะเป็น version ที่สมดุลที่สุดระหว่าง:

- Accuracy อยู่ในระดับใช้งานต่อได้
- ผลทดสอบบน clean dataset และ real-like dataset ใกล้เคียงกัน
- จำนวน feature ไม่เยอะเกินไปเมื่อเทียบกับ V4
- เหมาะเป็น baseline สำหรับนำไปต่อยอดเป็น production model
- รองรับแนวคิด real-time inference ได้ เพราะ feature หลักเป็นข้อมูลที่เตรียมจากประวัติก่อนหน้าและ snapshot ที่เกี่ยวข้อง

## ผลการประเมิน LightGBM V5

| Evaluation | Dataset | Rows | Accuracy | Recall | Precision | F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Holdout Test | SETC S1 | 1,000 | 82.00% | 59.56% | 78.84% | 67.86% | 83.27% |
| External Test | SETD S3 | 55,000 | 81.91% | 61.24% | 77.38% | 68.37% | 82.59% |

สรุป: ผลระหว่าง clean dataset และ real-like dataset ใกล้กัน จึงเหมาะใช้เป็น baseline สำหรับส่งต่องาน แต่ยังต้อง retrain ด้วยข้อมูลจริงของบริษัทก่อนนำไป production จริง

## โครงสร้างไฟล์สำคัญ

```text
return-risk-prediction/
  Model Use/
    01_Model_Artifacts/
    02_Datasets/
    03_Evaluation_Reports/
    04_User_Manual/
    05_Environment/
  docs/
    LightGBM/
      SETC/
      SETD/
  scripts/
  data/
  reports/
  docker-compose.yml
  requirements.txt
```

## โฟลเดอร์ Model Use สำหรับส่งต่องาน

โฟลเดอร์ `Model Use/` ถูกจัดไว้สำหรับให้คนอื่นรับงานต่อได้ง่าย โดยรวมไฟล์สำคัญของโมเดล LightGBM V5 ไว้แล้ว

### 1. Model Artifacts

```text
Model Use/01_Model_Artifacts/
  models/
    model_lgbm_s1_v5_lightgbm.pkl
    model_lgbm_s1_v5_metadata.json
  features/
    used_features_lgbm_s1_v5.csv
    train_validation_holdout_sets_lgbm_s1_v5.pkl
    df_featured_lgbm_s1_v5.csv
```

ความหมาย:

- `model_lgbm_s1_v5_lightgbm.pkl` = ไฟล์โมเดลตัวจริง
- `model_lgbm_s1_v5_metadata.json` = threshold, parameter, metric และข้อมูลประกอบโมเดล
- `used_features_lgbm_s1_v5.csv` = รายชื่อ feature 64 ตัวที่ต้องสร้างให้ตรงก่อน predict
- `df_featured_lgbm_s1_v5.csv` = dataset หลังทำ feature engineering

### 2. Datasets

```text
Model Use/02_Datasets/
  clean_dataset_s1.csv
  real_dataset_s1.csv
```

- `clean_dataset_s1.csv` = clean dataset 5,000 rows ที่ใช้ train
- `real_dataset_s1.csv` = real-like dataset 55,000 rows ที่ใช้ external test

### 3. Evaluation Reports

```text
Model Use/03_Evaluation_Reports/
  holdout_reports/
  external_test_reports/
  images/
  lgbm_s1_v1_to_v5_external_summary.csv
  lgbm_s1_v1_to_v5_external_summary.json
```

ใช้ตรวจสอบผลลัพธ์ของโมเดล เช่น Accuracy, Recall, F1, AUC และ prediction output

### 4. User Manual

```text
Model Use/04_User_Manual/
  LightGBM_V5_User_Manual_TH.docx
```

คู่มือภาษาไทยสำหรับคนรับงานต่อ อธิบายว่าโมเดลใช้งานอย่างไร ถ้าทำนายพลาดควรแก้ตรงไหน และถ้ามีข้อมูลจริงต้อง retrain/map feature อย่างไร

### 5. Environment

```text
Model Use/05_Environment/
  docker-compose.yml
  .env.example
  requirements.txt
  README_DOCKER_POSTGRES.md
```

ใช้สำหรับเปิด PostgreSQL ด้วย Docker และเตรียม environment สำหรับคนรับงานต่อ

## วิธีเปิด PostgreSQL ด้วย Docker

เข้าโฟลเดอร์ environment:

```powershell
cd "Model Use\05_Environment"
```

สร้างไฟล์ `.env` จากตัวอย่าง:

```powershell
copy .env.example .env
```

เปิด PostgreSQL และ pgAdmin:

```powershell
docker compose up -d
```

ตรวจว่า container ทำงาน:

```powershell
docker ps
```

ข้อมูลเชื่อมต่อจากเครื่อง:

```text
Host: 127.0.0.1
Port: 5433
Database: gmm_oshopping_db
User: admin
Password: ดูจากค่า DB_PASS ใน .env
```

pgAdmin:

```text
URL: http://localhost:5050
```

อ่านรายละเอียดเต็มได้ที่:

```text
Model Use/05_Environment/README_DOCKER_POSTGRES.md
```

## แนวคิด Production Flow

เมื่อมี order ใหม่เข้ามา ระบบ production ควรทำงานแบบนี้:

```text
Order ใหม่ 1 record
  -> Query เฉพาะ customer_id ที่เกี่ยวข้อง
  -> Query product/category/courier/payment/channel snapshot
  -> Build feature 1 row ให้ตรงกับ used_features_lgbm_s1_v5.csv
  -> Predict ด้วย LightGBM V5
  -> แสดงผล Low / Medium / High Risk
```

ระบบจริงไม่ควรคำนวณทั้ง dataset ทุกครั้ง แต่ควรใช้ Feature Store หรือ table snapshot เพื่อเก็บค่าที่คำนวณไว้ล่วงหน้า เช่น customer return history, product return rate, category return rate และ courier risk

## Feature หลักของ V5

V5 ใช้ 64 features โดยแบ่งกลุ่มได้ประมาณนี้:

- ข้อมูลลูกค้า เช่น `age`, `membership_tier`, `province`
- ข้อมูลสินค้า/order เช่น `category`, `brand`, `quantity`, `total_amount`
- ประวัติลูกค้า เช่น `total_orders_before`, `total_returns_before`, `customer_return_ratio`
- ประวัติย้อนหลัง เช่น `hist_return_rate_7d`, `hist_return_rate_30d`, `hist_return_rate_90d`, `hist_return_rate_365d`
- ความเสี่ยงสินค้า/หมวด/แบรนด์ เช่น `product_return_rate_pti`, `category_return_rate_pti`, `brand_return_rate_pti`
- ความเสี่ยงช่องทาง/ชำระเงิน/ขนส่ง เช่น `payment_return_rate_pti`, `channel_return_rate_pti`, `courier_return_rate_pti`
- feature ผสม เช่น `category_payment`, `category_channel`, `high_discount_cod`, `low_rating_high_discount`

## ถ้ามีข้อมูลจริงของบริษัทเข้ามา ต้องทำอะไรต่อ

เมื่อมีข้อมูลจริงจากบริษัท ไม่ควรเอาโมเดลเดิมไปใช้ทันทีโดยไม่ตรวจ ควรทำตามขั้นตอนนี้:

1. ตรวจ schema ของข้อมูลจริง เช่น order, customer, product, return, courier, payment
2. ทำ Data Dictionary เพื่อรู้ว่าแต่ละ column หมายถึงอะไร
3. Map column จริงให้ตรงกับ feature ที่ V5 ต้องใช้
4. ตรวจ missing, null, duplicate, outlier
5. สร้าง feature ด้วยสูตรเดียวกับ V5
6. ทดสอบ V5 เดิมกับข้อมูลจริงเพื่อดู baseline
7. Retrain LightGBM ด้วยข้อมูลจริง
8. Tune threshold และ parameter ใหม่
9. เปรียบเทียบ V5 เดิม vs V5 retrain หรือ V6
10. เลือกโมเดลสุดท้ายก่อนขึ้น production

## สถานะความสมบูรณ์ของโปรเจ็กต์

สิ่งที่พร้อมแล้ว:

- Dataset สำหรับ train/test แบบทดลอง
- Feature Engineering หลาย version
- LightGBM V1-V5
- Selected model: LightGBM V5
- Holdout test และ external test
- คู่มือภาษาไทยสำหรับส่งต่องาน
- ชุดไฟล์ `Model Use/`
- Docker/PostgreSQL handoff guide

สิ่งที่ยังต้องทำก่อน production จริง:

- ต่อ API เช่น FastAPI สำหรับรับ order ใหม่
- ทำ Feature Builder สำหรับสร้าง feature 1 row ใน production
- ทำ Feature Store ใน PostgreSQL
- Map ข้อมูลจริงของบริษัทกับ feature ของ V5
- Retrain ด้วยข้อมูลจริง
- ตั้งระบบ monitoring ว่าโมเดลทำนายพลาดตรงไหน
- เก็บ feedback หลัง order จบจริงเพื่อ retrain รอบต่อไป

## คำสั่ง Python เบื้องต้น

ติดตั้ง package:

```powershell
pip install -r requirements.txt
```

หมายเหตุ: ถ้าใช้ environment ใน `Model Use/05_Environment` ให้ใช้:

```powershell
pip install -r "Model Use\05_Environment\requirements.txt"
```

## ข้อควรระวัง

- ห้ามส่งไฟล์ `.env` ที่มี password จริงขึ้น GitHub
- ถ้าเป็นข้อมูลจริงของบริษัท ไม่ควร commit CSV จริงลง repo
- ไฟล์ CSV ขนาดใหญ่ควรใช้ Git LFS หรือเก็บแยกนอก repo
- Accuracy จาก dataset ทดลองไม่ใช่คำรับประกันว่า production จริงจะได้เท่ากัน
- เมื่อข้อมูลจริงมา ต้อง retrain และ validate ใหม่เสมอ

## สรุปสำหรับส่งต่องาน

ถ้าคนอื่นจะรับงานต่อ ให้เริ่มจาก:

```text
Model Use/README.md
Model Use/04_User_Manual/LightGBM_V5_User_Manual_TH.docx
Model Use/05_Environment/README_DOCKER_POSTGRES.md
```

โมเดลที่เลือกคือ:

```text
Model Use/01_Model_Artifacts/models/model_lgbm_s1_v5_lightgbm.pkl
```

รายชื่อ feature ที่ต้องสร้างให้ตรงคือ:

```text
Model Use/01_Model_Artifacts/features/used_features_lgbm_s1_v5.csv
```
