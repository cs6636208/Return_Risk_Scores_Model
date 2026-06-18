# วิธีรัน Test / Verify LightGBM V5

โฟลเดอร์นี้มีไฟล์ Python สำหรับตรวจสอบผลลัพธ์ของโมเดล LightGBM V5

```text
run_verify_v5_reference_metrics.py
run_test_v5_lightgbm.py
```

## 1. ไฟล์ที่ใช้ตรวจค่าให้ตรงรูป

ใช้ไฟล์นี้:

```text
run_verify_v5_reference_metrics.py
```

คำสั่ง:

```powershell
python "โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทส\run_verify_v5_reference_metrics.py"
```

ไฟล์นี้ทำหน้าที่:

```text
อ่านค่า metrics/predictions เดิม
ไม่ train ใหม่
ไม่ predict ใหม่
ใช้ยืนยันว่าค่า V5 ตรงกับรูปตารางอ้างอิง
```

ค่าที่ควรแสดง:

```text
Version: V5
Features: 64
Clean Accuracy: 82.00%
Real Accuracy: 81.91%
Gap: -0.09 pp
```

Output:

```text
โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทส\outputs\v5_reference_metrics_verification.csv
โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทส\outputs\v5_reference_metrics_verification.json
```

## 2. ไฟล์ที่ใช้รัน test/predict

ใช้ไฟล์นี้:

```text
run_test_v5_lightgbm.py
```

คำสั่ง default:

```powershell
python "โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทส\run_test_v5_lightgbm.py"
```

คำสั่งนี้จะอ่าน prediction เดิมของ real dataset S3 แล้วสรุป metric ให้ดู

ถ้าจะโหลดโมเดล `.pkl` แล้ว predict จากไฟล์ feature ที่พร้อมเข้าโมเดล:

```powershell
python "โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทส\run_test_v5_lightgbm.py" --mode model
```

## 3. ไฟล์ train อยู่ที่ไหน

ไฟล์ train อยู่คนละโฟลเดอร์:

```text
โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทรน\run_train_v5_lightgbm.py
```

คำสั่ง:

```powershell
python "โมเดลที่จะเอาไปใช้ต่อ\ไฟล์รันเทรน\run_train_v5_lightgbm.py"
```

## 4. ความแตกต่างของ verify / test / train

| ไฟล์ | ใช้ทำอะไร | ผลจะตรงรูปไหม |
| --- | --- | --- |
| `run_verify_v5_reference_metrics.py` | ตรวจค่าอ้างอิงเดิมจาก artifact | ตรงรูป |
| `run_test_v5_lightgbm.py` | รัน test/predict หรืออ่าน prediction เดิม | ถ้าใช้ saved prediction จะตรง Real Accuracy |
| `run_train_v5_lightgbm.py` | train model ใหม่ | อาจใกล้เคียง แต่ไม่รับประกันว่าตรง 82.00% ทุกครั้ง |

## 5. เหตุผลที่ train ใหม่อาจไม่เท่ารูป

ค่าในรูปมาจาก experiment เดิมที่มี split, parameter, threshold และ package version เฉพาะรอบนั้น

ถ้า train ใหม่ อาจคลาดเคลื่อนจาก:

```text
การแบ่ง train/validation/holdout ใหม่
LightGBM version ต่างกัน
threshold search ใหม่
การจัดการ categorical feature ต่างกัน
random seed หรือ environment ต่างกัน
```

ดังนั้นถ้าต้องการยืนยันค่าตามรูป ให้ใช้:

```text
run_verify_v5_reference_metrics.py
```

ถ้าต้องการ retrain เพื่อใช้งานจริง ให้ใช้:

```text
run_train_v5_lightgbm.py
```

## 6. หมายเหตุสำคัญเรื่อง real_dataset_s1.csv

`real_dataset_s1.csv` ยังเป็น clean/raw dataset ไม่ใช่ feature set พร้อมเข้าโมเดล

ถ้าจะเอา `real_dataset_s1.csv` ไป predict ใหม่ ต้องทำ Feature Engineering ก่อน ให้ได้ feature 64 ตัวเหมือน V5

ไฟล์ที่พร้อมเข้าโมเดลทันทีคือ:

```text
โมเดลที่จะเอาไปใช้ต่อ\โมเดล\features\df_featured_lgbm_s1_v5.csv
```
