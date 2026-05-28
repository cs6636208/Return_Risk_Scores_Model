# Comparison Version

โฟลเดอร์นี้ถูกจัดใหม่ให้แยก Version 1-4 ชัดเจน พร้อมส่วนภาพรวมสำหรับเทียบ performance รวม

## Folder Structure
- `00_Overall_Comparison/` - เอกสารและกราฟเปรียบเทียบรวม V1-V4
- `01_Version_1_Baseline_XGBoost/` - เอกสาร/feature/code ของ V1
- `02_Version_2_XGBoost_Safe_Rolling_HIGH_ACCURACY/` - เอกสาร/feature/code/metrics ของ V2 candidate หลัก
- `03_Version_3_Stacking_Model/` - เอกสาร/feature/code ของ V3 stacking
- `04_Version_4_Generated_SMOTE_Optuna/` - เอกสาร/feature/code ของ V4 generated data

## Current Decision
เลือก `Version 2 - XGBoost Safe Rolling HIGH_ACCURACY` เป็น candidate หลัก เพราะได้ performance สูงสุดและมี feature logic แบบ order-time safe/rolling history ที่สอดคล้องกับโจทย์ return-risk มากที่สุด

ไฟล์เดิมระดับบนยังคงอยู่เพื่อไม่ให้ reference เก่าเสีย แต่ไฟล์ที่จัดระเบียบสำหรับอ่านจริงอยู่ใน subfolder ด้านบน

Generated at: 2026-05-28T15:24:46