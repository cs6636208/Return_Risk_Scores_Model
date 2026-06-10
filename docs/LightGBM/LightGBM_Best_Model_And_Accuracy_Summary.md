# LightGBM Best Model And Accuracy Summary

เอกสารนี้สรุปผลล่าสุดหลังปรับ dataset ให้เป็น high-signal synthetic แบบสัดส่วนสมจริงมากขึ้น โดยไม่ balance เป็น 50/50 แล้ว

## Dataset Ratio ล่าสุด

| Dataset | Rows | Missing | Duplicate Order | Distinct Customers | Return Rate | Not Return Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SETC/S1 Train/Holdout Source | 5,000 | 0 | 0 | 1,169 | 32.84% | 67.16% |
| SETC/S2 Train/Holdout Source | 40,000 | 0 | 0 | 7,894 | 33.18% | 66.82% |
| SETD/S3 External Test Source | 10,000 | 0 | 0 | 1,198 | 33.50% | 66.50% |
| SETD/S4 External Test Source | 10,000 | 0 | 0 | 5,639 | 33.18% | 66.82% |

## ทำไมใช้สัดส่วนประมาณ 33% Return

- ใกล้ข้อมูลตั้งต้นของโปรเจ็กต์เดิมที่ return rate อยู่ประมาณ 29-30%
- สมจริงกว่า 50/50 เพราะในระบบขายจริง order ที่ไม่คืนควรมีมากกว่า order ที่คืน
- ยังมี positive class มากพอให้ LightGBM เรียนรู้เคส return ได้ ไม่ imbalance หนักเกินไป
- เหมาะกับโจทย์โปรเจ็กต์จบ เพราะอธิบายได้ว่าข้อมูลถูกปรับให้เสมือนจริง ไม่ได้ทำให้โมเดลง่ายเกินไป

## Feature Version ใหม่

| Version | Feature Count S2 | แนวคิด |
| --- | ---: | --- |
| V1 | 25 | Base order-time features: profile ลูกค้า, product, price, channel, payment, promotion, logistics expectation |
| V2 | 64 | Customer temporal history: เพิ่มประวัติลูกค้า, return ratio, spend/order behavior, rolling 7/30/60/90/180/365 วัน |
| V3 | 81 | Product/logistics risk: เพิ่ม product/category/brand/courier point-in-time risk และ quality/logistics score |
| V4 | 107 | Business interactions: เพิ่ม category-payment-channel-province interaction, bands, risk flags |
| V5 | 64 | Compact selected best: ตัดจาก V2-V4 ให้เหลือ feature ที่คุ้มและลด noise/resource |

## ผลลัพธ์หลัก

| Area | Best Version | Model | Feature Count | Accuracy | Recall | F1 | AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SETC/S1 holdout | V5 | LightGBM | 64 | 84.40% | 70.43% | 74.76% | 85.29% |
| SETC/S2 holdout | V3 | LightGBM | 81 | 87.20% | 76.00% | 79.75% | 88.69% |
| SETD/S3 external vs S1 | V4 | LightGBM | 107 | 85.16% | 73.55% | 76.86% | 87.73% |
| SETD/S4 external vs S2 | V3 | LightGBM | 81 | 87.23% | 76.52% | 79.91% | 88.83% |

## เลือก Model ไหนดีที่สุด

ถ้าวัดแบบ Accuracy-first ตามโจทย์:

- เลือก `SETC/S2 V3 LightGBM`
- External Accuracy on `SETD/S4`: `87.23%`
- เหตุผล: V3 เพิ่ม product/logistics risk แล้วทำ Accuracy บน external test ได้สูงสุด

ถ้าวัดแบบ return-risk business/cost:

- เลือก `SETC/S2 V5 LightGBM` เป็น candidate สำรอง
- External Accuracy on `SETD/S4`: `87.15%`
- Recall: `78.60%`
- F1: `80.23%`
- AUC: `88.96%`
- Cost ต่ำกว่า V3 เพราะจับเคส return ได้มากกว่า

## คำตัดสินแนะนำ

สำหรับรายงานหลักให้เลือก `SETC/S2 V3 LightGBM` เพราะชนะ Accuracy และอธิบาย feature strategy ได้ชัด

สำหรับ production/resource หรือ business-risk ให้เก็บ `SETC/S2 V5 LightGBM` เป็น alternative เพราะ performance ใกล้มาก ใช้ feature น้อยกว่า และ recall/F1 ดีกว่าเล็กน้อย

## ข้อควรระวัง

ตัวเลขนี้เป็น high-signal synthetic benchmark ยังไม่ใช่ production accuracy จากข้อมูลบริษัทจริง หากใช้ข้อมูลจริง ต้อง retrain และ test ใหม่ด้วย unseen real dataset
