# Business Insight Report — กราฟที่เลือกและ Feature ที่นำไปใช้

**โปรเจกต์:** GMM O Shopping — Return Risk Prediction  
**PDF:** [`reports/business_insights_report.pdf`](../../reports/business_insights_report.pdf)  
**สร้างใหม่:** `python scripts/export_business_insights_pdf.py`

---

## รายการกราฟที่เลือก (10 รายการ)

| ลำดับ | กราฟ | ไฟล์ |
|------|------|------|
| 1 | Return Rate by Channel / Payment / Category / Province / Tier | `01_categorical_impact.png` |
| 2 | Category × Payment | `02_cross_category_payment.png` |
| 3 | Province × Gender | `03_cross_province_gender.png` |
| 4 | Province × Payment | `04_cross_province_payment.png` |
| 5 | Top 20 Feature Importance (RF) | `05_feature_importance.png` |
| 6–10 | Return Rate by Province แยกหมวด | Cosmetics, Fashion, Electronics, Home_Appliance, Supplement |

---

## หลักการเลือก Insight → Feature

1. **Variance ชัด** — อัตราคืนต่างกันระหว่างกลุ่ม  
2. **ใช้ได้ตอนรับออเดอร์** — ไม่ใช้ข้อมูลหลังส่ง/หลังคืน (leakage)  
3. **แปลงเป็น feature ได้** — อยู่ใน `feature_engineering.py`  
4. **ยืนยันด้วยโมเดล** — สอดคล้อง Feature Importance

---

## สรุป Feature จาก Insight ทั้งหมด

| กลุ่ม Insight | Feature ในโมเดล |
|--------------|-----------------|
| ช่องทาง / จ่าย / หมวด / จังหวัด / Tier | `channel_type_*`, `payment_method_*`, `category_*`, `province_*`, `membership_tier` |
| Interaction | `category_payment_*`, `category_channel_*`, `province_payment_*`, `gender_province_*` |
| โปร / COD | `promo_discount_pct`, `is_high_discount`, `is_cod`, `is_low_commitment` |
| Fashion + TV | `is_fashion_tv`, `is_impulse_buy`, `is_peak_hour` |
| ประวัติลูกค้า | `customer_return_ratio`, `hist_*`, `days_since_last_return` |
| สินค้า / โลจิสติก | `product_rating`, `return_rate_by_category`, `delivery_time_expected_days` |

---

*รายละเอียดแต่ละกราฟอยู่ใน PDF ฉบับเต็ม*
