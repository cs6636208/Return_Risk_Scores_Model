# Business Insight + Feature Mapping Summary

รายงานนี้คัด insight จากชุดกราฟ EDA ที่มีอยู่ในโปรเจ็กต์ โดยเลือกเฉพาะกราฟที่ผูกกับ feature engineering และ model decision ได้จริง

- จำนวนกราฟ EDA ทั้งหมดที่ตรวจใน reports/business_insights, reports/Graph Item, reports/Graph Relation Feature: 45 รูป
- จำนวน insight ที่คัดเลือกมาทำ feature mapping: 10 ชุด
- Output chart folder: `docs\analysis\business_insight_selected_charts`

| No | Insight | Chart | Feature ที่ตามมา | เหตุผลที่เลือก | สิ่งที่เอาไปใช้ |
| --- | --- | --- | --- | --- | --- |
| 1 | Customer return history | `docs\analysis\business_insight_selected_charts\01_customer_return_history.png` | hist_order_count, hist_return_rate, customer_return_ratio, cust_return_rate_30d, cust_return_rate_180d | เลือกเพราะเชื่อมกับโจทย์ order ที่ 3 โดยใช้ order ก่อนหน้าเท่านั้น เช่น 1/2 = 0.5 | ใช้จัดระดับความเสี่ยงก่อนยืนยันหรือจัดส่ง และใช้ทำ Feature Store รายลูกค้า |
| 2 | Category and payment interaction | `docs\analysis\business_insight_selected_charts\02_category_and_payment_interaction.png` | category, payment_method, category_payment, is_cod | เลือกเพราะช่วยให้ model เห็น interaction ไม่ใช่ดู category หรือ payment แบบแยกกัน | ใช้เพิ่ม rule/feature สำหรับ COD หรือ category ที่เสี่ยงสูง |
| 3 | Category and channel interaction | `docs\analysis\business_insight_selected_charts\03_category_and_channel_interaction.png` | category, channel_type, category_channel, preferred_channel | เลือกเพราะ channel สะท้อนพฤติกรรมซื้อและความตั้งใจซื้อ เช่น TV/TikTok/App | ใช้จัด workflow confirm order แยกตาม channel |
| 4 | Province and payment risk | `docs\analysis\business_insight_selected_charts\04_province_and_payment_risk.png` | province, payment_method, province_payment | เลือกเพราะ province เป็นตัวแทน operational context เช่น พื้นที่ไกลหรือพื้นที่มีปัญหาคืนสูง | ใช้ให้ทีม operation เฝ้ากลุ่มจังหวัด/ช่องทางชำระเงินที่มี risk สูง |
| 5 | Product rating threshold | `docs\analysis\business_insight_selected_charts\05_product_rating_threshold.png` | product_rating, low_rating_alert | เลือกเพราะสร้าง feature binary ได้ง่ายและอธิบายเชิงธุรกิจชัด | ใช้ flag สินค้าที่ควรตรวจคุณภาพหรือให้ข้อมูลสินค้าเพิ่มก่อนขาย |
| 6 | Promotion and discount behavior | `docs\analysis\business_insight_selected_charts\06_promotion_and_discount_behavior.png` | promo_type, promo_discount_rate, total_discount_pct, discount_amount_ratio, is_high_discount | เลือกเพราะมีข้อมูลก่อน order เข้าและใช้กับ real-time scoring ได้ | ใช้แยก order ที่เกิดจาก promotion แรงเพื่อประเมิน risk เพิ่ม |
| 7 | Price and basket amount | `docs\analysis\business_insight_selected_charts\07_price_and_basket_amount.png` | unit_price, quantity, total_amount, log_unit_price, log_total_amount | เลือกเพราะเป็นข้อมูลพื้นฐานที่รู้ตอน order เข้าและสัมพันธ์กับ expected loss | ใช้ประกอบ threshold/cost matrix เมื่อ order มูลค่าสูง |
| 8 | Logistics expected risk | `docs\analysis\business_insight_selected_charts\08_logistics_expected_risk.png` | courier_type, avg_delivery_days, delivery_time_expected_days, damage_rate, logistics_risk | เลือกเฉพาะ feature ที่รู้ก่อนส่งสินค้า ส่วน delivery_days/delay_days เป็น post-event ต้องแยกใช้ | ใช้กับ order-time model โดยไม่ใช้ข้อมูลอนาคต |
| 9 | Repurchase behavior | `docs\analysis\business_insight_selected_charts\09_repurchase_behavior.png` | is_repurchased_item, days_since_last_order | เลือกเพราะเป็น signal ที่คำนวณจาก history ลูกค้าได้แบบ point-in-time | ใช้ประกอบ customer history เพื่อประเมินความมั่นใจของ order ใหม่ |
| 10 | Order time and weekend | `docs\analysis\business_insight_selected_charts\10_order_time_and_weekend.png` | order_hour, order_dayofweek, is_weekend | เลือกเพราะต้นทุน query ต่ำและรู้ได้ทันทีตอน order เข้า | ใช้เป็น feature เสริม ไม่ควรเป็นตัวหลักถ้า performance ไม่เพิ่ม |

## Leakage Note

- Feature ที่รู้หลังเหตุการณ์ เช่น `delivery_days`, `delay_days`, `return_date`, `refund_amount`, `return_reason`, `risk_score`, `risk_tier`, `shap_values` ไม่ควรใช้กับ real-time order-time model
- ถ้าใช้ feature กลุ่ม post-delivery ต้องระบุว่าเป็น model สำหรับ post-delivery scoring หรือ experiment เท่านั้น