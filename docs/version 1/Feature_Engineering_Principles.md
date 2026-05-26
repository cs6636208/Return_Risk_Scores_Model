---
title: หลักการทำ Feature Engineering และการคัดเลือก Feature
---
<style>
body { font-family: 'Tahoma', 'Sarabun', sans-serif; line-height: 1.6; }
h1, h2 { text-align: center; }
h2 { color: #555; }
.example { background-color: #f4f4f4; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 10px; font-family: monospace; }
.reason { background-color: #fdfdfd; padding: 10px; border-left: 4px solid #ffc107; margin-bottom: 20px; }
</style>

# หลักการทำ Feature Engineering และการคัดเลือก Feature
## เปรียบเทียบระหว่าง clean_dataset.csv และ df_engineered

---

### 1. ภาพรวมและวัตถุประสงค์ (Overview)
กระบวนการ **Feature Engineering** เป็นการเตรียมและคัดเลือกข้อมูลเพื่อนำไปสอน Machine Learning Model โดยเป้าหมายหลักคือการทำนายว่าคำสั่งซื้อจะถูกส่งคืนหรือไม่ (`is_returned`) เราเริ่มต้นจากข้อมูลดิบที่ผ่านการทำความสะอาดแล้ว (`clean_dataset.csv`) นำมาแปลงรูปแบบ สร้างตัวแปรใหม่ และตัดตัวแปรที่ไม่จำเป็นออก จนได้ชุดข้อมูลสุดท้ายคือ `df_engineered`

### 2. กลุ่ม Feature ที่ถูกตัดทิ้ง (Dropped Features)
หลักการในการพิจารณาตัด Feature ทิ้ง แบ่งออกเป็น 4 ประเภทหลัก ได้แก่:

#### 2.1. ข้อมูลระบุตัวตน (Identifiers & PII)
<div class="example">ตัวอย่าง: order_id, customer_id, customer_name, customer_phone, product_id, supplier_id</div>
<div class="reason"><strong>เหตุผล:</strong> ข้อมูลเหล่านี้เป็นข้อมูลเฉพาะเจาะจงของบุคคลหรือสิ่งของ ไม่ช่วยให้โมเดลเรียนรู้รูปแบบ (Pattern) ในภาพรวมได้ และหากนำไปสอนโมเดลอาจทำให้เกิดปัญหา Overfitting (จำข้อมูลได้แต่ทำนายของใหม่ไม่เป็น)</div>

#### 2.2. ข้อมูลในอนาคต หรือ ข้อมูลที่หลุดรอด (Data Leakage)
<div class="example">ตัวอย่าง: return_id, return_date, return_reason, item_condition, return_status, refund_amount, delivery_days</div>
<div class="reason"><strong>เหตุผล:</strong> ข้อมูลเหล่านี้จะเกิดขึ้นหรือรับรู้ได้ก็ต่อเมื่อ "มีการจัดส่งหรือคืนสินค้าไปแล้ว" การนำข้อมูลที่เกิดทีหลังมาใช้ทำนายล่วงหน้าถือเป็นการโกง (Leakage) เพราะในโลกความเป็นจริงตอนที่ลูกค้าเพิ่งกดสั่งซื้อ เราจะยังไม่มีข้อมูลเหล่านี้เลย</div>

#### 2.3. ผลลัพธ์จากโมเดลเดิม (Model Outputs)
<div class="example">ตัวอย่าง: risk_score, risk_tier, shap_values</div>
<div class="reason"><strong>เหตุผล:</strong> เป็นผลลัพธ์ที่ได้จากการคำนวณของโมเดลตัวอื่น ไม่ใช่ข้อมูลดิบตั้งต้น หากนำมาใช้จะทำให้เกิดความซ้ำซ้อนและไม่ได้ประโยชน์เชิงพฤติกรรม</div>

#### 2.4. ข้อมูลวันที่ดิบๆ (Raw Dates)
<div class="example">ตัวอย่าง: order_date, expected_delivery_date</div>
<div class="reason"><strong>เหตุผล:</strong> โมเดล Machine Learning แบบตารางมักไม่สามารถเข้าใจวันที่ตรงๆ ได้ จึงถูกตัดออกและนำไปสกัดเป็นช่วงเวลา หรือระยะห่างแทน (เช่น แปลงเป็น order_hour, days_since_last_order)</div>

### 3. กลุ่ม Feature ที่ถูกเก็บไว้และแปลงรูป (Kept & Transformed Features)
    
#### 3.1. ข้อมูลการสั่งซื้อหลัก
<div class="example">ได้แก่: quantity, unit_price, total_amount, payment_method, channel_type, category</div>
<div class="reason"><strong>การแปลงรูป:</strong> มีการทำ Log Transformation (เช่น <code>log_unit_price</code>, <code>log_total_amount</code>) เพื่อลดผลกระทบจากข้อมูลที่มีราคาสูงผิดปกติ (Outliers) ทำให้กราฟการกระจายตัวของข้อมูลมีความสมดุลมากขึ้น</div>

#### 3.2. ข้อมูลส่วนลด (Discounts)
<div class="example">ได้แก่: promo_discount_pct</div>
<div class="reason"><strong>การแปลงรูป:</strong> สร้าง Flag แจ้งเตือนเช่น <code>is_high_discount</code> (ส่วนลดมากกว่า 20%) เนื่องจากในเชิงธุรกิจพบว่าส่วนลดที่สูงมากมักกระตุ้นให้เกิดการซื้อแบบไม่ยั้งคิด และมีโอกาสขอคืนสินค้าในภายหลัง</div>

### 4. กลุ่ม Feature ที่สร้างขึ้นมาใหม่ (Engineered Features)
เรามีการสร้างข้อมูลใหม่จากความรู้เชิงธุรกิจ (Business Domain) แบ่งเป็น 4 กลุ่มหลัก:

#### 4.1. ข้อมูลสถิติย้อนหลัง (Point-in-Time Historical)
<div class="example">ได้แก่: total_orders_before, total_returns_before, customer_return_ratio, return_rate_by_category</div>
<div class="reason"><strong>เหตุผล:</strong> คำนวณจากประวัติลูกค้าหรือสินค้า <strong>"ณ วันก่อนที่จะเกิดคำสั่งซื้อปัจจุบัน"</strong> เท่านั้น เพื่อดักจับพฤติกรรมลูกค้าที่ชอบคืนสินค้าเป็นประจำ (Serial Returners) โดยรับประกันว่าจะไม่เกิด Data Leakage</div>

#### 4.2. ข้อมูลเชิงเวลา (Time-based)
<div class="example">ได้แก่: order_hour, order_dayofweek, is_weekend, is_peak_hour</div>
<div class="reason"><strong>เหตุผล:</strong> ช่วยดักจับพฤติกรรมการซื้อ เช่น การซื้อดึกๆ อาจมีความเสี่ยงในการคืนมากกว่าซื้อตอนกลางวัน</div>

#### 4.3. เงื่อนไขทางธุรกิจและความเสี่ยง (Business Rules / Risk Flags)
<div class="example">ได้แก่: is_long_distance_cod, is_impulse_buy, is_low_commitment, is_bracketing</div>
<div class="reason"><strong>เหตุผล:</strong> เป็นการจับคู่เงื่อนไขที่สะท้อนบริบททางธุรกิจ เช่น <code>is_long_distance_cod</code> (เก็บเงินปลายทางแต่ส่งข้ามจังหวัดไกลๆ) หรือ <code>is_bracketing</code> (ซื้อสินค้าแฟชั่นหลายชิ้นเพื่อลองไซส์แล้วส่งคืน) ซึ่งให้ประสิทธิภาพการทำนายที่แม่นยำกว่าการใช้ตัวแปรเดี่ยวๆ</div>

#### 4.4. การจับคู่ตัวแปร (Interaction Features)
<div class="example">ได้แก่: category_payment, category_channel, gender_province</div>
<div class="reason"><strong>เหตุผล:</strong> พฤติกรรมบางอย่างเกิดจากองค์ประกอบร่วมกัน เช่น การซื้อสินค้าแฟชั่นผ่านช่องทางทีวี (<code>is_fashion_tv</code>) อาจมีอัตราการคืนที่แตกต่างจากการซื้อเครื่องใช้ไฟฟ้าผ่านแพลตฟอร์มอื่น เป็นต้น</div>
