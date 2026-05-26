# 📈 รายงานแผนภูมิเปรียบเทียบการ Cross Categorical Features กับ Return Rate

รายงานฉบับนี้รวบรวมแผนภูมิแท่ง (Bar Charts) และแผนภูมิเส้น (Line Charts) ของตัวแปรกลุ่มสองตัวที่จับคู่ครอสกัน เพื่อเปรียบเทียบอัตราการคืนสินค้าเฉลี่ย (Return Rate) บนแกน Y ได้อย่างชัดเจนเชิงทัศนภาพครับ

---

## 📊 1. แผนภูมิแท่งเปรียบเทียบ Category & Payment Method กับ Return Rate

แสดงอัตราการคืนในแต่ละหมวดสินค้าแยกตามสีของช่องทางจ่ายเงิน:

![Category x Payment Method Return Rate](Graph%20Item/cross_chart_category_payment.png)

> **เจาะลึก:** เห็นได้ชัดว่า `Electronics` ที่จ่ายผ่านบัตรเครดิตพุ่งสูงถึง 35.1% ในขณะที่ `Cosmetics` ที่โอนเงินคืนต่ำสุดที่ 22.2%

---

## 📈 2. แผนภูมิเส้นแนวโน้ม Category & Channel Type กับ Return Rate

แสดงเส้นแนวโน้มอัตราการคืนสินค้าในแต่ละประเภทสินค้าแยกตามสื่อช่องทางขาย:

![Category x Channel Type Return Rate](Graph%20Item/cross_chart_category_channel.png)

> **เจาะลึก:** สินค้าหมวด `Electronics` แสดงแรงเหวี่ยงขึ้นเมื่อขายบน `TV Show` และ `TikTok` อย่างชัดเจน ส่วนหมวดหมู่อื่นๆ มีแรงเหวี่ยงที่แตกต่างกัน

---

## 📊 3. แผนภูมิแท่งเปรียบเทียบ Province & Payment Method กับ Return Rate

แสดงอัตราการคืนแยกตามแต่ละจังหวัดและสีของวิธีการชำระเงิน:

![Province x Payment Method Return Rate](Graph%20Item/cross_chart_province_payment.png)

> **เจาะลึก:** แสดงความเสี่ยงในพื้นที่ `Songkhla` เมื่อสั่งปลายทาง `COD` พุ่งขึ้นชัดเจน และ `Chonburi` ที่จ่ายบัตรเครดิต

---

## 📊 4. แผนภูมิแท่งเปรียบเทียบ Province & Gender กับ Return Rate

แสดงอัตราการคืนแยกตามจังหวัดปลายทางและสีระบุเพศของผู้ซื้อ:

![Province x Gender Return Rate](Graph%20Item/cross_chart_province_gender.png)

> **เจาะลึก:** ชี้ชัดถึงพฤติกรรมการตัดสินใจคืนสินค้าที่แยกกันคนละขั้วระหว่างเพศหญิงและชายในแต่ละภูมิภาค เช่น หญิงกรุงเทพฯ สูงถึง 32.6% แต่ชายกรุงเทพฯ คืนเพียง 15.1%
