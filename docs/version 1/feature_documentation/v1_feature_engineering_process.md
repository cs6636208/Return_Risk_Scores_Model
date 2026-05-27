# V1 Feature Engineering Process

Baseline feature engineering: สร้าง feature จำนวนมากและ encode จนได้ feature model 136 ตัว

## Code Files
- `docs\version 1\feature_engineering.py`
- `docs\version 1\model_training.py`
- `docs\version 1\model_evaluation.py`

## Steps
1. Load clean_dataset.csv และเลือก field ที่เกี่ยวกับ order, customer, product, promotion, channel, payment และ history
2. สร้าง numeric/log features เช่น log_unit_price และ log_total_amount เพื่อลด skew ของราคาและยอดรวม
3. สร้าง binary flags เช่น is_peak_hour, is_cod, is_high_discount, is_first_order, low_rating_alert
4. สร้าง interaction features เช่น category_payment, category_channel, province_payment
5. สร้าง customer history features เช่น total_orders_before, total_returns_before, customer_return_ratio และ rolling history 30/60/180 วัน
6. Encode categorical variables และสร้าง train_test_sets.pkl
7. Train หลาย model แล้ว save best_model.pkl

## Reasoning
V1 ใช้เป็น baseline กว้าง ๆ เพื่อดูว่าการสร้าง feature จำนวนมากช่วย model ได้แค่ไหน แต่ recall ต่ำและ cost สูง จึงยังไม่ใช่ candidate หลัก

## Pseudo-code
```python
df = read_csv("clean_dataset.csv")
df["log_unit_price"] = log1p(unit_price)
df["log_total_amount"] = log1p(total_amount)
df["is_cod"] = payment_method == "COD"
df["is_high_discount"] = total_discount_pct > 0.20
df["category_payment"] = category + "_" + payment_method
df["category_channel"] = category + "_" + channel_type
df = add_customer_history(df)
X = encode_and_scale(df.drop("is_returned"))
y = df["is_returned"]
train_test_sets = train_test_split(X, y, stratify=y, random_state=42)
```