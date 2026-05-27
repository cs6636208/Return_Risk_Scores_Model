# V3 Feature Engineering Process

V3 เป็น model architecture experiment: Stacking XGBoost + LightGBM + CatBoost โดย reuse feature จาก V2

## Code Files
- `docs\version 3\stacking_model_v3\scripts\model_training_v3_stacking.py`
- `docs\version 3\stacking_model_v3\scripts\model_evaluation_v3.py`

## Steps
1. Load train_test_sets_v2.pkl จาก V2
2. ไม่ได้สร้าง feature engineering ใหม่เอง แต่ใช้ V2 feature 38 ตัว
3. Train base models: XGBoost, LightGBM และ CatBoost
4. ใช้ LogisticRegression เป็น meta learner ใน StackingClassifier
5. Evaluate threshold scenarios, recall และ cost matrix

## Reasoning
V3 พิสูจน์ว่าการเปลี่ยน model architecture เป็น ensemble ช่วย recall ได้ แต่ยัง reuse V2 feature ที่มี delivery_days/delay_days และไม่ได้แก้ order-time safety

## Pseudo-code
```python
data = joblib.load("train_test_sets_v2.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
base_models = [XGBClassifier(), LGBMClassifier(), CatBoostClassifier()]
model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
model.fit(X_train, y_train)
evaluate_thresholds(model.predict_proba(X_test))
```