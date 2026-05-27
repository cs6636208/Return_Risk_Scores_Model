# Version 3 - Stacking Model Package

## Summary

V3 เป็น model experiment ที่ใช้ feature/split จาก V2 แล้วเปลี่ยน model architecture เป็น Stacking Ensemble

- Base models: `XGBoost`, `LightGBM`, `CatBoost`
- Meta learner: `LogisticRegression`
- Feature source: `train_test_sets_v2.pkl`
- Feature count: `38`
- Model file: `models/best_model_v3_stack.pkl`

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 66.67% |
| Recall | 63.76% |
| Precision | 44.84% |
| F1 | 52.65% |
| AUC | 71.90% |
| Cost | 20,400 |
| Rating | B |

## Process

1. Load `train_test_sets_v2.pkl`
2. Train `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`
3. Combine predictions with `StackingClassifier`
4. Use `LogisticRegression(class_weight='balanced')` as meta learner
5. Evaluate threshold scenarios and cost matrix

## Important Note

V3 ไม่ได้สร้าง feature engineering ใหม่เอง แต่ reuse V2 feature set ดังนั้นในรายงานควรอธิบายว่า V3 คือ model architecture experiment ไม่ใช่ feature version ใหม่
