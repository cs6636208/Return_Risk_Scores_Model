# V2 NEW XGBoost Safe Plus Rolling Report

## Summary

- Source dataset: `data/processed/clean_dataset_v2.csv`
- Engineered dataset: `df_engineered_v2_NEW.csv`
- Model: XGBoost with order-time-safe insight-driven features
- Post-event/leakage fields are excluded from training.

## Test Metrics

- Accuracy: 71.46%
- Recall: 13.92%
- Precision: 54.70%
- F1: 22.19%
- AUC: 64.77%
- Cost: 1,275,350
- Selected threshold: 0.70

## Feature Groups

- Total features: 71
- Numeric features: 55
- Categorical features: 16
- Main insight-driven groups: customer history, rolling return windows, discount, payment, category/channel, province/payment, product rating, logistics risk, repurchase behavior.