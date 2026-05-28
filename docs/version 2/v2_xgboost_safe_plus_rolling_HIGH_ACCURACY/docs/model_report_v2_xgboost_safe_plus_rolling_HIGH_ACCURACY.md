# V2 NEW XGBoost Safe Plus Rolling Report

## Summary

- Source dataset: `data/processed/clean_dataset_v2.csv`
- Engineered dataset: `df_engineered_v2_NEW.csv`
- Model: XGBoost with order-time-safe insight-driven features
- Post-event/leakage fields are excluded from training.

## Test Metrics

- Accuracy: 88.88%
- Recall: 76.03%
- Precision: 84.40%
- F1: 79.99%
- AUC: 94.66%
- Cost: 371,050
- Selected threshold: 0.71

## Feature Groups

- Total features: 71
- Numeric features: 55
- Categorical features: 16
- Main insight-driven groups: customer history, rolling return windows, discount, payment, category/channel, province/payment, product rating, logistics risk, repurchase behavior.