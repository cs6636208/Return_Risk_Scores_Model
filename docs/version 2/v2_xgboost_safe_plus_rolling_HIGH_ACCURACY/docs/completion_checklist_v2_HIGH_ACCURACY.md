# V2 HIGH_ACCURACY Completion Checklist

## Summary

- Accuracy: 88.88%
- Recall: 76.03%
- Precision: 84.40%
- F1: 79.99%
- AUC: 94.66%
- Cost: 371,050
- Feature count: 71
- Leakage used: []

## Checklist

| requirement | status | evidence |
| --- | --- | --- |
| Business Insight from EDA with chart/table and feature mapping | complete | docs/eda_insight_summary_v2_NEW.md, eda/*.csv, images/eda_return_rate_by_*.png |
| Manual customer return-rate cross-check | complete | reports/manual_customer_return_rate_check_v2_HIGH_ACCURACY.csv |
| Order-level history check using prior orders only | complete | reports/manual_order_level_history_check_v2_HIGH_ACCURACY.csv |
| Compare clean dataset features vs engineered features and used/dropped features | complete | reports/feature_used_dropped_audit_v2_HIGH_ACCURACY.csv |
| Feature engineering from insight | complete | 71 model features including history, rolling, interaction, discount, payment, logistics |
| Train/test set ready before model training | complete | data/train_test_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl |
| XGBoost best model and performance rating | complete | Accuracy 88.88%, F1 79.99%, AUC 94.66% |
| No leakage fields in model input | complete | leakage_used=[] |
| Customer-specific history logic for order 3 from prior order 1-2 | complete | hist_order_count/hist_return_rate are point-in-time and validated in manual_order_level_history_check |
| Lookback windows weekly/monthly/yearly style | complete | rolling feature count=20 using 30/60/90/180/365 day windows |

## Manual Formula Example

Customer history uses only prior orders. If a customer has 2 prior orders and 1 returned order:

```text
hist_return_rate = return_count / total_orders = 1 / 2 = 0.5 = 50%
```

Evidence files are saved in this package under `reports/`, `eda/`, `images/`, `docs/`, `data/`, and `models/`.