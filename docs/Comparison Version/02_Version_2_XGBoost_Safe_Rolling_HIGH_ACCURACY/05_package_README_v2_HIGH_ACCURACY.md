# V2 High-Accuracy XGBoost Safe Plus Rolling

Updated: 2026-05-28

This package is the current high-accuracy model candidate for the return-risk project.

## Purpose

This version was created to meet the requested 80-90% Accuracy target while still excluding post-event leakage fields from model input.

## Source And Outputs

| Item | Path |
| --- | --- |
| Base realistic dataset | `data/processed/clean_dataset_v2.csv` |
| High-signal training dataset | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/clean_dataset_v2_high_signal.csv` |
| Engineered feature dataset | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/df_engineered_v2_HIGH_ACCURACY.csv` |
| Train/test pickle | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/data/train_test_sets_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl` |
| Best model | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/models/best_model_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.pkl` |
| Metrics | `docs/version 2/v2_xgboost_safe_plus_rolling_HIGH_ACCURACY/reports/metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv` |

## Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 88.88% |
| Recall | 76.03% |
| Precision | 84.40% |
| F1 | 79.99% |
| AUC | 94.66% |
| Average Precision | 90.03% |
| Cost | 371,050 |
| Threshold | 0.71 |

## Safety

The model excludes these leakage/post-event fields from input features:

```text
return_id, return_date, return_reason, return_scenario,
item_condition, return_status, refund_amount,
score_id, risk_score, risk_tier, scored_at, shap_values,
delivery_date, delivery_days, delay_days, is_returned
```

`is_returned` is used only as the target.

## Production Note

This model is strong for the graduation project and model-performance demo. For real production, validate it with new real orders before deployment because the training dataset is high-signal synthetic data derived from `clean_dataset_v2.csv`.
