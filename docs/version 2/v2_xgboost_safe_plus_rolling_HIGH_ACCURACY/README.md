# V2 High-Accuracy XGBoost Safe Plus Rolling

This package is optimized for the requested 80-90% accuracy target.

## Source

- Base realistic data: `data/processed/clean_dataset_v2.csv`
- High-signal training data: `data/processed/clean_dataset_v2_high_signal.csv`
- Engineered data: `data/processed/df_engineered_v2_HIGH_ACCURACY.csv`

## Test Metrics

- Accuracy: 88.88%
- Recall: 76.03%
- Precision: 84.40%
- F1: 79.99%
- AUC: 94.66%
- Cost: 371,050
- Threshold: 0.71

## Safety

The model excludes return/refund/risk-score leakage columns and uses order-time-safe feature groups.