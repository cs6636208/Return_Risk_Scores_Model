# LightGBM SETC Rebuild

SETC is the train/validation/holdout side of the rebuilt LightGBM workflow. The latest dataset is high-signal synthetic data with a semi-realistic return ratio near 33%.

## Sources

- S1: generated clean high-signal dataset -> 5,000 rows
- S2: generated clean high-signal dataset -> 50,000 rows
- SETD keeps separate generated future datasets for external testing.

## Best Holdout Accuracy

- S1 best: `V4 = 77.70%`
- S2 best: `V5 = 79.03%`

## Evaluation Rule

Each version uses `64% fit / 16% validation / 20% holdout`. Validation is used for parameter/threshold selection. Holdout is used for reporting.

## New Data Policy

When testing SETD, feature engineering is rebuilt with SETC as historical context first, then only SETD rows are evaluated. This is closer to production because new orders need customer/category/product history from the feature store.
