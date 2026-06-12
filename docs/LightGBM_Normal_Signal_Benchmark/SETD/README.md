# LightGBM SETD Rebuild

SETD is the external test side of the rebuilt LightGBM workflow.

## External Dataset

S3 and S4 use generated future high-signal datasets with intentionally different row counts to test scale and model stability.

- S3 tests SETC/S1 models with 55,000 rows.
- S4 tests SETC/S2 models with 105,000 rows.
- S3 feature engineering uses SETC/S1 as history context.
- S4 feature engineering uses SETC/S2 as history context.

This avoids the old S3 issue where test data was aligned too closely with train data and produced unrealistic 92% Accuracy. Current S3/S4 are still synthetic benchmarks, but they are separated as external full-dataset tests.

## Best External Accuracy

- S3 vs S1 best: `V2 = 77.85%`
- S4 vs S2 best: `V5 = 79.29%`
