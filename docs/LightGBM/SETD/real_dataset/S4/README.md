# LightGBM SETC S2 Model Test on SETD real_dataset_s2.csv

Test data: `docs\LightGBM\SETD\real_dataset\real_dataset_s2.csv`

Train model source: `docs\LightGBM\SETC\clean_dataset\S2`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_dataset_s2.csv` file through already-trained LightGBM S2 V1-V5 models. It does not split the SETD test data again and does not retrain the models.

## Test Data Validation

- Rows: `105,000`
- Columns: `65`
- Missing/null cells: `0`
- Duplicate order_id: `0`
- Returned: `30,555`
- Not Returned: `74,445`
- Return rate: `29.10%`

## Results

| Version | Features | Threshold | S2 Holdout Accuracy | real_dataset_s2 Accuracy | Gap | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 0.42 | 69.77% | 72.38% | +2.61pt | 93.05% | 51.40% | 66.22% | 88.82% | 2,406,550 |
| V2 | 57 | 0.43 | 68.22% | 72.39% | +4.17pt | 88.17% | 51.50% | 65.02% | 86.88% | 3,076,500 |
| V3 | 72 | 0.44 | 71.65% | 73.49% | +1.84pt | 87.58% | 52.67% | 65.78% | 87.20% | 3,100,200 |
| V4 | 87 | 0.45 | 72.42% | 70.98% | -1.44pt | 88.94% | 50.08% | 64.08% | 85.62% | 3,043,450 |
| V5 | 67 | 0.43 | 71.61% | 67.71% | -3.90pt | 92.29% | 47.19% | 62.45% | 85.84% | 2,756,000 |

## Interpretation

`S2 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s2.csv`.

`real_dataset_s2 Accuracy` is the result from sending the full external test file through the same model after applying the matching V1-V5 feature engineering.

If the gap is small, the generated SETD test data is close to the S2 model distribution. If the gap is large, the test data distribution or signal strength is different from the S2 holdout split.
