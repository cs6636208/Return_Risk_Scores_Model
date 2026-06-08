# S2 Model Test on real_data_s2.csv

Test data: `docs\XGBoost\SETB\real_data\S4\real_data_s2.csv`

Train model source: `docs\XGBoost\SETA\clean_data\S2`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_data_s2.csv` file through already-trained S2 V1-V5 models. It does not split the test data again and does not retrain the models.

## Test Data Validation

- Rows: `50,000`
- Columns: `65`
- Missing/null cells: `0`
- Duplicate order_id: `0`
- Returned: `14,550`
- Not Returned: `35,450`
- Return rate: `29.10%`

## Results

| Version | Features | Threshold | S2 Holdout Accuracy | real_data_s2 Accuracy | Gap | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 0.37 | 46.59% | 46.72% | +0.13pt | 94.79% | 34.77% | 50.87% | 74.01% | 1,673,000 |
| V2 | 57 | 0.37 | 44.66% | 44.78% | +0.12pt | 94.65% | 33.92% | 49.94% | 72.41% | 1,730,650 |
| V3 | 72 | 0.38 | 48.68% | 47.86% | -0.82pt | 94.42% | 35.23% | 51.31% | 74.00% | 1,668,900 |
| V4 | 87 | 0.39 | 51.47% | 50.88% | -0.59pt | 94.85% | 36.69% | 52.91% | 76.18% | 1,565,100 |
| V5 | 67 | 0.39 | 50.82% | 49.61% | -1.21pt | 93.62% | 35.95% | 51.95% | 74.20% | 1,677,700 |

## Interpretation

`S2 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s2.csv`.

`real_data_s2 Accuracy` is the result from sending the full external test file through the same model after applying the matching V1-V5 feature engineering.

If the gap is small, the generated test data is close to the S2 model distribution. If the gap is large, the test data distribution or signal strength is different from the S2 holdout split.
