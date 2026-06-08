# S1 Model Test on real_data_s1.csv

Test data: `docs\XGBoost\SETB\real_data\real_data_s1.csv`

Train model source: `docs\XGBoost\SETA\clean_data\S1`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_data_s1.csv` file through already-trained S1 V1-V5 models. It does not split the test data again and does not retrain the models.

## Test Data Validation

- Rows: `55,000`
- Columns: `65`
- Missing/null cells: `0`
- Duplicate order_id: `0`
- Returned: `16,005`
- Not Returned: `38,995`
- Return rate: `29.10%`

## Results

| Version | Features | Threshold | S1 Holdout Accuracy | real_data_s1 Accuracy | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 0.44 | 64.20% | 74.03% | 85.29% | 53.36% | 65.65% | 86.45% | 1,773,950 |
| V2 | 57 | 0.43 | 63.10% | 76.84% | 86.64% | 56.68% | 68.52% | 88.57% | 1,599,450 |
| V3 | 72 | 0.44 | 64.70% | 78.02% | 85.84% | 58.32% | 69.45% | 89.14% | 1,624,500 |
| V4 | 87 | 0.46 | 66.00% | 77.66% | 82.15% | 58.23% | 68.15% | 87.72% | 1,900,100 |
| V5 | 67 | 0.45 | 64.60% | 76.81% | 83.18% | 56.95% | 67.61% | 87.33% | 1,849,250 |

## Interpretation

`S1 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s1.csv`.

`real_data_s1 Accuracy` is measured by sending all 55,000 rows through each already-trained S1 model after applying the matching version feature engineering. This external test uses the full file, not a 20% split.
