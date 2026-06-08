# LightGBM SETC S1 Model Test on SETD real_dataset_s1.csv

Test data: `docs\LightGBM\SETD\real_dataset\real_dataset_s1.csv`

Train model source: `docs\LightGBM\SETC\clean_dataset\S1`

Evaluation type: `external_full_dataset_accuracy`

Important: this test sends the full `real_dataset_s1.csv` file through already-trained LightGBM S1 V1-V5 models. It does not split the SETD test data again and does not retrain the models.

## Test Data Validation

- Rows: `55,000`
- Columns: `65`
- Missing/null cells: `0`
- Duplicate order_id: `0`
- Returned: `16,005`
- Not Returned: `38,995`
- Return rate: `29.10%`

## Results

| Version | Features | Threshold | S1 Holdout Accuracy | real_dataset_s1 Accuracy | Gap | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 0.56 | 68.40% | 92.18% | +23.78pt | 85.65% | 87.23% | 86.43% | 94.73% | 1,248,350 |
| V2 | 57 | 0.51 | 68.20% | 92.61% | +24.41pt | 87.45% | 87.19% | 87.32% | 95.42% | 1,107,300 |
| V3 | 72 | 0.55 | 68.80% | 93.16% | +24.36pt | 85.73% | 90.28% | 87.95% | 95.61% | 1,215,850 |
| V4 | 87 | 0.58 | 69.30% | 92.72% | +23.42pt | 83.07% | 91.12% | 86.91% | 95.25% | 1,419,750 |
| V5 | 67 | 0.58 | 67.90% | 92.12% | +24.22pt | 81.98% | 90.05% | 85.83% | 94.87% | 1,514,500 |

## Interpretation

`S1 Holdout Accuracy` is the original 20% holdout score from `clean_dataset_s1.csv`.

`real_dataset_s1 Accuracy` is the result from sending the full external test file through the same model after applying the matching V1-V5 feature engineering.

If the gap is small, the generated SETD test data is close to the S1 model distribution. If the gap is large, the test data distribution or signal strength is different from the S1 holdout split.
