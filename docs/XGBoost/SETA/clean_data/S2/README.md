# S2 Sequential V1-V5 - 80/20 Train-Test Split

Source clean dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s2.csv`

Evaluation follows the train/test split idea: train on 80% of the rows and evaluate on the held-out 20% rows that the model did not train on.

- Split method: `train_test_split(test_size=0.20, stratify=is_returned, random_state=42)`
- Model: `XGBoost`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Threshold: chosen on train split only, then applied to test split

| Version | Features | Train Rows | Test Rows | Test Accuracy | Test Recall | Test Precision | Test F1 | Test AUC | Test Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 40,000 | 10,000 | 46.59% | 94.78% | 34.70% | 50.81% | 73.81% | 335,450 |
| V2 | 57 | 40,000 | 10,000 | 44.66% | 94.60% | 33.86% | 49.87% | 72.00% | 347,350 |
| V3 | 72 | 40,000 | 10,000 | 48.68% | 93.37% | 35.49% | 51.43% | 74.24% | 343,450 |
| V4 | 87 | 40,000 | 10,000 | 51.47% | 93.85% | 36.88% | 52.95% | 76.80% | 323,200 |
| V5 | 67 | 40,000 | 10,000 | 50.82% | 92.23% | 36.39% | 52.19% | 74.50% | 347,600 |

## Version Flow

- V1: baseline raw/order-time features
- V2: V1 plus customer history and rolling history
- V3: V2 plus business interaction and point-in-time group return-rate features
- V4: V3 plus segment/operation risk features
- V5: compact feature set after V4 transformations

## Important

Previous SETA metrics used `full_training_in_sample_no_holdout`. These artifacts replace that approach with a true 80/20 holdout split, so Accuracy may be lower but is more defensible for explaining model performance.

