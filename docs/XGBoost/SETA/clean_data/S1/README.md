# S1 Sequential V1-V5 - 80/20 Train-Test Split

Source clean dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s1.csv`

Evaluation follows the train/test split idea: train on 80% of the rows and evaluate on the held-out 20% rows that the model did not train on.

- Split method: `train_test_split(test_size=0.20, stratify=is_returned, random_state=42)`
- Model: `XGBoost`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Threshold: chosen on train split only, then applied to test split

| Version | Features | Train Rows | Test Rows | Test Accuracy | Test Recall | Test Precision | Test F1 | Test AUC | Test Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 32 | 4,000 | 1,000 | 64.20% | 64.60% | 42.44% | 51.23% | 67.56% | 64,250 |
| V2 | 57 | 4,000 | 1,000 | 63.10% | 62.54% | 41.18% | 49.66% | 67.35% | 67,500 |
| V3 | 72 | 4,000 | 1,000 | 64.70% | 60.48% | 42.51% | 49.93% | 68.56% | 69,400 |
| V4 | 87 | 4,000 | 1,000 | 66.00% | 58.76% | 43.73% | 50.15% | 68.90% | 71,000 |
| V5 | 67 | 4,000 | 1,000 | 64.60% | 60.48% | 42.41% | 49.86% | 67.84% | 69,450 |

## Version Flow

- V1: baseline raw/order-time features
- V2: V1 plus customer history and rolling history
- V3: V2 plus business interaction and point-in-time group return-rate features
- V4: V3 plus segment/operation risk features
- V5: compact feature set after V4 transformations

## Important

Previous SETA metrics used `full_training_in_sample_no_holdout`. These artifacts replace that approach with a true 80/20 holdout split, so Accuracy may be lower but is more defensible for explaining model performance.

