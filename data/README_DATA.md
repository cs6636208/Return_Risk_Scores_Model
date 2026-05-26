# Data Folder

## `raw`

Original/mock source data before cleaning.

## `processed`

Cleaned datasets.

- `clean_dataset.csv` - original cleaned dataset.
- `clean_dataset_v4_generated.csv` - cleaned generated dataset for V4 imbalance experiment.

## `generated`

Synthetic/generated data.

- `v4_synthetic_orders_returns.csv` - generated imbalance dataset using original rows plus synthetic non-return rows.

## `features`

Feature matrices and train/test artifacts.

- `df_engineered.csv` - existing engineered feature dataset.
- `df_engineered_v4_generated.csv` - V4 generated engineered features.
- `train_test_sets_v4_generated.pkl` - V4 generated train/test/SMOTE artifact.

