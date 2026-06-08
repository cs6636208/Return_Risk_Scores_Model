# S2 V1 - XGBoost 80/20 Train-Test Split

V1 baseline: raw/order-time features from the clean dataset.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `XGBoost`
- Feature count: `32`
- Test Accuracy: `46.59%`
- Test Recall: `94.78%`
- Test Precision: `34.70%`
- Test F1: `50.81%`
- Test AUC: `73.81%`
- Test Cost: `335,450`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S2\V1\features\df_featured_s2_v1.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S2\V1\features\train_test_sets_s2_v1.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S2\V1\models\model_s2_v1_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S2\V1\reports\metrics_s2_v1.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S2\V1\reports\test_predictions_s2_v1.csv`

