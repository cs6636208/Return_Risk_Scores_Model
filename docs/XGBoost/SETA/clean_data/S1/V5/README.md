# S1 V5 - XGBoost 80/20 Train-Test Split

V5 compact: V4 transformations with a reduced compact feature set.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `XGBoost`
- Feature count: `67`
- Test Accuracy: `64.60%`
- Test Recall: `60.48%`
- Test Precision: `42.41%`
- Test F1: `49.86%`
- Test AUC: `67.84%`
- Test Cost: `69,450`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S1\V5\features\df_featured_s1_v5.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S1\V5\features\train_test_sets_s1_v5.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S1\V5\models\model_s1_v5_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S1\V5\reports\metrics_s1_v5.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S1\V5\reports\test_predictions_s1_v5.csv`

