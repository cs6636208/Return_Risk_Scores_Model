# S1 V2 - XGBoost 80/20 Train-Test Split

V2 history: V1 plus customer history and rolling point-in-time history.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `XGBoost`
- Feature count: `57`
- Test Accuracy: `63.10%`
- Test Recall: `62.54%`
- Test Precision: `41.18%`
- Test F1: `49.66%`
- Test AUC: `67.35%`
- Test Cost: `67,500`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S1\V2\features\df_featured_s1_v2.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S1\V2\features\train_test_sets_s1_v2.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S1\V2\models\model_s1_v2_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S1\V2\reports\metrics_s1_v2.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S1\V2\reports\test_predictions_s1_v2.csv`

