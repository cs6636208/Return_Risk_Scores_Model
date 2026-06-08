# S2 V2 - XGBoost 80/20 Train-Test Split

V2 history: V1 plus customer history and rolling point-in-time history.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `XGBoost`
- Feature count: `57`
- Test Accuracy: `44.66%`
- Test Recall: `94.60%`
- Test Precision: `33.86%`
- Test F1: `49.87%`
- Test AUC: `72.00%`
- Test Cost: `347,350`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S2\V2\features\df_featured_s2_v2.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S2\V2\features\train_test_sets_s2_v2.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S2\V2\models\model_s2_v2_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S2\V2\reports\metrics_s2_v2.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S2\V2\reports\test_predictions_s2_v2.csv`

