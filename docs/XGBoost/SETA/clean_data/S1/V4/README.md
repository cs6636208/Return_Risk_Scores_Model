# S1 V4 - XGBoost 80/20 Train-Test Split

V4 segment risk: V3 plus price, discount, rating, logistics, and segment-risk features.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `XGBoost`
- Feature count: `87`
- Test Accuracy: `66.00%`
- Test Recall: `58.76%`
- Test Precision: `43.73%`
- Test F1: `50.15%`
- Test AUC: `68.90%`
- Test Cost: `71,000`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S1\V4\features\df_featured_s1_v4.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S1\V4\features\train_test_sets_s1_v4.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S1\V4\models\model_s1_v4_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S1\V4\reports\metrics_s1_v4.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S1\V4\reports\test_predictions_s1_v4.csv`

