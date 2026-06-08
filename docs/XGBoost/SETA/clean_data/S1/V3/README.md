# S1 V3 - XGBoost 80/20 Train-Test Split

V3 interaction: V2 plus business interaction and group return-rate features.

- Source dataset: `docs\XGBoost\SETA\clean_data\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `XGBoost`
- Feature count: `72`
- Test Accuracy: `64.70%`
- Test Recall: `60.48%`
- Test Precision: `42.51%`
- Test F1: `49.93%`
- Test AUC: `68.56%`
- Test Cost: `69,400`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `docs\XGBoost\SETA\clean_data\S1\V3\features\df_featured_s1_v3.csv`
- Train/test artifact: `docs\XGBoost\SETA\clean_data\S1\V3\features\train_test_sets_s1_v3.pkl`
- Model: `docs\XGBoost\SETA\clean_data\S1\V3\models\model_s1_v3_xgboost.pkl`
- Metrics: `docs\XGBoost\SETA\clean_data\S1\V3\reports\metrics_s1_v3.csv`
- Test predictions: `docs\XGBoost\SETA\clean_data\S1\V3\reports\test_predictions_s1_v3.csv`

