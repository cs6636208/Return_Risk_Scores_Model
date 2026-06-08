# LightGBM SETC S2 V1 - 80/20 Train-Test Split

V1 baseline: raw/order-time features from the clean dataset.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S2\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `LightGBM`
- Feature count: `32`
- Test Accuracy: `69.77%`
- Test Recall: `91.13%`
- Test Precision: `48.96%`
- Test F1: `63.70%`
- Test AUC: `86.41%`
- Test Cost: `267,250`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S2\V1\features\df_featured_lgbm_s2_v1.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S2\V1\features\train_test_sets_lgbm_s2_v1.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S2\V1\models\model_lgbm_s2_v1_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S2\V1\reports\metrics_lgbm_s2_v1.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S2\V1\reports\test_predictions_lgbm_s2_v1.csv`
