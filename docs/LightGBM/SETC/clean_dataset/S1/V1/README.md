# LightGBM SETC S1 V1 - 80/20 Train-Test Split

V1 baseline: raw/order-time features from the clean dataset.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S1\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `LightGBM`
- Feature count: `32`
- Test Accuracy: `68.40%`
- Test Recall: `35.05%`
- Test Precision: `44.54%`
- Test F1: `39.23%`
- Test AUC: `66.89%`
- Test Cost: `100,850`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S1\V1\features\df_featured_lgbm_s1_v1.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S1\V1\features\train_test_sets_lgbm_s1_v1.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S1\V1\models\model_lgbm_s1_v1_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S1\V1\reports\metrics_lgbm_s1_v1.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S1\V1\reports\test_predictions_lgbm_s1_v1.csv`
