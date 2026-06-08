# LightGBM SETC S2 V5 - 80/20 Train-Test Split

V5 compact: V4 transformations with a reduced compact feature set.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S2\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `LightGBM`
- Feature count: `67`
- Test Accuracy: `71.61%`
- Test Recall: `88.83%`
- Test Precision: `50.70%`
- Test F1: `64.55%`
- Test AUC: `86.69%`
- Test Cost: `288,200`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S2\V5\features\df_featured_lgbm_s2_v5.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S2\V5\features\train_test_sets_lgbm_s2_v5.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S2\V5\models\model_lgbm_s2_v5_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S2\V5\reports\metrics_lgbm_s2_v5.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S2\V5\reports\test_predictions_lgbm_s2_v5.csv`
