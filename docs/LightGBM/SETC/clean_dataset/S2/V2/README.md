# LightGBM SETC S2 V2 - 80/20 Train-Test Split

V2 history: V1 plus customer history and rolling point-in-time history.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S2\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `LightGBM`
- Feature count: `57`
- Test Accuracy: `68.22%`
- Test Recall: `86.94%`
- Test Precision: `47.48%`
- Test F1: `61.42%`
- Test AUC: `83.71%`
- Test Cost: `329,900`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S2\V2\features\df_featured_lgbm_s2_v2.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S2\V2\features\train_test_sets_lgbm_s2_v2.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S2\V2\models\model_lgbm_s2_v2_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S2\V2\reports\metrics_lgbm_s2_v2.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S2\V2\reports\test_predictions_lgbm_s2_v2.csv`
