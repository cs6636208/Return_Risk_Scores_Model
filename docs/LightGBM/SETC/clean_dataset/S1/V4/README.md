# LightGBM SETC S1 V4 - 80/20 Train-Test Split

V4 segment risk: V3 plus price, discount, rating, logistics, and segment-risk features.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S1\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `LightGBM`
- Feature count: `87`
- Test Accuracy: `69.30%`
- Test Recall: `31.27%`
- Test Precision: `45.96%`
- Test F1: `37.22%`
- Test AUC: `68.25%`
- Test Cost: `105,350`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S1\V4\features\df_featured_lgbm_s1_v4.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S1\V4\features\train_test_sets_lgbm_s1_v4.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S1\V4\models\model_lgbm_s1_v4_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S1\V4\reports\metrics_lgbm_s1_v4.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S1\V4\reports\test_predictions_lgbm_s1_v4.csv`
