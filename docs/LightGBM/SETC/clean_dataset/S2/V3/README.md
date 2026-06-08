# LightGBM SETC S2 V3 - 80/20 Train-Test Split

V3 interaction: V2 plus business interaction and group return-rate features.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S2\clean_dataset_s2.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `40,000`
- Test rows: `10,000`
- Model: `LightGBM`
- Feature count: `72`
- Test Accuracy: `71.65%`
- Test Recall: `87.01%`
- Test Precision: `50.75%`
- Test F1: `64.11%`
- Test AUC: `85.72%`
- Test Cost: `311,850`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S2\V3\features\df_featured_lgbm_s2_v3.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S2\V3\features\train_test_sets_lgbm_s2_v3.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S2\V3\models\model_lgbm_s2_v3_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S2\V3\reports\metrics_lgbm_s2_v3.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S2\V3\reports\test_predictions_lgbm_s2_v3.csv`
