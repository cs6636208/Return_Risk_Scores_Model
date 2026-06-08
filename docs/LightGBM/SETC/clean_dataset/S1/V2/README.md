# LightGBM SETC S1 V2 - 80/20 Train-Test Split

V2 history: V1 plus customer history and rolling point-in-time history.

- Source dataset: `docs\LightGBM\SETC\clean_dataset\S1\clean_dataset_s1.csv`
- Evaluation type: `train_test_split_80_20_stratified_holdout`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `4,000`
- Test rows: `1,000`
- Model: `LightGBM`
- Feature count: `57`
- Test Accuracy: `68.20%`
- Test Recall: `38.49%`
- Test Precision: `44.62%`
- Test F1: `41.33%`
- Test AUC: `66.33%`
- Test Cost: `96,450`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `docs\LightGBM\SETC\clean_dataset\S1\V2\features\df_featured_lgbm_s1_v2.csv`
- Train/test artifact: `docs\LightGBM\SETC\clean_dataset\S1\V2\features\train_test_sets_lgbm_s1_v2.pkl`
- Model: `docs\LightGBM\SETC\clean_dataset\S1\V2\models\model_lgbm_s1_v2_lightgbm.pkl`
- Metrics: `docs\LightGBM\SETC\clean_dataset\S1\V2\reports\metrics_lgbm_s1_v2.csv`
- Test predictions: `docs\LightGBM\SETC\clean_dataset\S1\V2\reports\test_predictions_lgbm_s1_v2.csv`
