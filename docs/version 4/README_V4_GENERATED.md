# Version 4 - Generated Imbalanced Data Experiment

## Main Result

Best practical model: **XGBoost + SMOTE + Optuna**

| Model | Accuracy | Recall | F1 | AUC | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost + SMOTE + Optuna | 83.45% | 46.39% | 45.69% | 85.38% | 31,650 |

## Dataset Note

This version uses synthetic/generated data for an imbalance experiment.

- Original rows: 5,000
- Generated non-return rows: 4,700
- Final rows: 9,700
- Return rate: 15%
- SMOTE is applied only to the training split.

## Folder Map

- `data/generated/v4_synthetic_orders_returns.csv` - generated raw dataset
- `data/processed/clean_dataset_v4_generated.csv` - cleaned generated dataset
- `data/features/df_engineered_v4_generated.csv` - engineered feature dataset
- `data/features/train_test_sets_v4_generated.pkl` - train/test split with SMOTE artifacts
- `models/best_model_v4_generated.pkl` - best model artifact
- `models/best_model_v4_generated_metadata.json` - best model metadata
- `reports/model_evaluation/v4_generated_model_metrics.csv` - model metric table
- `reports/model_evaluation/v4_generated_shap_summary.png` - SHAP explainability chart
- `reports/eda/` - EDA charts
- `docs/v4_generated_end_to_end_report.pdf` - final PDF report
- `scripts/run_v4_generated_end_to_end_pipeline.py` - end-to-end pipeline script

## Pipeline Steps

1. Data Collection & Understanding: SQL template, schema/data dictionary, generated imbalance data, cleaning audit.
2. EDA: target distribution, category/channel/payment patterns, correlation heatmap.
3. Feature Engineering & Preprocessing: engineered features, encoding, train/test split, SMOTE.
4. Model Training & Evaluation: Logistic Regression, Random Forest, XGBoost, LightGBM, Optuna tuning, Cost Matrix, AUC, SHAP.
