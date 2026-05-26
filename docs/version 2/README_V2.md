# Version 2 - Selected Practical Model

V2 is the best practical model on the original data artifacts because it has the strongest balance of Recall, F1, AUC, and Cost.

## Main V2 Result

| Version | Model | Accuracy | Recall | F1 | AUC | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V2 | XGBoost | 67.87% | 65.60% | 54.27% | 73.66% | 19,550 |

## Open First

1. `Feature_Comparison_V2.pdf`
2. `reports/model_evaluation/metrics_comparison_bar.png`
3. `reports/model_evaluation/cost_optimization_v2.png`
4. `reports/model_evaluation/shap_summary_v2.png`
5. `reports/thresholds/v2_recommended_threshold_scenarios.csv`

## Folder Map

| Folder/File | Purpose |
| --- | --- |
| `feature_engineering_v2.py` | Original V2 feature engineering script. |
| `model_training_v2.py` | Original V2 model training script. |
| `model_evaluation_v2.py` | Original V2 evaluation script. |
| `scripts/` | Copies of V2 scripts grouped with the package. |
| `data/features/df_engineered_v2_preview.csv` | V2 feature preview dataset. |
| `data/features/df_engineered_v2_unencoded.csv` | V2 unencoded engineered dataset. |
| `data/features/train_test_sets_v2.pkl` | V2 train/test artifact. |
| `data/features/scaler_v2.pkl` | V2 scaler artifact. |
| `models/best_model_v2.pkl` | Main V2 model artifact. |
| `reports/model_evaluation/` | V2 metrics, cost, and SHAP images. |
| `reports/eda_v2/` | V2 EDA/feature relation charts. |
| `reports/thresholds/` | V2 threshold tuning CSVs. |
| `v2_1_accuracy_tuning/` | V2.1 experiment package, kept separate from main V2. |

## Why V2 Is Preferred

- V2 has the best real/original-data trade-off.
- V2 has the lowest Cost among V1-V4 original/generated comparison.
- V2 has the highest Recall and F1 among original-data model versions.
- V4 has higher Accuracy/AUC, but V4 uses generated/synthetic imbalance data.

