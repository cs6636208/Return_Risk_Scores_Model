# Project Structure

This file is the main map for the Return Risk Prediction project.

## Open First

1. `docs/version 4/README_V4_GENERATED.md`
   - Latest selected V4 generated experiment.
   - Main result: XGBoost + SMOTE + Optuna, Accuracy 83.45%, Recall 46.39%, F1 45.69%, AUC 85.38%, Cost 31,650.
2. `docs/model comparison/model_versions_v1_to_v4_comparison.pdf`
   - Comparison across V1-V4.
3. `docs/business insight/business_insight_feature_summary.md`
   - Business insight and feature mapping.

## Top-Level Folders

| Folder | Purpose | Notes |
| --- | --- | --- |
| `data/raw` | Original/mock raw source data | Do not edit manually. |
| `data/processed` | Cleaned datasets | Includes original clean data and V4 generated clean data. |
| `data/generated` | Synthetic/generated datasets | V4 generated imbalance data lives here. |
| `data/features` | Feature matrices and train/test artifacts | Used by training scripts. |
| `docs/reference` | Input materials and screenshots | PDFs, ER diagram, source reference images. |
| `docs/analysis` | Working analysis documents | Broader analysis outputs and report drafts. |
| `docs/business insight` | Business insight deliverables | Curated documents for presentation/reporting. |
| `docs/model comparison` | Version comparison deliverables | V1-V4 comparison report and chart. |
| `docs/version 1` | V1 code/report materials | Baseline feature engineering and model scripts. |
| `docs/version 2` | V2 code/report materials | V2 feature engineering, model scripts, and V2.1 tuning package. |
| `docs/version 3` | V3 code/report materials | Stacking model and threshold evaluation. |
| `docs/version 4` | Latest V4 generated package | Self-contained generated-data pipeline package. |
| `models` | Model artifacts | `.pkl` model files and metadata. |
| `reports` | Charts/metrics from experiments | Canonical output folder for scripts. |
| `scripts` | Re-runnable utility and experiment scripts | End-to-end pipelines and export scripts. |
| `src` | App/source package area | Project source code support. |

## Version Folders

### V1

`docs/version 1`

- `feature_engineering.py`
- `model_training.py`
- `model_evaluation.py`
- Feature engineering principle docs/PDF

### V2

`docs/version 2`

- `feature_engineering_v2.py`
- `model_training_v2.py`
- `model_evaluation_v2.py`
- `df_engineered_v2_preview.csv`
- `df_engineered_v2_unencoded.csv`
- `v2_1_accuracy_tuning/`
  - V2.1 tuning report, metrics CSV, selected features, and metadata.

### V3

`docs/version 3`

- `model_training_v3_stacking.py`
- `model_evaluation_v3.py`

### V4

`docs/version 4`

This is the latest selected V4 package and replaces the old V4 XGBoost feature-selection experiment.

- `data/generated/v4_synthetic_orders_returns.csv`
- `data/processed/clean_dataset_v4_generated.csv`
- `data/features/df_engineered_v4_generated.csv`
- `data/features/train_test_sets_v4_generated.pkl`
- `models/best_model_v4_generated.pkl`
- `reports/model_evaluation/v4_generated_model_metrics.csv`
- `reports/model_evaluation/v4_generated_shap_summary.png`
- `reports/eda/`
- `docs/v4_generated_end_to_end_report.pdf`
- `scripts/run_v4_generated_end_to_end_pipeline.py`

## Canonical Vs Deliverable Copies

- Canonical files used by scripts stay in `data/`, `reports/`, `models/`, and `scripts/`.
- Presentation/report copies are grouped under `docs/`.
- `docs/version 4` is self-contained for the latest V4 generated experiment.

