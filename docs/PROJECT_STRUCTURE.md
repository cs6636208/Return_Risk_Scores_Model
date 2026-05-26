# Project Structure

This document describes where each type of project file should live.

## Root

- `dashboard.py`: Streamlit application entry point.
- `README.md`: high-level project overview and run instructions.
- `requirements.txt`: Python dependencies.
- `docker-compose.yml`: local database stack.
- `package.json` / `package-lock.json`: Node-based tooling used for document/export utilities.

## Data

- `data/raw/`: source or mock source datasets.
- `data/processed/`: cleaned datasets ready for feature engineering.
- `data/features/`: engineered feature tables, train/test sets, and scalers.

## Models

- `models/`: only model binaries and model metadata.
- `models/*.pkl`: trained model artifacts.
- `models/best_model_metadata.json`: model selection and metric snapshot.

## Reports

- `reports/model_training/`: model comparison tables and charts.
- `reports/model_evaluation/`: V1 evaluation artifacts.
- `reports/model_evaluation_v2/`: V2 metrics and SHAP/cost artifacts.
- `reports/model_evaluation_v3/`: V3 threshold tradeoff and metrics CSVs.
- `reports/business_insights/`, `reports/eda_full/`, `reports/Graph Item/`, `reports/Graph Relation Feature/`: EDA and business insight visuals.

## Documentation

- `docs/version 1/`, `docs/version 2/`, `docs/version 3/`: versioned ML pipeline scripts and docs.
- `docs/analysis/`: markdown analysis notes and summaries.
- `docs/reference/input_materials/`: proposal PDFs, ERD input, and external reference files.
- `docs/reference/images/`: whiteboard/reference images.
- `docs/archive/conversations/`: conversation logs (ignored by git).

## Notebooks

- `notebooks/clean_process/`: data cleaning and schema-support notebooks/scripts.
- `notebooks/eda/`: EDA scripts and notebook exports.

## Source Scripts

- `src/setup_database.py`: mock data generation and database load.
- `src/sync_db_to_csv.py`: database to CSV sync utility.
- `scripts/print_pdf.js`: PDF rendering helper script.
