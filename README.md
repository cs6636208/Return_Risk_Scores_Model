# Return Risk Scores Model

Prototype return and refund risk scoring system for an e-commerce/order
operation. The project follows the planned workflow from data understanding,
EDA, feature engineering, model training, model evaluation, and a Streamlit
dashboard for operational review.

Folder map is documented in `docs/PROJECT_STRUCTURE.md`.

## Project Goal

The model estimates whether an order has elevated return risk before or during
fulfillment. The business objective is not only high accuracy: it is to reduce
avoidable return/refund cost by prioritizing high-risk orders for review.

Primary outputs:

- A clean order-level dataset for analysis and modeling.
- Engineered train/test feature sets.
- Model comparison artifacts and a saved best model.
- Cost/threshold evaluation reports.
- SHAP/feature-importance visuals for explainability.
- A Streamlit dashboard for risk score, risk tier, and order-level review.

## Current Status

| Plan phase | Status | Evidence |
| --- | --- | --- |
| Week 1-2 Data collection and cleaning | Done | `data/raw/mock_return_data.csv`, `data/processed/clean_dataset.csv` |
| Week 3 EDA and business insight | Done | `notebooks/eda/`, `reports/`, `docs/analysis/` |
| Week 4 Feature engineering | Done | `data/features/train_test_sets.pkl`, `data/features/train_test_sets_v2.pkl` |
| Week 5-6 Model training/evaluation | In progress/mostly done | `models/best_model*.pkl`, `reports/model_training/`, `reports/model_evaluation_v2/`, `reports/model_evaluation_v3/` |
| Week 7-8 Dashboard/reporting | Started | `dashboard.py` |

## Data Pipeline

1. Generate or load order, customer, product, courier, promotion, return, and
   risk metadata.
2. Clean transactional records into `data/processed/clean_dataset.csv`.
3. Build modeling features and train/test sets.
4. Train candidate models and tune the best model.
5. Evaluate with AUC-ROC, F1, precision, recall, and a business cost matrix.
6. Review order-level risk in the Streamlit dashboard.

Key scripts:

- `src/setup_database.py`: generates mock ERD-aligned data and uploads to PostgreSQL.
- `notebooks/clean_process/data_cleansing.py`: creates the clean dataset.
- `docs/version 1/feature_engineering.py`: feature engineering with SMOTE.
- `docs/version 2/feature_engineering_v2.py`: leakage-aware target encoding, no SMOTE.
- `docs/version 1/model_training.py`: baseline and tuned model comparison.
- `docs/version 3/model_evaluation_v3.py`: V3 threshold and metrics evaluation.
- `dashboard.py`: Streamlit risk review dashboard.

## Model Results

The current V3 stacking evaluation reports:

- AUC-ROC: 0.7190
- Average precision: 0.4878
- F1-score: 0.5265
- Precision: 0.4484
- Recall: 0.6376
- Optimal cost threshold: 0.45
- Expected cost at optimal threshold: 19,500 THB

The saved V1 metadata also reports XGBoost as the best tuned baseline:

- AUC-ROC: 0.6882
- F1-score: 0.3482
- Precision: 0.4968
- Recall: 0.2680

These results are suitable for a prototype, but they should be presented as a
cost-aware decision support tool rather than a highly accurate production
classifier. For this domain, recall and expected business cost matter more than
accuracy alone because missed high-risk returns can be expensive.

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

Run V3 evaluation without retraining:

```powershell
python "docs/version 3/model_evaluation_v3.py"
```

Optional database stack:

```powershell
docker compose up -d
```

## Repository Notes

- Conversation logs are kept under `docs/archive/conversations/` and ignored
  by git.
- Markdown analysis notes are under `docs/analysis/` and UTF-8 encoded. If Thai text looks broken in
  PowerShell, use an editor/browser with UTF-8 rendering.
- The dashboard reads local CSV/model-report artifacts and does not require a
  live PostgreSQL connection.
