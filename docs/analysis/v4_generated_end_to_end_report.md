# V4 Generated End-to-End Report

## 1.1 Data Collection & Understanding

- SQL template: `docs/analysis/v4_generated_data_collection_sql.sql`
- Data dictionary: `docs/analysis/data_dictionary_v4_generated.csv`
- Generated raw data: `data/generated/v4_synthetic_orders_returns.csv`
- Clean dataset: `data/processed/clean_dataset_v4_generated.csv`
- Rows before generation: `5000`
- Synthetic non-return rows added: `4700`
- Rows after generation: `9700`
- Return rate after generation: `0.1500`

## 1.2 EDA

- EDA charts folder: `reports/eda_v4_generated`
- Top category return rates: `{'Electronics': 0.16344086021505377, 'Supplement': 0.15810879511947126, 'Home_Appliance': 0.15649867374005305}`
- Top channel return rates: `{'TV_Show': 0.15598650927487354, 'Shopee': 0.15172988745310545, 'TikTok': 0.14777733279935923}`

## 1.3 Feature Engineering & Preprocessing

- Engineered feature CSV: `data/features/df_engineered_v4_generated.csv`
- Train/test/SMOTE artifact: `data/features/train_test_sets_v4_generated.pkl`
- SMOTE is applied only to the training split.
- Leakage fields such as return/refund/risk score and actual delivery outcome fields are excluded from model features.

## 1.4 Model Training & Evaluation

Best model: `XGBoost_SMOTE_Optuna`

- Threshold: `0.34`
- Accuracy: `0.8345`
- Precision: `0.4500`
- Recall: `0.4639`
- F1: `0.4569`
- AUC: `0.8538`
- Cost: `31,650` THB

## Metrics

| model | threshold | accuracy | precision | recall | f1 | auc | avg_precision | cost_thb | tn | fp | fn | tp | default_accuracy | default_recall | default_f1 | default_cost_thb | train_seconds | performance_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost_SMOTE_Optuna | 0.33999999999999997 | 0.8345360824742268 | 0.45 | 0.4639175257731959 | 0.45685279187817257 | 0.8538112237136325 | 0.4446689081268707 | 31650 | 1484 | 165 | 156 | 135 | 0.8458762886597938 | 0.23711340206185566 | 0.3157894736842105 | 37150 | 1.83899079999901 | 0.542080579831493 |
| LightGBM_SMOTE_Optuna | 0.58 | 0.8474226804123711 | 0.4782608695652174 | 0.18900343642611683 | 0.270935960591133 | 0.8525566885272549 | 0.4220074035576487 | 38400 | 1589 | 60 | 236 | 55 | 0.8448453608247423 | 0.23024054982817868 | 0.3080459770114943 | 37450 | 1.8833469999990484 | 0.4236216816481303 |
| RandomForest_SMOTE | 0.49999999999999994 | 0.8422680412371134 | 0.43478260869565216 | 0.1718213058419244 | 0.24630541871921183 | 0.7660937900508275 | 0.37577965355639453 | 39400 | 1584 | 65 | 241 | 50 | 0.8422680412371134 | 0.1718213058419244 | 0.24630541871921183 | 39400 | 3.269483500000206 | 0.40847650534084945 |
| LogisticRegression_SMOTE | 0.6399999999999999 | 0.8458762886597938 | 0.2777777777777778 | 0.01718213058419244 | 0.032362459546925564 | 0.7158790394678436 | 0.27561920265585316 | 43550 | 1636 | 13 | 286 | 5 | 0.8463917525773196 | 0.05154639175257732 | 0.09146341463414634 | 42500 | 0.8329940000003262 | 0.30668949387782335 |

## Cleaning Audit

| step | rows_affected |
| --- | --- |
| drop_exact_duplicates | 0 |
| drop_duplicate_order_id | 0 |
| fill_promo_type | 8967 |
| fill_return_id | 3545 |
| fill_numeric_risk_score | 4700 |
| recalculate_discount_applied_amount | 0 |
| recalculate_total_amount | 0 |
