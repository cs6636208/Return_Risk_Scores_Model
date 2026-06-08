# SETB Real Data S2 - S4 Test Data

File: `real_data_s2.csv`

Purpose: external test dataset for model evaluation. This file is generated as unseen-like, representative test data using the S2 distribution family, then assigned new order/customer/product/courier identifiers.

Important: this is synthetic real-like data, not actual company production data. It is designed to act like a final exam dataset because the model files in SETA were not trained on these exact rows or IDs.

## Dataset Summary

- Rows: `50,000`
- Columns: `65`
- Order range: `ORD_REAL_S2_055001` to `ORD_REAL_S2_105000`
- Date range: `2025-01-01 03:00:00` to `2026-05-07 23:00:00`
- Returned: `14,550`
- Not Returned: `35,450`
- Return rate: `29.10%`
- Missing/null cells: `0`
- Duplicate order_id: `0`

## Generation Logic

- Source distribution: `docs/XGBoost/SETA/clean_data/clean_dataset_s2.csv`
- Row count: 50,000 rows
- Order ID range: `ORD_REAL_S2_055001` to `ORD_REAL_S2_105000`
- Recreates order/customer timeline with a different random seed from S2
- Recomputes point-in-time history features such as `hist_order_count`, `hist_return_rate`, and `days_since_last_order`
- Keeps `is_returned` as ground truth for checking model predictions
- Keeps schema compatible with the 65-column clean dataset family

## How To Use

Use this file as full external test input. Before prediction, run the same feature engineering version as the model version being tested, then compare model prediction with `is_returned`.
