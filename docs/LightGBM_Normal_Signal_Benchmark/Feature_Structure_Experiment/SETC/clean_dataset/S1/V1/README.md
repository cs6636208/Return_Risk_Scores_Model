# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S1 V1

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V1 Order/Product Basic: baseline using simple order, customer profile, product, price, promotion, payment, and channel features.`
- Feature count: `24`
- Holdout Accuracy: `77.00%`
- Holdout Recall: `48.33%`
- Holdout F1: `50.22%`
- Holdout AUC: `71.79%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
