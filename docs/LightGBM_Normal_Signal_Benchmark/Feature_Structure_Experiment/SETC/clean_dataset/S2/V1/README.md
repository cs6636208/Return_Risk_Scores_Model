# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S2 V1

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V1 Order/Product Basic: baseline using simple order, customer profile, product, price, promotion, payment, and channel features.`
- Feature count: `24`
- Holdout Accuracy: `78.40%`
- Holdout Recall: `47.89%`
- Holdout F1: `51.70%`
- Holdout AUC: `72.34%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
