# LightGBM_SETC_S2_REBUILD V1

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V1 base order-time features: customer profile, product, price, channel, payment, promotion, and logistics expectation.`
- Feature count: `25`
- Holdout Accuracy: `78.15%`
- Holdout Recall: `48.59%`
- Holdout F1: `51.78%`
- Holdout AUC: `72.35%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
