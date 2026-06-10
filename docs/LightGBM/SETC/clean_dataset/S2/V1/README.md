# LightGBM_SETC_S2_REBUILD V1

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V1 base order-time features: customer profile, product, price, channel, payment, promotion, and logistics expectation.`
- Feature count: `25`
- Holdout Accuracy: `82.65%`
- Holdout Recall: `66.74%`
- Holdout F1: `71.23%`
- Holdout AUC: `83.39%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
