# LightGBM_SETC_S1_REBUILD V4

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V4 business interactions: V3 plus category-payment-channel-province interactions, bands, and risk flags.`
- Feature count: `107`
- Holdout Accuracy: `82.50%`
- Holdout Recall: `66.14%`
- Holdout F1: `70.69%`
- Holdout AUC: `83.13%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
