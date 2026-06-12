# LightGBM_SETC_S2_REBUILD V4

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V4 business interactions: V3 plus category-payment-channel-province interactions, bands, and risk flags.`
- Feature count: `107`
- Holdout Accuracy: `78.78%`
- Holdout Recall: `44.90%`
- Holdout F1: `50.54%`
- Holdout AUC: `72.49%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
