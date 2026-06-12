# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S1 V5

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V5 Hybrid Compact Best: compact hybrid selected from customer, product, logistics, payment, and interaction features.`
- Feature count: `65`
- Holdout Accuracy: `80.10%`
- Holdout Recall: `37.08%`
- Holdout F1: `47.21%`
- Holdout AUC: `73.97%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
