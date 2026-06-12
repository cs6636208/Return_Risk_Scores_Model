# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S2 V5

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V5 Hybrid Compact Best: compact hybrid selected from customer, product, logistics, payment, and interaction features.`
- Feature count: `65`
- Holdout Accuracy: `78.52%`
- Holdout Recall: `47.39%`
- Holdout F1: `51.58%`
- Holdout AUC: `72.61%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
