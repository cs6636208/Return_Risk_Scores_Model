# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S2 V4

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V4 Logistics & Payment Risk Focus: focuses on courier, logistics, payment, channel, province, COD, and remote-area risk.`
- Feature count: `32`
- Holdout Accuracy: `77.69%`
- Holdout Recall: `42.92%`
- Holdout F1: `48.15%`
- Holdout AUC: `71.63%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
