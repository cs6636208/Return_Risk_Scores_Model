# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S1 V4

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V4 Logistics & Payment Risk Focus: focuses on courier, logistics, payment, channel, province, COD, and remote-area risk.`
- Feature count: `32`
- Holdout Accuracy: `76.40%`
- Holdout Recall: `33.75%`
- Holdout F1: `40.70%`
- Holdout AUC: `68.83%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
