# LightGBM_SETC_S1_REBUILD V2

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V2 customer temporal history: V1 plus point-in-time customer return/spend/order behavior and rolling windows.`
- Feature count: `64`
- Holdout Accuracy: `77.00%`
- Holdout Recall: `49.58%`
- Holdout F1: `50.85%`
- Holdout AUC: `71.33%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
