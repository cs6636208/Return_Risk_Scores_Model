# LightGBM_SETC_S2_REBUILD V2

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V2 customer temporal history: V1 plus point-in-time customer return/spend/order behavior and rolling windows.`
- Feature count: `64`
- Holdout Accuracy: `78.94%`
- Holdout Recall: `44.90%`
- Holdout F1: `50.73%`
- Holdout AUC: `72.55%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
