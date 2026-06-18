# LightGBM_SETC_S1_REBUILD V2

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V2 customer temporal history: V1 plus point-in-time customer return/spend/order behavior and rolling windows.`
- Feature count: `64`
- Holdout Accuracy: `81.20%`
- Holdout Recall: `66.46%`
- Holdout F1: `69.28%`
- Holdout AUC: `82.57%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
