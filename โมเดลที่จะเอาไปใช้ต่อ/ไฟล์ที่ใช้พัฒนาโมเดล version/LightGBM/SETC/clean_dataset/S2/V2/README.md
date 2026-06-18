# LightGBM_SETC_S2_REBUILD V2

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V2 customer temporal history: V1 plus point-in-time customer return/spend/order behavior and rolling windows.`
- Feature count: `64`
- Holdout Accuracy: `82.17%`
- Holdout Recall: `70.12%`
- Holdout F1: `71.62%`
- Holdout AUC: `83.37%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
