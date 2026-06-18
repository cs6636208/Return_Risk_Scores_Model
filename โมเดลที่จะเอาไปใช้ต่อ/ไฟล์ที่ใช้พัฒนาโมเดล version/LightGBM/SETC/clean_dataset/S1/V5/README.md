# LightGBM_SETC_S1_REBUILD V5

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V5 compact selected best: reduced feature set selected from V2-V4 to keep performance high while reducing noise/resource use.`
- Feature count: `64`
- Holdout Accuracy: `82.00%`
- Holdout Recall: `59.56%`
- Holdout F1: `67.86%`
- Holdout AUC: `83.27%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
