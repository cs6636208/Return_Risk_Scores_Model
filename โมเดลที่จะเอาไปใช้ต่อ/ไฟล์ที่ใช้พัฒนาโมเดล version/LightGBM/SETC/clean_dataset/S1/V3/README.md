# LightGBM_SETC_S1_REBUILD V3

- Source dataset: `docs\LightGBM\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V3 product and logistics risk: V2 plus product/category/brand/courier point-in-time risk and quality/logistics scores.`
- Feature count: `81`
- Holdout Accuracy: `81.90%`
- Holdout Recall: `65.52%`
- Holdout F1: `69.78%`
- Holdout AUC: `82.71%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
