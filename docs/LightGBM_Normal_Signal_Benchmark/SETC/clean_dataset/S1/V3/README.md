# LightGBM_SETC_S1_REBUILD V3

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V3 product and logistics risk: V2 plus product/category/brand/courier point-in-time risk and quality/logistics scores.`
- Feature count: `81`
- Holdout Accuracy: `77.20%`
- Holdout Recall: `45.42%`
- Holdout F1: `48.88%`
- Holdout AUC: `72.28%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
