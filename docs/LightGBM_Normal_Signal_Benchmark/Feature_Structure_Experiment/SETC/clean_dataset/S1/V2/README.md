# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S1 V2

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V2 Customer Behavior Focus: focuses on customer history, return ratio, rolling behavior, spend, COD, and high-discount behavior.`
- Feature count: `40`
- Holdout Accuracy: `64.80%`
- Holdout Recall: `30.83%`
- Holdout F1: `29.60%`
- Holdout AUC: `57.73%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
