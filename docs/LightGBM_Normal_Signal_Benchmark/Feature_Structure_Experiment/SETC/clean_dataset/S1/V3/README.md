# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S1 V3

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s1.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V3 Product & Category Risk Focus: focuses on product/category/brand/supplier risk, quality, rating, damage, and price-index features.`
- Feature count: `22`
- Holdout Accuracy: `69.90%`
- Holdout Recall: `53.33%`
- Holdout F1: `45.96%`
- Holdout AUC: `69.59%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
