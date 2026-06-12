# LightGBM_NORMAL_SIGNAL_FEATURE_STRUCTURE_S2 V3

- Source dataset: `docs\LightGBM_Normal_Signal_Benchmark\SETC\clean_dataset\clean_dataset_s2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Model: `LightGBM`
- Feature strategy: `V3 Product & Category Risk Focus: focuses on product/category/brand/supplier risk, quality, rating, damage, and price-index features.`
- Feature count: `22`
- Holdout Accuracy: `75.48%`
- Holdout Recall: `38.90%`
- Holdout F1: `43.37%`
- Holdout AUC: `68.86%`

New data policy: the model can predict new rows only when the same feature schema is rebuilt. It does not learn automatically or jump to a new model without retraining.
