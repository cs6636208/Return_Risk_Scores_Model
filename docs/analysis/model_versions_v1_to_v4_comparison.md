# Model Version Comparison V1-V4

## Summary

- Highest Accuracy: `V1` with `0.7080`
- Best practical return-risk candidate by low Cost + F1/Recall: `V2`
- V4 is the cleanest/auditable pipeline, but V2 still has the strongest practical metric mix in the existing artifacts.

## Metric Comparison

| version | model | feature_count | threshold | accuracy | precision | recall | f1 | auc | cost_thb | performance_rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | XGBoost | 136 | 0.5000 | 0.7080 | 0.4968 | 0.2680 | 0.3482 | 0.6882 | 35,900 | D |
| V2 | XGBoost | 38 | 0.5000 | 0.6787 | 0.4628 | 0.6560 | 0.5427 | 0.7366 | 19,550 | B |
| V3 | Stacking (XGB+LGB+CAT) |  | 0.5000 | 0.6667 | 0.4484 | 0.6376 | 0.5265 | 0.7190 | 20,400 | B |
| V4 | XGBoost | 120 | 0.3300 | 0.6540 | 0.4331 | 0.6117 | 0.5071 | 0.6827 | 28,600 | B |

## Interpretation

- V1: Highest simple accuracy among V1-V3, but recall is low and cost is high.
- V2: Best overall practical version before V4: strong recall, F1, AUC, and lowest cost.
- V3: Stacking improves AUC/F1 over V1, but does not beat V2 on cost or recall.
- V4: Auditable cleaned-data XGBoost with selected features; balanced but still below V2 on F1/AUC/cost.

## Recommendation

If the report is judged by Accuracy only, V1 is highest among production-safe versions. If the goal is real return-risk usage, V2 is still the best overall from the saved results because it has the lowest Cost, best F1, best AUC, and high Recall. V4 should be kept as the clean/auditable XGBoost feature-selection path and can be improved further by borrowing the stronger V2 features and threshold strategy.
