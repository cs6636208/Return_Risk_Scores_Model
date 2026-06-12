# LightGBM SETC vs SETD Accuracy Stability Comparison

This report compares the same model version between clean holdout data and real external full-dataset data.

- S1_vs_S3 compares SETC/S1 clean 5,000 holdout with SETD/S3 real 55,000 full test.
- S2_vs_S4 compares SETC/S2 clean 50,000 holdout with SETD/S4 real 105,000 full test.

SETC accuracy is holdout accuracy from the clean dataset split. SETD accuracy is external full-dataset prediction accuracy; SETD is not retrained and is not split again.

## Summary Table

| comparison_pair | version | feature_structure | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | absolute_gap_pp | stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1_vs_S3 | V1 | Order/Product Basic | 77.0 | 77.8 | 0.8 | 0.8 | Very close |
| S1_vs_S3 | V2 | Customer Behavior Focus | 64.8 | 64.59 | -0.21 | 0.21 | Very close |
| S1_vs_S3 | V3 | Product & Category Risk Focus | 69.9 | 66.45 | -3.45 | 3.45 | Moderate gap |
| S1_vs_S3 | V4 | Logistics & Payment Risk Focus | 76.4 | 75.73 | -0.67 | 0.67 | Very close |
| S1_vs_S3 | V5 | Hybrid Compact Best | 80.1 | 78.36 | -1.74 | 1.74 | Close |
| S2_vs_S4 | V1 | Order/Product Basic | 78.4 | 78.18 | -0.22 | 0.22 | Very close |
| S2_vs_S4 | V2 | Customer Behavior Focus | 72.61 | 69.72 | -2.89 | 2.89 | Close |
| S2_vs_S4 | V3 | Product & Category Risk Focus | 75.48 | 75.89 | 0.41 | 0.41 | Very close |
| S2_vs_S4 | V4 | Logistics & Payment Risk Focus | 77.69 | 77.97 | 0.28 | 0.28 | Very close |
| S2_vs_S4 | V5 | Hybrid Compact Best | 78.52 | 78.49 | -0.03 | 0.03 | Very close |

## Best By Stability

| comparison_pair | version | feature_structure | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | absolute_gap_pp | stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2_vs_S4 | V5 | Hybrid Compact Best | 78.52 | 78.49 | -0.03 | 0.03 | Very close |
| S1_vs_S3 | V2 | Customer Behavior Focus | 64.8 | 64.59 | -0.21 | 0.21 | Very close |

## Best By Real External Accuracy

| comparison_pair | version | feature_structure | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | absolute_gap_pp | stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2_vs_S4 | V5 | Hybrid Compact Best | 78.52 | 78.49 | -0.03 | 0.03 | Very close |
| S1_vs_S3 | V5 | Hybrid Compact Best | 80.1 | 78.36 | -1.74 | 1.74 | Close |

## Interpretation By Version

### V1: Order/Product Basic

Baseline feature set. It uses simple order, customer profile, product, price, promotion, payment, and channel features. Accuracy is moderate because it does not see deeper customer history or group-level return risk.

| comparison_pair | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | stability |
| --- | --- | --- | --- | --- |
| S1_vs_S3 | 77.0 | 77.8 | 0.8 | Very close |
| S2_vs_S4 | 78.4 | 78.18 | -0.22 | Very close |

### V2: Customer Behavior Focus

Customer-history-heavy feature set. It drops when customers have little history or when the return risk comes from product, logistics, payment, or location context.

| comparison_pair | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | stability |
| --- | --- | --- | --- | --- |
| S1_vs_S3 | 64.8 | 64.59 | -0.21 | Very close |
| S2_vs_S4 | 72.61 | 69.72 | -2.89 | Close |

### V3: Product & Category Risk Focus

Product/category-focused feature set. It catches item and category risk better than V2, but misses customer and logistics/payment context.

| comparison_pair | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | stability |
| --- | --- | --- | --- | --- |
| S1_vs_S3 | 69.9 | 66.45 | -3.45 | Moderate gap |
| S2_vs_S4 | 75.48 | 75.89 | 0.41 | Very close |

### V4: Logistics & Payment Risk Focus

Logistics, payment, channel, province, COD, and remote-area feature set. It improves when risk is tied to delivery or payment context, but still lacks the full customer + product picture.

| comparison_pair | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | stability |
| --- | --- | --- | --- | --- |
| S1_vs_S3 | 76.4 | 75.73 | -0.67 | Very close |
| S2_vs_S4 | 77.69 | 77.97 | 0.28 | Very close |

### V5: Hybrid Compact Best

Balanced hybrid feature set. It combines customer, product/category, logistics/payment, promotion, and interaction features, so it is usually the most stable across clean holdout and real external tests.

| comparison_pair | clean_holdout_accuracy_pct | real_external_accuracy_pct | accuracy_gap_real_minus_clean_pp | stability |
| --- | --- | --- | --- | --- |
| S1_vs_S3 | 80.1 | 78.36 | -1.74 | Close |
| S2_vs_S4 | 78.52 | 78.49 | -0.03 | Very close |
