# LightGBM Rebuild Summary

This folder was rebuilt from scratch with a semi-realistic high-signal synthetic dataset. Row counts are preserved, but the generated data now has more varied orders/products/customers and a return ratio near 33%.

## Main Results

| Area | Best Result |
| --- | --- |
| SETC/S1 holdout | V3 = 77.70% |
| SETC/S2 holdout | V1 = 78.74% |
| SETD/S3 external vs S1 | V1 = 77.85% |
| SETD/S4 external vs S2 | V1 = 79.18% |

## Important Interpretation

Use holdout and SETD external values as the current high-signal benchmark results. Do not describe these numbers as real production accuracy because the dataset is still generated/synthetic.

SETD external feature engineering is rebuilt with SETC history context before prediction. This means S3/S4 are no longer tested as isolated CSV rows with no prior customer/product/category history.
