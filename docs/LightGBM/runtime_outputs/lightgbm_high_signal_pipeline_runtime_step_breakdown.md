# LightGBM High Signal Pipeline Runtime Step Breakdown

Pipeline Runtime = Query/Load Data + Clean/Prepare Data + Feature Engineering + Load Feature Schema + Load Model + Predict

## S1-like DB 5,000 rows

| Version | Features | Step 1 Query/Load | Step 2 Clean | Step 3 Feature Eng. | Step 4 Feature Schema | Step 5 Load Model | Step 6 Predict | Total Pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 25 | 0.4334s | 0.1230s | 0.1308s | 0.0017s | 0.0101s | 0.0537s | 0.6297s |
| V2 | 64 | 0.4334s | 0.1230s | 30.0664s | 0.0014s | 0.0109s | 0.0633s | 30.5754s |
| V3 | 81 | 0.4334s | 0.1230s | 32.5470s | 0.0016s | 0.0079s | 0.0577s | 33.0476s |
| V4 | 107 | 0.4334s | 0.1230s | 32.9234s | 0.0020s | 0.0064s | 0.1017s | 33.4670s |
| V5 | 64 | 0.4334s | 0.1230s | 32.9269s | 0.0019s | 0.0070s | 0.0708s | 33.4400s |

## S2-like DB 50,000 rows

| Version | Features | Step 1 Query/Load | Step 2 Clean | Step 3 Feature Eng. | Step 4 Feature Schema | Step 5 Load Model | Step 6 Predict | Total Pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 25 | 1.0301s | 0.7077s | 0.7411s | 0.0104s | 0.0246s | 0.3334s | 2.1396s |
| V2 | 64 | 1.0301s | 0.7077s | 228.6391s | 0.0021s | 0.0163s | 0.5675s | 230.2552s |
| V3 | 81 | 1.0301s | 0.7077s | 253.3153s | 0.0033s | 0.0144s | 0.4796s | 254.8428s |
| V4 | 107 | 1.0301s | 0.7077s | 256.8114s | 0.0037s | 0.0113s | 0.6849s | 258.5415s |
| V5 | 64 | 1.0301s | 0.7077s | 256.8475s | 0.0030s | 0.0138s | 0.5453s | 258.4398s |

## Interpretation
- Step 3 Feature Engineering ???????????????? V2-V5 ?????????????? customer history, rolling windows, point-in-time return rate ??? interaction features
- Step 6 Predict ??????????????????????????? pipeline ??????? ????????? LightGBM ????????????
- V1 ?????????????????? feature ??????? ???????????? history/risk aggregate ????
