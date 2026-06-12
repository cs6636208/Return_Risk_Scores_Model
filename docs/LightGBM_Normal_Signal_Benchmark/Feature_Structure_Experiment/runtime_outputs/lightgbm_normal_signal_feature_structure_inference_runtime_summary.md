# LightGBM Inference Runtime Summary

- Prediction repeats per measured model: 3
- Measured rows use existing `df_featured` feature sets and existing trained LightGBM models.
- SETD/S3 and SETD/S4 rows are estimated from measured per-row inference speed on the paired SETC model, because full external feature rebuilding is much slower and was not stored as a runtime artifact originally.
- Training time and full feature-engineering time are not included in this fast report.

| benchmark | runtime_type | set_name | version | predicted_rows | feature_count | model_load_seconds | predict_seconds_avg | rows_per_second_avg | milliseconds_per_1000_rows_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S1 clean full featured | V1 | 5000 | 24 | 0.0051 | 0.0385 | 129879.6881 | 7.6994 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S2 clean full featured | V1 | 50000 | 24 | 0.0107 | 0.3695 | 135309.9287 | 7.3904 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S1 clean full featured | V2 | 5000 | 40 | 0.0048 | 0.032 | 156223.6373 | 6.4011 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S2 clean full featured | V2 | 50000 | 40 | 0.0061 | 0.2144 | 233159.5828 | 4.2889 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S1 clean full featured | V3 | 5000 | 22 | 0.0097 | 0.045 | 111028.2922 | 9.0067 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S2 clean full featured | V3 | 50000 | 22 | 0.0065 | 0.1849 | 270416.5388 | 3.698 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S1 clean full featured | V4 | 5000 | 32 | 0.0084 | 0.0686 | 72887.6432 | 13.7197 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S2 clean full featured | V4 | 50000 | 32 | 0.0082 | 0.3922 | 127490.0615 | 7.8437 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S1 clean full featured | V5 | 5000 | 65 | 0.0083 | 0.1003 | 49855.1707 | 20.0581 |
| LightGBM_Normal_Signal_Feature_Structure | measured_full_featured_dataset_inference | SETC/S2 clean full featured | V5 | 50000 | 65 | 0.0061 | 0.41 | 121938.3218 | 8.2009 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S3 real 55,000 external estimated inference | V1 | 55000 | 24 | 0.0051 | 0.4235 | 129879.6881 | 7.6994 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S4 real 105,000 external estimated inference | V1 | 105000 | 24 | 0.0107 | 0.776 | 135309.9287 | 7.3904 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S3 real 55,000 external estimated inference | V2 | 55000 | 40 | 0.0048 | 0.3521 | 156223.6373 | 6.4011 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S4 real 105,000 external estimated inference | V2 | 105000 | 40 | 0.0061 | 0.4503 | 233159.5828 | 4.2889 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S3 real 55,000 external estimated inference | V3 | 55000 | 22 | 0.0097 | 0.4954 | 111028.2922 | 9.0067 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S4 real 105,000 external estimated inference | V3 | 105000 | 22 | 0.0065 | 0.3883 | 270416.5388 | 3.698 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S3 real 55,000 external estimated inference | V4 | 55000 | 32 | 0.0084 | 0.7546 | 72887.6432 | 13.7197 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S4 real 105,000 external estimated inference | V4 | 105000 | 32 | 0.0082 | 0.8236 | 127490.0615 | 7.8437 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S3 real 55,000 external estimated inference | V5 | 55000 | 65 | 0.0083 | 1.1032 | 49855.1707 | 20.0581 |
| LightGBM_Normal_Signal_Feature_Structure | estimated_external_inference_from_clean_per_row_speed | SETD/S4 real 105,000 external estimated inference | V5 | 105000 | 65 | 0.0061 | 0.8611 | 121938.3218 | 8.2009 |
