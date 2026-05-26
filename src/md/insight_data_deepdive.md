# Deep Dive Data Insight Report

## 1. Geographic & Logistics (Province Analysis)

             total_orders  avg_delivery_days  avg_expected_days  avg_gap  late_delivery_rate  return_rate  return_rate_when_late  return_rate_when_ontime
province                                                                                                                                                 
Bangkok               676               3.55               1.98     1.57               67.16        31.95                  39.43                    16.67
Chiang Mai            652               3.60               2.01     1.59               69.48        30.98                  35.76                    20.10
Chonburi              608               3.37               1.98     1.39               65.79        26.97                  33.00                    15.38
Khon Kaen             644               3.43               2.00     1.43               63.04        28.11                  33.50                    18.91
Nonthaburi            634               3.60               2.01     1.59               69.87        35.02                  39.50                    24.61
Phuket                593               3.62               1.97     1.65               69.98        33.90                  39.28                    21.35
Remote_Area           563               3.51               2.02     1.49               67.67        26.47                  32.55                    13.74
Songkhla              630               3.47               1.96     1.51               66.67        28.41                  32.62                    20.00


## 2. COD Risk by Province (Bangkok vs Regional)

             mean_Bank_Transfer  mean_COD  mean_Credit_Card  count_Bank_Transfer  count_COD  count_Credit_Card
province                                                                                                      
Bangkok                   31.72     28.63             36.06                  227        241                208
Chiang Mai                31.51     30.32             31.13                  219        221                212
Chonburi                  30.73     26.70             23.59                  192        221                195
Khon Kaen                 27.93     32.21             24.30                  222        208                214
Nonthaburi                32.43     38.39             33.78                  185        224                225
Phuket                    37.19     30.89             33.50                  199        191                203
Remote_Area               26.21     22.40             31.03                  206        183                174
Songkhla                  28.10     26.48             30.85                  210        219                201


## 3. Channel vs Category (Impulse Buying & Bracketing)

                             return_rate_%  total_orders
category       channel_type                             
Electronics    TV_Show               36.82           258
Home_Appliance Shopee                34.77           256
Electronics    TikTok                33.88           245
Supplement     Mobile_App            33.74           246
Home_Appliance Mobile_App            33.05           233
Supplement     TV_Show               32.30           257
Fashion        TV_Show               32.20           236
Supplement     TikTok                32.16           255
Electronics    Shopee                31.89           254
Fashion        TikTok                30.27           261
Home_Appliance TV_Show               29.84           248
Fashion        Shopee                29.41           238
               Mobile_App            28.57           231
Home_Appliance TikTok                28.35           254
Cosmetics      TikTok                28.02           257
               Shopee                27.84           255
Supplement     Shopee                27.72           267
Cosmetics      TV_Show               26.64           259
Electronics    Mobile_App            24.14           261
Cosmetics      Mobile_App            24.02           229


## 4. Return Reasons by Category

return_reason   Better Price Elsewhere  Changed Mind  Defective  Wrong Item
category                                                                   
Cosmetics                        23.97         20.97      28.46       26.59
Electronics                      22.98         26.40      30.43       20.19
Fashion                          22.68         28.18      28.18       20.96
Home_Appliance                   20.51         29.17      27.24       23.08
Supplement                       26.71         23.60      25.47       24.22

