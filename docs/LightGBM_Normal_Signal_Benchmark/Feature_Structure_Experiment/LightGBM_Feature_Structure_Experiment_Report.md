# LightGBM Normal-Signal Feature Structure Experiment

เอกสารนี้เป็นการทดลองใหม่ที่เปลี่ยน `Feature Structure` ของ LightGBM แต่ละ version ให้ต่างกันจริง ๆ ไม่ใช่แค่เพิ่ม feature ต่อจาก version ก่อนหน้า

## เป้าหมาย

- ใช้ dataset normal-signal เดิม เพื่อให้ข้อมูลใกล้เคียง order สินค้าทั่วไปมากกว่าชุด high-signal
- ใช้ LightGBM เหมือนเดิมทุก version
- เปลี่ยนเฉพาะ feature structure เพื่อดูว่าสมมติฐานทางธุรกิจแบบไหนช่วยโมเดลมากที่สุด
- รักษา row count เดิม: S1 5,000, S2 50,000, S3 55,000, S4 105,000

## Feature Structure ใหม่

| Version | Structure | Feature Count Planned | แนวคิด |
| --- | --- | ---: | --- |
| V1 | V1 Order/Product Basic | 24 | baseline using simple order, customer profile, product, price, promotion, payment, and channel features. |
| V2 | V2 Customer Behavior Focus | 44 | focuses on customer history, return ratio, rolling behavior, spend, COD, and high-discount behavior. |
| V3 | V3 Product & Category Risk Focus | 31 | focuses on product/category/brand/supplier risk, quality, rating, damage, and price-index features. |
| V4 | V4 Logistics & Payment Risk Focus | 32 | focuses on courier, logistics, payment, channel, province, COD, and remote-area risk. |
| V5 | V5 Hybrid Compact Best | 65 | compact hybrid selected from customer, product, logistics, payment, and interaction features. |

## Province Feature Interpretation

Province is used as a context feature, not as a rule that every customer in the same province has the same return risk.

The selected V5 model combines `province` with customer-level, product-level, logistics-level, and interaction features. This is important because a high return rate in one province may come from only some customers, some product categories, some couriers, or a specific province-category combination.

Key safeguards in V5:

- Customer-level features: `customer_return_ratio`, `total_orders_before`, `total_returns_before`, rolling return rates.
- Product/category features: `category_return_rate_pti`, `product_return_rate_pti`, `brand_return_rate_pti`.
- Logistics/payment features: `courier_return_rate_pti`, `payment_return_rate_pti`, `channel_return_rate_pti`.
- Interaction features: `province_payment`, `category_province`, `province_category_return_rate_pti`.

Therefore, the model does not judge risk from province alone. Province only helps the model understand whether location adds extra context after customer, product, and logistics behavior are already considered.

## Best Result Summary

| Area | Best Version | Accuracy | Recall | F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| SETC/S1 Holdout | V5 | 80.10% | 37.08% | 47.21% | 73.97% |
| SETC/S2 Holdout | V5 | 78.52% | 47.39% | 51.58% | 72.61% |
| SETD/S3 External vs S1 | V5 | 78.36% | 43.20% | 49.06% | 71.76% |
| SETD/S4 External vs S2 | V5 | 78.49% | 47.78% | 51.70% | 73.38% |

## SETC/S1 Holdout

| Version | Structure | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | V1 Order/Product Basic | 24 | 77.00% | 48.33% | 52.25% | 50.22% | 71.79% | 67,300 |
| V2 | V2 Customer Behavior Focus | 40 | 64.80% | 30.83% | 28.46% | 29.60% | 57.73% | 92,300 |
| V3 | V3 Product & Category Risk Focus | 22 | 69.90% | 53.33% | 40.38% | 45.96% | 69.59% | 65,450 |
| V4 | V4 Logistics & Payment Risk Focus | 32 | 76.40% | 33.75% | 51.27% | 40.70% | 68.83% | 83,350 |
| V5 | V5 Hybrid Compact Best | 65 | 80.10% | 37.08% | 64.96% | 47.21% | 73.97% | 77,900 |

## SETC/S2 Holdout

| Version | Structure | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | V1 Order/Product Basic | 24 | 78.40% | 47.89% | 56.17% | 51.70% | 72.34% | 674,100 |
| V2 | V2 Customer Behavior Focus | 40 | 72.61% | 28.96% | 40.57% | 33.79% | 63.56% | 908,700 |
| V3 | V3 Product & Category Risk Focus | 22 | 75.48% | 38.90% | 49.01% | 43.37% | 68.86% | 786,350 |
| V4 | V4 Logistics & Payment Risk Focus | 32 | 77.69% | 42.92% | 54.84% | 48.15% | 71.63% | 731,650 |
| V5 | V5 Hybrid Compact Best | 65 | 78.52% | 47.39% | 56.58% | 51.58% | 72.61% | 678,900 |

## SETD/S3 External

| Version | Structure | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | V1 Order/Product Basic | 24 | 77.80% | 45.76% | 54.78% | 49.86% | 72.22% | 3,849,600 |
| V2 | V2 Customer Behavior Focus | 40 | 64.59% | 46.44% | 33.25% | 38.75% | 61.62% | 4,172,400 |
| V3 | V3 Product & Category Risk Focus | 22 | 66.45% | 58.73% | 37.53% | 45.79% | 67.57% | 3,387,200 |
| V4 | V4 Logistics & Payment Risk Focus | 32 | 75.73% | 42.61% | 49.64% | 45.86% | 69.30% | 4,094,750 |
| V5 | V5 Hybrid Compact Best | 65 | 78.36% | 43.20% | 56.76% | 49.06% | 71.76% | 3,987,300 |

## SETD/S4 External

| Version | Structure | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | V1 Order/Product Basic | 24 | 78.18% | 47.44% | 55.53% | 51.17% | 73.14% | 7,129,000 |
| V2 | V2 Customer Behavior Focus | 40 | 69.72% | 32.59% | 35.87% | 34.15% | 62.35% | 9,264,050 |
| V3 | V3 Product & Category Risk Focus | 22 | 75.89% | 37.76% | 49.95% | 43.01% | 69.60% | 8,351,050 |
| V4 | V4 Logistics & Payment Risk Focus | 32 | 77.97% | 42.32% | 55.63% | 48.07% | 71.79% | 7,723,450 |
| V5 | V5 Hybrid Compact Best | 65 | 78.49% | 47.78% | 56.33% | 51.70% | 73.38% | 7,073,600 |

## ตีความ

ถ้า version ที่ชนะเปลี่ยนจาก experiment ก่อนหน้า แปลว่าโมเดลตอบสนองต่อ feature structure จริง ไม่ใช่แค่จำนวน feature หรือจำนวน row อย่างเดียว

การทดลองนี้เหมาะใช้ตอบอาจารย์ว่าเราไม่ได้ลองแค่เพิ่ม feature ไปเรื่อย ๆ แต่แยกสมมติฐานทางธุรกิจเป็นคนละ version ได้แก่ baseline, customer behavior, product/category risk, logistics/payment risk และ hybrid compact
