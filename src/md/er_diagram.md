# 📊 Entity Relationship Diagram (ERD)

This diagram visualizes the data structure of the Return Risk Prediction system.

```mermaid
erDiagram
    customers ||--o{ orders : places
    products ||--o{ orders : "included in"
    suppliers ||--o{ products : supplies
    couriers ||--o{ orders : delivers
    orders ||--o{ returns : "may have"
    orders ||--o{ risk_scores : "scored as"
    promotions ||--o{ orders : "applied to"


    customers {
        VARCHAR customer_id PK
        VARCHAR customer_name
        VARCHAR customer_phone
        VARCHAR membership_tier
        VARCHAR preferred_channel
        VARCHAR province
        DATE registration_date
    }

    products {
        VARCHAR product_id PK
        VARCHAR product_name
        VARCHAR category
        VARCHAR brand
        VARCHAR supplier_id FK
        NUMERIC unit_price
        BOOLEAN is_fragile
    }

    suppliers {
        VARCHAR supplier_id PK
        VARCHAR supplier_name
        VARCHAR contact
    }

    couriers {
        VARCHAR courier_id PK
        VARCHAR courier_name
    }

    promotions {
        VARCHAR promo_id PK
        VARCHAR promo_name
        VARCHAR promo_type
        NUMERIC discount_rate
        DATE start_date
        DATE end_date
    }

    orders {
        VARCHAR order_id PK
        VARCHAR customer_id FK
        VARCHAR product_id FK
        VARCHAR courier_id FK
        VARCHAR promo_id FK
        TIMESTAMP order_date
        TIMESTAMP delivery_time_expected_days
        TIMESTAMP delivery_date
        VARCHAR channel_type
        VARCHAR payment_method
        INTEGER quantity
        NUMERIC unit_price
        NUMERIC tier_discount_pct
        NUMERIC campaign_discount_pct
        NUMERIC total_amount
        INTEGER delivery_days
        BOOLEAN is_repurchased_item
        INTEGER order_hour
        INTEGER days_since_last_order
        INTEGER hist_order_count
        NUMERIC hist_return_rate
        SMALLINT is_returned
    }

    returns {
        VARCHAR return_id PK
        VARCHAR order_id FK
        VARCHAR customer_id FK
        DATE return_date
        VARCHAR return_reason
        VARCHAR return_scenario
        VARCHAR item_condition
        VARCHAR return_status
        NUMERIC refund_amount
    }

    risk_scores {
        VARCHAR score_id PK
        VARCHAR order_id FK
        NUMERIC risk_score
        VARCHAR risk_tier
        TIMESTAMP scored_at
        JSONB shap_values
    }
```

## Key Business Logic Notes
- **Order Table**: Acts as the central transaction hub, linking customers, products, logistics, and promotions.
- **Risk Scores**: Each order is evaluated by the ML model, generating a risk score and tier.
- **Returns**: A separate entity tracking post-delivery return events, linked back to the original order. Includes `item_condition` to distinguish between carrier damage and customer bracketing.
