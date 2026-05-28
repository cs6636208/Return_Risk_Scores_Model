/*
  Dataset query for:
  v2_xgboost_safe_plus_rolling_HIGH_ACCURACY

  Target table:
    public.order_history_complete_v2_new

  If your table was created with quoted uppercase NEW, replace every table name with:
    public."order_history_complete_v2_NEW"

  Goal:
    1) Pull the clean source dataset used by V2 HIGH_ACCURACY.
    2) Build order-time-safe engineered features with point-in-time customer history.
    3) Never use post-event leakage fields as model inputs.

  Leakage fields intentionally excluded from model features:
    return_id, return_date, return_reason, return_scenario, item_condition,
    return_status, refund_amount, score_id, risk_score, risk_tier, scored_at,
    shap_values, delivery_date, delivery_days, delay_days
*/

-- =========================================================
-- 0) Basic validation
-- =========================================================

SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT order_id) AS distinct_orders,
    COUNT(DISTINCT customer_id) AS distinct_customers,
    SUM(CASE WHEN is_returned = 1 THEN 1 ELSE 0 END) AS returned_orders,
    SUM(CASE WHEN is_returned = 0 THEN 1 ELSE 0 END) AS not_returned_orders,
    AVG(is_returned::numeric) AS return_rate,
    MIN(order_date) AS min_order_date,
    MAX(order_date) AS max_order_date
FROM public.order_history_complete_v2_new;


-- =========================================================
-- 1) Pull clean source dataset
--    This mirrors clean_dataset_v2_high_signal.csv.
-- =========================================================

SELECT
    order_id,
    order_date,
    expected_delivery_date,
    delivery_date,
    customer_id,
    customer_name,
    customer_phone,
    gender,
    age,
    membership_tier,
    preferred_channel,
    province,
    registration_date,
    customer_age_days,
    product_id,
    product_name,
    category,
    brand,
    is_fragile,
    product_rating,
    supplier_id,
    supplier_name,
    supplier_contact,
    courier_id,
    courier_name,
    courier_type,
    avg_delivery_days,
    damage_rate,
    coverage_region,
    promo_id,
    promo_name,
    promo_type,
    promo_discount_rate,
    promo_start_date,
    promo_end_date,
    channel_type,
    payment_method,
    quantity,
    unit_price,
    tier_discount_pct,
    campaign_discount_pct,
    total_discount_pct,
    discount_applied_amount,
    total_amount,
    delivery_time_expected_days,
    delivery_days,
    delay_days,
    is_repurchased_item,
    order_hour,
    days_since_last_order,
    hist_order_count,
    hist_return_rate,
    return_id,
    return_date,
    return_reason,
    return_scenario,
    item_condition,
    return_status,
    refund_amount,
    score_id,
    risk_score,
    risk_tier,
    scored_at,
    shap_values,
    is_returned
FROM public.order_history_complete_v2_new
ORDER BY order_date, order_id;


-- =========================================================
-- 2) Build df_engineered_v2_HIGH_ACCURACY-like dataset
--    This query creates the features used before XGBoost.
--
--    Important:
--      - History uses only rows where h.order_date < current order_date.
--      - Current order is not counted in its own history.
--      - customer_id/order_id/order_date are kept for audit only.
--      - Do not feed identifiers into the model.
-- =========================================================

WITH base AS (
    SELECT
        order_id,
        order_date::timestamp AS order_date,
        expected_delivery_date::timestamp AS expected_delivery_date,
        customer_id,
        gender,
        age,
        membership_tier,
        preferred_channel,
        province,
        registration_date::date AS registration_date,
        customer_age_days,
        category,
        brand,
        is_fragile,
        product_rating,
        courier_type,
        avg_delivery_days,
        damage_rate,
        coverage_region,
        promo_type,
        promo_discount_rate,
        channel_type,
        payment_method,
        quantity,
        unit_price,
        tier_discount_pct,
        campaign_discount_pct,
        total_discount_pct,
        discount_applied_amount,
        total_amount,
        delivery_time_expected_days,
        is_repurchased_item,
        order_hour,
        is_returned
    FROM public.order_history_complete_v2_new
),
engineered_base AS (
    SELECT
        b.order_id,
        b.customer_id,
        b.order_date,
        b.gender,
        b.age,
        b.membership_tier,
        b.preferred_channel,
        b.province,
        b.customer_age_days,
        GREATEST(EXTRACT(DAY FROM (b.order_date::date - b.registration_date)) / 30.44, 0) AS customer_tenure_months,
        b.category,
        b.brand,
        b.is_fragile,
        b.product_rating,
        b.courier_type,
        b.avg_delivery_days,
        b.damage_rate,
        b.coverage_region,
        b.promo_type,
        b.promo_discount_rate,
        b.channel_type,
        b.payment_method,
        b.quantity,
        b.unit_price,
        b.tier_discount_pct,
        b.campaign_discount_pct,
        b.total_discount_pct,
        b.discount_applied_amount,
        CASE
            WHEN (b.unit_price * b.quantity) > 0
                THEN b.discount_applied_amount::numeric / NULLIF((b.unit_price * b.quantity), 0)
            ELSE 0
        END AS discount_amount_ratio,
        b.total_amount,
        CASE
            WHEN b.quantity > 0 THEN b.total_amount::numeric / NULLIF(b.quantity, 0)
            ELSE 0
        END AS amount_per_item,
        LN(1 + GREATEST(b.total_amount::numeric, 0)) AS log_total_amount,
        CASE
            WHEN b.total_amount >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY b.total_amount) OVER ()
                THEN 1 ELSE 0
        END AS high_value_order,
        b.delivery_time_expected_days,
        b.is_repurchased_item,
        b.order_hour,
        EXTRACT(MONTH FROM b.order_date)::int AS order_month,
        EXTRACT(DOW FROM b.order_date)::int AS order_dayofweek,
        CASE WHEN EXTRACT(DOW FROM b.order_date)::int IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        CASE
            WHEN b.age < 25 THEN 'under_25'
            WHEN b.age < 35 THEN '25_34'
            WHEN b.age < 45 THEN '35_44'
            WHEN b.age < 55 THEN '45_54'
            ELSE '55_plus'
        END AS age_group,
        CASE WHEN LOWER(b.payment_method) LIKE '%cod%' OR LOWER(b.payment_method) LIKE '%cash%' THEN 1 ELSE 0 END AS is_cod,
        CASE WHEN LOWER(b.payment_method) LIKE '%bank%' OR LOWER(b.payment_method) LIKE '%transfer%' THEN 1 ELSE 0 END AS is_bank_transfer,
        CASE WHEN LOWER(b.payment_method) LIKE '%credit%' OR LOWER(b.payment_method) LIKE '%card%' THEN 1 ELSE 0 END AS is_credit_card,
        CASE WHEN b.total_discount_pct >= 0.20 THEN 1 ELSE 0 END AS is_high_discount,
        CASE WHEN b.product_rating < 3.5 THEN 1 ELSE 0 END AS low_rating_alert,
        CASE
            WHEN b.damage_rate >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY b.damage_rate) OVER ()
              OR b.avg_delivery_days >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY b.avg_delivery_days) OVER ()
                THEN 1 ELSE 0
        END AS logistics_risk,
        CONCAT(b.category, '__', b.payment_method) AS category_payment,
        CONCAT(b.category, '__', b.channel_type) AS category_channel,
        CONCAT(b.province, '__', b.payment_method) AS province_payment,
        CONCAT(b.membership_tier, '__', b.payment_method) AS tier_payment,
        b.is_returned
    FROM base b
),
with_history AS (
    SELECT
        e.*,
        COALESCE(life.hist_order_count, 0) AS hist_order_count,
        COALESCE(life.hist_return_rate, 0) AS hist_return_rate,
        COALESCE((e.order_date::date - life.last_order_date::date), 999) AS days_since_last_order,
        COALESCE((e.order_date::date - life.last_return_date::date), 999) AS days_since_last_return,

        COALESCE(w30.hist_order_count, 0) AS hist_order_count_30d,
        COALESCE(w30.hist_return_count, 0) AS hist_return_count_30d,
        COALESCE(w30.hist_return_rate, 0) AS hist_return_rate_30d,
        COALESCE(w30.hist_spend_sum, 0) AS hist_spend_sum_30d,

        COALESCE(w60.hist_order_count, 0) AS hist_order_count_60d,
        COALESCE(w60.hist_return_count, 0) AS hist_return_count_60d,
        COALESCE(w60.hist_return_rate, 0) AS hist_return_rate_60d,
        COALESCE(w60.hist_spend_sum, 0) AS hist_spend_sum_60d,

        COALESCE(w90.hist_order_count, 0) AS hist_order_count_90d,
        COALESCE(w90.hist_return_count, 0) AS hist_return_count_90d,
        COALESCE(w90.hist_return_rate, 0) AS hist_return_rate_90d,
        COALESCE(w90.hist_spend_sum, 0) AS hist_spend_sum_90d,

        COALESCE(w180.hist_order_count, 0) AS hist_order_count_180d,
        COALESCE(w180.hist_return_count, 0) AS hist_return_count_180d,
        COALESCE(w180.hist_return_rate, 0) AS hist_return_rate_180d,
        COALESCE(w180.hist_spend_sum, 0) AS hist_spend_sum_180d,

        COALESCE(w365.hist_order_count, 0) AS hist_order_count_365d,
        COALESCE(w365.hist_return_count, 0) AS hist_return_count_365d,
        COALESCE(w365.hist_return_rate, 0) AS hist_return_rate_365d,
        COALESCE(w365.hist_spend_sum, 0) AS hist_spend_sum_365d
    FROM engineered_base e
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            MAX(h.order_date) AS last_order_date,
            MAX(h.order_date) FILTER (WHERE h.is_returned = 1) AS last_return_date
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
    ) life ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            SUM(h.is_returned)::int AS hist_return_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            SUM(h.total_amount)::numeric AS hist_spend_sum
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
          AND h.order_date >= e.order_date - INTERVAL '30 days'
    ) w30 ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            SUM(h.is_returned)::int AS hist_return_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            SUM(h.total_amount)::numeric AS hist_spend_sum
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
          AND h.order_date >= e.order_date - INTERVAL '60 days'
    ) w60 ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            SUM(h.is_returned)::int AS hist_return_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            SUM(h.total_amount)::numeric AS hist_spend_sum
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
          AND h.order_date >= e.order_date - INTERVAL '90 days'
    ) w90 ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            SUM(h.is_returned)::int AS hist_return_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            SUM(h.total_amount)::numeric AS hist_spend_sum
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
          AND h.order_date >= e.order_date - INTERVAL '180 days'
    ) w180 ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*)::int AS hist_order_count,
            SUM(h.is_returned)::int AS hist_return_count,
            AVG(h.is_returned::numeric) AS hist_return_rate,
            SUM(h.total_amount)::numeric AS hist_spend_sum
        FROM base h
        WHERE h.customer_id = e.customer_id
          AND h.order_date < e.order_date
          AND h.order_date >= e.order_date - INTERVAL '365 days'
    ) w365 ON TRUE
)
SELECT
    order_id,
    customer_id,
    order_date,
    gender,
    age,
    membership_tier,
    preferred_channel,
    province,
    customer_age_days,
    customer_tenure_months,
    category,
    brand,
    is_fragile,
    product_rating,
    courier_type,
    avg_delivery_days,
    damage_rate,
    coverage_region,
    promo_type,
    promo_discount_rate,
    channel_type,
    payment_method,
    quantity,
    unit_price,
    tier_discount_pct,
    campaign_discount_pct,
    total_discount_pct,
    discount_applied_amount,
    discount_amount_ratio,
    total_amount,
    amount_per_item,
    log_total_amount,
    high_value_order,
    delivery_time_expected_days,
    is_repurchased_item,
    order_hour,
    days_since_last_order,
    days_since_last_return,
    hist_order_count,
    hist_return_rate,
    order_month,
    order_dayofweek,
    is_weekend,
    age_group,
    is_cod,
    is_bank_transfer,
    is_credit_card,
    is_high_discount,
    low_rating_alert,
    logistics_risk,
    category_payment,
    category_channel,
    province_payment,
    tier_payment,
    hist_order_count_30d,
    hist_return_count_30d,
    hist_return_rate_30d,
    hist_spend_sum_30d,
    hist_order_count_60d,
    hist_return_count_60d,
    hist_return_rate_60d,
    hist_spend_sum_60d,
    hist_order_count_90d,
    hist_return_count_90d,
    hist_return_rate_90d,
    hist_spend_sum_90d,
    hist_order_count_180d,
    hist_return_count_180d,
    hist_return_rate_180d,
    hist_spend_sum_180d,
    hist_order_count_365d,
    hist_return_count_365d,
    hist_return_rate_365d,
    hist_spend_sum_365d,
    is_returned,
    CASE
        WHEN MOD(ABS(HASHTEXT(order_id::text)), 10) < 8 THEN 'train'
        ELSE 'test'
    END AS dataset_split
FROM with_history
ORDER BY order_date, order_id;


-- =========================================================
-- 3) Real-time style query for one incoming order/customer
--
-- Replace:
--   :customer_id
--   :current_order_date
-- with bind parameters from API/application code.
-- =========================================================

SELECT
    COUNT(*)::int AS hist_order_count,
    COALESCE(SUM(is_returned), 0)::int AS hist_return_count,
    COALESCE(AVG(is_returned::numeric), 0) AS hist_return_rate,
    COALESCE(SUM(total_amount), 0) AS hist_spend_sum
FROM public.order_history_complete_v2_new
WHERE customer_id = :customer_id
  AND order_date < :current_order_date::timestamp;


SELECT
    COUNT(*)::int AS hist_order_count_90d,
    COALESCE(SUM(is_returned), 0)::int AS hist_return_count_90d,
    COALESCE(AVG(is_returned::numeric), 0) AS hist_return_rate_90d,
    COALESCE(SUM(total_amount), 0) AS hist_spend_sum_90d
FROM public.order_history_complete_v2_new
WHERE customer_id = :customer_id
  AND order_date < :current_order_date::timestamp
  AND order_date >= :current_order_date::timestamp - INTERVAL '90 days';


-- =========================================================
-- 4) Recommended indexes for fast history queries
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_order_history_v2_new_customer_order_date
    ON public.order_history_complete_v2_new (customer_id, order_date);

CREATE INDEX IF NOT EXISTS idx_order_history_v2_new_order_id
    ON public.order_history_complete_v2_new (order_id);

CREATE INDEX IF NOT EXISTS idx_order_history_v2_new_is_returned
    ON public.order_history_complete_v2_new (is_returned);
