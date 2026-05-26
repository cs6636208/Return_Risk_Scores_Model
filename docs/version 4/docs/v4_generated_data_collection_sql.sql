-- V4 Data Collection SQL Template
-- Pull order, return, customer, product, courier, promotion, and risk-scoring data.

SELECT
    o.order_id,
    o.order_date,
    o.expected_delivery_date,
    o.delivery_date,
    c.customer_id,
    c.gender,
    c.age,
    c.membership_tier,
    c.preferred_channel,
    c.province,
    c.registration_date,
    p.product_id,
    p.category,
    p.brand,
    p.is_fragile,
    p.product_rating,
    cr.courier_id,
    cr.courier_name,
    cr.courier_type,
    pr.promo_id,
    pr.promo_name,
    pr.promo_type,
    pr.promo_discount_rate,
    o.channel_type,
    o.payment_method,
    o.quantity,
    o.unit_price,
    o.total_discount_pct,
    o.total_amount,
    r.return_id,
    r.return_date,
    r.return_reason,
    r.refund_amount,
    CASE WHEN r.return_id IS NULL THEN 0 ELSE 1 END AS is_returned
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON o.product_id = p.product_id
LEFT JOIN couriers cr ON o.courier_id = cr.courier_id
LEFT JOIN promotions pr ON o.promo_id = pr.promo_id
LEFT JOIN returns r ON o.order_id = r.order_id;
