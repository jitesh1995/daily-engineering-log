-- dbt Model Patterns
-- Common data modeling patterns for analytics engineering

-- ============================================================
-- 1. STAGING MODEL (stg_orders.sql)
-- Light transformations, renaming, type casting
-- ============================================================

/*
{{ config(
    materialized='view',
    schema='staging'
) }}
*/

WITH source AS (
    SELECT * FROM /* {{ source('raw', 'orders') }} */ raw.orders
),

renamed AS (
    SELECT
        id AS order_id,
        user_id,
        LOWER(status) AS order_status,
        CAST(amount AS DECIMAL(10, 2)) AS order_amount,
        CAST(created_at AS TIMESTAMP) AS ordered_at,
        CAST(updated_at AS TIMESTAMP) AS updated_at,
        _loaded_at AS _etl_loaded_at
    FROM source
    WHERE id IS NOT NULL
)

SELECT * FROM renamed;


-- ============================================================
-- 2. INTERMEDIATE MODEL (int_order_items_enriched.sql)
-- Join and enrich, one clear purpose
-- ============================================================

/*
{{ config(materialized='ephemeral') }}
*/

WITH orders AS (
    SELECT * FROM /* {{ ref('stg_orders') }} */ staging.stg_orders
),

items AS (
    SELECT * FROM /* {{ ref('stg_order_items') }} */ staging.stg_order_items
),

products AS (
    SELECT * FROM /* {{ ref('stg_products') }} */ staging.stg_products
)

SELECT
    orders.order_id,
    orders.user_id,
    orders.ordered_at,
    items.product_id,
    products.product_name,
    products.category,
    items.quantity,
    items.unit_price,
    items.quantity * items.unit_price AS line_total
FROM orders
INNER JOIN items ON orders.order_id = items.order_id
INNER JOIN products ON items.product_id = products.product_id;


-- ============================================================
-- 3. MART MODEL (fct_daily_revenue.sql)
-- Business-level fact table
-- ============================================================

/*
{{ config(
    materialized='incremental',
    unique_key='revenue_date',
    schema='marts'
) }}
*/

WITH order_items AS (
    SELECT * FROM /* {{ ref('int_order_items_enriched') }} */ intermediate.int_order_items_enriched
    /*
    {% if is_incremental() %}
    WHERE ordered_at > (SELECT MAX(revenue_date) FROM {{ this }})
    {% endif %}
    */
)

SELECT
    DATE(ordered_at) AS revenue_date,
    category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS total_units,
    SUM(line_total) AS gross_revenue,
    AVG(line_total) AS avg_order_value,
    COUNT(DISTINCT user_id) AS unique_customers
FROM order_items
GROUP BY DATE(ordered_at), category;


-- ============================================================
-- 4. DATA QUALITY TESTS (schema.yml style, as SQL)
-- ============================================================

-- Test: No null order_ids
SELECT order_id
FROM staging.stg_orders
WHERE order_id IS NULL;
-- Expected: 0 rows

-- Test: Referential integrity
SELECT oi.order_id
FROM staging.stg_order_items oi
LEFT JOIN staging.stg_orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;
-- Expected: 0 rows

-- Test: Accepted values
SELECT order_status
FROM staging.stg_orders
WHERE order_status NOT IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled');
-- Expected: 0 rows
