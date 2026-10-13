-- SQL Window Functions Reference
-- Common patterns for analytics queries

-- 1. Running Total
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) AS running_total,
    SUM(revenue) OVER (
        PARTITION BY EXTRACT(YEAR FROM date)
        ORDER BY date
    ) AS ytd_revenue
FROM daily_revenue;

-- 2. Ranking with Ties
SELECT
    product_name,
    category,
    sales,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS row_num,
    RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS dense_rank
FROM product_sales;

-- 3. Moving Average (7-day)
SELECT
    date,
    daily_active_users,
    AVG(daily_active_users) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM user_metrics;

-- 4. Lead/Lag - Period over Period
SELECT
    month,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY month))
        / LAG(revenue, 1) OVER (ORDER BY month) * 100,
        2
    ) AS mom_growth_pct
FROM monthly_revenue;

-- 5. First/Last Value
SELECT
    user_id,
    event_date,
    event_type,
    FIRST_VALUE(event_type) OVER (
        PARTITION BY user_id ORDER BY event_date
    ) AS first_action,
    LAST_VALUE(event_type) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_action
FROM user_events;

-- 6. Percentile / Distribution
SELECT
    department,
    employee_name,
    salary,
    PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS percentile,
    NTILE(4) OVER (PARTITION BY department ORDER BY salary) AS salary_quartile
FROM employees;

-- 7. Gap Detection (session boundaries)
SELECT
    user_id,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event,
    EXTRACT(EPOCH FROM (
        event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time)
    )) / 60 AS minutes_since_last,
    CASE
        WHEN EXTRACT(EPOCH FROM (
            event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time)
        )) / 60 > 30 THEN 1
        ELSE 0
    END AS new_session_flag
FROM clickstream;
