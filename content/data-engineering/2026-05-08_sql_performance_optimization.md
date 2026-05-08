# SQL Performance Optimization Guide

## Index Strategy

### When to Index
- Columns in WHERE clauses
- JOIN columns
- ORDER BY columns
- High-cardinality columns

### When NOT to Index
- Small tables (< 1000 rows)
- Columns with low cardinality (boolean)
- Columns that are frequently updated
- Wide columns (long text)

### Composite Index Rules
Order matters! (a, b, c) index helps:
- WHERE a = 1
- WHERE a = 1 AND b = 2
- WHERE a = 1 AND b = 2 AND c = 3

Does NOT help:
- WHERE b = 2 (leftmost prefix missing)
- WHERE b = 2 AND c = 3

## Query Anti-Patterns

### 1. SELECT *
```sql
-- Bad: fetches all columns, prevents covering index usage
SELECT * FROM orders WHERE status = 'pending';

-- Good: fetch only what you need
SELECT order_id, amount FROM orders WHERE status = 'pending';
```

### 2. Functions on Indexed Columns
```sql
-- Bad: prevents index usage
WHERE YEAR(created_at) = 2024

-- Good: range scan uses index
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'
```

### 3. Implicit Type Casting
```sql
-- Bad: user_id is INT but compared to STRING
WHERE user_id = '12345'

-- Good: matching types
WHERE user_id = 12345
```

### 4. Correlated Subqueries
```sql
-- Bad: executes subquery per row
SELECT * FROM orders o
WHERE amount > (SELECT AVG(amount) FROM orders WHERE user_id = o.user_id);

-- Good: use window function or CTE
WITH user_avg AS (
    SELECT user_id, AVG(amount) AS avg_amount FROM orders GROUP BY user_id
)
SELECT o.* FROM orders o
JOIN user_avg ua ON o.user_id = ua.user_id
WHERE o.amount > ua.avg_amount;
```

## EXPLAIN ANALYZE Checklist
1. Look for Seq Scan on large tables → add index
2. Check estimated vs actual rows → update statistics
3. Look for Nested Loop on large joins → consider Hash Join
4. Watch for Sort operations → add index or increase work_mem
5. Check for high I/O → consider partitioning
