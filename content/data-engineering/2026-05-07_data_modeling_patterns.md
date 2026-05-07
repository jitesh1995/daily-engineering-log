# Data Modeling Patterns

## Star Schema
The classic dimensional model for analytics.

```
                    ┌──────────────┐
                    │  dim_date    │
                    └──────┬───────┘
                           │
┌──────────────┐   ┌──────┴───────┐   ┌──────────────┐
│ dim_customer │───│  fct_orders  │───│ dim_product  │
└──────────────┘   └──────┬───────┘   └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  dim_store   │
                    └──────────────┘
```

### Fact Tables
- Contain measurable events (orders, clicks, payments)
- Foreign keys to dimensions
- Additive metrics (amount, quantity, duration)
- Grain: one row per event

### Dimension Tables
- Descriptive attributes (name, category, location)
- Slowly changing (SCD Type 1, 2, or 3)
- Typically wide (many columns)
- Relatively small row count

## Slowly Changing Dimensions (SCD)

### Type 1: Overwrite
Simply update the value. No history preserved.
Use for: corrections, non-critical attributes.

### Type 2: Add New Row
Add new row with version tracking (valid_from, valid_to, is_current).
Use for: important attributes where history matters (price, status).

### Type 3: Add New Column
Add column for previous value (current_city, previous_city).
Use for: when you only need one level of history.

## One Big Table (OBT)
Pre-join everything for fast queries. Denormalized.
Trade storage for query speed.

### When to Use OBT
- Small-to-medium data volumes
- Simple, repetitive queries
- Self-serve analytics (no joins needed)
- Dashboard backing tables

## Activity Schema
For event-heavy workloads (product analytics, clickstream).

```sql
CREATE TABLE activity_stream (
    activity_id    BIGINT,
    entity_id      VARCHAR,  -- user, device, etc.
    activity_type  VARCHAR,  -- page_view, click, purchase
    occurred_at    TIMESTAMP,
    properties     JSONB     -- flexible payload
);
```

## Data Vault
For enterprise data warehousing at scale.

Components:
- **Hubs**: Business keys (customer_id, product_id)
- **Links**: Relationships between hubs
- **Satellites**: Descriptive attributes with history

Best for: large enterprises, multiple source systems, full audit trail.
