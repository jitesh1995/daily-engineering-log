# Data Lake Partitioning Guide

## Why Partition?
Partitioning enables **partition pruning** — skipping irrelevant files
during queries. A well-partitioned table can be 100x faster to query.

## Common Partition Strategies

### Time-Based (Most Common)
```
s3://datalake/events/
  ├── year=2024/
  │   ├── month=01/
  │   │   ├── day=01/
  │   │   │   └── part-00000.parquet
  │   │   └── day=02/
  │   └── month=02/
  └── year=2025/
```

Best for: time-series data, logs, events, most analytical queries.

### Category-Based
```
s3://datalake/transactions/
  ├── region=us-east/
  │   ├── type=purchase/
  │   └── type=refund/
  └── region=eu-west/
```

Best for: data commonly filtered by category.

### Hybrid (Time + Category)
```
s3://datalake/orders/
  ├── date=2024-01-15/
  │   ├── country=US/
  │   └── country=UK/
  └── date=2024-01-16/
```

## Partition Column Selection Rules

| Criteria | Good Partition Column | Bad Partition Column |
|----------|----------------------|---------------------|
| Cardinality | Low-medium (< 10K values) | High (user_id, UUID) |
| Query filter | Almost always filtered on | Rarely used in WHERE |
| Distribution | Even across values | Heavily skewed |
| Growth | Predictable over time | Unpredictable |

## File Size Targets

- **Target**: 128 MB - 1 GB per file (Parquet)
- **Too small** (< 10 MB): Too many files, slow listing
- **Too large** (> 2 GB): Slow to read, poor parallelism

## Anti-Patterns

1. **Over-partitioning**: Partitioning by high-cardinality columns
   creates millions of tiny files ("small file problem")
2. **Under-partitioning**: One giant partition that always gets full-scanned
3. **Partitioning by non-filter columns**: No benefit if queries don't
   filter on the partition column

## Compaction
Periodically merge small files into larger ones:
```sql
-- Spark compaction example
df.repartition(10)  -- target ~10 output files
  .write.mode("overwrite")
  .parquet("s3://datalake/events/date=2024-01-15/")
```
