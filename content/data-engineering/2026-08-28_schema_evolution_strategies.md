# Schema Evolution Strategies

## The Problem
Data schemas change over time. Columns are added, types change,
fields get renamed. Pipelines must handle this gracefully.

## Approaches

### 1. Backward Compatible Changes (Safe)
- Adding new nullable columns
- Adding columns with defaults
- Widening numeric types (INT → BIGINT)
- Adding new enum values

### 2. Breaking Changes (Dangerous)
- Removing columns
- Renaming columns
- Changing column types (STRING → INT)
- Changing nullability (NULL → NOT NULL)

## Strategies

### Schema Registry (Avro/Protobuf)
```
Producer → Schema Registry → Consumer
              ↓
    Compatibility Check
    (BACKWARD, FORWARD, FULL)
```

- **BACKWARD**: New schema can read old data
- **FORWARD**: Old schema can read new data
- **FULL**: Both directions compatible

### Schema-on-Read (Data Lake)
Store raw data, apply schema at query time.

```python
# Read with schema evolution enabled
df = spark.read \
    .option("mergeSchema", "true") \
    .parquet("s3://datalake/events/")
```

### Migration-Based (RDBMS)
Sequential, versioned migrations.

```sql
-- V001_add_email_column.sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
UPDATE users SET email = CONCAT(username, '@legacy.com') WHERE email IS NULL;
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- V002_rename_column.sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

## Best Practices

1. **Never delete columns immediately** — deprecate first, remove later
2. **Version your schemas** — track changes in a registry or migration tool
3. **Test with production-like data** before deploying schema changes
4. **Use FULL compatibility mode** when possible for streaming
5. **Document breaking changes** and coordinate with consumers
6. **Backfill nulls** before adding NOT NULL constraints
