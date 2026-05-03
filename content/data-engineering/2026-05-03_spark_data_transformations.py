"""
PySpark Data Transformation Patterns
Common transformations for data engineering pipelines.
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType


def create_spark_session(app_name="DataTransformations"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def deduplicate_events(df, id_col="event_id", timestamp_col="event_time"):
    """Keep the latest version of each event (SCD Type 1)."""
    window = Window.partitionBy(id_col).orderBy(F.col(timestamp_col).desc())
    return (
        df
        .withColumn("row_num", F.row_number().over(window))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )


def sessionize_events(df, user_col="user_id", time_col="event_time", gap_minutes=30):
    """Assign session IDs based on inactivity gaps."""
    window = Window.partitionBy(user_col).orderBy(time_col)

    return (
        df
        .withColumn("prev_time", F.lag(time_col).over(window))
        .withColumn(
            "gap_minutes",
            (F.unix_timestamp(time_col) - F.unix_timestamp("prev_time")) / 60,
        )
        .withColumn(
            "new_session",
            F.when(
                (F.col("gap_minutes") > gap_minutes) | F.col("prev_time").isNull(), 1
            ).otherwise(0),
        )
        .withColumn(
            "session_id",
            F.concat(
                F.col(user_col),
                F.lit("_"),
                F.sum("new_session").over(window),
            ),
        )
        .drop("prev_time", "gap_minutes", "new_session")
    )


def compute_rfm(df, user_col="user_id", date_col="order_date", amount_col="amount"):
    """Compute RFM (Recency, Frequency, Monetary) scores."""
    max_date = df.agg(F.max(date_col)).collect()[0][0]

    rfm = (
        df.groupBy(user_col)
        .agg(
            F.datediff(F.lit(max_date), F.max(date_col)).alias("recency"),
            F.countDistinct("order_id").alias("frequency"),
            F.sum(amount_col).alias("monetary"),
        )
    )

    # Score each metric 1-5
    for col_name in ["recency", "frequency", "monetary"]:
        ascending = col_name == "recency"  # lower recency = better
        rfm = rfm.withColumn(
            f"{col_name}_score",
            F.ntile(5).over(
                Window.orderBy(
                    F.col(col_name).asc() if ascending else F.col(col_name).desc()
                )
            ),
        )

    return rfm.withColumn(
        "rfm_segment",
        F.concat(
            F.col("recency_score"), F.col("frequency_score"), F.col("monetary_score")
        ),
    )


def pivot_metrics(df, group_col, pivot_col, value_col):
    """Dynamic pivot table for metrics."""
    pivot_values = [row[0] for row in df.select(pivot_col).distinct().collect()]

    return (
        df.groupBy(group_col)
        .pivot(pivot_col, pivot_values)
        .agg(F.sum(value_col))
        .fillna(0)
    )


def scd_type2_merge(current_df, new_df, key_cols, value_cols):
    """Slowly Changing Dimension Type 2 merge logic."""
    # Find changed records
    join_condition = [current_df[k] == new_df[k] for k in key_cols]
    value_condition = F.lit(False)
    for v in value_cols:
        value_condition = value_condition | (current_df[v] != new_df[v])

    changed = (
        current_df
        .join(new_df, join_condition, "inner")
        .filter(value_condition & (current_df["is_current"] == True))
        .select(current_df["*"])
    )

    # Close old records
    closed = changed.withColumn("is_current", F.lit(False)).withColumn(
        "valid_to", F.current_timestamp()
    )

    # New records from changes
    new_records = (
        new_df
        .join(changed.select(key_cols), key_cols, "inner")
        .withColumn("is_current", F.lit(True))
        .withColumn("valid_from", F.current_timestamp())
        .withColumn("valid_to", F.lit(None).cast(TimestampType()))
    )

    # Unchanged records
    unchanged = current_df.join(changed.select(key_cols), key_cols, "left_anti")

    # Truly new records (not in current at all)
    truly_new = (
        new_df
        .join(current_df.select(key_cols), key_cols, "left_anti")
        .withColumn("is_current", F.lit(True))
        .withColumn("valid_from", F.current_timestamp())
        .withColumn("valid_to", F.lit(None).cast(TimestampType()))
    )

    return unchanged.unionByName(closed).unionByName(new_records).unionByName(truly_new)
