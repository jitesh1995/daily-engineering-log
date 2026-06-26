"""
Airflow DAG: Daily ETL Pipeline
Extract from API, transform, load to warehouse.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-alerts@company.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


def extract_api_data(**context):
    """Extract data from source API."""
    import requests

    execution_date = context["ds"]
    url = f"https://api.source.com/events?date={execution_date}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="raw_data", value=data)
    context["ti"].xcom_push(key="record_count", value=len(data))
    return len(data)


def validate_data(**context):
    """Data quality checks on extracted data."""
    data = context["ti"].xcom_pull(key="raw_data", task_ids="extract.extract_api")
    record_count = len(data)

    # Assertions
    assert record_count > 0, "No records extracted!"
    assert all("id" in record for record in data), "Missing 'id' field"
    assert all("timestamp" in record for record in data), "Missing 'timestamp'"

    # Check for duplicates
    ids = [r["id"] for r in data]
    duplicates = len(ids) - len(set(ids))
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate IDs found")

    return {"records": record_count, "duplicates": duplicates}


def transform_data(**context):
    """Clean and transform raw data."""
    data = context["ti"].xcom_pull(key="raw_data", task_ids="extract.extract_api")

    transformed = []
    for record in data:
        transformed.append({
            "event_id": record["id"],
            "event_type": record.get("type", "unknown").lower(),
            "user_id": record.get("user_id"),
            "amount": float(record.get("amount", 0)),
            "event_timestamp": record["timestamp"],
            "processed_at": dt_class.utcnow().isoformat(),
        })

    # Deduplicate
    seen = set()
    deduplicated = []
    for row in transformed:
        if row["event_id"] not in seen:
            seen.add(row["event_id"])
            deduplicated.append(row)

    context["ti"].xcom_push(key="transformed_data", value=deduplicated)
    return len(deduplicated)


def load_to_warehouse(**context):
    """Load transformed data to PostgreSQL warehouse."""
    data = context["ti"].xcom_pull(
        key="transformed_data", task_ids="transform.transform_data"
    )

    hook = PostgresHook(postgres_conn_id="warehouse")
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO events (event_id, event_type, user_id, amount, event_timestamp, processed_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING
    """

    rows = [
        (r["event_id"], r["event_type"], r["user_id"],
         r["amount"], r["event_timestamp"], r["processed_at"])
        for r in data
    ]

    cursor.executemany(insert_sql, rows)
    conn.commit()
    cursor.close()
    conn.close()

    return f"Loaded {len(rows)} records"


with DAG(
    dag_id="daily_etl_pipeline",
    default_args=default_args,
    description="Daily ETL: API -> Transform -> Warehouse",
    schedule_interval="0 6 * * *",  # 6 AM UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["etl", "daily", "production"],
) as dag:

    start = EmptyOperator(task_id="start")

    with TaskGroup("extract") as extract_group:
        extract_task = PythonOperator(
            task_id="extract_api",
            python_callable=extract_api_data,
        )
        validate_task = PythonOperator(
            task_id="validate",
            python_callable=validate_data,
        )
        extract_task >> validate_task

    with TaskGroup("transform") as transform_group:
        transform_task = PythonOperator(
            task_id="transform_data",
            python_callable=transform_data,
        )

    with TaskGroup("load") as load_group:
        load_task = PythonOperator(
            task_id="load_warehouse",
            python_callable=load_to_warehouse,
        )

    end = EmptyOperator(task_id="end")

    start >> extract_group >> transform_group >> load_group >> end
