"""
Data Quality Framework
Reusable checks for data pipeline validation.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
from datetime import datetime
import json

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    metric_value: Any
    threshold: Any
    severity: str  # "critical", "warning", "info"
    message: str
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = dt_class.utcnow().isoformat()


class DataQualityChecker:
    """Framework for running data quality checks."""

    def __init__(self):
        self.results: list[CheckResult] = []

    def check_not_null(self, df, columns, severity="critical"):
        """Check that specified columns have no nulls."""
        for col in columns:
            null_count = df[col].isna().sum()
            total = len(df)
            self.results.append(CheckResult(
                check_name=f"not_null_{col}",
                passed=null_count == 0,
                metric_value=null_count,
                threshold=0,
                severity=severity,
                message=f"{col}: {null_count}/{total} nulls found",
            ))

    def check_unique(self, df, columns, severity="critical"):
        """Check that columns form a unique key."""
        key = columns if isinstance(columns, list) else [columns]
        duplicates = df.duplicated(subset=key).sum()
        self.results.append(CheckResult(
            check_name=f"unique_{'_'.join(key)}",
            passed=duplicates == 0,
            metric_value=duplicates,
            threshold=0,
            severity=severity,
            message=f"{'_'.join(key)}: {duplicates} duplicate rows",
        ))

    def check_accepted_values(self, df, column, accepted, severity="warning"):
        """Check that column values are within accepted set."""
        invalid = df[~df[column].isin(accepted)][column].unique()
        self.results.append(CheckResult(
            check_name=f"accepted_values_{column}",
            passed=len(invalid) == 0,
            metric_value=list(invalid),
            threshold=accepted,
            severity=severity,
            message=f"{column}: invalid values {list(invalid)[:5]}",
        ))

    def check_freshness(self, df, timestamp_col, max_hours=24, severity="critical"):
        """Check that data is recent enough."""
        max_ts = df[timestamp_col].max()
        age_hours = (dt_class.utcnow() - max_ts).total_seconds() / 3600
        self.results.append(CheckResult(
            check_name=f"freshness_{timestamp_col}",
            passed=age_hours <= max_hours,
            metric_value=round(age_hours, 2),
            threshold=max_hours,
            severity=severity,
            message=f"Data is {age_hours:.1f}h old (max: {max_hours}h)",
        ))

    def check_row_count(self, df, min_rows=1, max_rows=None, severity="critical"):
        """Check that row count is within expected range."""
        count = len(df)
        passed = count >= min_rows and (max_rows is None or count <= max_rows)
        self.results.append(CheckResult(
            check_name="row_count",
            passed=passed,
            metric_value=count,
            threshold={"min": min_rows, "max": max_rows},
            severity=severity,
            message=f"Row count: {count} (expected: {min_rows}-{max_rows or 'inf'})",
        ))

    def check_referential_integrity(self, df, column, reference_df, ref_column, severity="critical"):
        """Check foreign key relationship."""
        ref_values = set(reference_df[ref_column])
        orphans = df[~df[column].isin(ref_values)][column].nunique()
        self.results.append(CheckResult(
            check_name=f"ref_integrity_{column}",
            passed=orphans == 0,
            metric_value=orphans,
            threshold=0,
            severity=severity,
            message=f"{column}: {orphans} orphan values not in {ref_column}",
        ))

    def summary(self):
        """Return summary of all check results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        critical_failures = sum(
            1 for r in self.results if not r.passed and r.severity == "critical"
        )
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "critical_failures": critical_failures,
            "all_passed": failed == 0,
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in self.results
            ],
        }
