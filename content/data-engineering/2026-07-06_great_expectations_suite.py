"""
Data Quality with Great Expectations Pattern
Define and validate data contracts.
"""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Expectation:
    expectation_type: str
    kwargs: dict
    severity: str = "critical"

@dataclass
class ExpectationSuite:
    name: str
    expectations: list[Expectation] = field(default_factory=list)

    def expect_column_to_exist(self, column, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_to_exist",
            kwargs={"column": column},
            severity=severity,
        ))
        return self

    def expect_column_values_to_not_be_null(self, column, mostly=1.0, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": column, "mostly": mostly},
            severity=severity,
        ))
        return self

    def expect_column_values_to_be_between(self, column, min_value=None, max_value=None, severity="warning"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": column, "min_value": min_value, "max_value": max_value},
            severity=severity,
        ))
        return self

    def expect_column_values_to_be_in_set(self, column, value_set, severity="warning"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": column, "value_set": value_set},
            severity=severity,
        ))
        return self

    def expect_compound_columns_to_be_unique(self, column_list, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={"column_list": column_list},
            severity=severity,
        ))
        return self

    def expect_table_row_count_to_be_between(self, min_value, max_value=None, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": min_value, "max_value": max_value},
            severity=severity,
        ))
        return self

    def to_dict(self):
        return {
            "expectation_suite_name": self.name,
            "expectations": [
                {"expectation_type": e.expectation_type, "kwargs": e.kwargs, "meta": {"severity": e.severity}}
                for e in self.expectations
            ],
        }


# === Example: Define suite for orders table ===
def build_orders_suite():
    suite = ExpectationSuite(name="orders_suite")

    suite.expect_table_row_count_to_be_between(min_value=1)

    # Primary key
    suite.expect_column_to_exist("order_id")
    suite.expect_column_values_to_not_be_null("order_id")

    # Required fields
    for col in ["user_id", "order_date", "total_amount", "status"]:
        suite.expect_column_values_to_not_be_null(col)

    # Value ranges
    suite.expect_column_values_to_be_between("total_amount", min_value=0, max_value=100000)

    # Allowed values
    suite.expect_column_values_to_be_in_set(
        "status", ["pending", "processing", "shipped", "delivered", "cancelled"]
    )

    # Composite uniqueness
    suite.expect_compound_columns_to_be_unique(["order_id", "order_date"])

    return suite


if __name__ == "__main__":
    import json
    suite = build_orders_suite()
    print(json.dumps(suite.to_dict(), indent=2))
