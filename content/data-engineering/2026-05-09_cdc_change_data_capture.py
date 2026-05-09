"""
Change Data Capture (CDC) Pattern
Track and process database changes incrementally.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum
import json


class ChangeType(str, Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    table: str
    change_type: ChangeType
    timestamp: datetime
    primary_key: dict
    before: Optional[dict]  # None for INSERT
    after: Optional[dict]   # None for DELETE
    metadata: dict

    def to_dict(self):
        return {
            "table": self.table,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "primary_key": self.primary_key,
            "before": self.before,
            "after": self.after,
            "metadata": self.metadata,
        }


class CDCProcessor:
    """Process CDC events and apply them to a target."""

    def __init__(self):
        self.stats = {"inserts": 0, "updates": 0, "deletes": 0, "errors": 0}
        self.handlers = {}

    def register_handler(self, table: str, handler):
        """Register a handler function for a specific table."""
        self.handlers[table] = handler

    def process_event(self, event: ChangeEvent):
        """Route and process a single CDC event."""
        handler = self.handlers.get(event.table)

        if handler is None:
            return  # No handler registered for this table

        try:
            if event.change_type == ChangeType.INSERT:
                handler.handle_insert(event.primary_key, event.after)
                self.stats["inserts"] += 1

            elif event.change_type == ChangeType.UPDATE:
                changed_fields = self._detect_changes(event.before, event.after)
                handler.handle_update(event.primary_key, event.before, event.after, changed_fields)
                self.stats["updates"] += 1

            elif event.change_type == ChangeType.DELETE:
                handler.handle_delete(event.primary_key, event.before)
                self.stats["deletes"] += 1

        except Exception as e:
            self.stats["errors"] += 1
            raise

    @staticmethod
    def _detect_changes(before: dict, after: dict) -> list[str]:
        """Identify which fields changed between before and after."""
        if not before or not after:
            return list(after.keys()) if after else []

        changed = []
        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            if before.get(key) != after.get(key):
                changed.append(key)
        return changed

    def process_batch(self, events: list[ChangeEvent]):
        """Process a batch of CDC events in order."""
        for event in sorted(events, key=lambda e: e.timestamp):
            self.process_event(event)

    def get_stats(self):
        return {**self.stats, "total": sum(self.stats.values())}


class TimestampCDC:
    """Pull-based CDC using timestamp columns."""

    def __init__(self, table, timestamp_column="updated_at"):
        self.table = table
        self.timestamp_column = timestamp_column
        self.last_processed = None

    def build_query(self):
        """Build incremental extraction query."""
        if self.last_processed:
            return (
                f"SELECT * FROM {self.table} "
                f"WHERE {self.timestamp_column} > '{self.last_processed.isoformat()}' "
                f"ORDER BY {self.timestamp_column}"
            )
        return f"SELECT * FROM {self.table} ORDER BY {self.timestamp_column}"

    def update_watermark(self, max_timestamp):
        """Update the high watermark after successful processing."""
        self.last_processed = max_timestamp
