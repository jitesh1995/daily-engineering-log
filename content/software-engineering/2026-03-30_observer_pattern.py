"""
Observer Pattern (Event System)
Decouple event producers from consumers.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime


@dataclass
class Event:
    name: str
    data: dict
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = dt_class.utcnow().isoformat()


class EventBus:
    """Publish-subscribe event bus with support for:
    - Multiple handlers per event
    - Wildcard subscriptions
    - Handler priorities
    - Error isolation
    """

    def __init__(self):
        self._handlers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)
        self._history: list[Event] = []

    def subscribe(self, event_name: str, handler: Callable, priority: int = 0):
        """Subscribe to an event. Lower priority number = called first."""
        self._handlers[event_name].append((priority, handler))
        self._handlers[event_name].sort(key=lambda x: x[0])
        return lambda: self.unsubscribe(event_name, handler)

    def unsubscribe(self, event_name: str, handler: Callable):
        self._handlers[event_name] = [
            (p, h) for p, h in self._handlers[event_name] if h != handler
        ]

    def publish(self, event: Event):
        """Publish event to all matching handlers."""
        self._history.append(event)

        # Exact match handlers
        for _, handler in self._handlers.get(event.name, []):
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error for {event.name}: {e}")

        # Wildcard handlers
        for _, handler in self._handlers.get("*", []):
            try:
                handler(event)
            except Exception as e:
                print(f"Wildcard handler error: {e}")

    def get_history(self, event_name: str = None, limit: int = 10):
        events = self._history
        if event_name:
            events = [e for e in events if e.name == event_name]
        return events[-limit:]


# === Usage Example ===

class OrderService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def create_order(self, user_id, items, total):
        order_id = f"ord_{hash((user_id, total)) % 10000:04d}"
        # Business logic here...

        self.event_bus.publish(Event(
            name="order.created",
            data={"order_id": order_id, "user_id": user_id, "total": total},
        ))
        return order_id


class EmailNotifier:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("order.created", self.on_order_created)

    def on_order_created(self, event: Event):
        print(f"Sending confirmation email for order {event.data['order_id']}")


class AnalyticsTracker:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("*", self.track_all, priority=99)
        self.events_tracked = 0

    def track_all(self, event: Event):
        self.events_tracked += 1
        print(f"Analytics: tracked {event.name}")


class InventoryService:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("order.created", self.reserve_inventory, priority=-1)

    def reserve_inventory(self, event: Event):
        print(f"Reserving inventory for order {event.data['order_id']}")


if __name__ == "__main__":
    bus = EventBus()

    # Wire up services
    orders = OrderService(bus)
    email = EmailNotifier(bus)
    analytics = AnalyticsTracker(bus)
    inventory = InventoryService(bus)

    # Create an order — all observers react
    order_id = orders.create_order("usr_42", ["widget_a"], 29.99)
    print(f"Order created: {order_id}")
    print(f"Events tracked: {analytics.events_tracked}")
