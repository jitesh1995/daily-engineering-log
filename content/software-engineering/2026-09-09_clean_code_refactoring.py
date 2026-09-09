"""
Clean Code: Refactoring Examples
Before and after applying clean code principles.
"""

# ============================================================
# 1. EXTRACT METHOD — Break long functions into named steps
# ============================================================

# Before: One giant function
def process_order_before(order):
    # Validate
    if not order.get("items"):
        raise ValueError("No items")
    if order.get("total", 0) <= 0:
        raise ValueError("Invalid total")
    for item in order["items"]:
        if item["quantity"] <= 0:
            raise ValueError("Invalid quantity")

    # Calculate
    subtotal = sum(i["price"] * i["quantity"] for i in order["items"])
    tax = subtotal * 0.08
    shipping = 0 if subtotal > 50 else 5.99
    total = subtotal + tax + shipping

    # Format
    return {
        "order_id": order["id"],
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "shipping": round(shipping, 2),
        "total": round(total, 2),
    }


# After: Clear, named steps
def process_order_after(order):
    validate_order(order)
    pricing = calculate_pricing(order["items"])
    return format_order_summary(order["id"], pricing)


def validate_order(order):
    if not order.get("items"):
        raise ValueError("Order must contain at least one item")
    if order.get("total", 0) <= 0:
        raise ValueError("Order total must be positive")
    for item in order["items"]:
        if item["quantity"] <= 0:
            raise ValueError(f"Invalid quantity for item: {item.get('name', 'unknown')}")


def calculate_pricing(items):
    subtotal = sum(item["price"] * item["quantity"] for item in items)
    tax = subtotal * 0.08
    shipping = 0 if subtotal > 50 else 5.99
    return {"subtotal": subtotal, "tax": tax, "shipping": shipping}


def format_order_summary(order_id, pricing):
    total = sum(pricing.values())
    return {
        "order_id": order_id,
        **{k: round(v, 2) for k, v in pricing.items()},
        "total": round(total, 2),
    }


# ============================================================
# 2. REPLACE CONDITIONALS WITH POLYMORPHISM
# ============================================================

# Before: Switch on type
def calculate_area_before(shape):
    if shape["type"] == "circle":
        return 3.14159 * shape["radius"] ** 2
    elif shape["type"] == "rectangle":
        return shape["width"] * shape["height"]
    elif shape["type"] == "triangle":
        return 0.5 * shape["base"] * shape["height"]
    else:
        raise ValueError(f"Unknown shape: {shape['type']}")


# After: Polymorphism
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    def area(self):
        return 0.5 * self.base * self.height


# ============================================================
# 3. GUARD CLAUSES — Flatten nested conditionals
# ============================================================

# Before: Deep nesting
def get_discount_before(user, order):
    if user is not None:
        if user.is_active:
            if order.total > 100:
                if user.is_premium:
                    return 0.20
                else:
                    return 0.10
            else:
                return 0.05
        else:
            return 0
    else:
        return 0

# After: Guard clauses
def get_discount_after(user, order):
    if user is None or not user.is_active:
        return 0

    if order.total <= 100:
        return 0.05

    return 0.20 if user.is_premium else 0.10
