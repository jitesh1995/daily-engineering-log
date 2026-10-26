"""
Strategy Pattern
Define a family of algorithms, encapsulate each one, and make them
interchangeable at runtime.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


# === Strategy Interface ===
class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, base_price: float, quantity: int) -> float:
        pass


# === Concrete Strategies ===
class RegularPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * quantity


class BulkDiscountPricing(PricingStrategy):
    def __init__(self, threshold: int = 10, discount_pct: float = 0.15):
        self.threshold = threshold
        self.discount_pct = discount_pct

    def calculate_price(self, base_price: float, quantity: int) -> float:
        total = base_price * quantity
        if quantity >= self.threshold:
            total *= (1 - self.discount_pct)
        return total


class TieredPricing(PricingStrategy):
    """Different price per unit based on quantity tiers."""

    def __init__(self):
        self.tiers = [
            (10, 1.0),    # 1-10 units: full price
            (50, 0.9),    # 11-50 units: 10% off
            (100, 0.8),   # 51-100 units: 20% off
            (float("inf"), 0.7),  # 101+: 30% off
        ]

    def calculate_price(self, base_price: float, quantity: int) -> float:
        total = 0.0
        remaining = quantity
        prev_limit = 0

        for limit, multiplier in self.tiers:
            tier_qty = min(remaining, int(limit) - prev_limit)
            if tier_qty <= 0:
                break
            total += base_price * multiplier * tier_qty
            remaining -= tier_qty
            prev_limit = int(limit)

        return total


class SeasonalPricing(PricingStrategy):
    def __init__(self, season_multiplier: float = 1.25):
        self.season_multiplier = season_multiplier

    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * self.season_multiplier * quantity


# === Context ===
@dataclass
class ShoppingCart:
    pricing_strategy: PricingStrategy

    def set_strategy(self, strategy: PricingStrategy):
        self.pricing_strategy = strategy

    def checkout(self, items: list[tuple[str, float, int]]) -> dict:
        line_items = []
        total = 0.0

        for name, base_price, quantity in items:
            price = self.pricing_strategy.calculate_price(base_price, quantity)
            line_items.append({
                "item": name,
                "base_price": base_price,
                "quantity": quantity,
                "total": round(price, 2),
            })
            total += price

        return {"items": line_items, "total": round(total, 2)}


if __name__ == "__main__":
    items = [
        ("Widget A", 10.00, 5),
        ("Widget B", 25.00, 15),
        ("Widget C", 8.00, 100),
    ]

    cart = ShoppingCart(pricing_strategy=RegularPricing())
    print("Regular:", cart.checkout(items)["total"])

    cart.set_strategy(BulkDiscountPricing(threshold=10, discount_pct=0.15))
    print("Bulk Discount:", cart.checkout(items)["total"])

    cart.set_strategy(TieredPricing())
    print("Tiered:", cart.checkout(items)["total"])
