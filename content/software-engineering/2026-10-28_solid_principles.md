# SOLID Principles in Practice

## S — Single Responsibility Principle
A class should have only one reason to change.

```python
# Bad: One class does everything
class UserManager:
    def create_user(self, data): ...
    def send_welcome_email(self, user): ...
    def generate_report(self, users): ...

# Good: Separate responsibilities
class UserService:
    def create_user(self, data): ...

class EmailService:
    def send_welcome_email(self, user): ...

class ReportGenerator:
    def generate_report(self, users): ...
```

## O — Open/Closed Principle
Open for extension, closed for modification.

```python
# Bad: Modifying existing code for new shapes
def area(shape):
    if shape.type == "circle":
        return 3.14 * shape.radius ** 2
    elif shape.type == "rectangle":  # Adding new type = modifying existing code
        return shape.width * shape.height

# Good: Extend via new classes
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Circle(Shape):
    def area(self): return 3.14 * self.radius ** 2

class Rectangle(Shape):  # New shape = new class, no modification
    def area(self): return self.width * self.height
```

## L — Liskov Substitution Principle
Subtypes must be substitutable for their base types.

```python
# Bad: Square breaks Rectangle contract
class Rectangle:
    def set_width(self, w): self.width = w
    def set_height(self, h): self.height = h

class Square(Rectangle):  # Violates LSP
    def set_width(self, w): self.width = self.height = w  # Surprise!

# Good: Separate types or use composition
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
```

## I — Interface Segregation Principle
No client should be forced to depend on methods it doesn't use.

```python
# Bad: Fat interface
class Worker(ABC):
    @abstractmethod
    def code(self): ...
    @abstractmethod
    def test(self): ...
    @abstractmethod
    def manage(self): ...  # Not all workers manage

# Good: Segregated interfaces
class Coder(ABC):
    @abstractmethod
    def code(self): ...

class Tester(ABC):
    @abstractmethod
    def test(self): ...

class Developer(Coder, Tester):  # Compose as needed
    def code(self): ...
    def test(self): ...
```

## D — Dependency Inversion Principle
Depend on abstractions, not concretions.

```python
# Bad: Direct dependency on concrete class
class OrderService:
    def __init__(self):
        self.db = PostgresDatabase()  # Tightly coupled

# Good: Depend on abstraction
class OrderService:
    def __init__(self, db: Database):  # Accepts any Database impl
        self.db = db
```
