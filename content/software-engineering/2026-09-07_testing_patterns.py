"""
Testing Patterns
Unit tests, fixtures, mocking, parametrize, and TDD patterns.
"""
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, patch, MagicMock
import pytest


# === Code Under Test ===
@dataclass
class User:
    id: int
    name: str
    email: str
    is_active: bool = True


class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_id(self, user_id: int) -> Optional[User]:
        row = self.db.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        if row:
            return User(**row)
        return None

    def save(self, user: User) -> User:
        self.db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (user.name, user.email),
        )
        return user


class UserService:
    def __init__(self, repo: UserRepository, email_client):
        self.repo = repo
        self.email_client = email_client

    def register(self, name: str, email: str) -> User:
        if not email or "@" not in email:
            raise ValueError("Invalid email address")

        user = User(id=0, name=name, email=email)
        saved = self.repo.save(user)
        self.email_client.send_welcome(email)
        return saved

    def deactivate(self, user_id: int) -> bool:
        user = self.repo.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        user.is_active = False
        self.repo.save(user)
        return True


# === Test Fixtures ===
@pytest.fixture
def mock_db():
    return Mock()

@pytest.fixture
def mock_email():
    return Mock()

@pytest.fixture
def repo(mock_db):
    return UserRepository(mock_db)

@pytest.fixture
def service(repo, mock_email):
    return UserService(repo, mock_email)

@pytest.fixture
def sample_user():
    return User(id=1, name="Alice", email="alice@test.com")


# === Unit Tests ===
class TestUserService:
    def test_register_success(self, service, mock_email):
        """Happy path: valid registration."""
        user = service.register("Bob", "bob@test.com")
        assert user.name == "Bob"
        assert user.email == "bob@test.com"
        mock_email.send_welcome.assert_called_once_with("bob@test.com")

    def test_register_invalid_email(self, service):
        """Should raise ValueError for invalid email."""
        with pytest.raises(ValueError, match="Invalid email"):
            service.register("Bob", "not-an-email")

    def test_register_empty_email(self, service):
        with pytest.raises(ValueError):
            service.register("Bob", "")

    def test_deactivate_user(self, service, mock_db):
        """Should deactivate an existing user."""
        mock_db.execute.return_value = {
            "id": 1, "name": "Alice", "email": "alice@test.com", "is_active": True
        }
        result = service.deactivate(1)
        assert result is True

    def test_deactivate_nonexistent_user(self, service, mock_db):
        """Should raise ValueError for unknown user."""
        mock_db.execute.return_value = None
        with pytest.raises(ValueError, match="not found"):
            service.deactivate(999)


# === Parametrized Tests ===
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("a@b.co", True),
    ("no-at-sign", False),
    ("", False),
    ("spaces @test.com", True),  # Has @ so passes basic check
])
def test_email_validation(service, email, valid):
    if valid:
        user = service.register("Test", email)
        assert user.email == email
    else:
        with pytest.raises(ValueError):
            service.register("Test", email)


# === Test for Side Effects ===
class TestEmailIntegration:
    def test_welcome_email_sent_on_register(self, service, mock_email):
        service.register("New User", "new@test.com")
        mock_email.send_welcome.assert_called_once()

    def test_no_email_sent_on_failure(self, service, mock_email):
        with pytest.raises(ValueError):
            service.register("Bad", "")
        mock_email.send_welcome.assert_not_called()
