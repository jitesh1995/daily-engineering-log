# REST API Design Best Practices

## URL Structure

```
GET    /api/v1/users           # List users
POST   /api/v1/users           # Create user
GET    /api/v1/users/{id}      # Get user
PATCH  /api/v1/users/{id}      # Update user
DELETE /api/v1/users/{id}      # Delete user

# Sub-resources
GET    /api/v1/users/{id}/orders
POST   /api/v1/users/{id}/orders

# Filtering, sorting, pagination
GET    /api/v1/orders?status=pending&sort=-created_at&page=2&limit=25
```

## Naming Conventions
- Use **nouns**, not verbs: `/users` not `/getUsers`
- Use **plural**: `/users` not `/user`
- Use **kebab-case**: `/order-items` not `/orderItems`
- Max 2 levels of nesting: `/users/{id}/orders` (not deeper)

## HTTP Status Codes

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Validation error |
| 401 | Unauthorized | Missing/invalid auth |
| 403 | Forbidden | Authenticated but not allowed |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate, version conflict |
| 422 | Unprocessable | Valid syntax, invalid semantics |
| 429 | Too Many Requests | Rate limited |
| 500 | Server Error | Unexpected failure |

## Response Format

```json
{
  "data": {
    "id": "usr_123",
    "email": "user@example.com",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_abc123"
  }
}
```

## Pagination

```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 25,
    "total": 150,
    "has_more": true
  }
}
```

## Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {"field": "email", "message": "Must be a valid email address"},
      {"field": "age", "message": "Must be at least 18"}
    ]
  }
}
```

## Versioning
- URL path: `/api/v1/users` (most common, simplest)
- Header: `Accept: application/vnd.api.v1+json`
- Query: `/api/users?version=1`

## Rate Limiting Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1705312800
Retry-After: 30
```

## Idempotency
- GET, PUT, DELETE are naturally idempotent
- For POST: use `Idempotency-Key` header
- Store and return cached response for duplicate keys
