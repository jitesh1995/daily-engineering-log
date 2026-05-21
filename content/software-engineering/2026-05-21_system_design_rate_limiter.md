# System Design: Rate Limiter

## Requirements
- Limit requests per user/IP within a time window
- Low latency (< 1ms overhead)
- Distributed (works across multiple servers)
- Configurable rules (e.g., 100 req/min for API, 5 req/min for login)

## Algorithms

### 1. Token Bucket
Tokens refill at a fixed rate. Each request consumes a token.

```
Capacity: 10 tokens
Refill rate: 2 tokens/second

Request arrives → tokens > 0? → Allow (tokens -= 1)
                              → Deny (429 Too Many Requests)
```

**Pros**: Allows bursts up to bucket capacity
**Cons**: Memory per user (2 values: tokens + timestamp)

### 2. Sliding Window Log
Store timestamp of each request. Count requests in window.

**Pros**: Exact count, no boundary issues
**Cons**: High memory (stores every timestamp)

### 3. Sliding Window Counter
Hybrid: fixed window counts + weighted overlap.

```
Current window:  [====70%====]
Previous window: [====30%====]

Count = current_count * 0.7 + previous_count * 0.3
```

**Pros**: Low memory, smooth rate limiting
**Cons**: Approximate (but close enough)

## Distributed Implementation (Redis)

```python
def is_allowed(user_id, limit, window_seconds):
    key = f"rate:{user_id}"
    pipe = redis.pipeline()

    now = time.time()
    window_start = now - window_seconds

    pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
    pipe.zadd(key, {str(now): now})               # Add current request
    pipe.zcard(key)                                # Count in window
    pipe.expire(key, window_seconds)               # Auto-cleanup

    _, _, count, _ = pipe.execute()
    return count <= limit
```

## Architecture

```
Client → API Gateway → Rate Limiter → Backend
                            │
                        Redis Cluster
                       (shared state)
```

## Rules Configuration

```yaml
rate_limits:
  - name: api_default
    key: user_id
    limit: 100
    window: 60s

  - name: auth_login
    key: ip_address
    limit: 5
    window: 300s
    path: /api/auth/login

  - name: search
    key: user_id
    limit: 30
    window: 60s
    path: /api/search
```

## Response Headers

```
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705313400
Retry-After: 42
```
