# Observability Stack Design

## Three Pillars

### 1. Metrics (Prometheus + Grafana)
Quantitative measurements over time.
- **RED Method** (request-scoped): Rate, Errors, Duration
- **USE Method** (resource-scoped): Utilization, Saturation, Errors

Key metrics to always have:
- Request rate (rpm)
- Error rate (5xx / total)
- Latency percentiles (p50, p95, p99)
- CPU/Memory utilization
- Queue depth / processing lag
- Database connection pool usage

### 2. Logs (Loki / ELK / CloudWatch)
Structured events with context.

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "error",
  "service": "api-server",
  "trace_id": "abc123",
  "message": "Failed to process payment",
  "error": "timeout after 5s",
  "user_id": "usr_456",
  "amount": 99.99
}
```

Best practices:
- Always structured (JSON)
- Include trace_id for correlation
- Log at appropriate levels
- Never log secrets or PII

### 3. Traces (Jaeger / Tempo / X-Ray)
Distributed request flow across services.

```
[API Gateway] ──► [Auth Service] ──► [User DB]
      │
      └──► [Order Service] ──► [Payment Service] ──► [Stripe]
                  │
                  └──► [Inventory DB]
```

## SLI/SLO Framework

- **SLI**: What you measure (e.g., "% of requests < 200ms")
- **SLO**: Target for the SLI (e.g., "99.9% of requests < 200ms")
- **Error Budget**: 100% - SLO = budget for risky changes

## Alerting Philosophy

1. Alert on symptoms, not causes
2. Every alert must be actionable
3. Link to runbook in annotation
4. Page only for customer-facing impact
5. Use multi-window burn rate for SLO alerts
