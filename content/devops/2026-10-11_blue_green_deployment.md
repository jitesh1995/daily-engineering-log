# Blue-Green Deployment Strategy

## Overview
Blue-Green deployment maintains two identical production environments.
Only one (say "Blue") serves live traffic at any time. You deploy to
the idle environment ("Green"), test it, then switch traffic instantly.

## Architecture

```
                     ┌─────────────┐
                     │  Load       │
                     │  Balancer   │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐            ┌─────▼─────┐
        │   BLUE    │            │   GREEN   │
        │  (live)   │            │  (idle)   │
        │  v1.2.3   │            │  v1.2.4   │
        └───────────┘            └───────────┘
```

## Steps

1. **Deploy** new version to idle environment (Green)
2. **Smoke test** Green environment via internal URL
3. **Run integration tests** against Green
4. **Switch** load balancer to point to Green
5. **Monitor** error rates, latency, logs for 15 min
6. **Rollback** (if needed): switch back to Blue instantly

## Kubernetes Implementation

```yaml
# Blue deployment (currently live)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
  labels:
    app: myapp
    slot: blue
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: myapp:1.2.3

---
# Service targeting active slot
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    slot: blue  # ← Change to "green" to switch
```

## Rollback

Rollback is instant — just switch the selector back:
```bash
kubectl patch svc myapp -p '{"spec":{"selector":{"slot":"blue"}}}'
```

## Tradeoffs

| Pros | Cons |
|------|------|
| Instant rollback | 2x infrastructure cost |
| Zero downtime | Database migrations need care |
| Full production testing | More complex networking |
| Simple mental model | Stateful services are harder |
