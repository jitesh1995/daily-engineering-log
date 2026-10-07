# Production Security Hardening Checklist

## Container Security
- [ ] Use minimal base images (distroless/alpine)
- [ ] Run as non-root user
- [ ] Read-only root filesystem
- [ ] Drop all capabilities, add only needed
- [ ] Scan images for CVEs (Trivy, Snyk)
- [ ] Pin image versions (no `latest` tag)
- [ ] Use multi-stage builds
- [ ] No secrets in image layers

## Kubernetes Security
- [ ] Pod Security Standards (restricted profile)
- [ ] Network Policies (default deny, explicit allow)
- [ ] RBAC with least privilege
- [ ] Service mesh mTLS (Istio/Linkerd)
- [ ] Secrets encrypted at rest (KMS)
- [ ] Audit logging enabled
- [ ] Resource quotas and limits
- [ ] Pod disruption budgets

## Application Security
- [ ] Input validation on all endpoints
- [ ] Parameterized queries (no SQL injection)
- [ ] CORS properly configured
- [ ] Rate limiting on auth endpoints
- [ ] JWT with short expiry + refresh tokens
- [ ] HTTPS only (HSTS enabled)
- [ ] Security headers (CSP, X-Frame-Options, etc.)
- [ ] Dependency scanning (Dependabot, Renovate)

## Infrastructure Security
- [ ] VPC with private subnets for workloads
- [ ] Security groups: least privilege
- [ ] WAF in front of public endpoints
- [ ] CloudTrail / audit logging
- [ ] Automated backups with encryption
- [ ] SSH key rotation
- [ ] MFA on all admin accounts

## Monitoring & Response
- [ ] Centralized logging (ELK/Loki)
- [ ] Alert on anomalous patterns
- [ ] Incident response runbooks
- [ ] Regular penetration testing
- [ ] Automated compliance checks (OPA/Kyverno)
