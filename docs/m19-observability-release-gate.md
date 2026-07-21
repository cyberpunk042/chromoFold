# M19 — Observability and v1.0 qualification

M19 defines the final release-decision layer over M13–M18. It does not claim that hosted CI proves production GPU performance. It establishes stable telemetry, SLO, regression, evidence, incident, and promotion contracts, while reserving hardware qualification for the manual self-hosted workflow.

## Request correlation

Every production request must carry a trace ID, request ID, tenant namespace, model identity, sequence identity, worker identity, and release version across admission, routing, prefill, page encoding, transfer, decode, completion, usage accounting, logs, metrics, audit records, and qualification evidence.

Prompt and generated content are excluded from telemetry by default. Metric labels are restricted to bounded dimensions such as component role, model family, GPU class, SLO class, result, error code, and release version.

## Release gates

A candidate is eligible for stable promotion only when:

- correctness failures, cross-tenant contamination, and CUDA errors are zero;
- TTFT and inter-token latency are measured and within policy;
- the regression comparator accepts the candidate against a matching environment baseline;
- memory accounting and all page, snapshot, lease, and request references reconcile;
- cache-cold, cache-warm, overload, cancellation, worker loss, network fault, certificate rotation, rolling upgrade, rollback, and autoscaling scenarios pass;
- the incident bundle is generated with content redaction;
- the promoted artifact digest exactly equals the qualified digest.

## CI boundary

Pull-request CI compiles and executes the deterministic CPU contract, validates evidence structure, checks stable anchors, and rejects fabricated evidence. It does not prove GPU telemetry, multi-node tracing, 24-hour leak freedom, chaos recovery, production SLOs, or stable promotion.

The manual workflow requires a self-hosted distributed GPU runner and a qualification harness supplied by the patched runtime integration. It must produce strict M19 evidence plus a redacted incident bundle and benchmark report.

## Reproduction

```bash
make -f m19-qualification.mk m19-all
```

A passing M19 artifact supports a formal decision to create `v1.0.0-rc1`. Failures should produce targeted remediation pull requests rather than another automatic milestone.
