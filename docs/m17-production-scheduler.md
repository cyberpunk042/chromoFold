# M17 production scheduler

M17 turns the M16 disaggregated cluster into a bounded, tenant-aware production control plane.

## Scheduling path

```text
request
  -> tenant and model validation
  -> quota and budget admission
  -> prefill/decode/KV cost estimate
  -> deadline and weighted-fair score
  -> heterogeneous worker selection
  -> immediate admission or bounded queue
  -> completion/cancellation accounting
  -> circuit-breaker and drain lifecycle
  -> desired-capacity plan
```

## Invariants

- resource allocation follows admission;
- queues are bounded globally and per tenant;
- hard quotas fail closed;
- workers in draining, quarantined, failed, or open-circuit states receive no new work;
- equal scheduling scores resolve deterministically by worker ID;
- completion usage is recorded once per request ID;
- worker drain succeeds only after assigned requests leave;
- shutdown succeeds only with zero active requests;
- tenant cache namespaces must remain isolated unless a trusted shared namespace is explicitly configured;
- numerical quality and compression guarantees are never degraded as overload control.

## Kubernetes boundary

The deployment contract includes router replicas, GPU decode workers, readiness/liveness probes, graceful termination, a disruption budget, GPU node selection, and default-deny networking. Production deployments must add explicit allow policies, Secrets, persistent cache storage, metrics adapters, and environment-specific scheduling labels.

## Proof boundary

Hosted CI validates the deterministic CPU scheduler, quotas, bounded queues, cancellation, circuit transitions, drain safety, schema semantics, and Kubernetes structure. Production claims require real router/prefill/decode processes, heterogeneous GPUs, multi-tenant overload, continuous batching, checkpoint resume, rolling compatibility, circuit recovery, autoscaling decisions, zero contamination, and strict passing M17 evidence.
