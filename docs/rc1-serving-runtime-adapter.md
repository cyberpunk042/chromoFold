# RC1 serving-runtime qualification adapter

The adapter connects the repository-owned RC1 hardware harness to a serving runtime without allowing the harness to fabricate runtime operations.

## Security boundary

Qualification controls are disabled unless `CHROMOFOLD_QUALIFICATION_MODE=1`. Snapshot and scenario endpoints require `Authorization: Bearer $CHROMOFOLD_QUALIFICATION_ADMIN_TOKEN`. Bind to loopback or a restricted mTLS proxy; never expose qualification controls publicly.

## Endpoints

- `GET /live`: process liveness.
- `GET /ready`: requires a non-empty worker inventory and the configured immutable release digest.
- `GET /metrics`: bounded Prometheus metrics.
- `GET /qualification/snapshot`: observed runtime accounting and trust state.
- `POST /qualification/scenario`: invokes one explicitly configured runtime hook.

## Runtime snapshot producer

The serving runtime must atomically write the JSON snapshot passed with `--snapshot`. It must report observed values for requests, references, leases, transfer buffers, memory, CUDA and correctness failures, tenant isolation, dense fallback, mTLS, audit chaining, telemetry correlation, workers, and the release digest.

The adapter overrides `qualification_mode` and `release_digest` with its process configuration so a snapshot file cannot claim another digest.

## Scenario hooks

Pass `--hooks-dir /path/to/hooks`. Each executable file is named after a supported scenario:

```text
cache-cold
cache-warm
multi-tenant-overload
cancellation
decode-worker-failure
network-fault
certificate-rotation
rolling-upgrade
rollback
```

The request body is provided on stdin as JSON. The hook returns one JSON object on stdout. Missing hooks produce `INCOMPLETE`. Non-zero hooks cannot claim `PASS`.

Example response:

```json
{
  "status": "PASS",
  "requests_started": 100,
  "requests_completed": 99,
  "requests_failed": 1,
  "recovery_ms": 820,
  "references_reconciled": true,
  "observations": []
}
```

## Local contract

```bash
pip install jsonschema
make -f rc1-serving-adapter.mk rc1-serving-all
```

## Hardware sequence

After merge, run single-GPU smoke, single-GPU candidate, worker/network failure injection, rolling upgrade, rollback, and then the two-hour candidate soak. The 24-hour stable run remains last.

This adapter provides a secured protocol and hook boundary. Runtime-specific hooks must actually manipulate ChromoFold workers, networking, certificates, and release generations. A missing operation remains `INCOMPLETE`.
