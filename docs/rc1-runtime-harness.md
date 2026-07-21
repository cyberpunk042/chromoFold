# RC1 production runtime harness

The RC1 harness converts observations from a real ChromoFold serving deployment into the strict runtime evidence consumed by `chromofold_qualify.py`.

## Runtime protocol

The harness expects the configured runtime endpoint to implement:

- `GET /ready`
- `POST /qualification/scenario`
- `GET /qualification/snapshot`
- `GET /metrics` on the configured metrics endpoint

`/ready` returns either `{ "ready": true }` or `{ "status": "ready" }`.

`POST /qualification/scenario` receives a scenario object and returns:

```json
{
  "passed": true,
  "requests": 128,
  "errors": 0,
  "recovery_ms": 240,
  "reason": ""
}
```

The patched serving runtime must execute the requested operation. It must not acknowledge an injection that did not occur.

`GET /qualification/snapshot` returns runtime-observed state including:

```json
{
  "end_to_end_correlated": true,
  "mtls_observed": true,
  "audit_chain_verified": true,
  "prompt_content_leaks": 0,
  "correctness_failures": 0,
  "cuda_errors": 0,
  "cross_tenant_contamination": 0,
  "dense_fallback_launches": 0,
  "ttft_p95_ms": 412.0,
  "inter_token_p95_ms": 41.0,
  "memory": {
    "observed_vram_bytes": 0,
    "accounted_vram_bytes": 0,
    "unattributed_vram_bytes": 0
  },
  "references": {
    "pages": 0,
    "snapshots": 0,
    "leases": 0,
    "requests": 0,
    "transfer_buffers": 0
  },
  "promoted_digest": "sha256:..."
}
```

## Qualification scenarios

The harness requires these exact scenarios:

1. cache-cold
2. cache-warm
3. multi-tenant-overload
4. cancellation
5. decode-worker-failure
6. network-fault
7. certificate-rotation
8. rolling-upgrade
9. rollback

Each scenario must be executed and pass before the decision can be `PASS`.

## Modes

- `smoke`: approximately 10 minutes and intended for wiring validation.
- `candidate`: approximately two hours and intended for release-candidate qualification.
- `stable`: approximately 24 hours and intended for stable-release evidence.

The runtime adapter owns workload duration and repetition. The harness passes the selected mode to every scenario.

## Running manually

```bash
pip install jsonschema nvidia-ml-py
export CHROMOFOLD_RUNTIME_COMMAND='./deploy/start-qualified-runtime.sh'
python tools/chromofold_runtime_harness.py \
  --mode smoke \
  --model /models/model.gguf \
  --release-digest sha256:0123456789abcdef \
  --artifact qualification/rc1-runtime-evidence.json
python integrations/llama.cpp/qualification/validate_rc1_runtime_evidence.py \
  qualification/rc1-runtime-evidence.json
python tools/chromofold_qualify.py evaluate qualification/rc1-runtime-evidence.json
```

## Decision integrity

The harness returns:

- exit `0` for `PASS`;
- exit `1` for `FAIL`;
- exit `3` for `INCOMPLETE`.

Unavailable runtime endpoints, missing GPU telemetry, or missing end-to-end correlation can never become a passing result.

## Incident bundle

The generated archive contains redacted evidence, runtime logs, and metric snapshots. Keys associated with prompts, completions, passwords, secrets, tokens, private keys, and API keys are replaced. Bearer tokens and private-key blocks are also removed from free-form text.

The bundle does not prove redaction merely because it was created. Qualification environments should additionally scan the archive with organization-specific secret and PII detectors before release approval.

## CI policy

Pull requests run only hosted CPU contract tests. The hardware workflow is manual-only and requires a self-hosted GPU runner. No automatic pull-request check waits for private hardware.
