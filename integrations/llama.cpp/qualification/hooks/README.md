# RC1 qualification scenario hooks

The nine executable scenario hooks required by the RC1 local-machine qualification workflow
(`docs/_assets/chromofold_rc1_local_machine_handoff.docx`, §8). The **serving adapter**
(`tools/chromofold_qualification_adapter.py`) runs the hook whose filename matches the scenario name, feeds it the
scenario payload as JSON on **stdin**, and reads one JSON result object from **stdout**.

Point the adapter at this directory:

```sh
python3 tools/chromofold_qualification_adapter.py --listen 127.0.0.1 --port 8089 \
  --snapshot /run/chromofold/qualification-snapshot.json \
  --hooks-dir integrations/llama.cpp/qualification/hooks \
  --release-digest "$CHROMOFOLD_RELEASE_DIGEST"
```

## The honesty contract (why these don't just print PASS)

Per handoff §8–§9, **a hook must never return PASS from a no-op.** These hooks encode that rule in code:

- They do the runtime-agnostic work for real — reach the runtime, snapshot its reference counters **before and
  after**, time recovery, and issue real generation requests.
- The **destructive control action** for each scenario (kill a worker, inject a fault, rotate a cert, shift
  traffic, roll back) is executed via an **operator-provided command**, so nothing is faked.
- If the serving runtime, a generation endpoint, a snapshot source, or a required control command is **not
  available**, the hook returns **`INCOMPLETE`** with an explicit reason. It returns **`FAIL`** only on an
  **observed** invariant violation (a request failed, references drifted, a tenant was starved, cross-tenant
  contamination, a control command errored). It returns **`PASS`** only when the real operation *and* recovery
  *and* reference reconciliation were all observed.

On a machine with no serving runtime, **all nine correctly return `INCOMPLETE`** ("serving runtime not
reachable") — which is the truthful decision, not a failure of the hooks.

## Result contract (stdout)

```json
{ "status": "PASS|INCOMPLETE|FAIL",
  "requests_started": 0, "requests_completed": 0, "requests_failed": 0,
  "recovery_ms": 0, "references_reconciled": false,
  "observations": [ {"...": "..."} ] }
```

A `status:PASS` hook must exit 0 (the adapter downgrades a non-zero PASS to FAIL). All hooks exit 0 and carry the
verdict in `status`.

## Environment variables

| Variable | Meaning |
|---|---|
| `CHROMOFOLD_RUNTIME_ENDPOINT` | base URL of the serving runtime (default `http://127.0.0.1:8080`) |
| `CHROMOFOLD_SNAPSHOT_ENDPOINT`, `CHROMOFOLD_SNAPSHOT_TOKEN` | where to read the runtime state snapshot for reference reconciliation |
| `CHROMOFOLD_HOOK_GEN_CMD` | a shell command that issues **one** real generation request (exit 0 = completed); else a generic `/v1/completions` POST is tried |

Per-scenario control commands (unset ⇒ that scenario is `INCOMPLETE`, never `PASS`):

| Scenario | Control command(s) |
|---|---|
| `cache-cold` | `CHROMOFOLD_HOOK_CLEAR_CACHE_CMD` |
| `cache-warm` | *(none — primes then repeats via generation)* |
| `multi-tenant-overload` | *(none — drives two tenants; isolation checked from the snapshot)* |
| `cancellation` | `CHROMOFOLD_HOOK_CANCEL_CMD` |
| `decode-worker-failure` | `CHROMOFOLD_HOOK_KILL_DECODE_WORKER_CMD`, `CHROMOFOLD_HOOK_RESTORE_DECODE_WORKER_CMD` |
| `network-fault` | `CHROMOFOLD_HOOK_INJECT_NETWORK_FAULT_CMD`, `CHROMOFOLD_HOOK_CLEAR_NETWORK_FAULT_CMD` |
| `certificate-rotation` | `CHROMOFOLD_HOOK_ROTATE_CERT_CMD` |
| `rolling-upgrade` | `CHROMOFOLD_HOOK_ROLLING_UPGRADE_CMD` |
| `rollback` | `CHROMOFOLD_HOOK_ROLLBACK_CMD` |

Shared plumbing lives in [`_hooklib.py`](_hooklib.py). Contract-tested by
[`tests/test_qualification_hooks.py`](../../../../tests/test_qualification_hooks.py).
