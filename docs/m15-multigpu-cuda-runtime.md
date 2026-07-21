# M15 multi-GPU CUDA runtime

M15 converts the M14 distributed control-plane contract into a CUDA execution boundary.

## Runtime path

```text
sequence placement
  -> tensor-parallel shard planning
  -> device-local mixed-codec page lookup
  -> CUDA peer transport or explicit host staging
  -> per-device compressed attention
  -> collective output assembly
  -> generation-safe migration
  -> failure isolation and shutdown audit
```

## Safety rules

- The entire shard batch is validated before any transport or dispatch occurs.
- A failed device cannot receive new work.
- Remote pages require either CUDA peer access or explicitly enabled host staging.
- Dense attention fallback is prohibited.
- Page and snapshot references must reconcile to zero before shutdown succeeds.
- Device failure injection must fail closed before recovery can be claimed.

## CI policy

Routine pull requests run schema, script, anchor, false-evidence, and dense-fallback checks only. CUDA compilation is manual. The six-hour real multi-GPU proof is manual-only and requires a self-hosted runner with at least two NVIDIA GPUs and a runner-local GGUF.

## Production completion boundary

A production claim requires real peer transfers, real collective execution, mixed-codec remote page consumption, concurrent pinned-server requests, failure injection and recovery, four Compute Sanitizer tools, zero contamination, zero dense fallback, zero CUDA/transfer errors, and a clean shutdown audit.
