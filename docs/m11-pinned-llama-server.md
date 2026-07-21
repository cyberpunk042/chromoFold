# M11 pinned llama-server integration

M11 binds the M9 compressed CUDA data plane and the M10 multi-sequence resolver to one exact llama.cpp revision:

```text
76f46ad29d61fd8c1401e8221842934bf62a6064
```

## Runtime ownership

One `cf_llama_server_runtime` is owned per `llama_context`. It coordinates:

- the M10 host sequence cache;
- the M10 CUDA sequence resolver;
- slot and sequence generations;
- cancellation and slot reset;
- interleaved server batches;
- shutdown reclamation and auditing.

The dense KV cache is not a retry path. A ChromoFold execution failure is returned to the server.

## Upstream patch contract

`apply_pinned_server_patch.py` performs four checks before modification:

1. the checkout is at the exact pinned commit;
2. every required source file exists;
3. every expected source symbol is present;
4. no unsupported source drift has removed an integration anchor.

Every insertion carries M11 begin/end markers. Reapplying the generator must change no file and must report `idempotent: true`.

The generated hooks cover the upstream CLI parser, server context ownership, decode/KV append entry, sequence cache mutations, and attention graph construction. Hosted CI proves the pin, anchors, schemas, idempotency contract, and false-evidence rejection. The production claim remains gated on the self-hosted workflow applying the hooks to the pinned checkout and building the resulting server.

## Server lifecycle

```text
llama_context create
  -> ChromoFold runtime create
  -> HTTP slot create
  -> sequence-local append
  -> page seal and shared publication
  -> compressed sequence batch execution
  -> cancellation / copy / shift / reset
  -> stop scheduler
  -> release sequences
  -> reclaim device pages
  -> host and device audits
  -> evidence write
  -> runtime destroy
```

No new sequence or batch is accepted after shutdown starts.

## Production proof

The self-hosted workflow starts the actual patched `llama-server`, sends concurrent HTTP completion requests, exercises shared prefixes, independent prompts, cancellation, slot reuse, and branching, then performs a graceful shutdown.

A valid evidence artifact must prove:

```text
real_http_requests > 0
active_sequences_peak > 1
interleaved_batches > 0
device_append_calls > 0
seal_successes > 0
pages_registered > 0
shared_page_references > 0
compressed_batches_executed > 0
sequence_attention_launches > 1
sealed_values_consumed > 0
active_values_consumed > 0
cancelled_requests > 0
slot_reuses > 0
pages_reclaimed > 0
snapshots_acquired == snapshots_released
page_refs_at_shutdown == 0
snapshot_refs_at_shutdown == 0
cross_sequence_contamination == 0
dense_fallback_launches == 0
cuda_errors == 0
all sanitizer tools passed
shutdown audit passed
```

## Reproduction

```bash
make -f m11-pinned-llama-server.mk m11-verify
make -f m11-pinned-llama-server.mk m11-idempotency
make -f m11-pinned-llama-server.mk m11-schema
make -f m11-pinned-llama-server.mk m11-false-evidence
make -f m11-pinned-llama-server.mk m11-anchor-check
```

The final hardware workflow also requires a runner-local GGUF file and a self-hosted NVIDIA runner.

## Non-claims

This PR does not claim a passing production artifact before the self-hosted workflow has run. It also does not claim that the current correctness-oriented compressed attention implementation matches optimized dense-attention throughput. Those claims require measured server evidence rather than source-level contracts.
