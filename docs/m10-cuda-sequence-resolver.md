# M10 CUDA sequence resolver

This milestone connects the M9 compressed CUDA page format to the M10 multi-sequence ownership model.

## Execution path

```text
llama-server batch
  -> sequence-ID routing
  -> immutable shared-page lookup
  -> sequence descriptor snapshot
  -> sequence-local active K/V tail
  -> compressed attention launch
  -> CUDA event-backed snapshot release
  -> deferred page reclamation
```

## Implemented contracts

- device-resident shared compressed-page registration;
- per-sequence and per-layer descriptor tables;
- logical token positions independent from physical shared pages;
- sequence-local active K/V buffers;
- copy-on-write device active tails;
- all-item validation before batch launch;
- MHA and integer-ratio GQA routing;
- FP16, BF16 and FP32 query/output support;
- CUDA event-backed in-flight page references;
- cancellation that blocks new dispatch without freeing in-flight storage;
- reclamation only after logical and in-flight references reach zero;
- strict shutdown and isolation auditing;
- no dense fallback path.

## Batch transaction

A batch is validated and all sequence snapshots are acquired before any attention kernel is launched. Validation failure releases every acquired reference and launches nothing.

Once launches begin, an error fails the compressed batch. The implementation does not retry through dense KV attention.

## Copy semantics

Sealed pages are shared by logical reference. Mutable active K/V buffers are copied onto destination-owned allocations. Source and destination may then append and seal independently.

## Cancellation and reclamation

Cancellation makes the sequence non-dispatchable immediately. Logical references are removed during reset, while CUDA snapshots retain in-flight references until their completion events are observed. Retired pages are reclaimed only when both reference classes are zero.

## Evidence gate

A complete hardware artifact must prove:

```text
pages_registered > 0
descriptor_tables_built > 0
compressed_batches_executed > 0
sequences_per_batch_peak > 1
sequence_attention_launches > 1
snapshots_acquired == snapshots_released
in_flight_page_references_at_shutdown == 0
pages_reclaimed > 0
page_refs_at_shutdown == 0
cross_sequence_contamination == 0
dense_fallback_launches == 0
cuda_errors == 0
Compute Sanitizer passed
```

## Current boundary

The resolver, kernel, bridge, workload harness and proof gates are provided in this PR. A production claim still requires the self-hosted workflow to patch the exact pinned llama-server call sites, run a compatible GGUF, and emit evidence satisfying the strict validator. The hosted workflow proves compilation and contract integrity, not concurrent production throughput.
