# M9 Runtime Core

This document records the first executable layer beneath the M9 llama.cpp integration contract.

## Delivered in this round

### Public page and cache descriptors

`include/chromofold/kv_runtime.h` defines a C-compatible host-visible contract for:

- immutable sealed compressed-KV pages;
- the active appendable tail;
- cache topology across layers and grouped-query KV heads;
- explicit payload counts for validation and exact resident-byte accounting;
- a mergeable online-softmax CPU oracle.

The descriptors do not own memory. Payload pointers remain runtime-owned and are interpreted as device pointers when `CF_KV_PAGE_FLAG_DEVICE_RESIDENT` is set.

### Validation

`cf_kv_validate_page` rejects:

- wrong structure or ABI versions;
- unsealed, empty, oversized, or malformed pages;
- missing encoded streams, offsets, lookup tables, or scales;
- scale cardinalities inconsistent with K-per-channel and V-per-token quantization;
- insufficient block metadata;
- overflowing token ranges.

`cf_kv_validate_layout` additionally rejects:

- invalid layer/head topology;
- active tails larger than page capacity;
- under-accounted active K/V allocations;
- pages outside the configured topology;
- pages crossing into the active token range;
- overlapping sealed pages for the same layer and KV head.

### Exact accounting

`cf_kv_page_resident_bytes` computes encoded payload bytes from explicit element counts.

`cf_kv_cache_memory_stats` reports:

- sealed encoded payload bytes;
- host descriptor bytes;
- active-tail payload bytes;
- total accounted bytes;
- sealed and active token counts;
- page count.

This is accounting, not allocator telemetry. The llama.cpp adapter must separately report allocator-reserved and peak device memory in the M9 benchmark result.

### Online-softmax oracle

The CPU oracle supports:

1. initializing caller-owned accumulation storage;
2. accumulating score/value pairs without numerical overflow;
3. merging independently accumulated page states;
4. finalizing the normalized attention output.

The merge rule is the reference contract for the future paged CUDA kernel. Splitting a sequence at arbitrary page boundaries must produce the same result, within the declared tolerance, as dense attention over the unsplit sequence.

## CI and tests

`tests/kv_runtime_test.cpp` covers:

- valid and invalid page descriptors;
- exact payload-byte accounting;
- overlap detection;
- active-tail allocation bounds;
- page-wise online-softmax merge against a dense implementation;
- extreme score ranges;
- empty-state rejection.

`.github/workflows/m9-runtime.yml` builds the runtime with warnings-as-errors, AddressSanitizer, and UndefinedBehaviorSanitizer. It also compiles the public runtime header as C11 to protect the C ABI.

## Intentionally not implemented yet

- device allocation or ownership;
- page encoding/sealing;
- CUDA paged attention;
- active-tail plus sealed-history dispatch;
- llama.cpp source integration;
- GPU correctness or performance claims.

Those layers now have a concrete ABI, validation boundary, accounting model, and numerical oracle to build against.

## Next implementation boundary

The next kernel-facing entry point should consume an array of validated `cf_kv_page_view` descriptors and emit one online-softmax partial per query/head. The active tail should use the same partial representation so sealed and active results can be merged without separately normalized softmax outputs.
