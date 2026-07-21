# M9 llama.cpp integration contract

## Goal

Connect ChromoFold's compressed paged KV lifecycle to llama.cpp without making unsupported runtime or performance claims.

## Architectural boundary

The core cache manager owns per-layer and per-KV-head state. The C adapter is the only API llama.cpp should call. Version-sensitive upstream changes remain under `integrations/llama.cpp/`.

## Supported lifecycle

- append logically contiguous K/V rows;
- retain a bounded dense active page;
- seal every full page into an immutable compressed allocation;
- refresh the device descriptor array for attention;
- flush a final partial page;
- clear the cache;
- report exact active, compressed, and descriptor bytes.

## Failure semantics

- invalid topology fails during creation;
- non-contiguous append fails before publication;
- encode or CUDA upload failure is returned through `last_error`;
- sequence copy fails until shared immutable page ownership exists;
- nonzero context shift fails until position remapping is implemented;
- no operation silently switches to dense historical KV.

## Upstream compatibility

The integration manifest identifies the llama.cpp repository, integration mode, required paths, backend, and cache operations. It intentionally tracks `master` as a moving review target but does not claim compatibility with every upstream revision. A production release must replace that moving ref with an exact tested commit.

## Required downstream patch

The next patch layer must add:

- CLI/config parsing;
- adapter construction during context initialization;
- K/V append hooks;
- attention dispatch using ChromoFold descriptors;
- cache operation forwarding;
- diagnostics and JSON evidence;
- clean teardown and stream ordering.

## End-to-end proof gate

M9 is not complete until a pinned llama.cpp build demonstrates:

1. deterministic dense-versus-ChromoFold prompt ingestion;
2. logits or token agreement within declared tolerance;
3. compressed pages consumed by the CUDA kernel;
4. no resident dense historical copy;
5. measured KV and total VRAM;
6. fixed-VRAM maximum-context or batch improvement;
7. prompt and decode latency deltas;
8. reproducible machine-readable results.

## Known limitation

The current cache manager rebuilds the device descriptor array when an attention view is requested. This is correct but not tuned. Production work should use a capacity-managed descriptor buffer with incremental updates and stream-ordered reclamation.
