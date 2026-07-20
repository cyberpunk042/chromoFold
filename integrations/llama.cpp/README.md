# llama.cpp integration — M9

This directory is the integration boundary for ChromoFold's first real inference-runtime workload proof.

No llama.cpp source is vendored here. The eventual adapter is built against a pinned upstream commit and remains optional and disabled by default.

## Target command

```bash
llama-cli \
  -m /models/model.gguf \
  --ctx-size 65536 \
  --chromo-kv \
  --chromo-kv-page-size 256 \
  --chromo-kv-window 4096
```

The exact flag names may change during implementation, but the final integration must expose an unambiguous opt-in and must never silently claim ChromoFold when falling back to the baseline cache.

## Planned files

```text
integrations/llama.cpp/
├── README.md
├── chromofold_kv.h          adapter-facing ownership contract
├── chromofold_kv.cpp        cache lifecycle and llama.cpp translation
├── chromofold_kv_cuda.cu    runtime-specific CUDA glue where unavoidable
├── pinned-commit.txt        exact tested upstream revision
└── patches/                 minimal reviewable upstream patch series
```

Core codecs and kernels remain under `src/`; runtime-specific layout translation remains here.

## Adapter responsibilities

- Map llama.cpp layer/head/sequence metadata to ChromoFold cache identities.
- Allocate and append to the active KV page.
- Seal full pages into immutable encoded pages.
- Invoke the core paged-attention interface on llama.cpp's CUDA stream.
- Merge active-page and sealed-page online-softmax state correctly.
- Reset or fork per-sequence state without leaking allocations.
- Report active, sealed, metadata, scratch, and total resident bytes.
- Emit benchmark records conforming to `benchmarks/m9_result.schema.json`.

## Core-library responsibilities

- Validate page descriptors before launch.
- Encode and decode page payloads.
- Execute compressed paged attention.
- Provide a dense reference path for correctness tests.
- Preserve asynchronous caller-stream semantics.
- Return explicit unsupported/error statuses.

## Cache state machine

```text
EMPTY
  │ append
  ▼
ACTIVE(page partially filled)
  │ page full
  ▼
SEALING ──failure──► ACTIVE (unchanged, explicit error)
  │ success
  ▼
SEALED_HISTORY + new ACTIVE page
  │ reset
  ▼
EMPTY
```

A page is published only after all payload and metadata writes are ordered before the descriptor becomes visible to the query stream.

## Initial support envelope

The first implementation should deliberately support a narrow, testable envelope:

- NVIDIA CUDA backend;
- fp16/fp32 query and output accumulation as selected by the kernel;
- grouped-query attention;
- head dimension up to 128;
- one selected GGUF model family;
- batch sizes required by the benchmark matrix;
- page sizes 128 and 256;
- causal decoding and prefill paths explicitly identified.

Unsupported layouts fail with a clear message. Expansion comes only after the workload proof.

## Benchmark protocol

Every baseline/ChromoFold pair must use:

- the same model bytes and model SHA-256;
- the same runtime commit;
- the same prompt tokens, generation count, seed, and sampling settings;
- a clean process for peak-VRAM measurements;
- documented warmup runs;
- synchronized timing boundaries only in benchmark instrumentation, never in the core query API.

Results are checked with:

```bash
python3 benchmarks/validate_m9_result.py result.json --require-claim
```

The example JSON contains illustrative numbers and must never be cited as evidence.

## Required implementation tests

1. Active page only.
2. Exact seal boundary.
3. One token past a seal boundary.
4. Multiple sealed pages.
5. Query window beginning inside a sealed page.
6. Query window crossing active/sealed boundaries.
7. GQA head mapping.
8. Batch reset and cache reuse.
9. Baseline comparison across page sizes.
10. Allocation accounting and Compute Sanitizer fixture.

## Review rule

The adapter may not merge based only on successful text generation. It must include numerical comparison, memory accounting, and machine-readable benchmark output. The final M9 claim requires reproduction on a second compatible NVIDIA system.
