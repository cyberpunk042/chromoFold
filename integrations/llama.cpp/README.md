# ChromoFold KV adapter for llama.cpp

This directory is the version-sensitive boundary between llama.cpp and ChromoFold's compressed paged KV runtime. No llama.cpp source is vendored here, and ChromoFold remains optional and disabled by default.

## Intended runtime surface

```text
--kv-cache-backend dense
--kv-cache-backend chromofold
--chromofold-page-size 128
```

A downstream llama.cpp patch should never silently fall back after ChromoFold initialization. Unsupported models, layouts, backends, sequence operations, or GPU configurations must produce explicit errors.

## Implemented in this round

- C adapter creation and destruction;
- topology and option validation;
- per-layer/per-KV-head append;
- automatic sealing at the configured page size;
- explicit flush of partial active pages;
- move-only compressed page ownership;
- cache clearing and sequence-0 removal;
- explicit capability errors for sequence copy and nonzero context shift;
- memory and lifecycle statistics;
- no-GPU adapter seam tests;
- CUDA-container compilation;
- pinned upstream integration manifest.

## Cache lifecycle

```text
EMPTY
  │ append
  ▼
ACTIVE(page partially filled)
  │ page full
  ▼
ENCODE + UPLOAD
  │ success
  ▼
SEALED_HISTORY + new ACTIVE page
```

A failed encode or upload throws through the adapter and leaves no successful status. Historical dense KV is not silently retained under an untracked fallback mode.

## Wiring points in llama.cpp

The maintained downstream patch should remain narrow:

1. add backend and page-size options in `common/arg.cpp`;
2. construct `cf_llama_kv_adapter` beside llama.cpp's KV cache context;
3. forward newly produced per-layer K/V rows to `cf_llama_kv_append`;
4. route supported CUDA attention through the ChromoFold paged-attention descriptor;
5. forward clear, `seq_rm`, `seq_cp`, and shift operations;
6. expose `cf_llama_kv_stats` in diagnostics and benchmark JSON;
7. reject unsupported operations rather than mixing cache semantics.

Current upstream still centralizes cache operations in `src/llama-kv-cache.cpp`; the integration manifest records the paths and capabilities that must be reviewed whenever the pinned revision changes.

## Current envelope

- NVIDIA CUDA;
- fp32 K/V at the adapter boundary;
- head dimensions supported by `CF_KV_MAX_HEAD_DIM`;
- MHA and GQA topology validation;
- linear sequence `0` generation;
- fixed deterministic int4/Huffman page fixtures.

Sequence copy requires a reference-counted immutable page table. Nonzero context shift requires token-position remapping and RoPE-aware semantics. Both currently return explicit errors.

## Validation

```bash
make -f m9-llama.mk llama-adapter-compile
make -f m9-llama.mk llama-adapter-seam
```

The adapter may not be declared production-ready based only on compilation or successful text generation. Completion still requires a maintained llama.cpp patch, logits comparison, real-model memory accounting, and a fixed-VRAM capacity experiment.
