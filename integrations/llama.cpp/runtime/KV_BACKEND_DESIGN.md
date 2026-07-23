# End-to-end ChromoFold KV backend — design & phased plan

> ⚠️ **Read [`KV_BACKEND_FINDINGS.md`](KV_BACKEND_FINDINGS.md) first.** Scoping this backend surfaced that
> **llama.cpp already ships quantized KV (`-ctk q4_0`) + fused quantized flash attention** — i.e. the VRAM +
> decode-in-consumer thesis this backend targets. Against llama's q4_0 the storage/latency gap is ≈1×; ChromoFold's
> only measured KV edge is ~1.19× quantizer accuracy (KIVI vs per-block) and, more fundamentally, **searchability**
> (which KV-decode doesn't use). This design remains valid *if* the effort is justified on those differentiators —
> not on a VRAM win that llama already delivers. Below is the seam/plan; the recommendation is to redirect.

**Goal (P10 ship criterion).** Make `llama-server` *serve attention from the compressed KV cache instead of
allocating a dense one*, so the engine-level **2.67× capacity** (`gpu-capacity`) becomes a real **end-to-end VRAM
win** in a running server. The `cb_eval` shadow seam (append/replace, verified — see `CLI_EVIDENCE_PATCH.md`) proved
the compressed serving is *correct on a live model*, but it is a **shadow**: llama still allocates its full dense
KV, so no VRAM is saved. This backend removes the dense allocation.

## The prize, quantified (Qwen2.5-0.5B)
Dense KV per llama's allocator: `n_layer · n_embd_k_gqa · kv_size · (K+V) · sizeof(type_k)`.
For this model `n_layer=24`, `n_embd_k_gqa = kv_heads·head_dim = 2·64 = 128`, `type_k=f16`:
`24 · 128 · kv_size · 2 · 2 = 12288 · kv_size` bytes.

| context | dense f16 KV | ChromoFold int4 (3.2× logical / 2.67× real) |
|---|---|---|
| 8 192 | 96 MB | 30 / 36 MB |
| 32 768 | 402 MB | 126 / 151 MB |
| 131 072 | 1.6 GB | 0.5 / 0.6 GB |

The win scales with context — the long-context regime is exactly where it matters (and where the latency crossover
also lands, `gpu-latency`).

## The seam (pinned tree, commit 76f46ad2)
- **Construction / injection point:** `llama_model::create_memory(...)` — `src/llama-model.cpp:2040`. Dispatches to
  `llama_kv_cache` / `_dsa` / `_dsv4` / `_iswa`. A `chromofold` backend adds a branch here (gated on the already-
  scaffolded `common_chromofold_params.backend == "chromofold"`).
- **Dense allocation (the VRAM):** `src/llama-kv-cache.cpp:231-232` — per layer
  `k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)` (and `v`). This is what we replace with
  compressed page storage.
- **Write path:** `cpy_k`/`cpy_v` (`llama-kv-cache.cpp:1295/1330`) copy the current step's K/V into the cache slots.
  ChromoFold: **encode** (quantize→int4 pages via `CompressedKvCache::append`).
- **Read/attention path:** `get_k`/`get_v` (`1243/1263`) return dense tensor *views* consumed by
  `llm_graph_context::build_attn_mha` (`src/llama-graph.cpp:2384`), which builds `mul_mat(k,q) → softmax → mul_mat`.
  ChromoFold: replace this subgraph (for compressed layers) with a **single fused op** = `cf_kv_paged_attention`.
- **Interface surface:** `llama_memory_i` (`src/llama-memory.h`) — **21 virtual methods** (`init_batch`,
  `init_full`, `init_update`, `seq_rm/cp/keep/add/div`, `clear`, `get_can_shift`, state read/write, …). A backend
  must implement all; most map to `CompressedKvCache` ops we already have (`append`, `clear`, `remove/copy/shift_
  sequence`, `attention_view`).

## Architecture
Two pieces, each independently verifiable:

1. **Fused attention as a ggml custom op** (`ggml_custom` / `ggml_map_custom`). Wraps `cf_kv_paged_attention_async`:
   inputs are the query tensor + (via op userdata/params) the layer's `attention_view` (device page descriptors +
   active tail); output overwrites the attention result. This is the reusable primitive and can be verified in a
   **standalone ggml graph** (custom-op output vs the dense `mul_mat/softmax` path) before any llama wiring — the
   same "prove the primitive in isolation" approach that de-risked the replace step.
2. **`llama_kv_cache_chromofold : llama_memory_i`** — holds one `CompressedKvCache` (already GPU-resident,
   int4/int8, sealed pages + active tail). `cpy_k/cpy_v` → `append`; the graph attention → the custom op above.
   Allocates **no** dense `k`/`v` tensors → the VRAM saving.

## Phased plan (each phase ends green + measured)
- **P1 — custom attention op, standalone.** Register a ggml custom op wrapping `cf_kv_paged_attention`; verify
  bit-exact vs a dense ggml reference graph (reuse the `gpu-adapter-attention` fixtures). *No llama changes.*
- **P2 — `llama_kv_cache_chromofold` storage.** Implement the `llama_memory_i` surface over `CompressedKvCache`
  (append on write, no dense alloc); unit-test the lifecycle (append/clear/seq ops) off-graph.
- **P3 — graph routing.** Branch `create_memory` (2040) to build it under `--kv-cache-backend chromofold`; route
  `build_attn_mha` through the P1 custom op for compressed layers. Run `run_pair_server.py --parallel 1`:
  token parity (within the int4 tolerance from the frontier study) + **zero dense KV allocated**.
- **P4 — the payoff.** Measure real server VRAM (`nvidia-smi`) dense vs chromofold at long context → the end-to-end
  "N× more context / lower VRAM at equal context" number. Flip `--require-claim` to a real pass.

## Honest scope & risks
- **Big surface.** 21-method interface + graph surgery in `build_attn_mha` + a custom ggml op + the compressed
  storage lifecycle. Realistically **multiple sessions**; P1 is the bounded, high-value first increment.
- **Custom-op / device-pointer lifetime.** The op must read the layer's live `attention_view` at eval time; page
  pointers change as pages seal. Needs careful userdata/stream handling (the `cb_eval` replace path hit the same
  class of hazard and was solved with per-layer capture + sync).
- **Scope limits carry over.** Single-ubatch position alignment (`initial_support`), causal f32, GQA — the same
  envelope the shadow path proved; multi-ubatch/sequence-management is a follow-on.
- **Accuracy is the int4 quantizer's** (frontier study): parity is *within tolerance*, not bit-exact — the gate
  must be a tolerance bound, not `==` (see the strict-parity section of `CLI_EVIDENCE_PATCH.md`).
- **Latency:** the warp kernel makes compressed competitive (crossover ~1.04× at hd=64, long ctx); on this model
  it's ~break-even, so the honest headline of this backend is **capacity/VRAM**, not speed.

## What already exists to build on
- CLI flag `--kv-cache-backend {dense,chromofold}` + `--chromofold-page-size` + `common_chromofold_params`
  (scaffolded by `apply_runtime_patch.py`); many harness scripts already pass it (`performance/`, `multisequence/`,
  `distributed/`).
- `chromofold-runtime` static lib (adapter + `CompressedKvCache` + `paged_kv_attention`) wired by the patch cmake.
- The verified engine: `cf_kv_paged_attention` (warp-cooperative, bit-exact across head_dim 32–128 × int4/int8),
  `CompressedKvCache` (append + `attention_view` with active tail), `cf_llama_kv_attention` adapter entry point.
- The `cb_eval` shadow path is the **correctness oracle**: end-to-end compressed serving already matches dense on
  the live model (mean 1.2e-4 raw / int8 0.0024), so P3's parity target is known-achievable.
