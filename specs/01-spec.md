# ChromoFold Native Engine — Functional Specification

> **What** the native engine is and does, and how we will know it works. The **how** lives in
> [02-architecture.md](02-architecture.md); the **when** in [03-roadmap.md](03-roadmap.md).
> Governed by [00-constitution.md](00-constitution.md).

## 1. Problem

LLM serving is bounded by **GPU memory capacity and bandwidth**, not compute. The KV cache grows linearly with
context; weights, MoE experts, adapter libraries, and prompt caches all compete for the same VRAM. Ordinary
compression shrinks bytes but destroys navigation: to read one element you must inflate a whole block, on the
CPU, across PCIe. ChromoFold keeps LLM-shaped data **compressed *and* navigable, resident on the GPU**.

## 2. What it is

A native runtime that holds token-addressable objects as **succinct data structures** (packed/RRR wavelet
matrices, FM-index, reference-delta trees, block-entropy-coded value streams) in device memory, and answers
`access` / `rank` / `select` / `search` / `slice` **directly over the encoded form**, decoding only the exact
entries the next model operation consumes.

### 2.1 Object model

| Object | Encoding | Primary queries |
|---|---|---|
| Token stream / context | packed or RRR wavelet matrix (+ sampled suffix array) | `access(i)`, `rank(c,i)`, `select`, FM `count`/`locate`, slice decode |
| KV history | KIVI per-axis quantized + block-entropy-coded, windowed | `fetch_positions(window)`, attended-only decode |
| Prompt cache / adapter library | reference-delta tree (shared prefix + sparse deltas) | `decode(leaf)`, `fetch(leaf, pos)` |
| Quantized-weight side-channels | class-Huffman / rANS block streams + sparse outliers | `decode`, `fetch(idx)`, fused decode-in-matmul |

An **index view** is a plain-old-data descriptor over device pointers (bitplanes, superblocks, zero-counts,
token count, level count) — see the `ChromoFoldView` struct in the architecture doc. Views are immutable at
query time (P9).

### 2.2 Core operations

- `access(positions) -> token_ids` — random access into the compressed stream.
- `rank(symbol, positions) -> counts` — occurrences before a position; the FM-index primitive.
- `search(patterns) -> ranges/locations` — backward search over the BWT wavelet (count + locate).
- `slice(a, b) -> tokens` — contiguous range decode without touching the rest.
- `delta_apply(leaf, positions)` — reconstruct a reference-delta leaf, sparsely.
- **Fused consumers** (the thesis, P3): `access + embedding_gather`, `rank + candidate_retrieval`,
  `decode + kv_dequant`, `decode + delta_apply`, `RRR_decode + sparse_gather`.

## 3. Scope

### In scope
- A device-native, asynchronous, zero-host-copy query API over immutable GPU-resident indexes (P5).
- CUDA C++ production kernels for `access`, `rank`, RRR decode, FM backward-search, delta-apply, and the fused
  consumers above.
- A C++20 core owning the on-device binary format, builders, dispatch, CPU backend, and a stable C ABI (P8).
- An optimized CPU backend (AVX2/AVX-512) as a correctness oracle, build path, and honest baseline.
- A reproducible benchmark harness (P7) with the frozen Warp prototype as the reference oracle.

### Out of scope (for v1)
- Beating zstd/xz on raw compression ratio (explicitly not the objective — P10).
- Lossless compression of *dense* post-quantization weights (near-incompressible; compose with quantization
  and target structure instead — see risks).
- A from-scratch quantizer (ChromoFold composes on top of GPTQ/AWQ/KIVI; it is the entropy + addressing layer).
- Claiming to *replace* the KV cache; ChromoFold compresses/serves KV values and token memory, which are
  distinct from full contextual attention state.

## 4. Success criteria

Ordered by decisiveness (P10 first).

1. **Workload proof.** On one real inference path, ChromoFold fits a **larger batch or longer context in the
   same VRAM** at equal-or-better latency, output quality intact. This is the ship criterion.
2. **Device-native path.** A query completes with **no host allocation, no synchronization, no PCIe round trip**,
   on the caller's stream (validated by profiler).
3. **Fused win.** A fused decode-and-consume kernel beats decode-then-consume on **end-to-end** bytes-moved and
   throughput (e.g. embeddings/s), not just decoded-tokens/s.
4. **Frontier.** Against honest baselines (raw GPU gather, optimized CPU wavelet, zstd, GPU codec), ChromoFold
   is on the **memory–latency Pareto frontier** for sparse/random/search access, and says plainly where it is
   not (e.g. dense contiguous reads on small alphabets).
5. **Reproducibility.** Every headline ships with its envelope and passes on a second machine/GPU generation.

## 5. Evidence already in hand (the reference oracle)

Measured in the Warp prototype (`warp-solar-system-shaders`, 341 tests) — these are the **correctness and
lower-bound performance targets** the native port must match or beat; see [05-porting-map.md](05-porting-map.md).

- GPU wavelet `access`/`rank`: 4M tokens, V=256, ~1.1 B/tok index, ~400M access/s user-facing (~1.24 B/s kernel).
- Random access vs decompress-all: **68–111×** faster for sparse reads on a 4M-token stream.
- RRR wavelet on a BWT: **5.80 b/tok**, below the BWT's H₀ (5.96), two-level resident rank directory.
- KV (KIVI int4 + entropy): attn MSE 1.5e-4; **8.1–8.9× VRAM** vs fp16; window-attention latency **flat 1K→64K**
  while decode-all grows (**264× at 64K on Qwen2.5-1.5B**).
- GPU FM-index build (suffix array by prefix-doubling): **17–32×** faster than CPU, bit-identical.
- Fused decode-in-matmul (weights): **10.6× less VRAM held during compute** (Warp PoC, ~4 GFLOP/s — a
  correctness/memory proof, not yet a tensor-core kernel).
- Drop-in `transformers` KV cache: real Qwen2.5 generation, coherent, resident KV reduced.

The native engine exists to turn these validated primitives into a production, device-native, fused runtime —
and to close the measurement gaps the brief names (honest CPU/GPU-gather/zstd baselines, the four timing layers,
the workload proof).
