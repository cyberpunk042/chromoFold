# ChromoFold Constitution

> The non-negotiable principles that govern every design decision in the native engine.
> A proposed change that violates a principle here is out of scope until the principle is amended.
> Amendments require an explicit entry in the changelog at the bottom of this file.

ChromoFold is a **GPU-resident, random-access, searchable succinct-data-structure runtime** for the data an LLM
runs on — token streams, KV history, adapter/prompt deltas, and quantized weight side-channels. It is *not* a
file compressor and *not* a literal DNA-folding algorithm.

---

## Principles

### P1 — Compute-for-memory
The engine spends abundant GPU compute (FLOPs) to buy back the scarce resource: **VRAM capacity and memory
bandwidth**. Every feature must justify itself where memory — not compute — is the bottleneck. A change that
saves bytes but is not on a memory-bound path is not automatically a win.

### P2 — Navigable while compressed
Data stays **addressable in O(1)/O(log) and searchable in the compressed domain**. Random access (`access`),
counting/searching (`rank`, FM backward-search), and slice decode must survive every representation change.
If an optimization breaks navigation, it belongs in a general-purpose compressor, not here.

### P3 — Decode only inside the consumer
The compressed representation **survives until the exact operation that consumes the value**. The hot path must
not materialize a full decompressed intermediate buffer in global memory. Fused *decode-and-consume* kernels
(decode+embedding-gather, decode+dequant, rank+FM-search, decode+delta-apply) are the architectural thesis, not
an afterthought.

### P4 — Lossless over the chosen quantization
The entropy + index layer is **bit-exact over whatever quantization sits above it**. Accuracy is the quantizer's
job; keeping the quantized/encoded data small, addressable, and searchable is ours. Any lossy step is named,
owned by a layer above ChromoFold, and measured separately.

### P5 — Device-native, zero host round-trip
Production APIs **accept and return existing device buffers** and execute **asynchronously on the caller's
stream**. No implicit host allocation, no implicit synchronization, no PCIe round trip in the query path. The
Python prototype's copy-in/copy-out pattern is a research convenience, not the contract.

### P6 — Layered, not rewritten
The system is a **stack of languages, each used where it is strongest** — never a single-language rewrite:
Python (research/orchestration) · C++20 (core, format, builders, dispatch, bindings, stable C ABI) · CUDA C++
(production hot path) · optimized CPU AVX2/AVX-512 (baseline + build + fallback) · optional Rust (safe systems
integration) · optional Triton (experimental fused kernels). PTX only on profiler evidence; SASS never as
foundation.

### P7 — Measured, not asserted
Every performance claim carries a **reproducibility envelope**: hardware/driver/CUDA/compiler/commit SHA,
≥20 warm reps with median + p5/p95, and the four timing layers (kernel-only, GPU-API, upload+kernel, round-trip)
reported separately. Claims are stated against **honest baselines** (raw GPU gather, optimized CPU wavelet,
zstd, GPU codecs), never against a scalar Python loop. Reporting a negative result is a feature, not a failure.

### P8 — The C ABI is the contract
Internals evolve freely; the **C ABI is versioned and stable**. All interop — Python, Rust, Go, C#, llama.cpp,
TensorRT-class runtimes, custom servers — rides the ABI. The ABI is C-surface, C++-implementation.

### P9 — Build is not query
Expensive construction (suffix array, BWT, RRR encoding) may run **offline / on CPU / on a non-serving
accelerator**. The serving GPU carries a **compact, immutable query representation**. Append-heavy workloads use
reference-delta + chunked indexes rather than full reconstruction.

### P10 — The decisive proof is a workload, not a ratio
Success is **not** a headline compression ratio (zstd/xz win on bytes — different job). Success is a **real LLM
workload that fits a larger batch or longer context in the same GPU memory at equal-or-better latency**, with
the compressed data staying navigable. Ratio tables are supporting evidence; the workload is the proof.

---

## What this forbids

- A "just decompress it all first" fast path in the serving loop (violates P3).
- Any query API that allocates device memory or synchronizes implicitly (violates P5).
- A single-language rewrite "for purity" (violates P6).
- A benchmark headline without its reproducibility envelope and a real baseline (violates P7).
- Claiming a KV/token index "replaces the KV cache" without a model strategy that reconstructs or retains
  contextual state — token memory, compressed KV values, and full attention state are different objects (P10).

## Changelog
- 2026-07-20 — Initial constitution, derived from the ChromoFold Research Brief (July 2026) and the measured
  Warp prototype in `warp-solar-system-shaders/warp_compress`.
