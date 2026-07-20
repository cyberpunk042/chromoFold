# ChromoFold Native Engine — Roadmap

> **When**, in order. Each milestone has a goal, a concrete acceptance test, and its link to the frozen Warp
> reference ([05-porting-map.md](05-porting-map.md)). Governed by [00-constitution.md](00-constitution.md).
> The ordering is deliberate: **prove the primitive and the device-native path before breadth.**

## Guiding rule

Port **one operation at a time**, keep the Python API identical, and gate each step on beating (or matching, for
correctness) the frozen reference **and** an honest baseline (P7). Do not start a step until the previous step's
acceptance test is green and reproducible on a second run.

---

## M0 — Reproducible harness + frozen reference — ✅ DONE
**Goal.** Stand up the benchmark harness ([04-benchmarks.md](04-benchmarks.md)) and freeze the current Warp
result as the reference implementation and correctness oracle.
**Acceptance.** Harness emits the full reproducibility envelope (hardware/driver/CUDA/commit, ≥20 reps,
median+p5/p95, four timing layers) for Warp `access`/`rank`; results archived and versioned.
**Delivered.** `tools/export_reference.py` freezes a Warp `GPUWavelet` index + golden `access` output to a
portable `.cfwv` binary (1M tokens, V=256, 8 levels, 100K golden queries, 1.13 B/tok). `benchmarks/gpu_access.cu`
emits device metadata + 30-rep median/p95 across the kernel-only and round-trip timing layers.

## M1 — CUDA C++ `access`, device-native async API — ✅ DONE
**Goal.** Port `GPUWavelet::access` to a CUDA C++ kernel behind `cf_access_async` (device pointers, caller
stream, no host copy — P5), with compile-time specialization by vocabulary width (architecture §5).
**Acceptance.** Bit-identical to Warp `access`; kernel-only and round-trip times reported separately.
**Delivered (RTX 2080 Ti, sm_75, CUDA 12.6).** `src/cuda/access.cu` + `include/chromofold/chromofold.h`:
- correctness: **BIT-IDENTICAL** to the frozen Warp golden (0 / 100,000 mismatches).
- kernel-only: **1233 M access/s (0.81 ns/access)** — matches the Warp kernel; the device-native path allocates
  and synchronizes nothing.
- round-trip: 367 M access/s (2.73 ns) — isolates the H2D/D2H tax the device-native API (device pointers,
  caller stream) removes. This reproduces the brief's "kernel fast, user-facing transfer-bound" finding.
- templated `cf_access_kernel<LEVELS>` with a switch dispatch (2/4/8/15/16/17) + dynamic fallback.

**Next (M2).** Raw GPU gather baseline + Warp head-to-head across V and batch; wire pybind11 so the Python API
is unchanged.

## M2 — Baseline frontier for `access` — ◐ frontier DONE (pybind11 next)
**Goal.** Compare CUDA C++ `access` against **raw GPU gather** (uint8/16/32) and Warp, across vocabulary widths
and query patterns; compile-time specialization by `levels` (§5 of architecture).
**Delivered — Experiment A (`benchmarks/frontier.cu`, RTX 2080 Ti).** Raw gather vs wavelet access, kernel-only
median over 30 reps, three query patterns, six vocabularies. Honest findings:
- raw gather is the floor (~0.10 ns, flat across V — one memory load, no rank/search).
- addressability premium (wavelet ÷ gather), **uniform** queries: 1.7× (V=4) → 6.3× (V=256) → 13.4× (V=128K);
  it scales with `log₂(V)` (level count). **Sorted/contiguous roughly halves it** (V=256: 6.3× → 2.6×) via
  coalescing + rank-directory cache reuse — evidence for warp-cooperative queries (architecture §7).
- footprint: wavelet ≈ `log₂(V)` bits/tok, so it is *smaller* than raw where the type wastes bits (V=128K:
  19 b/tok vs raw uint32 32) and slightly larger on byte-aligned vocab (V=256: 9 vs uint8 8).
- **The honest line:** raw gather wins for dense, small-vocab, uniform reads; the wavelet's premium buys
  rank/select/FM-search (which gather cannot do) and large-vocab footprint. The win is *capabilities and
  sparse/search access*, not per-element gather speed — consistent with the constitution (P2, P10).
**Remaining.** pybind11 binding so `cf_access` is callable from Python with the prototype's API unchanged;
add the RRR-wavelet row and effective-bandwidth column.

## M3 — CUDA C++ `rank` + two-level rank directory
**Goal.** Port `rank`; implement and benchmark the two-level rank directory (§6) vs the eight-word-scan layout.
**Acceptance.** `rank` bit-identical to the naïve/CPU reference; two-level directory measured (latency vs
directory-memory tradeoff) with a recorded decision.

## M4 — RRR-coded bitplanes on GPU
**Goal.** Port RRR rank/decode so the resident index reaches the entropy frontier (the Warp prototype hit
5.80 b/tok on a BWT, below H₀). Include the two-level *encoding* (int32 anchors + uint16 deltas).
**Acceptance.** RRR `access`/`rank` bit-identical; measured memory–latency frontier vs packed wavelet and zstd,
honestly bounded (RRR wins on skewed planes, not on max-entropy).

## M5 — C++ builder + optimized CPU verifier
**Goal.** A C++20 offline builder (suffix array, BWT, RRR, serialization) and an AVX2/AVX-512 CPU backend as the
correctness oracle and honest CPU baseline (§9). Establishes the build ≠ query split (P9).
**Acceptance.** CPU backend matches GPU bit-for-bit on a fuzz corpus; CPU `access`/`rank` throughput reported as
a real baseline (not a Python loop); builder round-trips the on-device format.

## M6 — Fused `access` + embedding gather (the thesis)
**Goal.** The first fused decode-and-consume kernel (P3): decode token ID → gather embedding row → write
embedding, with **no decompressed token buffer** (Experiment D).
**Acceptance.** Beats decode-then-gather on **end-to-end embeddings/s and bytes moved** (not decoded-tokens/s);
output embeddings bit-identical to the unfused path.

## M7 — FM backward-search + locate on GPU
**Goal.** Port `count`/`locate` (backward search over the RRR/BWT wavelet). The Warp prototype already builds the
suffix array on-GPU (17–32× CPU) and does GPU count/locate/predict_next.
**Acceptance.** Search results bit-identical to CPU FM-index; latency + reproducibility envelope recorded.

## M8 — Reference-delta apply + shared-prefix serving demo
**Goal.** Port delta-apply; build the shared-prefix prompt-cache demonstration (Experiment B: N requests, one
big shared prefix, unique suffixes) — the naturally-redundant serving workload.
**Acceptance.** Resident + peak transient memory, TTFT/prefill, and max batch at fixed memory budget vs
duplicated and content-addressed storage; cost to add one suffix/turn recorded.

## M9 — Integrate with one real inference path (the ship criterion, P10)
**Goal.** Wire the KV path into one real runtime (start with the `transformers` `Cache` adapter already built;
target vLLM/llama.cpp KV hooks next) and measure **batch capacity / context length / TTFT / end-to-end
throughput** at equal VRAM.
**Acceptance.** A **larger batch or longer context fits in the same GPU memory at equal-or-better latency**, with
output quality intact. This is the decisive proof (P10).

## M10 — Rust wrapper (only now)
**Goal.** Add the Rust safe-systems layer (parsing, integrity, mmap, corpus, CLI, server glue) over the C ABI —
**after** the ABI and on-device format have stabilized (P8/P6).
**Acceptance.** Rust wrapper round-trips a build → serve cycle through the C ABI with zero changes to the core.

## M11 — PTX micro-optimization (evidence-gated)
**Goal.** Targeted inline PTX only where the profiler proves the compiler missed a needed instruction
(`popc`/`brev`/funnel shift/vector load/cache-control load).
**Acceptance.** A profiler trace identifying the specific bottleneck **before** any PTX is written; measured gain
that CUDA C++ could not reach. SASS remains inspection-only.

---

## Sequencing notes

- **M0–M2 are the credibility gate.** They convert "a primitive works in Warp" into "a device-native CUDA kernel
  on the measured frontier." Do not skip ahead to breadth.
- **M6 (fusion) is the differentiator** — it is where ChromoFold stops being "a faster compressor" and becomes a
  compressed-memory *runtime*. Prioritize it over porting every operation.
- **M9 is the only step that proves the thesis.** Everything before it is necessary machinery; the workload win
  is the deliverable.
- Rust (M10) and PTX (M11) are deliberately last — ecosystem/micro-optimization risk added only after value and
  the ABI are proven.
