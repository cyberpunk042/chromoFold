# ChromoFold Native Engine — Roadmap

> **When**, in order. Each milestone has a goal, a concrete acceptance test, and its link to the frozen Warp
> reference ([05-porting-map.md](05-porting-map.md)). Governed by [00-constitution.md](00-constitution.md).
> The ordering is deliberate: **prove the primitive and the device-native path before breadth.**

## Guiding rule

Port **one operation at a time**, keep the Python API identical, and gate each step on beating (or matching, for
correctness) the frozen reference **and** an honest baseline (P7). Do not start a step until the previous step's
acceptance test is green and reproducible on a second run.

---

## M0 — Reproducible harness + frozen reference
**Goal.** Stand up the benchmark harness ([04-benchmarks.md](04-benchmarks.md)) and freeze the current Warp
result as the reference implementation and correctness oracle.
**Acceptance.** Harness emits the full reproducibility envelope (hardware/driver/CUDA/commit, ≥20 reps,
median+p5/p95, four timing layers) for Warp `access`/`rank`; results archived and versioned.
**Depends on.** Warp prototype (already measured: ~1.1 B/tok, ~400M access/s user-facing).

## M1 — CUDA C++ `access`, device-native async API
**Goal.** Port `GPUWavelet::access` to a CUDA C++ kernel behind `chromofold_access_async` (device pointers,
caller stream, no host copy — P5). Preserve the exact Python API via pybind11.
**Acceptance.** Bit-identical to Warp `access` on all corpora; profiler shows **no host allocation / no implicit
sync** in the query path; kernel-only and round-trip times reported separately.

## M2 — Baseline frontier for `access`
**Goal.** Compare CUDA C++ `access` against **raw GPU gather** (uint8/16/32) and Warp, across vocabulary widths
and batch sizes; add compile-time specialization by `levels` (§5 of architecture).
**Acceptance.** An Experiment-A table (raw gather vs packed vs RRR wavelet) with resident B/token, ns/access,
p95, effective bandwidth — stating plainly where addressability costs more than raw gather (small V) and where
it wins (search/sparse).

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
