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

## M3 — CUDA C++ `rank` + two-level rank directory — ✅ DONE
**Goal.** Port `rank`; implement and benchmark the two-level rank directory (§6) vs the word-scan layout.
**Delivered (`src/cuda/rank.cu`, `benchmarks/rank_bench.cu`).** `cf_rank_async` (device-native, templated by
level count) **bit-identical** to the Warp golden. Two rank directories measured head-to-head:
- *linear* (M1 layout): superblock every 8 words + scan ≤7 words + one partial popcount — 0.50 B/word directory.
- *two-level*: coarse int32 superblock (every 64 words) + a **per-word uint16 within-superblock prefix** → zero
  word scan — 2.06 B/word directory.

**Result (RTX 2080 Ti), both bit-identical:** two-level rank is **1.63× faster at V=256 (0.73→0.45 ns)** and
**1.82× at V=32768 (1.46→0.80 ns)** — the win grows with level count (more rank1 calls, more scan eliminated).
**Recorded decision:** adopt the two-level directory for `rank`/FM-search. The 4× larger directory is only ~1.35×
the *total* index (the bitplane dominates at 4 B/word), a clear trade for a query-bound workload — and it is a
per-deployment knob (P9). Needed an `nwords+1` fine-array sentinel (a query at pos=n hits word=nwords). Next:
wire the two-level layout into `cf_access` too, and add the RRR-coded frontier (M4).

## M4 — RRR-coded bitplanes on GPU — ✅ DONE (rank1 **and** RRR-backed wavelet access/rank)
**Goal.** Port RRR rank/decode so the resident index reaches the entropy frontier (the Warp prototype hit
5.80 b/tok on a BWT, below H₀).
**Delivered (`src/cuda/rrr.cu`, `benchmarks/rrr_bench.cu`).** Faithful CUDA port of the RRR bitvector rank1:
superblock jump + in-block class scan + ONE combinatorial in-register block decode (`cf_decode_word` unrank via
a 16×16 binomial table) + `__popc`. `.cfrr` reference format + `cf_rrr_rank_async`. **Bit-identical to the Warp
golden at every density.**

**Result (RTX 2080 Ti), the entropy memory-latency frontier:**
| density | H₀ | RRR b/bit | vs packed | rank1 |
|---|---|---|---|---|
| 0.50 | 1.00 | 1.167 | 0.86× | 0.84 ns |
| 0.10 | 0.469 | 0.669 | 1.50× | 0.71 ns |
| 0.03 | 0.194 | 0.447 | 2.24× | 0.66 ns |
| 0.005 | 0.045 | 0.353 | 2.83× | 0.63 ns |

Skewed planes compress **2.2–2.8× below packed** (toward H₀) while rank1 stays **~0.7 ns** — the combinatorial
decode is nearly free and *faster* on skewed planes (smaller offset widths). Honest bound: on max-entropy planes
RRR is *bigger* than packed (1.167 b/bit) — keep those packed. This is the BWT's profile (skewed → RRR wins),
the memory the FM-index rides on.

### M4 wavelet wiring — RRR *under* the wavelet levels (the entropy-sized searchable self-index) — ✅ DONE
**Delivered (`src/cuda/rrr_wavelet.cu`, `include/chromofold/detail/rrr_wavelet_device.cuh`,
`benchmarks/rrr_wavelet.cu`).** Every wavelet level is now an **RRR bitvector with two-level superblock samples**
(int32 anchor per K=32 superblocks + uint16 delta — M3's directory composed with M4's RRR). `cf_rrrw_access_async`
/ `cf_rrrw_rank_async` walk the levels doing, per level, a superblock jump + short class scan + **one combinatorial
in-register block decode**; `access` reads each level's bit as `rank1(i+1) − rank1(i)` (reusing the decode path).
Faithful port of the prototype's `RRRWaveletGPU`; new `.cfrw` v1 reference format frozen from a **BWT'd order-1
Markov stream** (the skewed regime where RRR wins). Device inlines are header-only so a token can be decoded
in-register inside a future fused consumer (P3).

**Result (RTX 2080 Ti, 2M-token BWT, 100K access + 100K rank queries), both BIT-IDENTICAL to the Warp golden:**
| vocab | levels | RRR b/tok | packed b/tok | smaller | access | rank |
|---|---|---|---|---|---|---|
| 64 | 7 | **5.80** (H₀ 5.96) | 7.88 | 1.36× | 9.67 ns | 6.90 ns |
| 256 | 9 | **8.09** | 10.13 | 1.25× | 12.12 ns | 9.07 ns |

Reproduces the prototype's headline (**5.80 b/tok on the BWT, below H₀**) exactly. **The honest price:** the
in-register RRR decode makes access ~12–15× slower than the packed wavelet (M1 ~0.8 ns) and rank ~10× slower than
the two-level packed rank (M3 ~0.45 ns) — access pays for two rank1 decodes per level. The trade is memory **and
capability**: a searchable index 1.25–1.36× smaller, reaching below H₀, still fully GPU-resident. This is the
compact BWT self-index M7's FM-search rides on. **Next:** M7 FM backward-search (`count`/`locate`) over this
`cf_rrrw_*` index; and the large-intermediate fused op M6 wants (RRR-decode + sparse gather / KV-dequant), which
this unlocks. A possible access speedup — read the level bit directly (one decode) instead of two rank1 calls —
is left as a benchmark-gated optimization.

## M5 — C++ builder + CPU verifier — ◐ builder + scalar CPU oracle DONE (AVX + GPU-SA build next)
**Goal.** A C++20 offline builder (suffix array, BWT, RRR, serialization) and an AVX2/AVX-512 CPU backend as the
correctness oracle and honest CPU baseline (§9). Establishes the build ≠ query split (P9).
**Acceptance.** CPU backend matches GPU bit-for-bit on a fuzz corpus; CPU `access`/`rank` throughput reported as
a real baseline (not a Python loop); builder round-trips the on-device format.
**Delivered (`tools/build_index.cpp`, one self-contained C++20 translation unit, **no Warp/Python**).** A pure-CPU
offline builder: structured stream → suffix array (prefix-doubling) → BWT → RRR-wavelet (per-level `rrr_encode` +
two-level superblock samples, faithful CPU ports of `gpu_rrr`) → serialised to the **exact `.cfrw` v1 format** the
GPU kernels consume. A scalar CPU `access`/`rank` backend is both the oracle (computes the golden and self-checks
that `access` reconstructs the BWT) and an honest single-thread baseline.
**Result (RTX 2080 Ti host, 2M-token BWT, structured stream):**
- **self-check:** CPU `access` == BWT ✓ (the builder validates its own encode).
- **cross-check (the acceptance test):** feed the C++-built `.cfrw` to `build/rrr_wavelet` (GPU) → GPU `access` and
  `rank` **BIT-IDENTICAL** to the CPU oracle golden. The whole pipeline — build (C++), query (CUDA), oracle (C++)
  — agrees bit-for-bit with **the Warp prototype removed from the build path.**
- **build cost (offline, P9):** suffix array 1.2 s + RRR encode 0.1 s. Index 0.92 MB (**3.67 b/tok**, 2.15× below
  packed 7.88 on this run-heavy stream; H₀ 6.00).
- **honest CPU baseline (real, not a Python loop):** scalar 1-thread `access` **1371 ns**, `rank` **1245 ns** — so
  the GPU (M4: ~10 ns access, ~6 ns rank) is ~130–200× faster; the CPU path is the correctness oracle, not the
  throughput path. The CPU path is the correctness oracle, not the throughput path.
- **FM-index too (M7 self-hosted):** the builder now also emits the **`.cffm`** FM-index (C-table + sampled-SA mark
  plane + `sval`) with a CPU `count`/`locate` oracle (backward search + LF-walk). Self-check: CPU `count` == naive
  **and** CPU `locate` == naive ✓; cross-check: `build/fm_search` (GPU) on the C++-built `.cffm` is **BIT-IDENTICAL**
  (count + locate) to the CPU oracle. **The entire searchable-index stack — M4 RRR-wavelet + M7 FM — now builds in
  native C++ with no Warp anywhere in the build path**, GPU query verified against the CPU oracle end to end.
  (Side note: this run's stream makes 4-grams frequent → 1.25 M occurrences, so GPU locate is well-utilized at
  ~174 ns/occ — confirming M7's caveat that its earlier low-occupancy 0.9–9.7 µs was overhead, not throughput.)

- **GPU suffix-array build (the last host straggler) — ✅ DONE** (`src/cuda/suffix_array.cu`,
  `benchmarks/suffix_array.cu`, `include/chromofold/detail/suffix_cpu.hpp`). Prefix-doubling on the device, exactly
  the CPU algorithm parallelised: each round forms a 64-bit composite key `(rank[i]<<32)|(rank[i+k]+1)`, radix-sorts
  the suffixes by it (thrust `sort_by_key`) and re-ranks by adjacent-key differences (a scan). The SA of a
  sentinel-terminated string is unique, so the device build is verified **BIT-IDENTICAL** to the CPU SA. **Result
  (RTX 2080 Ti):** at 2M tokens **21.7–23.3× faster** than the CPU build (1.0 s → 0.05 s) — in the prototype's
  17–32× range; bit-identical across sizes/vocabs/seeds. Honest scaling: the win grows with n (9.3× at 500K, only
  2.0× at 100K where launch/allocator overhead dominates). Construction now leaves the CPU.

**Deferred (honest):** AVX2/AVX-512 vectorization of the CPU backend (the RRR decode is data-dependent + gather-heavy,
so thread-parallelism is the realistic CPU lever — SIMD yield is doubtful, unclaimed). **Next:** wire the GPU SA into
an nvcc-built builder (the g++ `build_index` still calls the CPU SA); vectorize/parallelize the CPU query backend.

## M6 — Fused decode-and-consume (the thesis) — ✅ DONE (large-intermediate win; embedding-gather boundary)
**Goal.** The first fused decode-and-consume kernel (P3): decode token ID → gather embedding row → write
embedding, with **no decompressed token buffer** (Experiment D).
**Delivered (`src/cuda/fused_embedding.cu`, `benchmarks/fused_embedding.cu`).** Two fused mappings, both
**bit-identical** to the unfused path, both exposed device-native (`cf_embedding_gather_async`).
**Honest result (RTX 2080 Ti, V=32768, 100K queries):** unfused (two kernels) **wins at every dim** —
99–427 M emb/s vs fused-block 23–151 vs fused-tpq 2–118. Why: decode is inherently sequential (best mapping =
one thread/query, fully parallel) but the row copy wants block/query coalescing — **opposite ideal mappings**,
which the unfused two-kernel path satisfies independently. And embedding gather's avoided intermediate is tiny
(token ids, 4 B), so fusion saves no meaningful memory.
**The sharpened thesis (this is the value):** P3 fusion pays **when the avoided intermediate is LARGE** — e.g.
the prototype's decode-in-matmul avoids materializing the full dequantized weight matrix (**10.6× less VRAM**) —
**or the consumer is LIGHT/SPARSE** (KV-page selection, sparse gather, FM-candidate retrieval). *Fuse by how
expensive the intermediate is, not by fusing everything.* Embedding gather was the wrong showcase.
### M6 large-intermediate re-target — fused int4 decode-in-GEMM (the positive case) — ✅ DONE
**Delivered (`src/cuda/fused_matmul.cu`, `include/chromofold/detail/block_huffman_device.cuh`,
`benchmarks/fused_matmul.cu`).** The op the boundary above pointed to: decode block-Huffman int4 weights **inside**
the GEMM (one thread per output column decodes W's row from the compressed stream and multiply-accumulates it), so
the dequantized (M×K) matrix **never exists in VRAM** — only the compressed store is resident during compute (P3).
Faithful port of the prototype's `FusedDecodeMatmul`; new `.cffw` v1 reference freezes the store + input + golden.
The shared `cf_bh_decode_at` LUT decode (DFloat11-style, one table hit → symbol + length) is header-only, so the
same inline decode feeds future fused ops (KV-dequant, sparse gather).

**Result (RTX 2080 Ti, int4, block=64, batch=8), fused BIT-IDENTICAL to decode-then-dense AND bit-exact to the
Warp golden (`rel = 0`):**
| W (int4) | resident (compressed) | dense (dequantized) | less VRAM | fused | dense (decode+GEMM) |
|---|---|---|---|---|---|
| 2048×2048 | 1.58 MB | 16.78 MB | **10.6×** | 4.42 ms | 2.42 ms |
| 4096×4096 | 6.30 MB | 67.11 MB | **10.6×** | 6.02 ms | 7.92 ms |

Reproduces the prototype's **10.6× less VRAM** headline. Fusion is **numerically free** (bit-identical to
materializing W). **The richer, honest finding on speed:** at 2048² fused is ~1.8× *slower* (the inline decode
serializes on the compute path), but at **4096² fused is ~1.3× *faster*** — the dense path's 67 MB intermediate
roundtrip (decode→write 67 MB→read 67 MB in the GEMM) dominates, and fusion eliminates it. So for a
**large-intermediate, memory-bound, single-use** matmul, fusion wins **memory *and* latency**; the standing caveat
is that a dense W **reused across many matmuls** amortizes its one decode and beats the fused path's re-decode
(compute-for-memory, P1). This is the positive mirror of the embedding-gather negative: **fuse when the
intermediate is large.** Next large-intermediate targets: decode + KV-dequant, and RRR-decode + sparse gather.

### M6 large-intermediate #2 — fused KV-dequant attention (the KV-path, M9 prep) — ✅ DONE
**Delivered (`src/cuda/fused_kv_attention.cu`, `benchmarks/fused_kv_attention.cu`, `tools/export_kv_attention.py`).**
The KV-cache instance of the fusion thesis — the long-context memory bottleneck. K,V are **KIVI per-axis-quantized**
(K per-channel, V per-token, arXiv 2402.02750) then **block-Huffman entropy-coded**; a causal-windowed attention
kernel decodes + dequantizes each attended K/V row **inside** the attention (Q·Kᵀ → softmax → ·V, one thread per
query, two passes over its window), so the dense dequantized K/V tiles **never materialise** — only the compressed
KV store is resident, and only the *windowed* positions are ever decoded (the light/sparse-consumer branch).
Reuses the header-only `cf_bh_decode_at`. A decode-then-dense reference uses identical float ops in the same order.

**Result (RTX 2080 Ti, int4, block=64), fused BIT-IDENTICAL to decode-then-dense; vs numpy golden rel ~1e-7:**
| K,V (int4) | window | resident (compressed KV) | dense (dequant K+V) | less VRAM | fused | dense |
|---|---|---|---|---|---|---|
| 4096×64 | 256 | 0.27 MB | 2.10 MB | **7.7×** | 10.3 ms | 2.5 ms |
| 8192×128 | 512 | 1.02 MB | 8.39 MB | **8.3×** | 57.2 ms | 21.1 ms |

Fusion is **numerically free** (bit-identical). **The honest trade here is the pure compute-for-memory one (P1):**
`dim` is a multiple of `block`, so the row skip-decode is zero — the fused path is slower *only* because it
re-decodes each attended K/V row per query with no dense cache (total fused decodes ≈ nq·window vs the dense
path's one-pass seq decodes), while the small dense tile (2–8 MB) stays cached. So **the win is KV capacity — fit a
longer context / more layers in the same VRAM (7.7–8.3× less resident KV) — not speed at these sizes.** It flips
toward fusion when the dense KV tile is too large to cache (very long context × layers, bandwidth-bound) or the
attention is genuinely sparse (few attended positions ⇒ decode is cheap vs materializing the whole tile). This is
the concrete KV brick for **M9** (wire into a real KV path and measure batch/context at equal VRAM).

## M7 — FM backward-search + locate on GPU — ✅ DONE (count + locate, over the RRR index)
**Goal.** Port `count`/`locate` (backward search over the RRR/BWT wavelet). The Warp prototype already builds the
suffix array on-GPU (17–32× CPU) and does GPU count/locate/predict_next.
**Delivered (`src/cuda/fm_search.cu`, `include/chromofold/detail/fm_search_device.cuh`,
`benchmarks/fm_search.cu`).** FM-index backward search running GPU-resident over the **M4 RRR-backed BWT wavelet**
— the same compact, entropy-sized index is now decoded, searched, and (via batched count) sampled from without
leaving VRAM. Three device-native entry points, all riding the header-only `cf_rrrw_*` inlines (P3):
- `cf_fm_count_async` — backward search, one thread/pattern (parallel across a batch).
- `cf_fm_ranges_async` — the SA range `[lo, hi)` per pattern (locate's first stage).
- `cf_fm_locate_async` — one LF-walk thread per occurrence over a **succinct sampled suffix array** (packed mark
  plane ranked by `cf_rank1` + sampled SA values); `text_pos = (SA[sampled] + steps) mod n`.

New `.cffm` v1 reference bundles the RRR-wavelet arrays + FM `C[]` + the SA sample, with golden count **and**
locate computed by **naive ground-truth matching** (stronger than a prototype match — the prototype's *RRR*
FM-index exposes count only; native RRR-backed **locate** is verified against truth).

**Result (RTX 2080 Ti, 2M-token BWT), both count and locate BIT-IDENTICAL to ground truth:**
| vocab | levels | index | count (200K patterns) | locate |
|---|---|---|---|---|
| 64 | 7 | 1.45 MB | **18.5 M patterns/s** | correct (4264 occ) |
| 256 | 9 | 2.02 MB | **19.0 M patterns/s** | correct (538 occ) |

count is internally consistent with M4: ~52 ns/pattern ÷ 8 rank calls (len-4 pattern × 2) ≈ 6.5 ns/rank, matching
the 6.9 ns RRR-wavelet rank. **Honest caveats:** (1) each locate LF step is a *full* RRR access + rank walk (≤
`sa_sample`=16 steps), so locate is decode-heavy — the entropy index's price on the walk. (2) The measured locate
ns/occurrence (0.9–9.7 µs) is **overhead-dominated**, not throughput: only 538–4264 occurrences means 538–4264
threads, which badly underutilizes the GPU; a representative locate-throughput number needs a large occurrence
batch (deferred). count, batched to 200K patterns, is the well-utilized figure. **Next:** `predict_next` is just a
batched count over candidate next-tokens (host-side today); a locate-throughput sweep; and GPU suffix-array build
(M5) to close the last host straggler.

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
