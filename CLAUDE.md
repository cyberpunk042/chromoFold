# CLAUDE.md — chromoFold native engine

> Onboarding for an AI coding session. Read this first, then [`specs/README.md`](specs/README.md).
> Vendor-neutral copy: [`AGENTS.md`](AGENTS.md). Cross-project sync: [`docs/PROJECT_SYNC.md`](docs/PROJECT_SYNC.md).

## What this is

**chromoFold** is a **GPU-resident, random-access, searchable succinct-data-structure runtime** for LLM data
(token streams, KV history, adapter/prompt deltas, quantized-weight side-channels). It is *not* a file
compressor and *not* literal DNA-folding. It spends abundant GPU compute to buy back scarce VRAM/bandwidth, keeps
data **navigable while compressed**, and — the thesis — decodes **only inside the consuming kernel** so a full
decompressed buffer never exists.

This repo is the **native C++20 / CUDA C++ engine**. Its measured Python/Warp **prototype** lives in the sibling
repo `~/warp-solar-system-shaders/warp_compress` (341 tests) and is the **correctness oracle + performance
floor**. See [`docs/PROJECT_SYNC.md`](docs/PROJECT_SYNC.md).

## Governing docs (read in order)

| Doc | Purpose |
|---|---|
| [`specs/00-constitution.md`](specs/00-constitution.md) | Principles P1–P10. **A change that violates one is out of scope until the principle is amended.** |
| [`specs/01-spec.md`](specs/01-spec.md) | Object model, operations, scope, success criteria. |
| [`specs/02-architecture.md`](specs/02-architecture.md) | Layered stack, device-native API, C ABI, fusion thesis. |
| [`specs/03-roadmap.md`](specs/03-roadmap.md) | **M0→M11 with acceptance tests + measured results.** The live status board. |
| [`specs/04-benchmarks.md`](specs/04-benchmarks.md) | Reproducibility envelope, timing layers, baselines, experiments A–E. |
| [`specs/05-porting-map.md`](specs/05-porting-map.md) | Prototype module → native component + numbers to match/beat. |

## The non-negotiables (constitution, short form)

1. **Compute-for-memory** — justify features where memory, not compute, is the bottleneck.
2. **Navigable while compressed** — never break O(1)/O(log) access + search.
3. **Decode only inside the consumer** — no standalone full-decompression pass in the hot path.
4. **Lossless over the chosen quantization** — accuracy is the quantizer's job; addressing is ours.
5. **Device-native** — query APIs take/return device pointers, run on the caller's stream, no host copy/alloc/sync.
6. **Layered, not rewritten** — Python (research) · C++20 core + C ABI · CUDA C++ hot path · CPU AVX · opt Rust/Triton.
7. **Measured, not asserted** — every claim carries its envelope + an honest baseline. **Report negatives.**
8. **The C ABI is the contract.** 9. **Build ≠ query.** 10. **The proof is a workload, not a ratio.**

The honesty discipline is the brand. When a result is negative (e.g. M6 fusion lost), **say so and extract the
lesson** — do not bury or spin it.

## Status (see roadmap for detail)

| Milestone | State | Headline (RTX 2080 Ti, sm_75) |
|---|---|---|
| M0 harness + frozen Warp reference | ✅ | `.cfwv`/`.cfrr` golden vectors + envelope harness |
| M1 CUDA `access` + device-native API | ✅ | **bit-identical**, 1233 M access/s kernel (matches Warp) |
| M2 frontier (raw gather vs wavelet) | ◐ | price-of-addressability measured; pybind11 remains |
| M3 CUDA `rank` + two-level directory | ✅ | bit-identical; two-level **1.6–1.8× faster** |
| M4 RRR rank1 (entropy frontier) | ✅ | bit-identical; skewed **2.2–2.8× below packed**, ~0.7 ns |
| M4 RRR-backed wavelet access/rank | ✅ | bit-identical; BWT **5.80 b/tok < H₀**, 1.25–1.36× smaller; access **5.0 ns** (1-decode/level) |
| M4 block-rANS decode (near-entropy coder) | ✅ | bit-identical; **rANS 2.2× < Huffman** on low-entropy+large-block (0.55 vs 1.21 b/val); honest crossover |
| M7 FM count + locate (over RRR index) | ✅ | bit-identical to ground truth; count **~19 M patterns/s**, GPU-resident locate |
| M6 fused decode-in-GEMM (large-intermediate) | ✅ | bit-exact to golden; **10.6× less VRAM**, faster at 4096² (positive case) |
| M6/M9 fused KV-dequant attention | ✅ | bit-identical; **7.7–8.3× less KV VRAM** (KIVI int4+Huffman); capacity win, M9 brick |
| M6 fused sparse gather (P2, sparse consumer) | ✅ | bit-identical; **17.8× vs decompress-all** at 0.1% touched (~N/K); still 3.7× at K=N |
| M8 reference-delta cluster (cross-seq dedup) | ◐ | bit-identical reconstruct; **25.1× less VRAM**, 63× cheaper/turn; TTFT/max-batch need M9 |
| M6 fused embedding-gather (small-intermediate) | ◐ | built, bit-identical; **honest negative** (fuse only large-intermediate) |
| M5 C++ builder + CPU oracle (RRR-wavelet **+ FM**) | ◐ | native C++ build, **no Warp**; GPU==CPU bit-identical (access/rank **+ count/locate**) |
| M5 GPU suffix-array build | ✅ | bit-identical to CPU SA; **21–23× faster** at 2M (construction leaves the CPU) |
| M5 AVX/threaded CPU · Rust · PTX | ☐ | not started |

**Next:** RRR is wired under the wavelet levels (M4, 5.80 b/tok on the BWT) **and** FM backward-search rides on it
(M7: `cf_fm_count_async`/`cf_fm_ranges_async`/`cf_fm_locate_async`, count + locate bit-identical to ground truth,
~19 M patterns/s). The compact index is now decoded, searched, and sampled from without leaving VRAM. Remaining:
the native C++ builder (M5, `tools/build_index.cpp`) now builds **both** the RRR-wavelet index **and** the FM-index
with **no Warp**, and the GPU kernels are **bit-identical to its CPU oracle** (access/rank **and** count/locate) —
the whole searchable stack is build≠query, self-hosted; the **suffix-array build now runs on the GPU too** (M5,
bit-identical to CPU, **21–23× faster** — construction leaves the CPU). Remaining: AVX/threaded CPU backend;
pybind11; wiring the KV fusion into a real inference path (M9). Two
large-intermediate fusions have now landed, both reusing `cf_bh_decode_at`: int4 decode-in-GEMM (M6, **10.6× less
VRAM**, faster at 4096²) and **fused KV-dequant windowed attention (M6/M9, 7.7–8.3× less KV VRAM**, bit-identical —
the long-context capacity brick for M9).

## Build & run

```sh
# needs nvcc (tested 12.6) + an NVIDIA GPU. Default ARCH=sm_75; override e.g. ARCH=sm_80.
make all                                  # build every benchmark
# reference generation imports the prototype -> pass its venv python:
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python bench          # M1 verify + timing
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python experiment-a   # M2 frontier
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rank           # M3 rank + two-level
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rrr            # M4 RRR frontier
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rrr-wavelet    # M4 RRR-backed wavelet access/rank
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rans           # M4 block-rANS vs Huffman (entropy crossover)
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python fm-search      # M7 FM count + locate
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python fused-matmul   # M6 fused decode-in-GEMM (large-intermediate)
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python kv-attention   # M6/M9 fused KV-dequant windowed attention
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python sparse-gather  # M6/P2 fused sparse gather vs decompress-all
make delta                                # M8 reference-delta cluster decode (cross-seq dedup; exporter is Warp-free)
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python fused          # M6 fused embedding vs unfused
make build-index                          # M5 native C++ build (no Warp): RRR-wavelet + FM, GPU==CPU bit-identical
make suffix-array                         # M5 GPU suffix-array build vs CPU (bit-identical + speedup), no Python
```

## Layout

```
include/chromofold/chromofold.h            the stable C ABI (cf_wavelet_view, cf_access_async, cf_rank_async, ...)
include/chromofold/detail/access_device.cuh  device-side packed-wavelet decode (shared by access + fused kernels)
include/chromofold/detail/rrr_wavelet_device.cuh  device-side RRR-backed wavelet decode (cf_rrrw_view + access/rank)
include/chromofold/detail/fm_search_device.cuh  device-side FM backward-search + LF-walk locate (cf_fm_view)
include/chromofold/detail/block_huffman_device.cuh  device-side LUT decode-at-bit (feeds fused decode-in-GEMM)
src/cuda/{access,rank,rrr,rrr_wavelet,rans,fm_search,fused_embedding,fused_matmul,fused_kv_attention,sparse_gather,delta_apply}.cu  the CUDA kernels
tools/build_index.cpp                      native C++20 offline builder (SA→BWT→RRR→.cfrw + FM→.cffm), no Warp; CPU oracle
src/cuda/suffix_array.cu                    GPU suffix-array build (thrust prefix-doubling); include/.../detail/suffix_cpu.hpp = CPU golden
benchmarks/{gpu_access,frontier,rank_bench,rrr_bench,rrr_wavelet,fm_search,fused_embedding,fused_matmul}.cu  verify + measure
benchmarks/reference_io.h                  the .cfwv v3 loader (shared)
tools/{export_reference,export_rrr,export_rrr_wavelet,export_fm_index,export_fused_matmul}.py  freeze Warp golden
specs/                                     the SDD document set (constitution → roadmap → porting map)
```

## Gotchas (learned the hard way)

- **`make reference`/`experiment-a`/`rank`/`rrr` import `warp_compress`** from `~/warp-solar-system-shaders` —
  always pass `PYTHON=~/warp-solar-system-shaders/.venv/bin/python` (the system python lacks warp/torch).
- **The `Makefile` is force-added** — the C++ `.gitignore` ignores `Makefile` (CMake convention). Use
  `git add -f Makefile`. `build/` and `*.cfwv`/`*.cfrr` are git-ignored (regenerate them).
- **Format bumps invalidate old reference files** — `.cfwv` is at v3, `.cfrr` at v1. On a bump, `rm` the stale
  `benchmarks/*.cfwv benchmarks/refs/*` so the loader (which rejects other versions) regenerates them.
- **Don't put the `CK(...)` macro inside a typed lambda** — it expands to `return <int>`, conflicting with a
  `double`-returning timing lambda. Use plain CUDA calls inside `timeit`.
- **Two-level rank / RRR boundary:** a query at `pos == n` (n a multiple of 32) hits `word == nwords`; the fine
  array needs an `nwords+1` sentinel. The golden check catches this — trust it.
- **Every new kernel must verify bit-for-bit against a frozen Warp golden before any timing is trusted** (M0).

## Commit convention

Branch off `main` only if asked to; otherwise commit to `main`. Messages: what + the measured result + honest
caveat. End with:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
Git identity for this repo: `Cyberpunk 042 <jfm.devops.expert@gmail.com>`.
