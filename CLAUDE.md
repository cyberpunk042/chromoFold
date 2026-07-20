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
| M6 fused decode+embedding-gather | ◐ | built, bit-identical; **honest negative** (fuse only large-intermediate) |
| M5 C++ builder + AVX CPU · M7 FM-search · Rust · PTX | ☐ | not started |

**Next:** wire RRR under the wavelet levels (+ two-level superblocks) → a natively entropy-sized searchable
index; then M5 (C++ builder + CPU backend), M7 (FM backward-search `count`/`locate`), pybind11. The
large-intermediate fused op (decode + sparse gather / KV-dequant) becomes buildable once RRR-wavelet lands.

## Build & run

```sh
# needs nvcc (tested 12.6) + an NVIDIA GPU. Default ARCH=sm_75; override e.g. ARCH=sm_80.
make all                                  # build every benchmark
# reference generation imports the prototype -> pass its venv python:
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python bench          # M1 verify + timing
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python experiment-a   # M2 frontier
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rank           # M3 rank + two-level
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python rrr            # M4 RRR frontier
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python fused          # M6 fused vs unfused
```

## Layout

```
include/chromofold/chromofold.h            the stable C ABI (cf_wavelet_view, cf_access_async, cf_rank_async, ...)
include/chromofold/detail/access_device.cuh  device-side wavelet decode (shared by access + fused kernels)
src/cuda/{access,rank,rrr,fused_embedding}.cu  the CUDA kernels
benchmarks/{gpu_access,frontier,rank_bench,rrr_bench,fused_embedding}.cu  verify + measure vs the frozen golden
benchmarks/reference_io.h                  the .cfwv v3 loader (shared)
tools/{export_reference,export_rrr}.py     freeze Warp golden vectors (import warp_compress from the prototype)
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
