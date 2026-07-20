# ChromoFold — Specification Set (SDD)

Spec-driven development artifacts for the native ChromoFold engine. Read in order; each governs the next.
Grounded in the **ChromoFold Research Brief (July 2026)** and the measured Warp prototype
(`warp-solar-system-shaders/warp_compress`, 341 tests).

| # | Document | Answers |
|---|---|---|
| 00 | [Constitution](00-constitution.md) | The non-negotiable principles (P1–P10). Amend before violating. |
| 01 | [Specification](01-spec.md) | **What** the engine is, its object model, operations, scope, and success criteria. |
| 02 | [Architecture](02-architecture.md) | **How** it is built: layered stack, device-native API, C ABI, fusion thesis. |
| 03 | [Roadmap](03-roadmap.md) | **When**: the M0→M11 migration, each with an acceptance test. |
| 04 | [Benchmarks](04-benchmarks.md) | **How we prove it**: envelope, timing layers, baselines, experiments A–E. |
| 05 | [Porting Map](05-porting-map.md) | Warp prototype → native component, with frozen reference numbers. |

## The one-paragraph thesis

ChromoFold is a **GPU-resident, random-access, searchable succinct-data-structure runtime** for LLM data. It
spends abundant GPU compute to buy back scarce VRAM and bandwidth (P1), keeps data **navigable while compressed**
(P2), and — the decisive idea — makes decompression a **private step inside the consuming kernel** so a full
decompressed buffer never exists (P3). It is built as a **layered stack** (Python research · C++20 core + C ABI ·
CUDA C++ hot path · CPU AVX baseline · optional Rust/Triton — P6), not a single-language rewrite. The proof is
**not a compression ratio**; it is a real LLM fitting a larger batch or context in the same GPU memory at
equal-or-better latency (P10).

## Status

- **Prototype:** measured, honest, 341 tests (Warp). It validates the primitives and is the correctness oracle.
- **Native engine:** specified here; implementation begins at **M0** (reproducible harness + frozen reference),
  then **M1** (CUDA C++ `access`, device-native async API). Do not skip the credibility gate (M0–M2).

## Reference

- `docs/_PDFs/ChromoFold_Research_Brief.pdf` (in the prototype repo) — the source brief.
- The prototype: `warp-solar-system-shaders/warp_compress` and its `docs/chromofold*.md`.
