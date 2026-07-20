# chromoFold

Best understood as a GPU-resident succinct-data-structure runtime, not as a conventional compressor and not as a
literal DNA-folding. Its strongest idea is to keep token-addressable data compressed in GPU memory, support rank,
access, and search directly over that representation, and decode only the exact entries consumed by the next
model ops.

```
                 Python API / experiments
                          │
                          ▼
                 C++20 core + stable C ABI
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  CUDA C++ kernels   CPU AVX backend   optional Triton
  (hot path)         (baseline·build)  (experimental)
```

**The thesis:** decompression becomes a private step *inside* the consuming kernel — a full decompressed buffer
never exists. The proof is not a compression ratio; it is a real LLM fitting a larger batch or context in the
same GPU memory at equal-or-better latency.

## Start here (AI or human onboarding)

- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — how to build, current status, gotchas, the non-negotiables.
- [`docs/PROJECT_SYNC.md`](docs/PROJECT_SYNC.md) — how this native engine and the Python/Warp prototype
  (`~/warp-solar-system-shaders`) stay in sync and get compared.
- [`benchmarks/results.json`](benchmarks/results.json) — the measured ledger (per machine, Python vs C++), and
  [`benchmarks/dashboard.html`](benchmarks/dashboard.html) — the comparison dashboard it feeds.

## Planning & specification

Spec-driven development artifacts live in [`specs/`](specs/) — read the [index](specs/README.md) first:

- [00 · Constitution](specs/00-constitution.md) — the non-negotiable principles.
- [01 · Specification](specs/01-spec.md) — what it is, the object model, success criteria.
- [02 · Architecture](specs/02-architecture.md) — the layered stack, device-native API, C ABI, fusion thesis.
- [03 · Roadmap](specs/03-roadmap.md) — the M0→M11 migration, each with an acceptance test.
- [04 · Benchmarks](specs/04-benchmarks.md) — reproducibility envelope, baselines, experiments A–E.
- [05 · Porting Map](specs/05-porting-map.md) — the measured Warp prototype → native components.

## Status

Specified, pre-implementation. The measured prototype (Warp, 341 tests) lives in the sibling
`warp-solar-system-shaders/warp_compress` repository and is the correctness oracle and performance floor.
Implementation begins at **M0** (reproducible harness + frozen reference), then **M1** (CUDA C++ `access` behind
a device-native async API).
