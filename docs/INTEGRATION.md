# From engine to runtime — the integration, packaging, and release-qualification layers

The rest of the docs ([README](../README.md), [CONCEPTS](CONCEPTS.md), [RESULTS](RESULTS.md)) describe the
**engine**: the compressed, searchable, fusible data structures and their bit-verified kernels (milestones M0–M8).
This page maps the layers built **on top** of that engine to turn it into a real serving runtime and to gate a
release — the work that corresponds to the spec's **M9** ("integrate with one real inference path — the ship
criterion") and beyond.

> **Scope & honesty.** This is a survey. I (the engine-track author) **verified the RC1 CPU contract suites and
> the scenario hooks** directly (see the bottom of this page). The llama.cpp KV backend and the M10–M19
> productionization milestones were authored in a parallel track; I describe their **structure and intent from
> their own code, READMEs, schemas, and validators**, and do **not** claim to have independently reproduced their
> hardware results. Each area's authoritative artifact is linked so you can check it yourself.

---

## The three layers

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  RELEASE QUALIFICATION   RC1 serving adapter + runtime harness    │  <- gates a release (PASS/FAIL/INCOMPLETE)
   │                          + evidence validators + scenario hooks   │
   ├─────────────────────────────────────────────────────────────────┤
   │  RUNTIME INTEGRATION     integrations/llama.cpp/                   │  <- a real inference runtime uses the
   │                          compressed paged-KV backend + milestones  │     engine as its KV cache
   ├─────────────────────────────────────────────────────────────────┤
   │  PACKAGING               packaging/ libchromofold.so + C ABI       │  <- the linkable library boundary
   ├─────────────────────────────────────────────────────────────────┤
   │  ENGINE (this repo's core, M0–M8)   access · rank · RRR wavelet ·  │  <- what the other docs describe
   │                          FM-index · fused matmul/KV/gather · delta │
   └─────────────────────────────────────────────────────────────────┘
```

---

## 1. Packaging — `libchromofold.so` and the C ABI (`packaging/`)

The outward-facing library boundary a native runtime links against, built for **sovereign-os SDD-500**. It is
**additive** — it does not change the engine's algorithms or the benchmark build. Highlights:

- **`include/chromofold/chromofold.h`** — the primary stable C ABI: packed-wavelet `access`/`rank`, fused
  embedding gather, and the **compressed-KV causal windowed attention** (`cf_kv_attn_fused_async` + the
  bit-identical dense reference).
- **`include/chromofold/chromofold_search.h`** — the compressed-domain search surface: RRR-backed wavelet
  self-index + FM-index (`cf_fm_count_async`/`cf_fm_ranges_async`/`cf_fm_locate_async`).
- **`packaging/chromofold_capability.json`** — a committed capability descriptor (ABI version, exported functions,
  view structs, fixture formats, honest-degrade contract) a consumer can inspect **without loading CUDA**.
- **`packaging/conformance.c`** — a CPU-only FFI-signature seam: links the `.so` and asserts every entry point is
  exported with the right signature and NULL-argument error contract (guards return `CF_ERR_INVALID_ARGUMENT`
  before any CUDA call). **`seam_check.c` + `fixtures/tiny.cf*`** validate the on-disk header formats; a
  hardware-gated **`functional.cu`** runs FM-index search through the `.so` on a GPU.

Authoritative: [`packaging/README.md`](../packaging/README.md).

## 2. Runtime integration — the llama.cpp compressed-KV backend (`integrations/llama.cpp/`)

The version-sensitive boundary where a llama.cpp build can use ChromoFold as its **paged KV cache backend**
(`--kv-cache-backend chromofold`), instead of a dense cache. No llama.cpp source is vendored; ChromoFold is
optional and **disabled by default**, and a core rule is **honest-degrade**: after ChromoFold initializes it must
**never silently fall back** — unsupported models/layouts/GPU configs raise explicit errors (a `dense_fallback`
is a qualification FAIL). The KV lifecycle is page-based: `EMPTY → ACTIVE(partial page) → ENCODE+UPLOAD (sealed
compressed page)`, with move-only compressed-page ownership and explicit capability errors for unsupported
sequence ops.

On top of that backend sit a series of productionization milestones, each with an **evidence schema + validator**
(so their results are machine-checkable, in the same spirit as the engine's bit-identical golden checks):

| Area | Milestone | Evidence artifact |
|---|---|---|
| Multi-GPU | m15 | `multigpu/m15-evidence.schema.json` + `validate_m15_evidence.py` |
| Disaggregated prefill/decode | m16 | `disaggregated/m16-evidence.schema.json` |
| Scheduler | m17 | `scheduler/m17-evidence.schema.json` + `validate_m17_evidence.py` |
| Security | m18 | `security/m18-evidence.schema.json` |
| Multi-sequence / CUDA resolver | — | `multisequence/*-evidence.schema.json` + validators + server bridges |
| Distributed · adaptive · performance · e2e | — | `distributed/ · adaptive/ · performance/ · e2e/` schemas |
| Production server | — | `server/production-evidence.schema.json` + `run_production_workload.py` + `validate_production_evidence.py` |

Authoritative: [`integrations/llama.cpp/README.md`](../integrations/llama.cpp/README.md) and each area's schema +
validator. (These correspond to the parallel track's own M10–M19 numbering, which is distinct from the engine
[roadmap](../specs/03-roadmap.md)'s M0–M11.)

## 3. Release qualification — RC1 (`rc1-*.mk`, `tools/chromofold_*`, `integrations/llama.cpp/qualification/`)

The workflow that decides whether a `v1.0.0-rc1` tag is justified, driven by the operator guide
`docs/_assets/chromofold_rc1_local_machine_handoff.docx`. It has three parts:

- **Serving adapter** (`tools/chromofold_qualification_adapter.py`) — a secured, localhost-bound HTTP surface
  (`/live`, `/ready`, `/metrics`, `/qualification/snapshot`, `/qualification/scenario`) that reads a live runtime
  snapshot and runs scenario **hooks**.
- **Runtime harness** (`tools/chromofold_runtime_harness.py`) — drives a smoke (600 s) / candidate (2 h) / stable
  (24 h) run, collects **real GPU telemetry** (pynvml), executes the nine required scenarios, and emits a signed
  **evidence artifact** with a `PASS` / `FAIL` / `INCOMPLETE` decision + an incident bundle.
- **Evidence validators + CLI** (`integrations/llama.cpp/qualification/validate_rc1_runtime_evidence.py`,
  `tools/chromofold_qualify.py`) — the gate. Fabricated or incomplete evidence is **rejected**; a PASS requires
  matching digests, real GPU observations, executed scenarios, zero correctness/CUDA/tenant/dense-fallback
  violations, reconciled references, and verified mTLS + audit chain.
- **The nine scenario hooks** ([`integrations/llama.cpp/qualification/hooks/`](../integrations/llama.cpp/qualification/hooks/))
  — cache-cold/warm, multi-tenant-overload, cancellation, decode-worker-failure, network-fault,
  certificate-rotation, rolling-upgrade, rollback. They perform real operations and **never return PASS from a
  no-op** (see the [hooks README](../integrations/llama.cpp/qualification/hooks/README.md)).

### What I verified on this machine

- **RC1 CPU contract suites pass** — `make -f rc1-runtime-harness.mk rc1-harness-all`, `-f rc1-serving-adapter.mk
  rc1-serving-all`, `-f rc1-qualification.mk rc1-all` all green, including the **negative tests** that assert
  fabricated evidence is rejected.
- **The scenario hooks are contract-valid** — all nine emit valid JSON and, with no serving runtime present,
  correctly return `INCOMPLETE` (never a fabricated PASS), both standalone and through the adapter.
- **A smoke qualification run here is `INCOMPLETE`** — honestly, because this machine has **no serving-runtime
  binary and no weight model** (only tokenizer-vocab `gguf` files). The `/ready` endpoint returns 503 (no
  workers) and the harness writes `decision: INCOMPLETE, "runtime not ready"`. This is the correct outcome: the
  tooling refuses to fabricate a PASS. A real PASS needs a machine that owns the GPU, a weight model, and the
  running serving runtime, per the handoff.

### What remains for a real RC1 PASS

A validated `PASS` evidence artifact on real hardware: launch the serving runtime with a weight model, wire the
snapshot producer to observed state, configure the operator control commands the hooks call
(`CHROMOFOLD_HOOK_*_CMD`), run the smoke harness, and validate the evidence. Nothing about a green CPU contract
gate authorizes a release — only that validated artifact does.
