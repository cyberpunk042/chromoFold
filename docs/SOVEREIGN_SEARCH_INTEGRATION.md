# ChromoFold ↔ sovereign-os — the FM-search integration (reconciled + verified)

The **one live ChromoFold integration**, reconciled 2026-07-23. Unlike the weight-fold consumer M21 names
(`sovereign-os` SDD-401/402, which [does not exist yet](m21-super-elastic-recursive-fold.md)), the **compressed-
domain search** lane is real on both sides: the ABI matches, and the chromoFold half is verified through the
shared library. This is what *both* projects prioritized — chromoFold's M7 + the spec-draft demo, and sovereign-os
SDD-400 **Lane A (FM-index-search-first)**.

## The ABI is aligned (chromoFold ships = sovereign-os mirrors)

| capability | chromoFold — [`chromofold_search.h`](../include/chromofold/chromofold_search.h) | sovereign-os (SDD-400/500) |
|---|---|---|
| `fm_count` (flagged `sovereign_os_first`) | `cf_fm_count_async(cf_fm_view, …)` | mirrored `cf_fm_count` in its `chromofold_search.h` |
| `fm_ranges` | `cf_fm_ranges_async(…)` — `[lo,hi)` per pattern | mirrored `cf_fm_ranges` |
| `fm_locate` | `cf_fm_locate_async(…)` — text positions | mirrored `cf_fm_locate` |
| capability descriptor | [`packaging/chromofold_capability.json`](../packaging/chromofold_capability.json) — `null_arg_contract`, `conformance_requires_gpu:false` | mirrored as the sovereign `CapabilityDescriptor` |

`packaging/Makefile` exists **for this**: "the `sovereign-chromofold-sys` FFI crate links this `.so`; CI runs
`make conformance` with no GPU."

## Verified live (this session, RTX 2080 Ti)
- **`make -C packaging functional`** → `libchromofold.so` serves `cf_fm_count` + `cf_fm_locate` **bit-identical to
  golden** over `fixtures/tiny.cffm` (n=8193, σ=65, 528 patterns). The engine side of the Lane-A binding works
  end-to-end **through the shared library** — the exact surface sovereign links.
- **`make conformance`** → the no-GPU ABI seam (`null_arg_contract`) sovereign CI runs.

## `predict` = a *derived* capability, resolved on both sides
Neither side has a native `cf_predict` (an early assumption both corrected). The speculative-decode **n-gram draft
model is derived** from `count`/`ranges`/`locate`:
- **chromoFold:** [`tests/spec_draft_demo.cu`](../tests/spec_draft_demo.cu) (`make spec-draft`) is the GPU-side
  proof — FM backward-search → `locate` → argmax next-token, **bit-matching an uncompressed hash n-gram baseline**
  (identical hit-rates), one compressed index for any context length.
- **sovereign-os:** provenance-B (CPU-native Rust `FmIndex`) has its own derived `predict`, verified against a
  reference cross-check (~4000 assertions).
- Both derive from the same FM primitive, so they agree — chromoFold's demo is a natural shared reference/golden.

## The step-7 gap — now unblocked from the engine side (2026-07-23)
sovereign-os's **linked** backend returned `NotImplemented` for `count`/`ranges`/`locate` because the host→device
marshalling (SDD-400 "step 7") wasn't wired, and the `-sys` crate exposes only the **device-native** async API
(which needs cudart + `cudaMalloc`/`Memcpy` on the caller). Rather than push that unsafe CUDA marshalling into the
Rust FFI, the engine now offers a **host-pointer FM-search layer** that hides it:

- **`cf_fm_host_load` / `cf_fm_host_count` / `cf_fm_host_ranges` / `cf_fm_host_locate` / `cf_fm_host_free`**
  ([`chromofold_search.h`](../include/chromofold/chromofold_search.h)) — build a device-resident index from a
  `.cffm` blob once (P9), then query with plain host arrays. All device marshalling is inside the (tested) engine.
- **Verified through the `.so` by a PURE-C caller** (no CUDA in the source — exactly a Rust `-sys`'s position):
  `make -C packaging functional-host` → `cf_fm_host_count` **bit-identical to golden** over `fixtures/tiny.cffm`
  (528 patterns), null-arg contract holds. See [`packaging/functional_host.c`](../packaging/functional_host.c);
  the capability is registered as `fm_host_search` in `chromofold_capability.json`.

So sovereign's step-7 collapses to: read the `.cffm` bytes → `cf_fm_host_load` → `cf_fm_host_count`/`locate` with
Rust slices → done, **no cudart in the FFI crate.** The device-native async API stays for a future zero-copy path.

### Wired + verified end-to-end on hardware (2026-07-23)
The sovereign side is now bound and the linked path **verified on a GPU**, not just compiled — in
`sovereign-chromofold-sys` (the sole unsafe carve-out; `sovereign-chromofold` forbids `unsafe`):
- the five `cf_fm_host_*` `extern "C"` decls + thin `unsafe` wrappers + opaque `CfFmHostIndex` (`cargo check`
  default **and** `--features linked`, `cargo test` 6 pass);
- a **safe `HostFmIndex` RAII wrapper** — host slices in / `Result<Vec<…>>` out, frees on drop, no `unsafe` crosses
  its API (clippy-clean);
- an `examples/host_fm_smoke.rs` that **ran against `libchromofold.so` on an RTX 2080 Ti**: loaded `tiny.cffm`,
  counted every single-symbol pattern via `HostFmIndex`, and confirmed the counts sum to `n=8193` (the global FM
  invariant) — **PASS.**

So the full cross-repo path is proven: sovereign Rust (safe `HostFmIndex`) → `-sys` FFI → `libchromofold.so` → GPU
FM-search → correct results. SDD-400 Lane A (FM-index-search-first) is demonstrably wired and correct. Reproduce:
`make -C packaging functional-host` (engine side) + the `cargo run … --example host_fm_smoke` command in that
file's header (linked side).

**The surface is now on the *safe* crate too** — `sovereign-chromofold` (the unsafe-forbidding crate the rest of
the workspace depends on) exposes `HostFmSearch` (`count`/`ranges`/`locate` over a `.cffm` blob), delegating to the
`-sys` `HostFmIndex`; opt-in via its `linked` feature, honest-degrading to `HostSearchError::Unavailable` otherwise.
Its `examples/host_search_smoke.rs` ran the **safe** surface end-to-end on the same GPU (invariant `n=8193` holds).
So provenance-A is no longer `-sys`-only: it sits beside the CPU-native `FmIndex` (provenance-B) as a peer backend,
both index-scoped, both agreeing with the same oracle. The only open item is the operator's call on whether the
`sovereign-chromofold` *default* search path routes through provenance-A (it stays opt-in, off by default).

## When to flip provenance-A (measured, RTX 2080 Ti)
The two backends are correctness-equivalent (`parity_smoke`: A == B == oracle, count **and** locate, 309 patterns
including the zero-occurrence path). So the routing decision is purely cost, and it has two axes — both measured, so
the operator decides on numbers, not vibes:

- **Memory.** The shippable searchable index is ~**6.8 b/tok** (converged, sa=1/16) — ~0.8 b/tok over bit-packed raw,
  the honest price of `count`/`locate`. See [`SEARCHABLE_WORKLOADS.md`](SEARCHABLE_WORKLOADS.md) ("Index footprint").
- **Throughput** (`sovereign-chromofold` `bench_search`, batched count, build/load excluded):

  | corpus | batch=1 | batch=4096 | crossover |
  |---|---|---|---|
  | 50 000 | CPU ~2000× | CPU 2.64M vs GPU 2.06M pat/s | **CPU wins everywhere** |
  | 1 000 000 | CPU ~2000× | **GPU 1.8M vs CPU 0.73M pat/s (2.5×)** | GPU wins only at large batch |

  The GPU path has a fixed ~1.5–2.3 ms per-call floor, so it plateaus ~1.8–2M pat/s regardless of batch; the CPU
  index degrades with corpus size (cache). **So provenance-A is a scale/capacity play — large GPU-resident corpus +
  large query batches — not a latency play.** For interactive or small-corpus search, provenance-B (CPU) is the
  right default.

  **What the fixed floor is — measured, not assumed (`bench_search_device`).** We built the device-native path
  (`DeviceFmIndex` over resident device buffers, `cf_fm_count_async` on a stream, zero host round-trip) expecting
  the host layer's per-call malloc/memcpy to be that floor — i.e. that removing it would move the crossover much
  earlier. **The measurement refuted that:** the device-native hot loop (query pre-resident, result kept on device)
  beats the `HostFmSearch` host path by only **1.07–1.41×**, and both sit at the same ~1.2–1.8 ms floor that is
  **batch-independent for K≥16**. So the floor is **launch/kernel-bound, not host-marshalling-bound** — zero-copy is
  a marginal win here, not the lever. The device-native path is correct, wired, and verified (`device_search`), and
  it is the right API for a device-resident producer/consumer pipeline; it just does **not** materially move this
  search workload's latency. Fewer, larger launches matter more than removing host copies. (An honest negative —
  it replaces the earlier speculation that the device-native API would shift the crossover.)

## Honest bottom line
Search-while-compressed is the through-line of everything this cycle produced (the O(n) memo, the spec-draft demo)
**and** it's the only ChromoFold capability sovereign-os has committed to bind first — because it's the one with no
existing analogue. The ABI is aligned, the engine half is verified through the `.so`, and `predict` reduces to the
FM primitive on both sides. Weight-folding (`.cfold`/`.secfold`) remains a promising, later, design-stage track.

## Reproduce
```sh
# chromoFold side (this repo)
make -C packaging functional     # GPU: cf_fm_count/locate through libchromofold.so vs golden
make -C packaging conformance    # no-GPU: the null_arg ABI seam sovereign CI runs
make spec-draft                  # the derived n-gram predict (FM draft vs hash baseline)
```
sovereign-os side: `docs/sdd/400-chromofold-compressed-domain-integration.md` + the `sovereign-chromofold-sys` crate.
